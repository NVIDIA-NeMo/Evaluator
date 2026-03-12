"""Core evaluation loop: seed -> solve -> verify, with async parallelism."""
from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from nemo_evaluator.environments.base import EvalEnvironment, VerifyResult
from nemo_evaluator.metrics.aggregation import category_breakdown, summary_stats
from nemo_evaluator.metrics.confidence import bootstrap_ci
from nemo_evaluator.metrics.pass_at_k import aggregate_pass_at_k, pass_at_k
from nemo_evaluator.observability.collector import ArtifactCollector
from nemo_evaluator.observability.progress import NoOpProgress, ProgressTracker
from nemo_evaluator.observability.types import StepRecord
from nemo_evaluator.runner.artifacts import build_artifact_bundle
from nemo_evaluator.solvers import Solver, SolveResult
from nemo_evaluator.runner.step_log import (
    INFERENCE_LOG,
    VERIFIED_LOG,
    StepLog,
    config_hash,
)

if TYPE_CHECKING:
    from nemo_evaluator.sandbox.base import OutsideEndpoint
    from nemo_evaluator.sandbox.manager import SandboxManager

logger = logging.getLogger(__name__)

DEFAULT_MAX_CONCURRENT = 32


async def run_evaluation(
    env: EvalEnvironment,
    solver: Solver,
    n_repeats: int = 1,
    max_problems: int | None = None,
    config: dict[str, Any] | None = None,
    progress: ProgressTracker | None = None,
    problem_range: tuple[int, int] | None = None,
    max_concurrent: int = DEFAULT_MAX_CONCURRENT,
    judge_client: Any = None,
    sandbox_manager: SandboxManager | None = None,
    model_url: str | None = None,
    step_log_dir: Path | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    config = config or {}

    name = env.name
    ds_size = await env.dataset_size()
    if ds_size < 0:
        raise ValueError(f"Environment {name!r} returned invalid dataset_size={ds_size}. "
                         "Ensure the environment is reachable and has a valid dataset.")
    if max_problems is not None:
        ds_size = min(ds_size, max_problems)

    if problem_range:
        start, end = problem_range
        end = min(end, ds_size)
        config["shard_range"] = [start, end]
    else:
        start, end = 0, ds_size

    # Pre-pull sandbox images if available
    if sandbox_manager:
        specs = await env.sandbox_specs()
        if specs:
            await sandbox_manager.pre_pull(specs)

    # --- Step log setup ---
    inference_log: StepLog | None = None
    verified_log: StepLog | None = None
    inferred_cache: dict[tuple[int, int], dict[str, Any]] = {}
    verified_cache: dict[tuple[int, int], dict[str, Any]] = {}

    if step_log_dir is not None:
        inference_log = StepLog(step_log_dir / INFERENCE_LOG)
        verified_log = StepLog(step_log_dir / VERIFIED_LOG)

        cfg_hash = config_hash(config)

        if resume:
            old_meta = inference_log.load_meta()
            if old_meta and old_meta.get("config_hash") != cfg_hash:
                logger.warning(
                    "Config changed since last run (old=%s new=%s). "
                    "Inference cache invalidated; verified cache retained.",
                    old_meta.get("config_hash", "?"), cfg_hash,
                )
                verified_cache = verified_log.load()
                if verified_cache:
                    verified_log.compact(verified_cache)
                verified_log.open()
                inference_log.open(truncate=True)
                inference_log.write_meta({"config_hash": cfg_hash})
            else:
                inferred_cache = inference_log.load()
                verified_cache = verified_log.load()
                if inferred_cache:
                    meta = old_meta or {"config_hash": cfg_hash}
                    inference_log.compact(inferred_cache, meta=meta)
                if verified_cache:
                    verified_log.compact(verified_cache)
                inference_log.open()
                verified_log.open()

            n_from_cache = len(verified_cache)
            n_verify_only = len(inferred_cache) - len(
                set(inferred_cache) & set(verified_cache))
            if n_from_cache or n_verify_only:
                logger.info("resume: %d fully cached, %d verify-only, rest from scratch",
                            n_from_cache, n_verify_only)
        else:
            inference_log.open(truncate=True)
            inference_log.write_meta({"config_hash": cfg_hash})
            verified_log.open(truncate=True)

    pg = progress or NoOpProgress()
    collector = ArtifactCollector()

    n_problems = end - start
    pg.on_start(name, n_problems, n_repeats)
    logger.info("eval start: %s problems=%d [%d:%d] repeats=%d concurrency=%d",
                name, n_problems, start, end, n_repeats, max_concurrent)

    results: list[dict[str, Any]] = []
    problem_correct: dict[int, list[float]] = {}
    lock = asyncio.Lock()
    cum_correct = 0
    cum_total = 0
    t0 = time.monotonic()

    sem = asyncio.Semaphore(max_concurrent)

    async def _run_step(idx: int, rep: int, seed_result, seed_ms: float):
        nonlocal cum_correct, cum_total
        async with sem:
            key = (idx, rep)

            # --- Fast path: fully verified in cache ---
            cached_verified = verified_cache.get(key)
            if cached_verified is not None:
                reward = cached_verified.get("reward", 0.0)
                tokens = cached_verified.get("tokens", 0)
                result_dict = {
                    "problem_idx": idx, "repeat": rep,
                    "reward": reward,
                    "model_response": cached_verified.get("response", ""),
                    "extracted_answer": cached_verified.get("extracted_answer"),
                    "expected_answer": seed_result.expected_answer,
                    "scoring_details": cached_verified.get("scoring_details", {}),
                    "metadata": {**seed_result.metadata,
                                 **cached_verified.get("scoring_metadata", {})},
                    "tokens": tokens,
                    "latency_ms": cached_verified.get("latency_ms", 0),
                }
                async with lock:
                    results.append(result_dict)
                    if reward > 0:
                        cum_correct += 1
                    cum_total += 1
                    problem_correct.setdefault(idx, []).append(reward)
                pg.on_step(idx - start, rep, n_problems, n_repeats,
                           reward, tokens, 0)
                return

            step = StepRecord(
                problem_idx=idx, repeat=rep,
                prompt=seed_result.prompt,
                expected_answer=seed_result.expected_answer,
                seed_metadata=seed_result.metadata,
                seed_ms=seed_ms if rep == 0 else 0,
            )

            step_t0 = time.monotonic()
            solve_result: SolveResult | None = None
            sandbox = None
            response_text = ""
            tokens = 0
            latency_ms = 0.0

            # Acquire per-problem sandbox if configured
            if sandbox_manager:
                spec = sandbox_manager.resolve_spec(seed_result)
                if spec:
                    outside_eps: list[OutsideEndpoint] = []
                    if model_url:
                        from nemo_evaluator.sandbox.base import OutsideEndpoint as OE
                        outside_eps.append(OE(url=model_url, env_var="MODEL_BASE_URL"))
                    sandbox = await sandbox_manager.acquire(spec, outside_endpoints=outside_eps)

            try:
                # --- Inference phase ---
                cached_inferred = inferred_cache.get(key)
                if cached_inferred is not None:
                    response_text = cached_inferred.get("response", "")
                    tokens = cached_inferred.get("tokens", 0)
                    latency_ms = cached_inferred.get("latency_ms", 0)
                    logger.debug("resume p%d r%d: using cached inference", idx, rep)
                else:
                    try:
                        if sandbox is not None and hasattr(solver, "solve") and _solver_accepts_sandbox(solver):
                            solve_result = await solver.solve(seed_result, sandbox=sandbox)
                        else:
                            solve_result = await solver.solve(seed_result)
                        if solve_result.model_response:
                            step.model_response = solve_result.model_response
                            step.model_ms = solve_result.model_response.latency_ms
                    except Exception as e:
                        step.model_error = str(e)
                        logger.warning("solve error p%d r%d: %s", idx, rep, e)

                    response_text = solve_result.response if solve_result else ""
                    tokens = solve_result.model_response.total_tokens if solve_result and solve_result.model_response else 0
                    latency_ms = solve_result.model_response.latency_ms if solve_result and solve_result.model_response else 0

                    if inference_log is not None:
                        inf_record = {
                            "problem_idx": idx, "repeat": rep,
                            "response": response_text,
                            "tokens": tokens, "latency_ms": latency_ms,
                            "prompt": seed_result.prompt,
                            "expected_answer": seed_result.expected_answer,
                            "seed_metadata": seed_result.metadata,
                            "trajectory": solve_result.trajectory if solve_result else None,
                        }
                        await inference_log.append(inf_record)

                # --- Verify phase ---
                tv = time.monotonic()
                try:
                    vr = await env.verify(
                        response_text, seed_result.expected_answer,
                        sandbox=sandbox, **seed_result.metadata,
                    )
                except Exception as e:
                    logger.warning("verify error p%d r%d: %s", idx, rep, e)
                    vr = VerifyResult(reward=0.0, scoring_details={"error": str(e), "method": "verify_failed"})
                step.verify_ms = (time.monotonic() - tv) * 1000
                step.total_ms = (time.monotonic() - step_t0) * 1000

                step.reward = vr.reward
                step.extracted_answer = vr.extracted_answer
                step.scoring_details = vr.scoring_details
                step.scoring_method = vr.scoring_details.get("method", "")

                if judge_client and vr.scoring_details.get("needs_judge"):
                    try:
                        from nemo_evaluator.scoring.judge import judge_score
                        judge_result = await judge_score(
                            instruction=seed_result.prompt,
                            response=response_text,
                            expected=seed_result.expected_answer,
                            client=judge_client,
                        )
                        step.scoring_details["judge"] = judge_result
                        if "normalized" in judge_result:
                            step.reward = judge_result["normalized"]
                            vr.reward = judge_result["normalized"]
                    except Exception as e:
                        logger.warning("judge error p%d r%d: %s", idx, rep, e)

                if verified_log is not None:
                    ver_record = {
                        "problem_idx": idx, "repeat": rep,
                        "reward": vr.reward,
                        "extracted_answer": vr.extracted_answer,
                        "scoring_details": vr.scoring_details,
                        "scoring_metadata": vr.metadata,
                        "response": response_text,
                        "tokens": tokens, "latency_ms": latency_ms,
                    }
                    await verified_log.append(ver_record)

                result_dict = {
                    "problem_idx": idx, "repeat": rep,
                    "reward": vr.reward,
                    "model_response": response_text,
                    "extracted_answer": vr.extracted_answer,
                    "expected_answer": seed_result.expected_answer,
                    "scoring_details": vr.scoring_details,
                    "metadata": {**seed_result.metadata, **vr.metadata},
                    "tokens": tokens,
                    "latency_ms": latency_ms,
                }
                if solve_result and solve_result.trajectory:
                    result_dict["trajectory"] = solve_result.trajectory

                async with lock:
                    collector.record(step)
                    results.append(result_dict)
                    if vr.reward > 0:
                        cum_correct += 1
                    cum_total += 1
                    problem_correct.setdefault(idx, []).append(vr.reward)

                pg.on_step(idx - start, rep, n_problems, n_repeats, vr.reward, tokens, step.total_ms)

            finally:
                if sandbox and sandbox_manager:
                    await sandbox_manager.release(sandbox)

    # Pipeline: seed problems and fan out repeats with back-pressure
    max_buffered = max_concurrent * 2
    pending: set[asyncio.Task] = set()

    try:
        for idx in range(start, end):
            while len(pending) >= max_buffered:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for t in done:
                    if t.exception():
                        logger.error("Step failed: %s", t.exception())

            ts = time.monotonic()
            seed_result = await env.seed(idx)
            seed_ms = (time.monotonic() - ts) * 1000

            for rep in range(n_repeats):
                task = asyncio.create_task(_run_step(idx, rep, seed_result, seed_ms))
                pending.add(task)

        if pending:
            done, _ = await asyncio.wait(pending)
            for t in done:
                if t.exception():
                    logger.error("Step failed: %s", t.exception())

    finally:
        if inference_log is not None:
            inference_log.close()
        if verified_log is not None:
            verified_log.close()
        if sandbox_manager:
            await sandbox_manager.shutdown()
        if hasattr(solver, "close"):
            await solver.close()
        if hasattr(env, "close"):
            await env.close()

    elapsed = time.monotonic() - t0
    pg.on_done(cum_correct, cum_total, elapsed, sum(
        s.model_response.total_tokens for s in collector.steps if s.model_response))

    problem_list = [
        (n_repeats, sum(1 for r in rewards if r > 0))
        for rewards in problem_correct.values()
    ]
    metrics: dict[str, Any] = {}
    for k in [1] + ([n_repeats] if n_repeats > 1 else []):
        if k <= n_repeats and problem_list:
            pak = aggregate_pass_at_k(problem_list, k)
            ci = bootstrap_ci([pass_at_k(n, c, k) for n, c in problem_list])
            metrics[f"pass@{k}"] = {
                "value": round(pak, 4),
                "ci_lower": round(ci.ci_lower, 4),
                "ci_upper": round(ci.ci_upper, 4),
            }

    metrics["summary"] = summary_stats([r["reward"] for r in results])

    cats = None
    if results and "category" in results[0].get("metadata", {}):
        cat_results = category_breakdown(results, "category")
        cats = [{"category": c.category, "n_samples": c.n_samples,
                 "mean_reward": round(c.mean_reward, 4)} for c in cat_results]

    artifacts = collector.build(elapsed)
    metrics["runtime"] = artifacts.runtime.to_dict()
    metrics["failures"] = artifacts.failures.to_dict()

    bundle = build_artifact_bundle(
        benchmark_name=name, results=results, metrics=metrics,
        config=config, categories=cats,
    )
    bundle["_results"] = results
    bundle["_artifacts"] = artifacts
    return bundle


def _solver_accepts_sandbox(solver: Any) -> bool:
    """Check if a solver's solve() method accepts a 'sandbox' keyword argument."""
    import inspect
    try:
        sig = inspect.signature(solver.solve)
        return "sandbox" in sig.parameters
    except (ValueError, TypeError):
        return False
