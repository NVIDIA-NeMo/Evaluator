# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Core evaluation loop: seed -> solve -> verify, with async parallelism."""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from nemo_evaluator.environments.base import EvalEnvironment, VerifyResult
from nemo_evaluator.errors import GracefulError
from nemo_evaluator.metrics.aggregation import category_breakdown, scoring_details_breakdown, summary_stats
from nemo_evaluator.metrics.confidence import bootstrap_ci
from nemo_evaluator.metrics.pass_at_k import aggregate_pass_at_k, pass_at_k
from nemo_evaluator.observability.collector import ArtifactCollector
from nemo_evaluator.observability.progress import NoOpProgress, ProgressTracker
from nemo_evaluator.observability.types import StepRecord
from nemo_evaluator.engine.artifacts import build_artifact_bundle
from nemo_evaluator.sandbox.strategies import pick_lifecycle
from nemo_evaluator.solvers import Solver
from nemo_evaluator.engine.step_log import (
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
    skip_failed: bool = False,
    max_system_retries: int = 3,
    shard_info: tuple[int, int] | None = None,
    instruction_template: Path | None = None,
) -> dict[str, Any]:
    config = config or {}
    max_system_retries = max(1, max_system_retries)

    name = env.name
    ds_size = await env.dataset_size()
    if ds_size < 0:
        raise ValueError(
            f"Environment {name!r} returned invalid dataset_size={ds_size}. "
            "Ensure the environment is reachable and has a valid dataset."
        )
    if max_problems is not None:
        ds_size = min(ds_size, max_problems)

    if shard_info and not problem_range:
        from nemo_evaluator.engine.sharding import get_shard_range

        shard_idx, total_shards = shard_info
        problem_range = get_shard_range(ds_size, shard_idx, total_shards)
        config["shard"] = {
            "idx": shard_idx,
            "total": total_shards,
            "range": list(problem_range),
        }

    if problem_range:
        start, end = problem_range
        end = min(end, ds_size)
        config["shard_range"] = [start, end]
    else:
        start, end = 0, ds_size

    await env.prepare()

    if sandbox_manager:
        build_reqs = await env.image_build_requests()
        if build_reqs:
            await sandbox_manager.provision(build_reqs)
        specs = await env.sandbox_specs()
        if specs:
            await sandbox_manager.pre_pull(specs)

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
                    old_meta.get("config_hash", "?"),
                    cfg_hash,
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
            n_verify_only = len(inferred_cache) - len(set(inferred_cache) & set(verified_cache))
            if n_from_cache or n_verify_only:
                logger.info("resume: %d fully cached, %d verify-only, rest from scratch", n_from_cache, n_verify_only)
        else:
            inference_log.open(truncate=True)
            inference_log.write_meta({"config_hash": cfg_hash})
            verified_log.open(truncate=True)

    pg = progress or NoOpProgress()
    collector = ArtifactCollector()

    n_problems = end - start
    pg.on_start(name, n_problems, n_repeats)
    logger.info(
        "eval start: %s problems=%d [%d:%d] repeats=%d concurrency=%d",
        name,
        n_problems,
        start,
        end,
        n_repeats,
        max_concurrent,
    )

    if n_problems == 0:
        logger.warning("0 problems to evaluate — dataset may be missing or empty")
        pg.on_done(0, 0, 0.0, 0)
        return build_artifact_bundle(
            benchmark_name=name,
            results=[],
            metrics={
                "summary": summary_stats([]),
                "runtime": ArtifactCollector().build(0.0).runtime.to_dict(),
                "failures": ArtifactCollector().build(0.0).failures.to_dict(),
            },
            config=config,
            categories=None,
        )

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
            cached_verified = verified_cache.get(key)
            if cached_verified is not None:
                reward = cached_verified.get("reward", 0.0)
                tokens = cached_verified.get("tokens", 0)
                result_dict = {
                    "problem_idx": idx,
                    "repeat": rep,
                    "reward": reward,
                    "model_response": cached_verified.get("response", ""),
                    "extracted_answer": cached_verified.get("extracted_answer"),
                    "expected_answer": seed_result.expected_answer,
                    "scoring_details": cached_verified.get("scoring_details", {}),
                    "metadata": {**seed_result.metadata, **cached_verified.get("scoring_metadata", {})},
                    "tokens": tokens,
                    "latency_ms": cached_verified.get("latency_ms", 0),
                }
                async with lock:
                    results.append(result_dict)
                    if reward > 0:
                        cum_correct += 1
                    cum_total += 1
                    problem_correct.setdefault(idx, []).append(reward)
                pg.on_step(idx - start, rep, n_problems, n_repeats, reward, tokens, 0)
                return

            step = StepRecord(
                problem_idx=idx,
                repeat=rep,
                prompt=seed_result.prompt,
                expected_answer=seed_result.expected_answer,
                seed_metadata=seed_result.metadata,
                seed_ms=seed_ms if rep == 0 else 0,
            )

            step_t0 = time.monotonic()

            outside_eps: list[OutsideEndpoint] = []
            if model_url:
                from nemo_evaluator.sandbox.base import OutsideEndpoint as OE

                outside_eps.append(OE(url=model_url, env_var="MODEL_BASE_URL"))

            sandbox_cfg = config.get("_sandbox_config")
            vr: VerifyResult | None = None
            tv = step_t0

            for _attempt in range(1, max_system_retries + 1):
                solve_result = None
                sandbox = None
                response_text = ""
                tokens = 0
                latency_ms = 0.0
                step.model_error = None
                vr = None

                lifecycle = pick_lifecycle(
                    seed_result,
                    sandbox_manager,
                    outside_endpoints=outside_eps,
                    config_capture_cmd=sandbox_cfg.capture_cmd if sandbox_cfg else None,
                    verify_timeout=sandbox_cfg.verify_timeout if sandbox_cfg else 600.0,
                )

                try:
                    await lifecycle.setup()

                    # ── Solve ────────────────────────────────────────
                    cached_inferred = inferred_cache.get(key)
                    if cached_inferred is not None:
                        response_text = cached_inferred.get("response", "")
                        tokens = cached_inferred.get("tokens", 0)
                        latency_ms = cached_inferred.get("latency_ms", 0)
                        logger.debug("resume p%d r%d: using cached inference", idx, rep)
                    else:
                        pg.on_phase(idx - start, rep, n_problems, n_repeats, "solving")
                        try:
                            if _solver_accepts_sandbox(solver):
                                sandbox = await lifecycle.get_agent_sandbox()
                            if sandbox is not None:
                                solve_result = await solver.solve(seed_result, sandbox=sandbox)
                            else:
                                solve_result = await solver.solve(seed_result)
                            if solve_result.model_response:
                                step.model_response = solve_result.model_response
                                step.model_ms = solve_result.model_response.latency_ms
                            if solve_result.error:
                                logger.warning("solve error p%d r%d (graceful): %s", idx, rep, solve_result.error)
                                step.model_error = solve_result.error
                        except GracefulError as e:
                            step.model_error = str(e)
                            logger.warning("solve error p%d r%d (graceful): %s", idx, rep, e)
                        except Exception:
                            raise  # system error — outer loop will retry

                        response_text = solve_result.response if solve_result else ""
                        tokens = (
                            solve_result.model_response.total_tokens
                            if solve_result and solve_result.model_response
                            else 0
                        )
                        latency_ms = (
                            solve_result.model_response.latency_ms
                            if solve_result and solve_result.model_response
                            else 0
                        )

                        if inference_log is not None:
                            inf_record = {
                                "problem_idx": idx,
                                "repeat": rep,
                                "response": response_text,
                                "tokens": tokens,
                                "latency_ms": latency_ms,
                                "prompt": seed_result.prompt,
                                "expected_answer": seed_result.expected_answer,
                                "seed_metadata": seed_result.metadata,
                                "trajectory": solve_result.trajectory if solve_result else None,
                            }
                            await inference_log.append(inf_record)

                    # ── Verify ───────────────────────────────────────
                    _solve_failed = step.model_error is not None
                    if _solve_failed:
                        vr = VerifyResult(
                            reward=0.0,
                            scoring_details={
                                "error": step.model_error,
                                "error_category": "graceful",
                                "method": "solve_failed",
                            },
                        )
                        logger.info(
                            "p%d r%d: solver failed — grading 0.0 (skipping verify)",
                            idx,
                            rep,
                        )
                    else:
                        await lifecycle.transition_to_verify(
                            response_text,
                            solver_modified=(sandbox is not None),
                        )
                        pg.on_phase(idx - start, rep, n_problems, n_repeats, "verifying")
                    tv = time.monotonic()
                    if not _solve_failed and solve_result and solve_result.reward is not None:
                        vr = VerifyResult(
                            reward=solve_result.reward,
                            scoring_details=solve_result.scoring_details,
                        )
                        logger.debug("p%d r%d: using pre-computed reward=%.4f", idx, rep, vr.reward)
                    elif not _solve_failed:
                        verify_sandbox = await lifecycle.get_verify_sandbox()
                        vr = await env.verify(
                            response_text,
                            seed_result.expected_answer,
                            sandbox=verify_sandbox,
                            **seed_result.metadata,
                        )
                    break  # success — exit retry loop

                except GracefulError as e:
                    await lifecycle.teardown()
                    logger.warning(
                        "p%d r%d: graceful error, grading 0.0: %s",
                        idx,
                        rep,
                        e,
                    )
                    vr = VerifyResult(
                        reward=0.0,
                        scoring_details={
                            "error": str(e),
                            "error_category": "graceful",
                            "method": "graceful_error",
                        },
                    )
                    break

                except Exception as e:
                    await lifecycle.teardown()
                    if _attempt < max_system_retries:
                        delay = min(30, 5 * (2 ** (_attempt - 1)))
                        logger.warning(
                            "p%d r%d: system error (attempt %d/%d), retrying in %ds: %s",
                            idx,
                            rep,
                            _attempt,
                            max_system_retries,
                            delay,
                            e,
                        )
                        await asyncio.sleep(delay)
                        continue
                    logger.error(
                        "p%d r%d: system error exhausted %d retries: %s",
                        idx,
                        rep,
                        max_system_retries,
                        e,
                    )
                    if not skip_failed:
                        raise
                    vr = VerifyResult(
                        reward=0.0,
                        scoring_details={
                            "error": str(e),
                            "error_category": "system",
                            "retries_exhausted": max_system_retries,
                            "method": "system_error",
                        },
                    )
                    break
            try:
                step.verify_ms = (time.monotonic() - tv) * 1000
                step.total_ms = (time.monotonic() - step_t0) * 1000

                step.reward = vr.reward
                step.extracted_answer = vr.extracted_answer
                step.scoring_details = vr.scoring_details
                step.scoring_method = vr.scoring_details.get("method", "")

                extra_scorers = (config or {}).get("scorers", [])
                if extra_scorers:
                    from nemo_evaluator.scoring import get_scorer, ScorerInput

                    for scorer_name in extra_scorers:
                        try:
                            sfn = get_scorer(scorer_name)
                            sinput = ScorerInput(
                                response=response_text,
                                target=seed_result.expected_answer,
                                metadata=seed_result.metadata,
                            )
                            sresult = sfn(sinput)
                            sresult["reward"] = 1.0 if sresult.get("correct") else 0.0
                            vr.scoring_details[f"scorer:{scorer_name}"] = sresult
                        except Exception as e:
                            logger.warning("scorer %s error p%d r%d: %s", scorer_name, idx, rep, e)
                            vr.scoring_details[f"scorer:{scorer_name}"] = {"error": str(e), "reward": 0.0}

                if judge_client and vr.scoring_details.get("needs_judge"):
                    pg.on_phase(idx - start, rep, n_problems, n_repeats, "judging")
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
                        "problem_idx": idx,
                        "repeat": rep,
                        "reward": vr.reward,
                        "extracted_answer": vr.extracted_answer,
                        "scoring_details": vr.scoring_details,
                        "scoring_metadata": vr.metadata,
                        "response": response_text,
                        "tokens": tokens,
                        "latency_ms": latency_ms,
                    }
                    await verified_log.append(ver_record)

                scorer_rewards = {}
                for sk, sv in vr.scoring_details.items():
                    if sk.startswith("scorer:") and isinstance(sv, dict):
                        scorer_rewards[sk] = sv.get("reward", 0.0)

                result_dict = {
                    "problem_idx": idx,
                    "repeat": rep,
                    "reward": vr.reward,
                    "scorer_rewards": scorer_rewards,
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
                    step.trajectory = solve_result.trajectory

                async with lock:
                    collector.record(step)
                    results.append(result_dict)
                    if vr.reward > 0:
                        cum_correct += 1
                    cum_total += 1
                    problem_correct.setdefault(idx, []).append(vr.reward)

                pg.on_step(idx - start, rep, n_problems, n_repeats, vr.reward, tokens, step.total_ms)

            finally:
                await lifecycle.teardown()

    max_buffered = max_concurrent * 2
    pending: set[asyncio.Task] = set()

    first_error: BaseException | None = None

    def _check_done(done_tasks: set[asyncio.Task]) -> None:
        nonlocal first_error
        for t in done_tasks:
            if t.cancelled():
                continue
            exc = t.exception()
            if exc is None:
                continue
            logger.error(
                "Step failed (retries exhausted, will abort run): %s",
                exc,
            )
            if first_error is None:
                first_error = exc

    interrupted = False

    try:
        for idx in range(start, end):
            if first_error is not None:
                break

            while len(pending) >= max_buffered:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                _check_done(done)
                if first_error is not None:
                    break

            if first_error is not None:
                break

            ts = time.monotonic()
            seed_result = await env.seed(idx)
            seed_ms = (time.monotonic() - ts) * 1000

            if instruction_template is not None:
                from nemo_evaluator.templates import render_template

                wd = seed_result.sandbox_spec.workdir if seed_result.sandbox_spec else "/testbed"
                seed_result.prompt = render_template(
                    instruction_template,
                    original_prompt=seed_result.prompt,
                    workspace_path=wd,
                    metadata=seed_result.metadata,
                )

            for rep in range(n_repeats):
                task = asyncio.create_task(_run_step(idx, rep, seed_result, seed_ms))
                pending.add(task)

        if first_error is not None:
            for t in pending:
                t.cancel()
            if pending:
                await asyncio.wait(pending)
        elif pending:
            done, _ = await asyncio.wait(pending)
            _check_done(done)

    except (KeyboardInterrupt, asyncio.CancelledError):
        interrupted = True
        logger.warning("Interrupted — cancelling %d pending tasks", len(pending))
        for t in pending:
            t.cancel()
        if pending:
            await asyncio.wait(pending, timeout=30)

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
        if interrupted:
            raise KeyboardInterrupt()

    elapsed = time.monotonic() - t0

    if first_error is not None:
        raise RuntimeError(f"Evaluation aborted (skip_failed=False): {first_error}") from first_error

    all_rewards = [r["reward"] for r in results]
    per_problem_means = [sum(rewards) / len(rewards) for rewards in problem_correct.values() if rewards]
    has_fractional = any(r not in (0, 0.0, 1, 1.0) for r in all_rewards)

    metrics: dict[str, Any] = {}
    if per_problem_means and has_fractional:
        ci = bootstrap_ci(per_problem_means)
        metrics["mean_reward"] = {
            "value": round(ci.mean, 4),
            "ci_lower": round(ci.ci_lower, 4),
            "ci_upper": round(ci.ci_upper, 4),
        }

    overall_mean = metrics["mean_reward"]["value"] if "mean_reward" in metrics else None
    pg.on_done(
        cum_correct,
        cum_total,
        elapsed,
        sum(s.model_response.total_tokens for s in collector.steps if s.model_response),
        mean_reward=overall_mean,
    )

    # pass@k: useful for binary-scored benchmarks (code generation, etc.)
    problem_list = [(n_repeats, sum(1 for r in rewards if r > 0)) for rewards in problem_correct.values()]
    for k in [1] + ([n_repeats] if n_repeats > 1 else []):
        if k <= n_repeats and problem_list:
            pak = aggregate_pass_at_k(problem_list, k)
            ci = bootstrap_ci([pass_at_k(n, c, k) for n, c in problem_list])
            metrics[f"pass@{k}"] = {
                "value": round(pak, 4),
                "ci_lower": round(ci.ci_lower, 4),
                "ci_upper": round(ci.ci_upper, 4),
            }

    metrics["summary"] = summary_stats(all_rewards)

    cats = None
    if results and "category" in results[0].get("metadata", {}):
        cat_results = category_breakdown(results, "category")
        cats = [
            {"category": c.category, "n_samples": c.n_samples, "mean_reward": round(c.mean_reward, 4)}
            for c in cat_results
        ]

    sd_breakdowns = scoring_details_breakdown(results)
    if sd_breakdowns:
        metrics["breakdowns"] = {
            field: [
                {
                    "group": c.category,
                    "n": c.n_samples,
                    "mean_reward": round(c.mean_reward, 4),
                    "ci_lower": round(c.ci.ci_lower, 4),
                    "ci_upper": round(c.ci.ci_upper, 4),
                }
                for c in groups
            ]
            for field, groups in sd_breakdowns.items()
        }

    scorer_agg: dict[str, dict[str, Any]] = {}
    for r in results:
        sd = r.get("scoring_details", {})
        for key, val in sd.items():
            if not key.startswith("scorer:"):
                continue
            if not isinstance(val, dict):
                continue
            bucket = scorer_agg.setdefault(key, {"correct": 0, "total": 0})
            bucket["total"] += 1
            if val.get("correct"):
                bucket["correct"] += 1
    for key, agg in scorer_agg.items():
        n = agg["total"]
        c = agg["correct"]
        acc = c / n if n else 0.0
        ci = bootstrap_ci([1.0] * c + [0.0] * (n - c)) if n else bootstrap_ci([])
        metrics[key] = {
            "value": round(acc, 4),
            "ci_lower": round(ci.ci_lower, 4),
            "ci_upper": round(ci.ci_upper, 4),
            "correct": c,
            "total": n,
        }

    artifacts = collector.build(elapsed)
    metrics["runtime"] = artifacts.runtime.to_dict()
    metrics["failures"] = artifacts.failures.to_dict()

    bundle = build_artifact_bundle(
        benchmark_name=name,
        results=results,
        metrics=metrics,
        config=config,
        categories=cats,
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
