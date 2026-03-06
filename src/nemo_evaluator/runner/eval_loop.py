from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Union

from nemo_evaluator.adapters.base import EnvironmentAdapter
from nemo_evaluator.environments.base import EvalEnvironment
from nemo_evaluator.metrics.aggregation import category_breakdown, summary_stats
from nemo_evaluator.metrics.confidence import bootstrap_ci
from nemo_evaluator.metrics.pass_at_k import aggregate_pass_at_k, pass_at_k
from nemo_evaluator.observability.collector import ArtifactCollector
from nemo_evaluator.observability.progress import NoOpProgress, ProgressTracker
from nemo_evaluator.observability.types import ModelResponse, StepRecord
from nemo_evaluator.runner.artifacts import build_artifact_bundle
from nemo_evaluator.runner.model_client import ModelClient

logger = logging.getLogger(__name__)
EvalSource = Union[EvalEnvironment, EnvironmentAdapter]


async def _seed(src: EvalSource, idx: int):
    if isinstance(src, EnvironmentAdapter):
        return await src.seed(idx)
    return await asyncio.to_thread(src.seed, idx)


async def _verify(src: EvalSource, response: str, expected: str, **meta):
    if isinstance(src, EnvironmentAdapter):
        return await src.verify(response, expected, **meta)
    return await asyncio.to_thread(src.verify, response, expected, **meta)


async def _size(src: EvalSource) -> int:
    if isinstance(src, EnvironmentAdapter):
        return await src.dataset_size()
    return len(src)


async def run_evaluation(
    env: EvalSource,
    client: ModelClient,
    n_repeats: int = 1,
    max_problems: int | None = None,
    system_prompt: str | None = None,
    config: dict[str, Any] | None = None,
    progress: ProgressTracker | None = None,
    problem_range: tuple[int, int] | None = None,
) -> dict[str, Any]:
    config = config or {}
    config.setdefault("repeats", n_repeats)
    config.setdefault("model", client.model)
    config.setdefault("base_url", client.base_url)

    name = env.name
    ds_size = await _size(env)
    if max_problems is not None:
        ds_size = min(ds_size, max_problems)

    if problem_range:
        start, end = problem_range
        end = min(end, ds_size)
        config["shard_range"] = [start, end]
    else:
        start, end = 0, ds_size

    pg = progress or NoOpProgress()
    collector = ArtifactCollector()

    n_problems = end - start
    pg.on_start(name, n_problems, n_repeats)
    logger.info("eval start: %s problems=%d [%d:%d] repeats=%d", name, n_problems, start, end, n_repeats)

    results: list[dict[str, Any]] = []
    problem_correct: dict[int, tuple[int, int]] = {}
    cum_correct = 0
    cum_total = 0
    t0 = time.monotonic()

    try:
        for idx in range(start, end):
            ts = time.monotonic()
            seed_result = await _seed(env, idx)
            seed_ms = (time.monotonic() - ts) * 1000

            n_correct = 0
            for rep in range(n_repeats):
                step = StepRecord(
                    problem_idx=idx,
                    repeat=rep,
                    prompt=seed_result.prompt,
                    system_prompt=system_prompt,
                    expected_answer=seed_result.expected_answer,
                    seed_metadata=seed_result.metadata,
                    seed_ms=seed_ms if rep == 0 else 0,
                )

                step_t0 = time.monotonic()
                resp: ModelResponse | None = None
                effective_system = system_prompt or seed_result.system
                try:
                    if seed_result.messages:
                        resp = await client.chat(messages=seed_result.messages)
                    else:
                        resp = await client.chat(seed_result.prompt, system=effective_system)
                    step.model_response = resp
                    step.model_ms = resp.latency_ms
                except Exception as e:
                    step.model_error = str(e)
                    logger.warning("model error p%d r%d: %s", idx, rep, e)

                response_text = resp.content if resp else ""

                tv = time.monotonic()
                vr = await _verify(env, response_text, seed_result.expected_answer, **seed_result.metadata)
                step.verify_ms = (time.monotonic() - tv) * 1000
                step.total_ms = (time.monotonic() - step_t0) * 1000

                step.reward = vr.reward
                step.extracted_answer = vr.extracted_answer
                step.scoring_details = vr.scoring_details
                step.scoring_method = vr.scoring_details.get("method", "")

                collector.record(step)

                if vr.reward > 0:
                    n_correct += 1
                    cum_correct += 1
                cum_total += 1

                results.append({
                    "problem_idx": idx,
                    "repeat": rep,
                    "reward": vr.reward,
                    "model_response": response_text,
                    "extracted_answer": vr.extracted_answer,
                    "expected_answer": seed_result.expected_answer,
                    "scoring_details": vr.scoring_details,
                    "metadata": {**seed_result.metadata, **vr.metadata},
                    "tokens": resp.total_tokens if resp else 0,
                    "latency_ms": resp.latency_ms if resp else 0,
                })

                pg.on_step(idx - start, rep, n_problems, n_repeats,
                           vr.reward, resp.total_tokens if resp else 0, step.total_ms)

            problem_correct[idx] = (n_repeats, n_correct)

    finally:
        await client.close()

    elapsed = time.monotonic() - t0
    pg.on_done(cum_correct, cum_total, elapsed, sum(
        s.model_response.total_tokens for s in collector.steps if s.model_response))

    problem_list = list(problem_correct.values())
    metrics: dict[str, Any] = {}
    for k in [1] + ([n_repeats] if n_repeats > 1 else []):
        if k <= n_repeats:
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
