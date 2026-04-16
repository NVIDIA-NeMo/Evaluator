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
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

from nemo_evaluator.metrics.aggregation import category_breakdown, summary_stats
from nemo_evaluator.metrics.confidence import bootstrap_ci, sample_level_ci
from nemo_evaluator.metrics.pass_at_k import aggregate_pass_at_k, pass_at_k

logger = logging.getLogger(__name__)


def get_shard_range(total: int, shard_idx: int, total_shards: int) -> tuple[int, int]:
    if total <= 0 or total_shards <= 0:
        return 0, 0
    if shard_idx < 0 or shard_idx >= total_shards:
        raise ValueError(f"shard_idx={shard_idx} out of range [0, {total_shards})")
    per_shard = total // total_shards
    remainder = total % total_shards
    start = shard_idx * per_shard + min(shard_idx, remainder)
    end = start + per_shard + (1 if shard_idx < remainder else 0)
    return start, min(end, total)


def shard_from_env() -> tuple[int, int] | None:
    idx = os.environ.get("NEL_SHARD_IDX") or os.environ.get("SLURM_ARRAY_TASK_ID")
    count = os.environ.get("NEL_TOTAL_SHARDS") or os.environ.get("SLURM_ARRAY_TASK_COUNT")
    if idx is None or count is None:
        return None
    try:
        return int(idx), int(count)
    except ValueError:
        logger.warning("Invalid shard env vars: NEL_SHARD_IDX=%s NEL_TOTAL_SHARDS=%s", idx, count)
        return None


def merge_results(shard_dirs: list[str | Path], output_dir: str | Path, n_repeats: int) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_results: list[dict[str, Any]] = []
    all_trajectories: list[str] = []
    configs: list[dict] = []
    all_runtime_stats: list[dict] = []

    for d in sorted(shard_dirs, key=str):
        d = Path(d)

        rp = d / "results.jsonl"
        if rp.exists():
            for line in rp.read_text(encoding="utf-8").strip().split("\n"):
                if line:
                    all_results.append(json.loads(line))

        tp = d / "trajectories.jsonl"
        if tp.exists():
            all_trajectories.extend(tp.read_text(encoding="utf-8").strip().split("\n"))

        for bp in d.glob("eval-*.json"):
            b = json.loads(bp.read_text(encoding="utf-8"))
            configs.append(b.get("config", {}))

        sp = d / "runtime_stats.json"
        if sp.exists():
            all_runtime_stats.append(json.loads(sp.read_text(encoding="utf-8")))

    if not all_results:
        logger.warning("No results found in %d shard dirs", len(shard_dirs))
        return {}

    # Use ACTUAL per-problem repeat counts, not the global n_repeats arg.
    # This handles partial shards correctly.
    problem_counts: dict[int, list[int]] = {}  # {idx: [correct, total]}
    for r in all_results:
        idx = r["problem_idx"]
        if idx not in problem_counts:
            problem_counts[idx] = [0, 0]
        problem_counts[idx][1] += 1
        if r.get("reward", 0) > 0:
            problem_counts[idx][0] += 1

    problem_list = [(actual_total, correct) for correct, actual_total in problem_counts.values()]

    metrics: dict[str, Any] = {}
    for k in [1] + ([n_repeats] if n_repeats > 1 else []):
        valid = [(n, c) for n, c in problem_list if n >= k]
        if valid:
            pak = aggregate_pass_at_k(valid, k)
            entry: dict[str, Any] = {"value": round(pak, 4)}
            if k == 1:
                sci = sample_level_ci(valid)
                if sci is not None:
                    entry["ci_lower"] = round(sci.ci_lower, 4)
                    entry["ci_upper"] = round(sci.ci_upper, 4)
            bci = bootstrap_ci([pass_at_k(n, c, k) for n, c in valid])
            entry["bootstrap_ci_lower"] = round(bci.ci_lower, 4)
            entry["bootstrap_ci_upper"] = round(bci.ci_upper, 4)
            metrics[f"pass@{k}"] = entry

    metrics["summary"] = summary_stats([r.get("reward", 0) for r in all_results])

    if all_runtime_stats:
        merged_rt: dict[str, Any] = {
            "total_steps": sum(s.get("total_steps", 0) for s in all_runtime_stats),
            "total_tokens": sum(s.get("total_tokens", 0) for s in all_runtime_stats),
            "total_prompt_tokens": sum(s.get("total_prompt_tokens", 0) for s in all_runtime_stats),
            "total_completion_tokens": sum(s.get("total_completion_tokens", 0) for s in all_runtime_stats),
            "elapsed_seconds": max((s.get("elapsed_seconds", 0) for s in all_runtime_stats), default=0),
            "model_errors": sum(s.get("model_errors", 0) for s in all_runtime_stats),
            "total_retries": sum(s.get("total_retries", 0) for s in all_runtime_stats),
            "n_shards": len(all_runtime_stats),
        }
        # Average the percentile latencies across shards
        percs = [s.get("latency_percentiles_ms", {}) for s in all_runtime_stats]
        valid_percs = [p for p in percs if p]
        if valid_percs:
            merged_rt["latency_percentiles_ms"] = {
                "p50": round(float(np.mean([p.get("p50", 0) for p in valid_percs])), 2),
                "p90": round(float(np.mean([p.get("p90", 0) for p in valid_percs])), 2),
                "p99": round(float(np.mean([p.get("p99", 0) for p in valid_percs])), 2),
            }
        total_elapsed = merged_rt["elapsed_seconds"]
        if total_elapsed > 0:
            merged_rt["tokens_per_second"] = round(merged_rt["total_tokens"] / total_elapsed, 2)
            merged_rt["steps_per_second"] = round(merged_rt["total_steps"] / total_elapsed, 2)

        metrics["runtime"] = merged_rt

    cats = None
    if all_results and "category" in all_results[0].get("metadata", {}):
        cr = category_breakdown(all_results, "category")
        cats = [{"category": c.category, "n_samples": c.n_samples, "mean_reward": round(c.mean_reward, 4)} for c in cr]

    from nemo_evaluator.engine.artifacts import build_artifact_bundle

    config = configs[0] if configs else {}
    config["n_shards"] = len(shard_dirs)
    bundle = build_artifact_bundle(
        benchmark_name=config.get("benchmark", "merged"),
        results=all_results,
        metrics=metrics,
        config=config,
        categories=cats,
    )

    bp = out / f"{bundle['run_id']}.json"
    bp.write_text(json.dumps(bundle, indent=2, default=str), encoding="utf-8")

    rp = out / "results.jsonl"
    with rp.open("w", encoding="utf-8") as f:
        for r in all_results:
            f.write(json.dumps(r, default=str) + "\n")

    if all_trajectories:
        tp = out / "trajectories.jsonl"
        tp.write_text("\n".join(all_trajectories) + "\n", encoding="utf-8")

    return bundle
