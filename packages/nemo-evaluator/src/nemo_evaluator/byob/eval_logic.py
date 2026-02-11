# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Shared BYOB evaluation logic for both subprocess and native modes."""

import importlib
import sys
from pathlib import Path
from typing import Callable, Dict, List

from nemo_evaluator.api.api_dataclasses import (
    EvaluationResult,
    MetricResult,
    Score,
    ScoreStats,
    TaskResult,
)
from nemo_evaluator.byob.decorators import (
    BenchmarkDefinition,
    clear_registry,
    get_registered_benchmarks,
)


def import_benchmark(
    benchmark_module: str, benchmark_name: str
) -> BenchmarkDefinition:
    """Import benchmark module and look up benchmark by name.

    Handles both file paths and module names. Clears registry first
    for a fresh state to prevent cross-evaluation pollution.

    Args:
        benchmark_module: Path to .py file or dotted module name.
        benchmark_name: Normalized benchmark name to look up.

    Returns:
        BenchmarkDefinition for the requested benchmark.

    Raises:
        ValueError: If benchmark not found after import.
    """
    clear_registry()

    module_path = Path(benchmark_module)
    if module_path.exists() and module_path.is_file():
        parent_dir = str(module_path.parent.absolute())
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        module_name = module_path.stem
    else:
        module_name = benchmark_module

    # Always reload to ensure fresh decorator execution
    if module_name in sys.modules:
        importlib.reload(sys.modules[module_name])
    else:
        importlib.import_module(module_name)

    benchmarks = get_registered_benchmarks()
    if benchmark_name not in benchmarks:
        available = ", ".join(benchmarks.keys())
        raise ValueError(
            f"Benchmark '{benchmark_name}' not found. Available: {available}"
        )
    return benchmarks[benchmark_name]


def run_eval_loop(
    bench: BenchmarkDefinition,
    dataset: List[Dict],
    model_call_fn: Callable[[str, str], str],
    endpoint_type: str,
) -> List[Dict]:
    """Core evaluation loop shared between subprocess and native modes.

    For each sample: renders prompt template with sample fields, calls model
    via model_call_fn, scores response with bench.scorer_fn. Skips samples
    on error (missing fields, model failures) rather than raising.

    Args:
        bench: Benchmark definition with prompt template and scorer.
        dataset: List of sample dicts loaded from JSONL.
        model_call_fn: Callable(prompt, endpoint_type) -> response_text.
        endpoint_type: "chat" or "completions".

    Returns:
        List of score dicts from scorer function (one per successfully-evaluated sample).
    """
    all_scores = []
    for idx, row in enumerate(dataset):
        # Render prompt
        try:
            prompt = bench.prompt.format(**row)
        except KeyError as e:
            # Sample missing required field - skip it
            import sys
            print(
                f"Warning: Sample {idx} missing field {e}, skipping",
                file=sys.stderr,
            )
            continue

        # Call model
        try:
            response = model_call_fn(prompt, endpoint_type)
        except Exception as e:
            # Model call failed - skip sample
            import sys
            print(
                f"Warning: Model call failed for sample {idx}: {e}, skipping",
                file=sys.stderr,
            )
            continue

        # Score
        target = row.get(bench.target_field, "")
        sample_scores = bench.scorer_fn(response, str(target), row)
        all_scores.append(sample_scores)

    return all_scores


def build_evaluation_result(
    scores: List[Dict],
    benchmark_name: str,
) -> EvaluationResult:
    """Convert aggregated scores into an EvaluationResult.

    Uses runner.aggregate_scores() for statistics computation, then constructs
    proper Pydantic models matching the engine's expected structure.

    Args:
        scores: Per-sample score dicts from run_eval_loop.
        benchmark_name: Name for the task in the output.

    Returns:
        EvaluationResult with tasks populated.
    """
    # Import here to avoid circular dependency
    from nemo_evaluator.byob.runner import aggregate_scores

    raw = aggregate_scores(scores, benchmark_name)
    if not raw or "tasks" not in raw:
        return EvaluationResult(tasks={})

    tasks = {}
    for task_name, task_data in raw["tasks"].items():
        metrics = {}
        for metric_name, metric_data in task_data.get("metrics", {}).items():
            metric_scores = {}
            for score_name, score_data in metric_data.get("scores", {}).items():
                metric_scores[score_name] = Score(
                    value=score_data["value"],
                    stats=ScoreStats(
                        count=score_data.get("count"),
                        mean=score_data.get("mean"),
                        stderr=score_data.get("stderr"),
                        stddev=score_data.get("stddev"),
                    ),
                )
            metrics[metric_name] = MetricResult(scores=metric_scores)
        tasks[task_name] = TaskResult(metrics=metrics)

    return EvaluationResult(tasks=tasks)
