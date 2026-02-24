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
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Protocol, Tuple, Union

from nemo_evaluator.api.api_dataclasses import (
    EvaluationResult,
    MetricResult,
    Score,
    ScoreStats,
    TaskResult,
)
from nemo_evaluator.byob.decorators import (
    BenchmarkDefinition,
    PromptType,
    ScorerInput,
    clear_registry,
    get_registered_benchmarks,
)
from nemo_evaluator.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SampleResult:
    """Per-sample evaluation result for --save-predictions output.

    Attributes:
        sample_id: Zero-based index of the sample in the dataset.
        prompt: Rendered prompt string or message array sent to the model.
        response: Model response text (last turn), or None if model call failed.
        target: Ground-truth target value from the dataset.
        scores: Score dict from the scorer function, or None if not scored.
        status: One of "scored", "skipped_missing_field", "skipped_model_error".
        error: Error message string if the sample was skipped.
        metadata: The full sample row from the dataset.
        trajectory: Full multi-turn conversation history, if applicable.
    """

    sample_id: int
    prompt: PromptType
    response: Optional[str]
    target: str
    scores: Optional[dict]
    status: str
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    trajectory: Optional[List[Dict[str, str]]] = None


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


class EvalStrategy(Protocol):
    """Protocol for evaluation strategies.

    Different evaluation modes (standard, judge, multi-turn, agentic)
    implement this protocol to define how each sample is evaluated.
    """

    def evaluate_sample(
        self,
        idx: int,
        row: Dict,
        bench: BenchmarkDefinition,
        model_call_fn: Callable[[PromptType, str], str],
        endpoint_type: str,
    ) -> Tuple[Optional[Dict], Optional[SampleResult]]:
        """Evaluate a single sample.

        Returns:
            Tuple of (scores_dict_or_None, sample_result_or_None).
            Return (None, SampleResult with error status) on failure.
        """
        ...


class StandardStrategy:
    """Standard single-turn evaluation: render prompt -> call model -> score."""

    def evaluate_sample(
        self,
        idx: int,
        row: Dict,
        bench: BenchmarkDefinition,
        model_call_fn: Callable[[PromptType, str], str],
        endpoint_type: str,
    ) -> Tuple[Optional[Dict], Optional[SampleResult]]:
        # Render prompt
        try:
            prompt = bench.prompt.format(**row)
        except KeyError as e:
            target = row.get(bench.target_field, "")
            return None, SampleResult(
                sample_id=idx, prompt="", response=None,
                target=str(target), scores=None,
                status="skipped_missing_field", error=str(e), metadata=row,
            )

        # Call model
        try:
            response = model_call_fn(prompt, endpoint_type)
        except Exception as e:
            target = row.get(bench.target_field, "")
            return None, SampleResult(
                sample_id=idx, prompt=prompt, response=None,
                target=str(target), scores=None,
                status="skipped_model_error", error=str(e), metadata=row,
            )

        # Score using ScorerInput
        target = row.get(bench.target_field, "")
        scorer_input = ScorerInput(
            response=response,
            target=str(target),
            metadata=row,
            model_call_fn=model_call_fn,
            config=bench.extra_config,
        )
        try:
            sample_scores = bench.scorer_fn(scorer_input)
        except Exception as e:
            return None, SampleResult(
                sample_id=idx, prompt=prompt, response=response,
                target=str(target), scores=None,
                status="skipped_scorer_error", error=str(e), metadata=row,
            )

        prediction = SampleResult(
            sample_id=idx, prompt=prompt, response=response,
            target=str(target), scores=sample_scores,
            status="scored", metadata=row,
        )
        return sample_scores, prediction


class MultiTurnStrategy:
    """Multi-turn evaluation: iterate through turns, build conversation, score at end.

    Expects the dataset row to have a ``turns`` field (list of prompt strings)
    or a ``prompt`` field that is a list. Each turn is sent as a message array
    with full conversation history. The scorer receives the final response and
    the full trajectory.
    """

    def __init__(self, turns_field: str = "prompt"):
        self.turns_field = turns_field

    def evaluate_sample(
        self,
        idx: int,
        row: Dict,
        bench: BenchmarkDefinition,
        model_call_fn: Callable[[PromptType, str], str],
        endpoint_type: str,
    ) -> Tuple[Optional[Dict], Optional[SampleResult]]:
        turns = row.get(self.turns_field, [])
        if not turns or not isinstance(turns, list):
            return None, SampleResult(
                sample_id=idx, prompt="", response=None,
                target=str(row.get(bench.target_field, "")), scores=None,
                status="skipped_missing_field",
                error=f"No turns found in field '{self.turns_field}'",
                metadata=row,
            )

        conversation: List[Dict[str, str]] = []
        responses: List[str] = []

        for turn_idx, turn_prompt in enumerate(turns):
            conversation.append({"role": "user", "content": str(turn_prompt)})
            try:
                response = model_call_fn(conversation, endpoint_type)
            except Exception as e:
                return None, SampleResult(
                    sample_id=idx, prompt=conversation, response=None,
                    target=str(row.get(bench.target_field, "")), scores=None,
                    status="skipped_model_error", error=str(e),
                    metadata=row, trajectory=list(conversation),
                )
            conversation.append({"role": "assistant", "content": response})
            responses.append(response)

        # Score using the last response and full trajectory
        target = row.get(bench.target_field, "")
        scorer_input = ScorerInput(
            response=responses[-1],
            target=str(target),
            metadata=row,
            model_call_fn=model_call_fn,
            config=bench.extra_config,
            conversation=list(conversation),
            turn_index=len(turns),
        )
        try:
            sample_scores = bench.scorer_fn(scorer_input)
        except Exception as e:
            return None, SampleResult(
                sample_id=idx, prompt=conversation, response=responses[-1],
                target=str(target), scores=None,
                status="skipped_scorer_error", error=str(e),
                metadata=row, trajectory=list(conversation),
            )

        prediction = SampleResult(
            sample_id=idx, prompt=conversation, response=responses[-1],
            target=str(target), scores=sample_scores,
            status="scored", metadata=row, trajectory=list(conversation),
        )
        return sample_scores, prediction


def run_eval_loop(
    bench: BenchmarkDefinition,
    dataset: List[Dict],
    model_call_fn: Callable[[PromptType, str], str],
    endpoint_type: str,
    strategy: Optional[EvalStrategy] = None,
    save_predictions: bool = False,
    show_progress: bool = True,
    max_consecutive_errors: int = 0,
    fail_on_skip: bool = False,
    parallelism: int = 1,
) -> Tuple[List[Dict], List[SampleResult]]:
    """Core evaluation loop shared between subprocess and native modes.

    Delegates per-sample evaluation to the provided strategy (defaults to
    StandardStrategy).  The loop handles bookkeeping: progress logging,
    consecutive-error tracking, fail-on-skip, and prediction collection.

    Args:
        bench: Benchmark definition with prompt template and scorer.
        dataset: List of sample dicts loaded from JSONL.
        model_call_fn: Callable(prompt, endpoint_type) -> response_text.
        endpoint_type: "chat" or "completions".
        strategy: EvalStrategy implementation for per-sample evaluation.
            Defaults to StandardStrategy when None.
        save_predictions: If True, collect per-sample SampleResult objects.
        show_progress: If True, log progress periodically.
        max_consecutive_errors: If > 0, abort after N consecutive failures.
        fail_on_skip: If True, raise RuntimeError on any skipped sample.
        parallelism: Number of concurrent evaluation threads. When > 1,
            samples are evaluated in parallel using a ThreadPoolExecutor.

    Returns:
        Tuple of (all_scores, all_predictions).
    """
    if strategy is None:
        strategy = StandardStrategy()

    if parallelism > 1 and len(dataset) > 1:
        return _run_eval_loop_parallel(
            bench=bench,
            dataset=dataset,
            model_call_fn=model_call_fn,
            endpoint_type=endpoint_type,
            strategy=strategy,
            save_predictions=save_predictions,
            show_progress=show_progress,
            max_consecutive_errors=max_consecutive_errors,
            fail_on_skip=fail_on_skip,
            parallelism=parallelism,
        )

    return _run_eval_loop_sequential(
        bench=bench,
        dataset=dataset,
        model_call_fn=model_call_fn,
        endpoint_type=endpoint_type,
        strategy=strategy,
        save_predictions=save_predictions,
        show_progress=show_progress,
        max_consecutive_errors=max_consecutive_errors,
        fail_on_skip=fail_on_skip,
    )


def _run_eval_loop_sequential(
    bench: BenchmarkDefinition,
    dataset: List[Dict],
    model_call_fn: Callable[[PromptType, str], str],
    endpoint_type: str,
    strategy: EvalStrategy,
    save_predictions: bool = False,
    show_progress: bool = True,
    max_consecutive_errors: int = 0,
    fail_on_skip: bool = False,
) -> Tuple[List[Dict], List[SampleResult]]:
    """Sequential evaluation loop (original behavior)."""
    all_scores = []
    all_predictions: List[SampleResult] = []
    total = len(dataset)
    consecutive_errors = 0
    scored_count = 0
    skipped_count = 0

    progress_interval = max(1, min(10, total // 10)) if total > 0 else 1

    for idx, row in enumerate(dataset):
        scores, prediction = strategy.evaluate_sample(
            idx, row, bench, model_call_fn, endpoint_type
        )

        if scores is None:
            # Sample failed
            skipped_count += 1
            consecutive_errors += 1
            if save_predictions and prediction:
                all_predictions.append(prediction)
            msg = f"Sample {idx} failed: {prediction.error if prediction else 'unknown'}"
            logger.warning("Sample failed", sample_id=idx, error=prediction.error if prediction else "unknown")
            if fail_on_skip:
                raise RuntimeError(msg)
            if max_consecutive_errors > 0 and consecutive_errors >= max_consecutive_errors:
                raise RuntimeError(
                    f"Aborting: {consecutive_errors} consecutive errors "
                    f"reached limit of {max_consecutive_errors}"
                )
            continue

        # Success
        consecutive_errors = 0
        all_scores.append(scores)
        scored_count += 1
        if save_predictions and prediction:
            all_predictions.append(prediction)

        # Progress
        if show_progress and total > 0:
            processed = idx + 1
            if processed % progress_interval == 0 or processed == total:
                logger.info(
                    "Evaluation progress",
                    processed=processed, total=total,
                    pct=round((processed / total) * 100),
                    scored=scored_count, skipped=skipped_count,
                )

    if show_progress:
        logger.info(
            "Evaluation complete",
            scored=scored_count, skipped=skipped_count, total=total,
        )

    return all_scores, all_predictions


def _run_eval_loop_parallel(
    bench: BenchmarkDefinition,
    dataset: List[Dict],
    model_call_fn: Callable[[PromptType, str], str],
    endpoint_type: str,
    strategy: EvalStrategy,
    save_predictions: bool = False,
    show_progress: bool = True,
    max_consecutive_errors: int = 0,
    fail_on_skip: bool = False,
    parallelism: int = 4,
) -> Tuple[List[Dict], List[SampleResult]]:
    """Parallel evaluation loop using ThreadPoolExecutor.

    Maintains sample ordering in results regardless of completion order.
    Thread-safe bookkeeping via a lock for counters and error tracking.
    """
    total = len(dataset)

    # Pre-allocate result slots indexed by sample position
    results: List[Optional[Tuple[Optional[Dict], Optional[SampleResult]]]] = [None] * total

    # Thread-safe counters
    lock = threading.Lock()
    scored_count = 0
    skipped_count = 0
    consecutive_errors = 0
    abort_error: Optional[RuntimeError] = None

    progress_interval = max(1, min(10, total // 10)) if total > 0 else 1

    def evaluate_one(idx: int, row: Dict) -> Tuple[int, Optional[Dict], Optional[SampleResult]]:
        scores, prediction = strategy.evaluate_sample(
            idx, row, bench, model_call_fn, endpoint_type
        )
        return idx, scores, prediction

    with ThreadPoolExecutor(max_workers=parallelism) as executor:
        futures = {}
        for idx, row in enumerate(dataset):
            future = executor.submit(evaluate_one, idx, row)
            futures[future] = idx

        for future in as_completed(futures):
            idx = futures[future]
            try:
                _, scores, prediction = future.result()
            except Exception as e:
                # Unexpected exception from the thread itself
                scores = None
                prediction = SampleResult(
                    sample_id=idx, prompt="", response=None,
                    target="", scores=None,
                    status="skipped_model_error", error=str(e),
                )

            results[idx] = (scores, prediction)

            with lock:
                if scores is None:
                    skipped_count += 1
                    consecutive_errors += 1
                    msg = f"Sample {idx} failed: {prediction.error if prediction else 'unknown'}"
                    logger.warning("Sample failed", sample_id=idx, error=prediction.error if prediction else "unknown")
                    if fail_on_skip and abort_error is None:
                        abort_error = RuntimeError(msg)
                    if (max_consecutive_errors > 0
                            and consecutive_errors >= max_consecutive_errors
                            and abort_error is None):
                        abort_error = RuntimeError(
                            f"Aborting: {consecutive_errors} consecutive errors "
                            f"reached limit of {max_consecutive_errors}"
                        )
                else:
                    consecutive_errors = 0
                    scored_count += 1

                if show_progress and total > 0:
                    processed = scored_count + skipped_count
                    if processed % progress_interval == 0 or processed == total:
                        logger.info(
                            "Evaluation progress",
                            processed=processed, total=total,
                            pct=round((processed / total) * 100),
                            scored=scored_count, skipped=skipped_count,
                        )

    # Check for abort conditions after all futures complete
    if abort_error is not None:
        raise abort_error

    # Collect results in original order
    all_scores = []
    all_predictions: List[SampleResult] = []
    for idx in range(total):
        result_pair = results[idx]
        if result_pair is None:
            continue
        scores, prediction = result_pair
        if scores is not None:
            all_scores.append(scores)
        if save_predictions and prediction is not None:
            all_predictions.append(prediction)

    if show_progress:
        logger.info(
            "Evaluation complete",
            scored=scored_count, skipped=skipped_count, total=total,
        )

    return all_scores, all_predictions


def build_evaluation_result(
    scores: List[Dict],
    benchmark_name: str,
) -> EvaluationResult:
    """Convert aggregated scores into an EvaluationResult.

    Uses runner.aggregate_scores() for statistics computation, then constructs
    proper Pydantic models matching the engine's expected structure.
    """
    from nemo_evaluator.byob.aggregation import aggregate_scores

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
