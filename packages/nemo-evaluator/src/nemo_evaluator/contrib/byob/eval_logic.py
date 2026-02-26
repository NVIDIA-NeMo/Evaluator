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
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

from nemo_evaluator.contrib.byob.decorators import (
    BenchmarkDefinition,
    ScorerInput,
    clear_registry,
    get_registered_benchmarks,
)
from nemo_evaluator.logging import get_logger

logger = get_logger(__name__)

# Module-level Jinja2 environment singleton (lazy-initialized)
_jinja2_env = None


def _get_jinja2_env():
    """Return a cached jinja2.Environment (created on first call)."""
    global _jinja2_env
    if _jinja2_env is None:
        import jinja2

        _jinja2_env = jinja2.Environment(undefined=jinja2.StrictUndefined)
    return _jinja2_env


def render_prompt(template: str, row: Dict, is_jinja2: bool = False) -> str:
    """Render a prompt template with the given row data.

    Args:
        template: The prompt template string.
        row: Dict of field values to substitute.
        is_jinja2: If True, render with Jinja2 engine; otherwise use str.format().

    Returns:
        Rendered prompt string.
    """
    if is_jinja2:
        return _get_jinja2_env().from_string(template).render(**row)
    return template.format(**row)


@dataclass
class SampleResult:
    """Per-sample evaluation result for --save-predictions output.

    Attributes:
        sample_id: Zero-based index of the sample in the dataset.
        prompt: Rendered prompt string sent to the model.
        response: Model response text, or None if model call failed.
        target: Ground-truth target value from the dataset.
        scores: Score dict from the scorer function, or None if not scored.
        status: One of "scored", "skipped_missing_field", "skipped_model_error".
        error: Error message string if the sample was skipped.
        metadata: The full sample row from the dataset.
    """

    sample_id: int
    prompt: str
    response: Optional[str]
    target: Any
    scores: Optional[dict]
    status: str
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)


def import_benchmark(benchmark_module: str, benchmark_name: str) -> BenchmarkDefinition:
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
        model_call_fn: Callable[[str, str], str],
        endpoint_type: str,
        request_timeout: Optional[float] = None,
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
        model_call_fn: Callable[[str, str], str],
        endpoint_type: str,
        request_timeout: Optional[float] = None,
    ) -> Tuple[Optional[Dict], Optional[SampleResult]]:
        # Render prompt
        try:
            prompt = render_prompt(bench.prompt, row, bench._is_jinja2)
        except KeyError as e:
            target = row.get(bench.target_field, "")
            return None, SampleResult(
                sample_id=idx,
                prompt="",
                response=None,
                target=target,
                scores=None,
                status="skipped_missing_field",
                error=str(e),
                metadata=row,
            )

        # Render system prompt if configured
        rendered_system_prompt = None
        if bench.system_prompt:
            try:
                rendered_system_prompt = render_prompt(
                    bench.system_prompt, row, bench._is_system_prompt_jinja2
                )
            except KeyError as e:
                logger.warning(
                    "System prompt rendering failed, skipping",
                    sample_id=idx,
                    error=str(e),
                )

        # Call model
        try:
            response = model_call_fn(
                prompt,
                endpoint_type,
                system_prompt=rendered_system_prompt,
                timeout=request_timeout,
            )
        except Exception as e:
            target = row.get(bench.target_field, "")
            return None, SampleResult(
                sample_id=idx,
                prompt=prompt,
                response=None,
                target=target,
                scores=None,
                status="skipped_model_error",
                error=str(e),
                metadata=row,
            )

        # Score using ScorerInput
        target = row.get(bench.target_field, "")
        scorer_input = ScorerInput(
            response=response,
            target=target,
            metadata=row,
            model_call_fn=model_call_fn,
            config=bench.extra_config,
        )
        try:
            sample_scores = bench.scorer_fn(scorer_input)
        except Exception as e:
            return None, SampleResult(
                sample_id=idx,
                prompt=prompt,
                response=response,
                target=target,
                scores=None,
                status="skipped_scorer_error",
                error=str(e),
                metadata=row,
            )

        prediction = SampleResult(
            sample_id=idx,
            prompt=prompt,
            response=response,
            target=target,
            scores=sample_scores,
            status="scored",
            metadata=row,
        )
        return sample_scores, prediction


class EvalOnlyStrategy:
    """Eval-only strategy: use pre-generated responses from the dataset instead of calling the model."""

    def __init__(self, response_field: str):
        self.response_field = response_field

    def evaluate_sample(
        self,
        idx: int,
        row: Dict,
        bench: BenchmarkDefinition,
        model_call_fn: Callable[[str, str], str],
        endpoint_type: str,
        request_timeout: Optional[float] = None,
    ) -> Tuple[Optional[Dict], Optional[SampleResult]]:
        # Get pre-generated response from dataset
        if self.response_field not in row:
            target = row.get(bench.target_field, "")
            return None, SampleResult(
                sample_id=idx,
                prompt="",
                response=None,
                target=target,
                scores=None,
                status="skipped_missing_field",
                error=f"Response field '{self.response_field}' not found in row",
                metadata=row,
            )

        response = str(row[self.response_field])

        # Render prompt (best-effort, for predictions output)
        try:
            prompt = render_prompt(bench.prompt, row, bench._is_jinja2)
        except KeyError:
            prompt = ""

        # Score using ScorerInput
        target = row.get(bench.target_field, "")
        scorer_input = ScorerInput(
            response=response,
            target=target,
            metadata=row,
            model_call_fn=model_call_fn,
            config=bench.extra_config,
        )
        try:
            sample_scores = bench.scorer_fn(scorer_input)
        except Exception as e:
            return None, SampleResult(
                sample_id=idx,
                prompt=prompt,
                response=response,
                target=target,
                scores=None,
                status="skipped_scorer_error",
                error=str(e),
                metadata=row,
            )

        prediction = SampleResult(
            sample_id=idx,
            prompt=prompt,
            response=response,
            target=target,
            scores=sample_scores,
            status="scored",
            metadata=row,
        )
        return sample_scores, prediction


def run_eval_loop(
    bench: BenchmarkDefinition,
    dataset: List[Dict],
    model_call_fn: Callable[[str, str], str],
    endpoint_type: str,
    strategy: Optional[EvalStrategy] = None,
    save_predictions: bool = False,
    show_progress: bool = True,
    max_consecutive_errors: int = 0,
    fail_on_skip: bool = False,
    parallelism: int = 1,
    request_timeout: Optional[float] = None,
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
        request_timeout: Per-request timeout in seconds, passed to
            model_call_fn as the HTTP timeout for each model call.

    Returns:
        Tuple of (all_scores, all_predictions).
    """
    if strategy is None:
        if bench.response_field:
            strategy = EvalOnlyStrategy(bench.response_field)
        else:
            strategy = StandardStrategy()

    if parallelism > 1 and len(dataset) > 1:
        if max_consecutive_errors > 0:
            logger.warning(
                "max_consecutive_errors is unreliable with parallelism > 1 "
                "(completion order != sample order)"
            )
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
            request_timeout=request_timeout,
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
        request_timeout=request_timeout,
    )


def _run_eval_loop_sequential(
    bench: BenchmarkDefinition,
    dataset: List[Dict],
    model_call_fn: Callable[[str, str], str],
    endpoint_type: str,
    strategy: EvalStrategy,
    save_predictions: bool = False,
    show_progress: bool = True,
    max_consecutive_errors: int = 0,
    fail_on_skip: bool = False,
    request_timeout: Optional[float] = None,
) -> Tuple[List[Dict], List[SampleResult]]:
    """Sequential evaluation loop (original behavior)."""
    if request_timeout is not None:
        logger.info("Per-request timeout", request_timeout=request_timeout)
    all_scores = []
    all_predictions: List[SampleResult] = []
    total = len(dataset)
    consecutive_errors = 0
    scored_count = 0
    skipped_count = 0

    progress_interval = max(1, min(10, total // 10)) if total > 0 else 1

    for idx, row in enumerate(dataset):
        scores, prediction = strategy.evaluate_sample(
            idx, row, bench, model_call_fn, endpoint_type, request_timeout
        )

        if scores is None:
            # Sample failed
            skipped_count += 1
            consecutive_errors += 1
            if save_predictions and prediction:
                all_predictions.append(prediction)
            msg = (
                f"Sample {idx} failed: {prediction.error if prediction else 'unknown'}"
            )
            logger.warning(
                "Sample failed",
                sample_id=idx,
                error=prediction.error if prediction else "unknown",
            )
            if fail_on_skip:
                raise RuntimeError(msg)
            if (
                max_consecutive_errors > 0
                and consecutive_errors >= max_consecutive_errors
            ):
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
                    processed=processed,
                    total=total,
                    pct=round((processed / total) * 100),
                    scored=scored_count,
                    skipped=skipped_count,
                )

    if show_progress:
        logger.info(
            "Evaluation complete",
            scored=scored_count,
            skipped=skipped_count,
            total=total,
        )

    return all_scores, all_predictions


def _run_eval_loop_parallel(
    bench: BenchmarkDefinition,
    dataset: List[Dict],
    model_call_fn: Callable[[str, str], str],
    endpoint_type: str,
    strategy: EvalStrategy,
    save_predictions: bool = False,
    show_progress: bool = True,
    max_consecutive_errors: int = 0,
    fail_on_skip: bool = False,
    parallelism: int = 4,
    request_timeout: Optional[float] = None,
) -> Tuple[List[Dict], List[SampleResult]]:
    """Parallel evaluation loop using ThreadPoolExecutor.

    Maintains sample ordering in results regardless of completion order.
    Thread-safe bookkeeping via a lock for counters and error tracking.
    """
    if request_timeout is not None:
        logger.info("Per-request timeout", request_timeout=request_timeout)
    total = len(dataset)

    # Pre-allocate result slots indexed by sample position
    results: List[Optional[Tuple[Optional[Dict], Optional[SampleResult]]]] = [
        None
    ] * total

    # Thread-safe counters
    lock = threading.Lock()
    scored_count = 0
    skipped_count = 0
    consecutive_errors = 0
    abort_error: Optional[RuntimeError] = None

    progress_interval = max(1, min(10, total // 10)) if total > 0 else 1

    def evaluate_one(
        idx: int, row: Dict
    ) -> Tuple[int, Optional[Dict], Optional[SampleResult]]:
        scores, prediction = strategy.evaluate_sample(
            idx, row, bench, model_call_fn, endpoint_type, request_timeout
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
                    sample_id=idx,
                    prompt="",
                    response=None,
                    target="",
                    scores=None,
                    status="skipped_model_error",
                    error=str(e),
                )

            results[idx] = (scores, prediction)

            with lock:
                if scores is None:
                    skipped_count += 1
                    consecutive_errors += 1
                    msg = f"Sample {idx} failed: {prediction.error if prediction else 'unknown'}"
                    logger.warning(
                        "Sample failed",
                        sample_id=idx,
                        error=prediction.error if prediction else "unknown",
                    )
                    if fail_on_skip and abort_error is None:
                        abort_error = RuntimeError(msg)
                    if (
                        max_consecutive_errors > 0
                        and consecutive_errors >= max_consecutive_errors
                        and abort_error is None
                    ):
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
                            processed=processed,
                            total=total,
                            pct=round((processed / total) * 100),
                            scored=scored_count,
                            skipped=skipped_count,
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
            scored=scored_count,
            skipped=skipped_count,
            total=total,
        )

    return all_scores, all_predictions
