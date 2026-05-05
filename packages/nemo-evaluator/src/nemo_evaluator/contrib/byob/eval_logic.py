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
import random
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

    Different evaluation modes (standard, judge, multi-turn, agentic,
    multiple-choice loglikelihood) implement this protocol to define how
    each sample is evaluated.
    """

    def evaluate_sample(
        self,
        idx: int,
        row: Dict,
        bench: BenchmarkDefinition,
        model_call_fn: Callable[..., Any],
        endpoint_type: str,
        request_timeout: Optional[float] = None,
        fewshot_prefix: str = "",
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
        model_call_fn: Callable[..., Any],
        endpoint_type: str,
        request_timeout: Optional[float] = None,
        fewshot_prefix: str = "",
    ) -> Tuple[Optional[Dict], Optional[SampleResult]]:
        """Render prompt, call model, score the response.

        Args:
            idx: Zero-based sample index.
            row: Dataset row dict.
            bench: Benchmark definition with prompt template and scorer.
            model_call_fn: Callable(prompt, endpoint_type, ...) -> response.
            endpoint_type: ``"chat"`` or ``"completions"``.
            request_timeout: Per-request HTTP timeout in seconds.

        Returns:
            Tuple of (scores_dict, SampleResult) on success, or
            (None, SampleResult with error status) on failure.
        """
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

        if fewshot_prefix:
            prompt = fewshot_prefix + prompt

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
    """Eval-only strategy: score pre-generated responses from the dataset."""

    def __init__(self, response_field: str):
        """Initialize eval-only strategy.

        Args:
            response_field: Name of the dataset field containing pre-generated responses.
        """
        self.response_field = response_field

    def evaluate_sample(
        self,
        idx: int,
        row: Dict,
        bench: BenchmarkDefinition,
        model_call_fn: Callable[..., Any],
        endpoint_type: str,
        request_timeout: Optional[float] = None,
        fewshot_prefix: str = "",
    ) -> Tuple[Optional[Dict], Optional[SampleResult]]:
        """Score a pre-generated response from the dataset (no model call).

        Args:
            idx: Zero-based sample index.
            row: Dataset row dict (must contain ``response_field``).
            bench: Benchmark definition with scorer.
            model_call_fn: Not used (present for protocol compliance).
            endpoint_type: Not used (present for protocol compliance).
            request_timeout: Not used (present for protocol compliance).
            fewshot_prefix: Not used (present for protocol compliance).

        Returns:
            Tuple of (scores_dict, SampleResult) on success, or
            (None, SampleResult with error status) on failure.
        """
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


def _resolve_choices_from_row(row: Dict, choices_field: str) -> Optional[List[str]]:
    """Resolve per-row choices from a field name or dotted path.

    Resolution order:

    1. Treat ``choices_field`` as a dotted path (e.g. ``"choices.text"``)
       and walk it through nested dicts.
    2. If that fails, look up ``choices_field`` as a literal key
       (supports keys that themselves contain dots).
    3. **Convenience:** if the resolved value is a dict (HuggingFace's
       ``{"label": [...], "text": [...]}`` schema), pull the ``"text"``
       field automatically. Datasets that store choices under a
       different sub-key (e.g. ``"options"``) should use the dotted
       path form (``"choices.options"``).

    Returns the list of stringified choices, or ``None`` if the field
    cannot be resolved to a list.
    """
    raw: Any = row
    for part in choices_field.split("."):
        if isinstance(raw, dict) and part in raw:
            raw = raw[part]
        else:
            raw = None
            break

    if raw is None:
        raw = row.get(choices_field)

    if isinstance(raw, dict):
        raw = raw.get("text")

    if isinstance(raw, list):
        return [str(c) for c in raw]
    return None


class MultipleChoiceStrategy:
    """Per-choice loglikelihood ranking (lm-eval-harness ``local-completions`` parity).

    For each row:

    1. Render the context from ``bench.prompt``.
    2. Resolve the candidate continuations (from ``bench.choices`` or
       ``row[bench.choices_field]``).
    3. Call ``model_call_fn(context, "completions_logprob",
       continuation=cont)`` for each candidate, collecting
       ``(sum_logprob, is_greedy)`` tuples.
    4. Inject ``_choices``, ``_choices_logprobs``, ``_choices_is_greedy``
       into ``ScorerInput.metadata`` (the shared row + response-metadata
       bag). ``response`` is set to the argmax choice so legacy
       text-based scorers (``exact_match`` etc.) still work.
    5. Hand off to the user scorer (commonly :func:`multiple_choice_acc`).
    """

    def evaluate_sample(
        self,
        idx: int,
        row: Dict,
        bench: BenchmarkDefinition,
        model_call_fn: Callable[..., Any],
        endpoint_type: str,
        request_timeout: Optional[float] = None,
        fewshot_prefix: str = "",
    ) -> Tuple[Optional[Dict], Optional[SampleResult]]:
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

        if fewshot_prefix:
            prompt = fewshot_prefix + prompt

        # Resolve choices: per-row field takes precedence over static list
        choices: Optional[List[str]] = None
        if bench.choices_field:
            choices = _resolve_choices_from_row(row, bench.choices_field)
        if not choices and bench.choices:
            choices = list(bench.choices)
        if not choices:
            target = row.get(bench.target_field, "")
            return None, SampleResult(
                sample_id=idx,
                prompt=prompt,
                response=None,
                target=target,
                scores=None,
                status="skipped_missing_field",
                error=(
                    f"No choices available: choices_field "
                    f"'{bench.choices_field}' missing from row and no static "
                    f"choices declared on @benchmark."
                ),
                metadata=row,
            )

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

        choices_logprobs: List[float] = []
        choices_is_greedy: List[bool] = []
        for cont in choices:
            try:
                ll, greedy = model_call_fn(
                    prompt,
                    "completions_logprob",
                    system_prompt=rendered_system_prompt,
                    timeout=request_timeout,
                    continuation=cont,
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
            choices_logprobs.append(float(ll))
            choices_is_greedy.append(bool(greedy))

        argmax_idx = max(range(len(choices)), key=lambda i: choices_logprobs[i])
        response = choices[argmax_idx]

        target = row.get(bench.target_field, "")
        # Build a shared metadata dict carrying both row fields and per-call
        # response metadata under reserved ``_`` keys. Both the scorer and
        # the prediction record see the same shape.
        merged_metadata = {
            **row,
            "_choices": choices,
            "_choices_logprobs": choices_logprobs,
            "_choices_is_greedy": choices_is_greedy,
        }
        scorer_input = ScorerInput(
            response=response,
            target=target,
            metadata=merged_metadata,
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
            metadata=merged_metadata,
        )
        return sample_scores, prediction


def render_fewshot_example(bench: BenchmarkDefinition, row: Dict) -> Optional[str]:
    """Render a single few-shot example for *row*.

    If ``bench.fewshot_template`` is set, render that with the row dict.
    Otherwise reuse the main ``bench.prompt`` template and append the
    target value (fetched from ``bench.target_field``).

    Returns None on missing-field errors so the caller can skip cleanly.
    """
    try:
        if bench.fewshot_template:
            return render_prompt(bench.fewshot_template, row, bench._is_jinja2)
        prompt_part = render_prompt(bench.prompt, row, bench._is_jinja2)
        target_part = row.get(bench.target_field, "")
        return f"{prompt_part} {target_part}".rstrip()
    except KeyError:
        return None


def build_fewshot_prefix(bench: BenchmarkDefinition, examples: List[Dict]) -> str:
    """Render *examples* into a prefix string ready to prepend to each prompt.

    Skips examples that fail to render (missing fields). Always appends the
    benchmark's ``fewshot_separator`` after the last example so the test
    prompt starts on a fresh boundary.
    """
    if not examples:
        return ""
    rendered: List[str] = []
    for ex in examples:
        text = render_fewshot_example(bench, ex)
        if text is not None:
            rendered.append(text)
    if not rendered:
        return ""
    # Use ``is None`` rather than ``or`` so an explicit empty-string
    # separator (concat with no delimiter) is honoured.
    sep = bench.fewshot_separator if bench.fewshot_separator is not None else "\n\n"
    return sep.join(rendered) + sep


def build_fewshot_examples(
    primary_dataset_uri: str,
    primary_dataset: List[Dict],
    num_fewshot: int,
    fewshot_split: Optional[str],
    field_mapping: Optional[Dict[str, str]] = None,
    seed: int = 42,
) -> List[Dict]:
    """Sample ``num_fewshot`` examples deterministically (lm-eval style).

    Selection rules (in order):

    1. If ``fewshot_split`` is set and the primary dataset URI is an
       ``hf://`` URI, load that split via the dataset module and sample
       ``num_fewshot`` rows. This is the **safe** path — examples come
       from a different split than the test set, so there is no
       contamination.
    2. Otherwise, sample ``num_fewshot`` rows from the **tail** of
       ``primary_dataset`` (i.e. the rows least likely to be evaluated
       when ``--limit-samples`` is set). A loud warning is logged
       because the fewshot pool overlaps with the evaluation set when
       running the full dataset, which can leak gold answers into the
       prompt.

    To guarantee no contamination, declare a ``fewshot_split`` on the
    ``@benchmark`` (e.g. ``"train"`` or ``"dev"``) so this function
    samples from a disjoint split.
    """
    if num_fewshot <= 0:
        return []

    pool: List[Dict] = []
    if fewshot_split and primary_dataset_uri.startswith("hf://"):
        try:
            from nemo_evaluator.contrib.byob.dataset import load_dataset

            # Replace or inject ?split= in the URI
            if "?" in primary_dataset_uri:
                base, _ = primary_dataset_uri.split("?", 1)
            else:
                base = primary_dataset_uri
            fs_uri = f"{base}?split={fewshot_split}"
            pool = load_dataset(
                fs_uri, limit=max(num_fewshot * 4, 16), field_mapping=field_mapping
            )
        except Exception as e:
            logger.warning(
                "Failed to load fewshot_split, falling back to primary dataset",
                fewshot_split=fewshot_split,
                error=str(e),
            )
            pool = []

    if not pool:
        # Fallback: no separate fewshot split is available. Sample from
        # the tail of the primary dataset to minimise overlap with the
        # eval set when the user passes --limit-samples (which iterates
        # from the head). When the full dataset is evaluated, the
        # fewshot pool is a strict subset of the eval set and gold
        # answers can leak — warn loudly so the user knows to declare
        # ``fewshot_split=`` on the @benchmark.
        logger.warning(
            "fewshot_split not available; sampling from primary dataset. "
            "This risks test-set contamination because the fewshot pool "
            "overlaps with rows being evaluated. Declare a separate "
            "fewshot_split on @benchmark (e.g. 'train' or 'dev') to "
            "eliminate the overlap.",
            num_fewshot=num_fewshot,
            primary_dataset_size=len(primary_dataset),
        )
        pool_size = max(num_fewshot * 4, num_fewshot)
        # Tail slice — falls back to the head only if the dataset is
        # smaller than the desired pool.
        if len(primary_dataset) > pool_size:
            pool = primary_dataset[-pool_size:]
        else:
            pool = list(primary_dataset)

    if not pool:
        return []

    rng = random.Random(seed)
    if len(pool) <= num_fewshot:
        return list(pool)
    return rng.sample(pool, num_fewshot)


def run_eval_loop(
    bench: BenchmarkDefinition,
    dataset: List[Dict],
    model_call_fn: Callable[..., Any],
    endpoint_type: str,
    strategy: Optional[EvalStrategy] = None,
    save_predictions: bool = False,
    show_progress: bool = True,
    max_consecutive_errors: int = 0,
    fail_on_skip: bool = False,
    parallelism: int = 1,
    request_timeout: Optional[float] = None,
    fewshot_examples: Optional[List[Dict]] = None,
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
        elif (
            endpoint_type == "completions_logprob"
            or bench.endpoint_type == "completions_logprob"
        ):
            # Logprob-mode MCQ ranking is the only strategy that requires
            # ``choices`` / ``choices_field``; the @benchmark decorator
            # already validates that pairing. Don't auto-pick MCQ just
            # because choices are declared — a user may declare them as
            # informational metadata while running the chat endpoint.
            strategy = MultipleChoiceStrategy()
        else:
            strategy = StandardStrategy()

    fewshot_prefix = (
        build_fewshot_prefix(bench, fewshot_examples) if fewshot_examples else ""
    )

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
            fewshot_prefix=fewshot_prefix,
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
        fewshot_prefix=fewshot_prefix,
    )


def _run_eval_loop_sequential(
    bench: BenchmarkDefinition,
    dataset: List[Dict],
    model_call_fn: Callable[..., Any],
    endpoint_type: str,
    strategy: EvalStrategy,
    save_predictions: bool = False,
    show_progress: bool = True,
    max_consecutive_errors: int = 0,
    fail_on_skip: bool = False,
    request_timeout: Optional[float] = None,
    fewshot_prefix: str = "",
) -> Tuple[List[Dict], List[SampleResult]]:
    """Sequential evaluation loop. See :func:`run_eval_loop` for parameter docs."""
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
        # Pass fewshot_prefix only when non-empty so legacy strategy
        # implementations (without the kwarg) continue to work.
        kwargs = {"fewshot_prefix": fewshot_prefix} if fewshot_prefix else {}
        scores, prediction = strategy.evaluate_sample(
            idx,
            row,
            bench,
            model_call_fn,
            endpoint_type,
            request_timeout,
            **kwargs,
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
    model_call_fn: Callable[..., Any],
    endpoint_type: str,
    strategy: EvalStrategy,
    save_predictions: bool = False,
    show_progress: bool = True,
    max_consecutive_errors: int = 0,
    fail_on_skip: bool = False,
    parallelism: int = 4,
    request_timeout: Optional[float] = None,
    fewshot_prefix: str = "",
) -> Tuple[List[Dict], List[SampleResult]]:
    """Parallel evaluation loop. See :func:`run_eval_loop` for parameter docs."""
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
        kwargs = {"fewshot_prefix": fewshot_prefix} if fewshot_prefix else {}
        scores, prediction = strategy.evaluate_sample(
            idx,
            row,
            bench,
            model_call_fn,
            endpoint_type,
            request_timeout,
            **kwargs,
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
