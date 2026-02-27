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

"""BYOB-GYM adapter: bridges BYOB benchmarks to the NeMo Gym resource server interface.

This module provides ``ByobGymHarness``, a concrete implementation of the
``GymHarness`` ABC that adapts any BYOB benchmark (defined via ``@benchmark``
and ``@scorer`` decorators) to the Gym reward-server protocol.

Architecture overview::

    Gym resource server          ByobGymHarness               BYOB
    ==================          ================           =========
    get_dataset()        --->   load + render prompts  --> dataset.py, eval_logic.py
    verify(response)     --->   build ScorerInput      --> scorer_fn()
                                map scores to reward   <-- {"correct": True, ...}
                                (float, answer)        <-- return to Gym

Design decisions
----------------

1. **Takes a BenchmarkDefinition directly.** The harness is constructed from
   a ``BenchmarkDefinition`` rather than dynamically importing a module.  This
   is deliberate: ``import_benchmark()`` clears the global registry as a
   side-effect, which is unsafe in a long-lived Gym server that might host
   multiple harnesses.  A factory classmethod ``from_module()`` is provided
   for convenience when you need to load from a module path.

2. **Score-to-reward mapping.** BYOB scorers return arbitrary dicts (e.g.
   ``{"correct": True}``, ``{"f1": 0.85, "precision": 0.9}``).  The Gym
   interface requires a single float reward in [0, 1].  The mapping strategy
   is configurable via ``reward_key``:

   - If ``reward_key`` is set, that specific key from the scorer dict is
     used (bool -> 0.0/1.0, numeric -> clamped to [0, 1]).
   - If ``reward_key`` is None (default), the adapter uses a deterministic
     priority: ``"correct"`` > ``"judge_score"`` > first boolean key > first
     float key > mean of all numeric values.

   This covers the common cases: exact_match returns ``{"correct": True}``,
   judges return ``{"judge_score": 0.8}``, and multi-metric scorers
   (f1_token, rouge) get a reasonable single-number summary.

3. **Async wrapping.** BYOB scorers are synchronous and typically CPU-bound
   (string comparison, regex).  The ``verify()`` method calls the scorer
   synchronously inside the async method rather than using
   ``asyncio.to_thread()``, because:
   - Thread-pool dispatch adds ~50us overhead per call for no benefit when
     the scorer itself takes <1ms.
   - Judge-based scorers are I/O-bound but use their own HTTP session
     internally (requests library), so they already release the GIL.
   An escape hatch is available: if the benchmark has
   ``extra_config["gym_offload_scorer"] = True``, the scorer is dispatched
   to the default executor via ``asyncio.to_thread()``.

4. **Judge support.** Judge scorers access their endpoint config through
   ``ScorerInput.config``, which is populated from
   ``BenchmarkDefinition.extra_config``.  Gym-level judge configuration
   can be injected via the ``judge_config`` constructor parameter, which
   merges into ``extra_config`` under the ``"judge"`` key.  This allows
   Gym deployments to override the judge URL/model/API-key without
   modifying the benchmark source.

5. **Metadata passthrough.** Each dataset row's fields (beyond the rendered
   prompt and target) are passed through as ``**kwargs`` to ``verify()``
   via the ``_byob_row`` key.  This preserves the full row for the scorer's
   ``metadata`` parameter.

Example usage::

    from nemo_evaluator.contrib.byob.gym_adapter import ByobGymHarness

    # From a BenchmarkDefinition (when already imported)
    harness = ByobGymHarness(bench=my_bench_definition)
    dataset = harness.get_dataset()
    reward, answer = await harness.verify(response_text="42", expected_answer="42")

    # From a module path (convenience factory)
    harness = ByobGymHarness.from_module(
        benchmark_module="examples/byob/global_mmlu_lite/benchmark.py",
        benchmark_name="global_mmlu_lite",
    )
"""

from __future__ import annotations

import asyncio
import importlib
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from nemo_evaluator.contrib.byob.dataset import load_dataset
from nemo_evaluator.contrib.byob.decorators import (
    BenchmarkDefinition,
    ScorerInput,
    _normalize_name,
    clear_registry,
    get_registered_benchmarks,
)
from nemo_evaluator.contrib.byob.eval_logic import render_prompt
from nemo_evaluator.gym_harness import GymHarness
from nemo_evaluator.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Score-to-reward mapping
# ---------------------------------------------------------------------------

# Priority order for auto-detecting the reward key from a scorer result dict.
_REWARD_KEY_PRIORITY = ("correct", "judge_score")


def scores_to_reward(
    scores: Dict[str, Any],
    reward_key: Optional[str] = None,
) -> float:
    """Convert a BYOB scorer result dict to a single float reward in [0, 1].

    Strategy (applied in order):

    1. If ``reward_key`` is specified, use ``scores[reward_key]``.
    2. Look for well-known keys in priority order: ``"correct"``,
       ``"judge_score"``.
    3. Use the first boolean-valued key found.
    4. Use the first float/int-valued key found.
    5. Compute the mean of all numeric values.
    6. Return 0.0 if no numeric values exist.

    Booleans are mapped to 0.0/1.0.  Numeric values are clamped to [0, 1].

    Args:
        scores: The dict returned by a BYOB scorer function.
        reward_key: Optional explicit key to extract from scores.

    Returns:
        A float reward in [0, 1].

    Raises:
        KeyError: If ``reward_key`` is specified but not present in scores.
    """
    if not scores:
        return 0.0

    # --- Step 1: explicit key ---
    if reward_key is not None:
        value = scores[reward_key]
        return _coerce_to_reward(value)

    # --- Step 2: well-known priority keys ---
    for key in _REWARD_KEY_PRIORITY:
        if key in scores:
            return _coerce_to_reward(scores[key])

    # --- Step 3: first boolean key ---
    for key, value in scores.items():
        if isinstance(value, bool):
            return 1.0 if value else 0.0

    # --- Step 4: first numeric key ---
    for key, value in scores.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return _clamp01(float(value))

    # --- Step 5: mean of all numeric values ---
    numeric_values = [
        float(v) for v in scores.values() if isinstance(v, (bool, int, float))
    ]
    if numeric_values:
        mean = sum(numeric_values) / len(numeric_values)
        return _clamp01(mean)

    # --- Step 6: nothing usable ---
    return 0.0


def _coerce_to_reward(value: Any) -> float:
    """Convert a single score value to a float reward in [0, 1]."""
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return _clamp01(float(value))
    # Last resort: try float conversion
    try:
        return _clamp01(float(value))
    except (TypeError, ValueError):
        return 0.0


def _clamp01(value: float) -> float:
    """Clamp a float to the [0, 1] interval."""
    return max(0.0, min(1.0, value))


# ---------------------------------------------------------------------------
# Extracted-answer heuristic
# ---------------------------------------------------------------------------


def extract_answer_from_scores(scores: Dict[str, Any]) -> Optional[str]:
    """Attempt to extract a human-readable "answer" from scorer results.

    Looks for common keys that contain the extracted/parsed answer rather
    than a numeric score.  Returns None if no such key is found (which is
    the common case for simple scorers).

    Args:
        scores: The dict returned by a BYOB scorer function.

    Returns:
        The extracted answer string, or None.
    """
    for key in ("extracted_answer", "predicted", "prediction", "judge_grade"):
        if key in scores:
            return str(scores[key])
    return None


# ---------------------------------------------------------------------------
# ByobGymHarness
# ---------------------------------------------------------------------------


class ByobGymHarness(GymHarness):
    """Adapts a BYOB benchmark to the Gym resource-server interface.

    Args:
        bench: A ``BenchmarkDefinition`` from the BYOB decorator registry.
        reward_key: Optional explicit scorer-dict key to use as the reward.
            When None, the adapter auto-detects using ``scores_to_reward()``.
        judge_config: Optional dict of judge endpoint configuration.  Merged
            into ``bench.extra_config["judge"]`` so judge-based scorers pick
            it up automatically.  Allows Gym-level override of judge URL,
            model, and API key without touching the benchmark source.
        dataset_limit: Optional limit on dataset size (number of rows).
        model_call_fn: Optional callable for judge/multi-turn scorers that
            need to call a model during verification.  Signature:
            ``(prompt: str, endpoint_type: str, **kw) -> str``.
    """

    # Metadata key used to pass the full BYOB dataset row through Gym's
    # kwargs mechanism so that verify() can reconstruct ScorerInput.metadata.
    _ROW_KEY = "_byob_row"

    def __init__(
        self,
        bench: BenchmarkDefinition,
        *,
        reward_key: Optional[str] = None,
        judge_config: Optional[Dict[str, Any]] = None,
        dataset_limit: Optional[int] = None,
        model_call_fn: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        # Construct eval_type from the benchmark name, matching the pattern
        # used by the BYOB compiler: "byob_<normalized_name>.<name>"
        eval_type = f"byob_{bench.normalized_name}.{bench.name}"
        super().__init__(eval_type=eval_type, **kwargs)

        self._bench = bench
        self._reward_key = reward_key
        self._dataset_limit = dataset_limit
        self._model_call_fn = model_call_fn
        self._offload_scorer = bench.extra_config.get("gym_offload_scorer", False)

        # Merge Gym-level judge config into extra_config if provided
        if judge_config is not None:
            self._bench.extra_config.setdefault("judge", {})
            self._bench.extra_config["judge"].update(judge_config)

        # Lazily loaded dataset cache
        self._dataset_cache: Optional[List[dict]] = None

    # -- Factory classmethods -----------------------------------------------

    @classmethod
    def from_module(
        cls,
        benchmark_module: str,
        benchmark_name: str,
        **kwargs: Any,
    ) -> "ByobGymHarness":
        """Create a ByobGymHarness by importing a benchmark module.

        This is the convenience path for Gym configurations that reference
        BYOB benchmarks by module path + name (the same identifiers used
        by the BYOB CLI and compiler).

        Note: this method clears the global BYOB benchmark registry as a
        side-effect of import.  It is safe to call once during server
        startup but should not be called concurrently.

        Args:
            benchmark_module: Path to ``.py`` file or dotted module name.
            benchmark_name: Human-readable benchmark name (will be
                normalized internally).
            **kwargs: Forwarded to the ``ByobGymHarness`` constructor.

        Returns:
            A configured ``ByobGymHarness`` instance.

        Raises:
            ValueError: If the benchmark is not found after import.
        """
        normalized = _normalize_name(benchmark_name)
        bench = _import_benchmark_safe(benchmark_module, normalized)
        return cls(bench=bench, **kwargs)

    @classmethod
    def from_eval_type(
        cls,
        eval_type: str,
        **kwargs: Any,
    ) -> "ByobGymHarness":
        """Create a ByobGymHarness from a Gym eval_type string.

        Expected format: ``"byob_<normalized_name>.<benchmark_name>"``

        The method imports the compiled BYOB package (which must be on
        ``sys.path`` or in ``nemo_evaluator.__path__``) and extracts the
        ``BenchmarkDefinition`` from the registry.

        Args:
            eval_type: The Gym eval_type string.
            **kwargs: Forwarded to the ``ByobGymHarness`` constructor.

        Returns:
            A configured ``ByobGymHarness`` instance.

        Raises:
            ValueError: If eval_type format is invalid or benchmark not found.
        """
        if "." not in eval_type:
            raise ValueError(
                f"Invalid BYOB eval_type format: '{eval_type}'. "
                f"Expected 'byob_<name>.<benchmark_name>'."
            )
        pkg_name, _ = eval_type.split(".", 1)
        if not pkg_name.startswith("byob_"):
            raise ValueError(
                f"Invalid BYOB eval_type format: '{eval_type}'. "
                f"Package name must start with 'byob_'."
            )
        normalized_name = pkg_name[len("byob_") :]

        # Import the compiled benchmark package to trigger registration
        module_name = f"nemo_evaluator.{pkg_name}"
        bench = _import_benchmark_safe(module_name, normalized_name)
        return cls(bench=bench, **kwargs)

    # -- GymHarness interface -----------------------------------------------

    def get_dataset(self) -> list[dict]:
        """Load the BYOB dataset and transform it into Gym's expected format.

        Each returned row contains:
        - ``responses_create_params``: dict with ``"input"`` key holding
          the rendered prompt as OpenAI chat messages (with optional system
          message).
        - ``expected_answer``: str representation of the target value.
        - ``_byob_row``: the full original dataset row dict, passed through
          to ``verify()`` as a kwarg for ScorerInput.metadata reconstruction.

        Returns:
            List of dicts in Gym dataset format.
        """
        if self._dataset_cache is not None:
            return self._dataset_cache

        bench = self._bench

        # Load raw dataset
        raw_data = load_dataset(
            bench.dataset,
            limit=self._dataset_limit,
            field_mapping=bench.field_mapping,
        )

        logger.info(
            "Building Gym dataset from BYOB benchmark",
            benchmark=bench.name,
            raw_rows=len(raw_data),
        )

        gym_dataset: list[dict] = []

        for row in raw_data:
            # Render user prompt
            try:
                rendered_prompt = render_prompt(bench.prompt, row, bench._is_jinja2)
            except KeyError as e:
                logger.warning(
                    "Skipping row: prompt rendering failed",
                    error=str(e),
                    benchmark=bench.name,
                )
                continue

            # Build messages list (OpenAI chat format)
            messages: list[dict] = []

            # System prompt (optional)
            if bench.system_prompt:
                try:
                    rendered_system = render_prompt(
                        bench.system_prompt,
                        row,
                        bench._is_system_prompt_jinja2,
                    )
                    messages.append({"role": "system", "content": rendered_system})
                except KeyError as e:
                    logger.warning(
                        "System prompt rendering failed, omitting",
                        error=str(e),
                        benchmark=bench.name,
                    )

            # User message
            messages.append({"role": "user", "content": rendered_prompt})

            # Extract target
            target = row.get(bench.target_field, "")
            expected_answer = str(target) if target is not None else ""

            gym_row = {
                "responses_create_params": {"input": messages},
                "expected_answer": expected_answer,
                self._ROW_KEY: row,
            }
            gym_dataset.append(gym_row)

        logger.info(
            "Gym dataset ready",
            benchmark=bench.name,
            gym_rows=len(gym_dataset),
            skipped=len(raw_data) - len(gym_dataset),
        )

        self._dataset_cache = gym_dataset
        return gym_dataset

    async def verify(
        self,
        response_text: str,
        expected_answer: str,
        **kwargs: Any,
    ) -> tuple[float, str | None]:
        """Score a model response using the BYOB scorer.

        Constructs a ``ScorerInput`` from the response, expected_answer,
        and the original dataset row (passed through via ``_byob_row``),
        then invokes the benchmark's scorer function and maps the result
        to a single float reward.

        Args:
            response_text: The model's generated response.
            expected_answer: The ground-truth string from the dataset row.
            **kwargs: Must include ``_byob_row`` (the original dataset row).
                Any other kwargs are ignored.

        Returns:
            A ``(reward, extracted_answer)`` tuple.
        """
        bench = self._bench

        # Reconstruct metadata from the original row.  Fall back to a
        # minimal dict if the row was not passed through (should not happen
        # in normal operation, but makes the method robust to direct calls).
        row: dict = kwargs.get(self._ROW_KEY, {})

        # Reconstruct the target from the original row if available,
        # falling back to expected_answer.  This preserves structured
        # targets (lists, dicts) that were stringified for Gym transport.
        target: Any = row.get(bench.target_field, expected_answer)

        # Build ScorerInput
        scorer_input = ScorerInput(
            response=response_text,
            target=target,
            metadata=row,
            model_call_fn=self._model_call_fn,
            config=bench.extra_config,
        )

        # Invoke scorer (sync function, optionally offloaded to thread pool)
        try:
            if self._offload_scorer:
                scores = await asyncio.to_thread(bench.scorer_fn, scorer_input)
            else:
                scores = bench.scorer_fn(scorer_input)
        except Exception as e:
            logger.warning(
                "Scorer failed during verify()",
                benchmark=bench.name,
                error=str(e),
            )
            return 0.0, None

        if not isinstance(scores, dict):
            logger.warning(
                "Scorer returned non-dict result",
                benchmark=bench.name,
                result_type=type(scores).__name__,
            )
            return 0.0, None

        # Map scores to reward
        try:
            reward = scores_to_reward(scores, self._reward_key)
        except KeyError:
            logger.warning(
                "Configured reward_key not found in scorer output",
                reward_key=self._reward_key,
                available_keys=list(scores.keys()),
                benchmark=bench.name,
            )
            reward = 0.0

        # Attempt to extract a human-readable answer
        extracted = extract_answer_from_scores(scores)

        return reward, extracted

    # -- Accessors ----------------------------------------------------------

    @property
    def benchmark(self) -> BenchmarkDefinition:
        """Return the underlying BenchmarkDefinition."""
        return self._bench

    @property
    def reward_key(self) -> Optional[str]:
        """Return the configured reward key (or None for auto-detect)."""
        return self._reward_key


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _import_benchmark_safe(
    benchmark_module: str,
    normalized_name: str,
) -> BenchmarkDefinition:
    """Import a benchmark module and return the BenchmarkDefinition.

    Unlike ``eval_logic.import_benchmark()``, this function does NOT
    call ``clear_registry()`` before import.  Instead, it saves the
    registry state, imports the module, extracts the target benchmark,
    and then restores the registry.  This makes it safe to use in
    long-lived processes where other benchmarks may already be registered.

    Args:
        benchmark_module: Path to ``.py`` file or dotted module name.
        normalized_name: The normalized benchmark name to look up.

    Returns:
        The ``BenchmarkDefinition`` for the requested benchmark.

    Raises:
        ValueError: If the benchmark is not found after import.
    """
    # Save existing registry state
    saved_registry = dict(get_registered_benchmarks())

    try:
        # Clear for a clean import
        clear_registry()

        module_path = Path(benchmark_module)
        if module_path.exists() and module_path.is_file():
            parent_dir = str(module_path.parent.absolute())
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)
            module_name = module_path.stem
        else:
            module_name = benchmark_module

        # Import or reload
        if module_name in sys.modules:
            importlib.reload(sys.modules[module_name])
        else:
            importlib.import_module(module_name)

        benchmarks = get_registered_benchmarks()
        if normalized_name not in benchmarks:
            available = ", ".join(benchmarks.keys()) or "(none)"
            raise ValueError(
                f"Benchmark '{normalized_name}' not found after importing "
                f"'{benchmark_module}'. Available: {available}"
            )

        return benchmarks[normalized_name]

    finally:
        # Restore the previous registry state so other benchmarks are
        # not lost.  The target benchmark is returned by value above,
        # so restoring the registry does not invalidate it.
        clear_registry()
        from nemo_evaluator.contrib.byob.decorators import _BENCHMARK_REGISTRY

        _BENCHMARK_REGISTRY.update(saved_registry)
