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
"""Bring-Your-Own-Benchmark API: @benchmark + @scorer + @image_builder.

All benchmarks -- built-in and user-provided -- use this API.

    from nemo_evaluator.environments.custom import benchmark, scorer
    from nemo_evaluator.scoring import ScorerInput, exact_match

    @benchmark(name="my-bench", dataset="hf://my/data?split=test",
               prompt="Question: {q}\\nAnswer:", target_field="answer")
    @scorer
    def score(sample: ScorerInput) -> dict:
        return exact_match(sample)

Extension hooks:
  - prepare_row(row, idx, rng) -> row:  transform each dataset row after loading
  - seed_fn(row, idx) -> SeedResult:    fully custom seed (overrides prompt template)
  - @image_builder(fn):                 declare images that need building (Docker)

Dataset specs:
  - "hf://dataset?split=test&config=cfg"  (HuggingFace)
  - "path/to/data.jsonl"                  (local JSONL)
  - callable returning list[dict]         (programmatic)
"""

from __future__ import annotations

import json
import logging
import random
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, TypeVar, cast, overload

from pydantic import BaseModel

if TYPE_CHECKING:
    from nemo_evaluator.sandbox.base import Sandbox

from nemo_evaluator.environments.base import EvalEnvironment, SeedResult, VerifyResult
from nemo_evaluator.environments.registry import register
from nemo_evaluator.sandbox.base import ImageBuildRequest, SandboxSpec
from nemo_evaluator.scoring.metric import (
    CandidateOutput,
    DatasetRow,
    Metric,
    MetricDescriptor,
    MetricInput,
    MetricOutputSpec,
    MetricResult,
    MetricScorerFunction,
    ScorerCallable,
    ScorerConfig,
    ScorerFunctionMetric,
    ScorerReturn,
    score_names_from_output_spec,
)
from nemo_evaluator.scoring.types import ScorerInput

logger = logging.getLogger(__name__)


# ── Data types ────────────────────────────────────────────────────────────


ConfigT = TypeVar("ConfigT", bound=Mapping[str, object] | BaseModel)
ConfigModelT = TypeVar("ConfigModelT", bound=BaseModel)


@dataclass
class BenchmarkDefinition:
    name: str
    dataset: str | Callable[..., list[dict[str, Any]]]
    prompt: str
    target_field: str = "target"
    endpoint_type: str = "chat"
    system_prompt: str | None = None
    field_mapping: dict[str, str] | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    requirements: list[str] | None = None
    scorer_fn: Callable[..., ScorerReturn] | None = None
    prepare_row: Callable[[dict[str, Any], int, random.Random], dict[str, Any]] | None = None
    seed_fn: Callable[[dict[str, Any], int], SeedResult] | None = None
    image_builder_fn: Callable[[list[dict[str, Any]]], ImageBuildRequest] | None = None


_BYOB_REGISTRY: dict[str, BenchmarkDefinition] = {}


# ── Dataset loading ───────────────────────────────────────────────────────


def _load_dataset_from_spec(
    spec: str | Callable[..., list[dict[str, Any]]],
    num_examples: int | None = None,
) -> list[dict[str, Any]]:
    if not isinstance(spec, str):
        import inspect

        sig = inspect.signature(spec)
        if "num_examples" in sig.parameters:
            rows = spec(num_examples=num_examples)
        else:
            rows = spec()
        return rows[:num_examples] if num_examples else rows

    if spec.startswith("hf://"):
        return _load_hf(spec[5:], num_examples=num_examples)

    path = Path(spec)
    if path.exists():
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return _load_csv(path, delimiter=",")
        if suffix in (".tsv", ".tab"):
            return _load_csv(path, delimiter="\t")
        rows = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
                    if num_examples and len(rows) >= num_examples:
                        break
        return rows

    return _load_hf(spec, num_examples=num_examples)


def _load_csv(path: Path, delimiter: str = ",") -> list[dict[str, Any]]:
    import csv

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        return [dict(row) for row in reader]


def _load_hf(spec: str, num_examples: int | None = None) -> list[dict[str, Any]]:
    from datasets import load_dataset

    parts = spec.split("?")
    dataset_name = parts[0]
    params: dict[str, str] = {}
    if len(parts) > 1:
        for kv in parts[1].split("&"):
            if "=" in kv:
                k, v = kv.split("=", 1)
                params[k] = v

    split = params.get("split", "test")
    config = params.get("config")

    if num_examples and "[" not in split:
        split = f"{split}[:{num_examples}]"

    args = [dataset_name]
    if config:
        args.append(config)
    ds = load_dataset(*args, split=split)
    return [dict(row) for row in ds]


def _format_prompt(template: str, row: dict[str, Any], field_mapping: dict[str, str] | None = None) -> str:
    data = dict(row)
    if field_mapping:
        for src, dst in field_mapping.items():
            if src in data:
                data[dst] = data[src]
    if template.endswith((".txt", ".md", ".jinja")) and Path(template).exists():
        template = Path(template).read_text()
    try:
        return template.format(**data)
    except KeyError as e:
        logger.warning("Prompt template has unknown field %s (available: %s)", e, list(data.keys()))
        return template


# ── Environment class ─────────────────────────────────────────────────────


class ByobEnvironment(EvalEnvironment):
    """Auto-generated from @benchmark + @scorer."""

    def __init__(self, definition: BenchmarkDefinition, num_examples: int | None = None) -> None:
        super().__init__()
        self._defn = definition
        self.name = definition.name

        raw = _load_dataset_from_spec(definition.dataset, num_examples=num_examples)

        rng = random.Random(42)
        if definition.prepare_row:
            raw = [definition.prepare_row(row, i, rng) for i, row in enumerate(raw)]

        self._dataset = raw
        logger.info("BYOB %s: %d samples", definition.name, len(raw))

    async def image_build_requests(self) -> list[ImageBuildRequest] | None:
        if not self._defn.image_builder_fn:
            return None
        result = self._defn.image_builder_fn(self._dataset)
        if not isinstance(result, ImageBuildRequest):
            raise TypeError(f"@image_builder must return ImageBuildRequest, got {type(result).__name__}")
        return [result]

    async def sandbox_specs(self) -> list[SandboxSpec] | None:
        if not self._defn.seed_fn:
            return None
        specs: list[SandboxSpec] = []
        for i, row in enumerate(self._dataset):
            seed = self._defn.seed_fn(row, i)
            if seed.sandbox_spec:
                specs.append(seed.sandbox_spec)
            if seed.verify_sandbox_spec:
                specs.append(seed.verify_sandbox_spec)
        return specs or None

    async def seed(self, idx: int) -> SeedResult:
        row = self._dataset[idx]

        if self._defn.seed_fn:
            return self._defn.seed_fn(row, idx)

        prompt = _format_prompt(self._defn.prompt, row, self._defn.field_mapping)
        target = str(row.get(self._defn.target_field, ""))

        meta: dict[str, Any] = {"source": "byob", "benchmark": self._defn.name}
        for k, v in row.items():
            meta[k] = v

        messages = [{"role": "user", "content": prompt}]
        if self._defn.system_prompt:
            messages.insert(0, {"role": "system", "content": self._defn.system_prompt})

        return SeedResult(
            prompt=prompt, expected_answer=target, messages=messages, system=self._defn.system_prompt, metadata=meta
        )

    async def verify(self, response: str, expected: str, sandbox: Sandbox | None = None, **meta: Any) -> VerifyResult:
        if self._defn.scorer_fn is None:
            correct = response.strip().lower() == expected.strip().lower()
            return VerifyResult(
                reward=1.0 if correct else 0.0,
                extracted_answer=response.strip()[:200],
                scoring_details={"method": "default_exact_match"},
            )

        import asyncio

        to_metric = getattr(self._defn.scorer_fn, "to_metric", None)
        if callable(to_metric):
            metric = cast(ScorerFunctionMetric[ScorerConfig], to_metric()).bind_raw_config(
                config=self._defn.extra,
                sandbox=sandbox,
                target=expected,
            )
            metric_input = _metric_input_from_verify(
                response=response,
                metadata=meta,
            )
            result = await metric.compute_scores(metric_input)
            return _metric_result_to_verify_result(
                metric=metric,
                result=result,
                benchmark_name=self._defn.name,
                response=response,
            )

        sample = ScorerInput(
            response=response, target=expected, metadata=meta, config=self._defn.extra, sandbox=sandbox
        )
        scores_result = self._defn.scorer_fn(sample)
        if asyncio.iscoroutine(scores_result):
            scores_result = await scores_result
        scores = cast(Mapping[str, object], scores_result)

        reward_value = scores.get("correct", scores.get("reward", next(iter(scores.values()), 0)))
        reward = float(reward_value) if isinstance(reward_value, bool | int | float) else 0.0
        extracted = scores.get("extracted")
        return VerifyResult(
            reward=reward,
            extracted_answer=extracted if isinstance(extracted, str) else response.strip()[:200],
            scoring_details={"method": f"byob_{self._defn.name}", **dict(scores)},
        )


def _metric_input_from_verify(
    *,
    response: str,
    metadata: dict[str, Any],
) -> MetricInput:
    row_data: dict[str, object] = dict(metadata)
    return MetricInput(
        row=DatasetRow(data=row_data),
        candidate=CandidateOutput(output_text=response),
    )


def _metric_result_to_verify_result(
    *,
    metric: Metric,
    result: MetricResult,
    benchmark_name: str,
    response: str,
) -> VerifyResult:
    outputs = {output.name: output.value for output in result.outputs}
    score_names = score_names_from_output_spec(metric.output_spec())
    scores = {name: _score_value(outputs[name]) for name in score_names if name in outputs}
    reward_name = _select_reward_score_name(scores=scores, declared=score_names)
    extracted = outputs.get("extracted")

    scoring_details: dict[str, Any] = {
        "method": f"byob_{benchmark_name}",
        "metric_type": metric.type,
        "outputs": outputs,
    }
    for name, value in outputs.items():
        scoring_details.setdefault(name, value)

    return VerifyResult(
        reward=scores[reward_name] if reward_name is not None else 0.0,
        extracted_answer=extracted if isinstance(extracted, str) else response.strip()[:200],
        scoring_details=scoring_details,
    )


def _select_reward_score_name(*, scores: dict[str, float], declared: list[str]) -> str | None:
    for preferred in ("reward", "correct"):
        if preferred in scores:
            return preferred
    for name in declared:
        if name in scores:
            return name
    return next(iter(scores), None)


def _score_value(value: object) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, int | float):
        return float(value)
    raise TypeError(f"Metric score output must be bool, int, or float, got {type(value).__name__}")


# ── Decorators ────────────────────────────────────────────────────────────


def benchmark(
    name: str,
    dataset: str | Callable[..., list[dict[str, Any]]],
    prompt: str = "",
    target_field: str = "target",
    endpoint_type: str = "chat",
    system_prompt: str | None = None,
    field_mapping: dict[str, str] | None = None,
    extra: dict[str, Any] | None = None,
    requirements: list[str] | None = None,
    prepare_row: Callable[[dict[str, Any], int, random.Random], dict[str, Any]] | None = None,
    seed_fn: Callable[[dict[str, Any], int], SeedResult] | None = None,
    **kwargs: Any,
):
    """Register a benchmark. Decorate a scorer function."""
    defn = BenchmarkDefinition(
        name=name,
        dataset=dataset,
        prompt=prompt,
        target_field=target_field,
        endpoint_type=endpoint_type,
        system_prompt=system_prompt,
        field_mapping=field_mapping,
        extra={**(extra or {}), **kwargs},
        requirements=requirements,
        prepare_row=prepare_row,
        seed_fn=seed_fn,
    )

    def decorator(fn):
        defn.scorer_fn = fn
        if hasattr(fn, "_image_builder_fn"):
            defn.image_builder_fn = fn._image_builder_fn
        _BYOB_REGISTRY[name] = defn

        @register(name)
        class _Env(ByobEnvironment):
            def __init__(self, num_examples: int | None = None):
                super().__init__(defn, num_examples)

        _Env.__name__ = f"Bench_{name}"
        _Env.__qualname__ = f"Bench_{name}"
        return fn

    return decorator


@overload
def scorer(
    fn: None = None,
    *,
    metric_type: str,
    outputs: list[MetricOutputSpec],
    config_schema: type[ConfigModelT],
) -> Callable[[ScorerCallable[ConfigModelT]], MetricScorerFunction[ConfigModelT]]: ...


@overload
def scorer(
    fn: ScorerCallable[ConfigModelT],
    *,
    metric_type: str,
    outputs: list[MetricOutputSpec],
    config_schema: type[ConfigModelT],
) -> MetricScorerFunction[ConfigModelT]: ...


@overload
def scorer(
    fn: None = None,
    *,
    metric_type: str,
    outputs: list[MetricOutputSpec],
    config_schema: None = None,
) -> Callable[[ScorerCallable[ConfigT]], MetricScorerFunction[ConfigT]]: ...


@overload
def scorer(
    fn: ScorerCallable[ConfigT],
    *,
    metric_type: str,
    outputs: list[MetricOutputSpec],
    config_schema: None = None,
) -> MetricScorerFunction[ConfigT]: ...


@overload
def scorer(fn: ScorerCallable[ConfigT]) -> ScorerCallable[ConfigT]: ...


@overload
def scorer(
    fn: None = None,
    *,
    metric_type: None = None,
    outputs: None = None,
    config_schema: None = None,
) -> Callable[[ScorerCallable[ConfigT]], ScorerCallable[ConfigT]]: ...


def scorer(
    fn: Callable[..., ScorerReturn] | None = None,
    *,
    metric_type: str | None = None,
    outputs: list[MetricOutputSpec] | None = None,
    config_schema: type[BaseModel] | None = None,
) -> object:
    """Marks a function as a scorer.

    Plain ``@scorer`` keeps the current ``ScorerInput -> dict`` behavior.
    ``@scorer(metric_type=..., outputs=...)`` exposes ``descriptor`` and
    ``to_metric()`` for adapting scorer functions to the shared Metric protocol.
    """
    if outputs is None and (metric_type is not None or config_schema is not None):
        metric_options = [
            option
            for option, value in (
                ("metric_type=...", metric_type),
                ("config_schema=...", config_schema),
            )
            if value is not None
        ]
        raise ValueError(
            f"@scorer({', '.join(metric_options)}) opts into the Metric contract, but no outputs were declared. "
            "Pass outputs=[MetricOutputSpec(...)] so the metric descriptor can declare and validate outputs."
        )
    if outputs is not None and metric_type is None:
        raise ValueError(
            "@scorer(outputs=...) opts into the Metric contract, but no metric_type was declared. "
            "Pass metric_type='...' so the metric has a stable identity across refactors."
        )

    def decorate(fn: Callable[..., ScorerReturn]) -> object:
        return _decorate_scorer(
            cast(ScorerCallable[ScorerConfig], fn),
            metric_type=metric_type,
            outputs=outputs,
            config_schema=config_schema,
        )

    return decorate(fn) if fn is not None else decorate


def _decorate_scorer(
    fn: ScorerCallable[ConfigT],
    *,
    metric_type: str | None = None,
    outputs: list[MetricOutputSpec] | None = None,
    config_schema: type[BaseModel] | None = None,
):
    setattr(fn, "_is_scorer", True)
    if outputs is None:
        return fn
    if metric_type is None:
        raise ValueError("metric_type is required when outputs are declared")

    descriptor = MetricDescriptor(
        type=metric_type,
        outputs=outputs,
        config_schema=config_schema,
    )

    def to_metric() -> ScorerFunctionMetric[ConfigT]:
        return ScorerFunctionMetric(
            descriptor=descriptor,
            scorer_fn=fn,
        )

    setattr(fn, "descriptor", descriptor)
    setattr(fn, "to_metric", to_metric)
    return fn

def image_builder(builder_fn: Callable[[list[dict[str, Any]]], ImageBuildRequest]):
    """Declare images that need building, stacked with ``@benchmark``.

    ``builder_fn`` receives the dataset rows and returns an
    :class:`ImageBuildRequest` (image names + Docker build callable).
    The :class:`SandboxManager` handles backend-specific conversion.

    Stack in the decorator chain between ``@benchmark`` and ``@scorer``::

        @benchmark(name="swebench-verified", ...)
        @image_builder(swebench_image_build_request)
        @scorer
        async def score(sample): ...
    """

    def decorator(fn):
        fn._image_builder_fn = builder_fn  # type: ignore[attr-defined]
        return fn

    return decorator
