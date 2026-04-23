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
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from nemo_evaluator.sandbox.base import Sandbox

from nemo_evaluator.environments.base import EvalEnvironment, SeedResult, VerifyResult
from nemo_evaluator.sandbox.base import ImageBuildRequest, SandboxSpec
from nemo_evaluator.environments.registry import register
from nemo_evaluator.scoring.contracts import Metric, metric_as_scorer
from nemo_evaluator.scoring.types import ScorerInput

logger = logging.getLogger(__name__)


# ── Data types ────────────────────────────────────────────────────────────


@dataclass
class BenchmarkDefinition:
    name: str
    dataset: str | Callable[[], list[dict]]
    prompt: str
    target_field: str = "target"
    endpoint_type: str = "chat"
    system_prompt: str | None = None
    field_mapping: dict[str, str] | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    requirements: list[str] | None = None
    scorer_fn: Callable[[ScorerInput], dict] | None = None
    prepare_row: Callable[[dict, int, random.Random], dict] | None = None
    seed_fn: Callable[[dict, int], SeedResult] | None = None
    image_builder_fn: Callable[[list[dict]], ImageBuildRequest] | None = None


_BYOB_REGISTRY: dict[str, BenchmarkDefinition] = {}


# ── Dataset loading ───────────────────────────────────────────────────────


def _load_dataset_from_spec(spec: str | Callable, num_examples: int | None = None) -> list[dict[str, Any]]:
    if callable(spec):
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


def _format_prompt(template: str, row: dict, field_mapping: dict | None = None) -> str:
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

        sample = ScorerInput(
            response=response, target=expected, metadata=meta, config=self._defn.extra, sandbox=sandbox
        )
        scores = self._defn.scorer_fn(sample)
        if asyncio.iscoroutine(scores):
            scores = await scores

        reward = float(scores.get("correct", scores.get("reward", next(iter(scores.values()), 0))))
        return VerifyResult(
            reward=reward,
            extracted_answer=scores.get("extracted", response.strip()[:200]),
            scoring_details={"method": f"byob_{self._defn.name}", **scores},
        )


# ── Decorators ────────────────────────────────────────────────────────────


def benchmark(
    name: str,
    dataset: str | Callable,
    prompt: str = "",
    target_field: str = "target",
    endpoint_type: str = "chat",
    system_prompt: str | None = None,
    field_mapping: dict[str, str] | None = None,
    extra: dict[str, Any] | None = None,
    requirements: list[str] | None = None,
    prepare_row: Callable | None = None,
    seed_fn: Callable | None = None,
    metric: Metric | None = None,
    **kwargs,
):
    """Register a benchmark. Decorate a scorer function or a :class:`Metric` class.

    The decoration target may be:

    - a **function** ``(ScorerInput) -> dict`` (classic NEL BYOB)
    - a **class** that satisfies :class:`Metric` (e.g. a :class:`TemplateMetric`
      subclass with a no-arg constructor)
    - a **function**, with an **instance** passed via ``metric=`` kwarg

    Examples::

        # classic — function scorer
        @benchmark(name="my-bench", dataset="hf://...", prompt="{q}", target_field="a")
        @scorer
        def score(s): return {"correct": s.response == s.target}

        # object-style — TemplateMetric class
        @benchmark(name="my-bench", dataset="hf://...", prompt="{q}", target_field="a")
        class MyMetric(TemplateMetric):
            type: Literal["my-metric"] = "my-metric"
            def _score(self, input): return 1.0 if input.response == input.target else 0.0

        # object-style — pre-configured instance
        @benchmark(name="my-bench", ..., metric=BLEU(references="{{ reference }}"))
        def _placeholder(): ...
    """
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

    def decorator(target):
        # Resolve the scorer_fn from whichever form we got
        scorer_fn: Callable[[ScorerInput], dict] | None = None

        if metric is not None:
            # explicit instance via kwarg
            scorer_fn = metric_as_scorer(metric)
            image_src = target
        elif isinstance(target, type):
            # A class: must satisfy the Metric surface, else reject.
            if not _class_satisfies_metric(target):
                raise TypeError(
                    f"@benchmark target is a class ({target.__name__}) but does "
                    f"not satisfy the Metric contract (needs 'type', "
                    f"'compute_scores', 'score_names'). Provide a TemplateMetric "
                    f"subclass, a function scorer, or pass metric= kwarg."
                )
            # a Metric class — instantiate with no args, wrap
            try:
                instance = target()
            except Exception as e:  # ValidationError, TypeError, etc.
                raise TypeError(
                    f"@benchmark can only auto-instantiate Metric classes with "
                    f"a no-arg constructor. {target.__name__} requires "
                    f"arguments; pass a pre-configured instance via "
                    f"@benchmark(..., metric=MyMetric(...))"
                ) from e
            scorer_fn = metric_as_scorer(instance)
            image_src = target
        elif callable(target):
            # function-style scorer
            scorer_fn = target
            image_src = target
        else:
            raise TypeError(
                f"@benchmark target must be a callable scorer function, a "
                f"Metric class, or called with metric= kwarg. Got {type(target).__name__}."
            )

        defn.scorer_fn = scorer_fn
        if hasattr(image_src, "_image_builder_fn"):
            defn.image_builder_fn = image_src._image_builder_fn
        _BYOB_REGISTRY[name] = defn

        @register(name)
        class _Env(ByobEnvironment):
            def __init__(self, num_examples: int | None = None):
                super().__init__(defn, num_examples)

        _Env.__name__ = f"Bench_{name}"
        _Env.__qualname__ = f"Bench_{name}"
        return target

    return decorator


def _class_satisfies_metric(cls: type) -> bool:
    """Return True if a class has the method-name surface of :class:`Metric`.

    Avoids ``isinstance()`` which requires an instance. Handles Pydantic
    BaseModel subclasses where ``type`` is a field (not a class attr): in
    that case we check ``model_fields`` instead of ``hasattr``. Runtime
    Protocol checks happen post-instantiation inside the decorator.
    """
    if not (hasattr(cls, "compute_scores") and hasattr(cls, "score_names")):
        return False
    # Pydantic v2 fields live in model_fields, not as class-level attrs
    model_fields = getattr(cls, "model_fields", None)
    if model_fields and "type" in model_fields:
        return True
    return hasattr(cls, "type")


def scorer(fn: Callable[[ScorerInput], dict]) -> Callable[[ScorerInput], dict]:
    """Marks a function as a scorer."""
    fn._is_scorer = True  # type: ignore[attr-defined]
    return fn


def image_builder(builder_fn: Callable[[list[dict]], ImageBuildRequest]):
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
