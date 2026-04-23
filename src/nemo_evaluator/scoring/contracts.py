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

"""Metric abstractions for NEL and downstream providers (NMP SDK, etc.).

This module defines the shared contract that lets NeMo Evaluator (NEL) and
downstream metric providers (NMP's ``nemo_evaluator_sdk``, third-party
plugins) interoperate without duplicating evaluation logic.

## Concepts

The design centers four entities:

- :class:`MetricInput` — unified input dataclass (alias of :class:`ScorerInput`)
- :class:`MetricResult` — unified output Pydantic model (a.k.a. ``MetricOutput``)
- :class:`Metric` — object-style runtime contract (Protocol)
- :data:`Scorer` — function-style runtime contract (``Callable[[MetricInput], dict]``)

Function-style and object-style are two idioms over the *same* input/output
shape. The :func:`metric_as_scorer` helper bridges a :class:`Metric` into a
``Scorer``, so object-style metrics can register in NEL's
``_SCORER_REGISTRY`` and run inside NEL's existing evaluation loop.

## Authoring a metric

For ~20-30 LOC per metric, subclass :class:`TemplateMetric` and implement
``_score``::

    from typing import Literal
    from nemo_evaluator.scoring import TemplateMetric, MetricInput, register_metric

    @register_metric
    class MyMetric(TemplateMetric):
        type: Literal["my-metric"] = "my-metric"

        def _score(self, input: MetricInput) -> float:
            return 1.0 if input.response == input.target else 0.0

## ERD

::

    MetricInput ── consumed by ──> Scorer (callable)
                └─ consumed by ──> Metric (Protocol)
                                    │
    MetricResult <── returned by ───┤
                                    │
    TemplateMetric ──implements──────┘
    (concrete base class; users subclass)

## Departures from NMP SDK's pre-reshape Metric Protocol

- ``metric(item, sample, trace)`` is **not** part of the Protocol. SDK
  concrete metrics that keep it as an internal helper still satisfy the
  Protocol structurally; consumers must not rely on it.
- ``compute_scores`` takes a single :class:`MetricInput` instead of a pair
  of ``item``/``sample`` dicts. The input is the same dataclass NEL's
  BYOB function-style scorers already use, unifying the contract.
"""

from __future__ import annotations

import math
from abc import abstractmethod
from typing import Any, Awaitable, Callable, ClassVar, Protocol, runtime_checkable

from pydantic import BaseModel, Field, field_serializer, field_validator

from nemo_evaluator.scoring.types import ScorerInput


__all__ = [
    # Input/output
    "MetricInput",
    "MetricOutput",
    "MetricResult",
    "MetricScore",
    # Protocols
    "Metric",
    "CorpusMetric",
    "MetricWithSecrets",
    "Scorer",
    "SecretRefLike",
    "SecretResolver",
    # Base class for easy authoring
    "TemplateMetric",
    # Bridge
    "metric_as_scorer",
    # Registry
    "register_metric",
    "get_metric",
    "list_metrics",
]


# ============================================================================
# Input: unified contract for Scorer and Metric
# ============================================================================

MetricInput = ScorerInput
"""Unified input type for both function-style scorers and object-style metrics.

Aliased to :class:`nemo_evaluator.scoring.types.ScorerInput` so code that
already uses ``ScorerInput`` works unchanged. Exposing it under the
``MetricInput`` name here makes the ``Metric`` Protocol signatures read
naturally.
"""


# ============================================================================
# Output types (Pydantic, dep-light — no pyarrow/pandas/openai)
# ============================================================================


class MetricScore(BaseModel):
    """One named score emitted by a metric call.

    Providers that attach aggregate statistics (count, mean, stddev, rubric
    distributions, etc.) should subclass this model in their own package —
    those are provider-specific concerns, not part of NEL's contract.
    """

    name: str
    value: float

    @field_validator("value", mode="before")
    @classmethod
    def _parse_value(cls, v: Any) -> Any:
        if isinstance(v, str):
            if v.strip().lower() == "nan":
                return float("nan")
            raise ValueError("The only string value allowed for 'value' is NaN")
        return v

    @field_serializer("value")
    def _serialize_value(self, v: float) -> float | str:
        if isinstance(v, float) and math.isnan(v):
            return "NaN"
        return v


class MetricResult(BaseModel):
    """Result of one metric call: one or more named scores.

    Also exposed as :data:`MetricOutput` for symmetry with :data:`MetricInput`.
    """

    scores: list[MetricScore]


MetricOutput = MetricResult
"""Alias for :class:`MetricResult`. Use in signatures for readability."""


# ============================================================================
# Secrets (abstract)
# ============================================================================


@runtime_checkable
class SecretRefLike(Protocol):
    """Abstract secret reference.

    Concrete ``SecretRef`` types (with Pydantic root validation) live in
    NMP's SDK. NEL only needs the ``root`` accessor for the env-var name.
    """

    @property
    def root(self) -> str: ...


SecretResolver = Callable[[str], Awaitable[str | None]]
"""Async callable that resolves a secret name to its value (or None)."""


# ============================================================================
# Protocols: Scorer (function-style) and Metric (object-style)
# ============================================================================


Scorer = Callable[[MetricInput], dict]
"""Function-style scorer: ``(MetricInput) -> dict``.

This is NEL's native shape, used by the :func:`nemo_evaluator.scoring.get_scorer`
registry and by the ``@scorer`` decorator in the BYOB framework.
"""


@runtime_checkable
class Metric(Protocol):
    """Object-style metric contract.

    Concrete implementations (BLEU, ROUGE, LLM judge, RAGAS, etc.) satisfy
    this Protocol structurally — no base class required. For authoring
    ergonomics, NEL provides :class:`TemplateMetric` as a concrete starting
    point.

    The ``type`` attribute is a public string identifier. Providers may
    implement it with a plain string, a ``str``-backed ``Enum``, or a
    ``StrEnum``; consumers must treat it as a string.
    """

    @property
    def type(self) -> str:
        """Public string identifier for this metric."""
        ...

    async def compute_scores(self, input: MetricInput) -> MetricResult:
        """Compute structured scores for one input."""
        ...

    def score_names(self) -> list[str]:
        """Return the canonical score names this metric emits."""
        ...


@runtime_checkable
class CorpusMetric(Protocol):
    """Metrics that also emit corpus-level scores (e.g., BLEU corpus)."""

    async def compute_corpus_scores(
        self, inputs: list[MetricInput]
    ) -> MetricResult | None:
        """Compute corpus-level scores across all evaluated inputs."""
        ...


@runtime_checkable
class MetricWithSecrets(Protocol):
    """Metrics that depend on secrets (API keys for remote judges, etc.)."""

    def secrets(self) -> dict[str, SecretRefLike]:
        """Environment-variable names mapped to secret references."""
        ...

    async def resolve_secrets(self, resolver: SecretResolver) -> None:
        """Resolve declared secrets via the provided resolver before use."""
        ...


# ============================================================================
# TemplateMetric — concrete base for ~20-30 LOC metrics
# ============================================================================


class TemplateMetric(BaseModel):
    """Concrete base class for user-authored metrics.

    Subclasses declare Pydantic config fields + implement :meth:`_score`.
    Everything else (``compute_scores`` wrapping, ``score_names`` default,
    Pydantic validation) is inherited.

    Example::

        from typing import Literal
        from nemo_evaluator.scoring import TemplateMetric, MetricInput, register_metric

        @register_metric
        class ExactMatch(TemplateMetric):
            type: Literal["exact-match"] = "exact-match"

            def _score(self, input: MetricInput) -> float:
                return 1.0 if str(input.response).strip() == str(input.target).strip() else 0.0

    Subclasses MUST define:

    - a ``type`` field (typically ``Literal["..."] = "..."`` for discriminated-union Pydantic dispatch)
    - ``_score(input)`` — returns a single float score

    Subclasses MAY override:

    - ``compute_scores(input)`` — for multi-score output
    - ``score_names()`` — if emitting scores beyond ``[self.type]``
    """

    type: str = Field(description="Public string identifier for this metric.")

    @abstractmethod
    def _score(self, input: MetricInput) -> float:
        """Compute one raw score value. Subclass implements this."""
        ...

    async def compute_scores(self, input: MetricInput) -> MetricResult:
        """Default implementation: wrap ``_score`` in a single-score MetricResult."""
        return MetricResult(scores=[MetricScore(name=self.type, value=self._score(input))])

    def score_names(self) -> list[str]:
        """Default implementation: a single score named after ``self.type``."""
        return [self.type]


# ============================================================================
# Bridge: Metric -> function-style Scorer
# ============================================================================


def metric_as_scorer(metric: Metric) -> Scorer:
    """Adapt an object-style :class:`Metric` to a function-style :data:`Scorer`.

    NEL's scoring registry (``_SCORER_REGISTRY``) holds ``(MetricInput) -> dict``
    callables. This helper wraps an object-style :class:`Metric` so it can
    register there without glue code. The wrapper runs ``compute_scores``
    synchronously via ``asyncio.run`` or a fresh thread if a loop is active.
    """
    import asyncio
    import threading

    def _run_sync(coro_factory):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro_factory())

        result: list = []
        errors: list = []

        def _runner():
            try:
                result.append(asyncio.run(coro_factory()))
            except BaseException as e:  # pragma: no cover
                errors.append(e)

        t = threading.Thread(target=_runner, name="metric-as-scorer")
        t.start()
        t.join()
        if errors:
            raise errors[0]
        return result[0]

    def _scorer(input: MetricInput) -> dict:
        result = _run_sync(lambda: metric.compute_scores(input))
        out: dict[str, Any] = {"metric_type": metric.type}
        for score in result.scores:
            out[score.name] = score.value
        if len(result.scores) == 1:
            out["score"] = float(result.scores[0].value)
        return out

    return _scorer


# ============================================================================
# Registry: @register_metric class decorator
# ============================================================================


_METRIC_REGISTRY: dict[str, type[Metric]] = {}


def register_metric(cls: type[Metric]) -> type[Metric]:
    """Class decorator that registers a :class:`Metric` class by its ``type``.

    The class must have a ``type`` attribute (class-level or default field
    value) that returns the public identifier. Reads either a plain
    ``ClassVar`` or a Pydantic field default.

    Usage::

        @register_metric
        class MyMetric(TemplateMetric):
            type: Literal["my-metric"] = "my-metric"
            ...
    """
    type_id = _read_type_identifier(cls)
    if not type_id:
        raise ValueError(
            f"{cls.__name__}.type must be set as a class-level attribute or "
            "Pydantic field with a default value for @register_metric to work."
        )
    _METRIC_REGISTRY[type_id] = cls
    return cls


def _read_type_identifier(cls: type) -> str | None:
    """Best-effort read of ``cls.type`` without instantiating."""
    # Pydantic field with default
    fields = getattr(cls, "model_fields", None)
    if fields and "type" in fields:
        default = fields["type"].default
        if default is not None and default is not ...:
            return str(default)
    # Plain class attribute
    value = cls.__dict__.get("type")
    if isinstance(value, str):
        return value
    return None


def get_metric(name: str) -> type[Metric]:
    """Look up a registered metric class by its type identifier."""
    if name not in _METRIC_REGISTRY:
        raise KeyError(
            f"Unknown metric {name!r}. Registered: {sorted(_METRIC_REGISTRY)}"
        )
    return _METRIC_REGISTRY[name]


def list_metrics() -> list[str]:
    """Return the names of all registered metrics."""
    return sorted(_METRIC_REGISTRY)
