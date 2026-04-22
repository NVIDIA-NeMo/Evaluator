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

"""Contracts between NeMo Evaluator and downstream metric providers.

This module defines the abstract interface that NMP's ``nemo_evaluator_sdk``
and other metric providers satisfy. Concrete metric implementations (BLEU,
ROUGE, LLM judge, RAGAS, etc.) live **outside** NEL — they are runtime
code with external dependencies (sacrebleu, openai, ragas, etc.). NEL
only owns the contract:

- **Metric protocols** describing how NEL calls a metric
- **Result types** describing what NEL receives back

The design mirrors Python's ``typing.Protocol`` approach: concrete classes
satisfy the contract structurally, no inheritance required. A metric
provider library depends on NEL for these contracts; NEL never depends on
any provider.

See also ``nemo_evaluator.scoring.types.ScorerInput`` — the companion
contract for function-style scorers ``(ScorerInput) -> dict``. Object-style
metrics (this module) and function-style scorers coexist; ``metric_as_scorer``
bridges the two.
"""

from __future__ import annotations

import math
from typing import Any, Awaitable, Callable, Protocol, runtime_checkable

from pydantic import BaseModel, Field, field_serializer, field_validator


__all__ = [
    "CorpusMetric",
    "Metric",
    "MetricResult",
    "MetricScore",
    "MetricWithPreflight",
    "MetricWithSecrets",
    "RubricScoreStat",
    "RubricScoreValue",
    "ScoreStats",
    "SecretRefLike",
    "SecretResolver",
    "metric_as_scorer",
]


# -----------------------------------------------------------------------------
# Score value types
# -----------------------------------------------------------------------------


class RubricScoreValue(BaseModel):
    """Rubric-based score definition for grading criteria."""

    label: str = Field(description="Label for this rubric level.")
    description: str | None = Field(
        default=None,
        description="Semantic meaning of the rubric level.",
    )
    value: float | int = Field(description="Score value assigned to this rubric level.")


class RubricScoreStat(RubricScoreValue):
    """Rubric score with sample-count statistics."""

    count: int = Field(default=0, description="Number of samples at this rubric level.")


class ScoreStats(BaseModel):
    """Stats for a score. NaN floats serialize as the string ``"NaN"`` for JSON portability."""

    count: int | None = None
    sum: float | None = None
    sum_squared: float | None = None
    min: float | None = None
    max: float | None = None
    mean: float | None = None
    variance: float | None = None
    stddev: float | None = None
    stderr: float | None = None
    nan_count: int | None = None
    rubric_distribution: list[RubricScoreStat] | None = None

    @field_serializer("sum", "sum_squared", "min", "max", "mean", "variance", "stddev", "stderr")
    def _serialize_nan(self, v: float | None) -> float | str | None:
        if v is None:
            return None
        if isinstance(v, float) and math.isnan(v):
            return "NaN"
        return v


class MetricScore(BaseModel):
    """One named score emitted by a metric call."""

    name: str
    value: float
    stats: ScoreStats | None = Field(
        default=None,
        description="Aggregate statistics for this score, if any.",
    )

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
    """Result of one metric call: one or more named scores."""

    scores: list[MetricScore]


# -----------------------------------------------------------------------------
# Secret references (abstract)
# -----------------------------------------------------------------------------


@runtime_checkable
class SecretRefLike(Protocol):
    """Abstract secret reference.

    Concrete ``SecretRef`` types (with Pydantic root validation, etc.) live in
    NMP's ``nemo_evaluator_sdk.values.common``. NEL only needs the ``root``
    accessor to describe the environment-variable name.
    """

    @property
    def root(self) -> str: ...


SecretResolver = Callable[[str], Awaitable[str | None]]


# -----------------------------------------------------------------------------
# Metric protocols
# -----------------------------------------------------------------------------


@runtime_checkable
class Metric(Protocol):
    """Structural contract for object-style metrics.

    Concrete implementations (BLEU, ROUGE, F1, ExactMatch, StringCheck,
    NumberCheck, ToolCalling, LLMJudge, Remote, RAGAS variants) live in
    ``nemo_evaluator_sdk`` and other provider libraries. They satisfy this
    Protocol by having the four members below — no base class required.

    ``type`` is a public string identifier. Providers may implement it with a
    plain string, a ``str``-backed ``Enum``, or a ``StrEnum``; consumers must
    treat it as a string and not rely on enum-only APIs (``.value`` etc.).
    """

    @property
    def type(self) -> str:
        """Public string identifier for this metric."""
        ...

    def metric(self, item: dict, sample: dict, trace: Any = None) -> float | bool:
        """Compute a single raw score for an (item, sample) pair."""
        ...

    async def compute_scores(self, item: dict, sample: dict) -> MetricResult:
        """Compute structured scores for an (item, sample) pair."""
        ...

    def score_names(self) -> list[str]:
        """Return the canonical score names this metric emits."""
        ...


@runtime_checkable
class CorpusMetric(Protocol):
    """Metrics that also emit corpus-level scores (e.g., BLEU corpus)."""

    async def compute_corpus_scores(
        self, items: list[dict], samples: list[dict]
    ) -> MetricResult | None:
        """Compute corpus-level scores across all evaluated rows."""
        ...


@runtime_checkable
class MetricWithSecrets(Protocol):
    """Metrics that depend on secrets (e.g., API keys for remote judges)."""

    def secrets(self) -> dict[str, SecretRefLike]:
        """Environment-variable names mapped to secret references."""
        ...

    async def resolve_secrets(self, resolver: SecretResolver) -> None:
        """Resolve declared secrets via the provided resolver before use."""
        ...


@runtime_checkable
class MetricWithPreflight(Protocol):
    """Metrics that need one-time setup before parallel evaluation starts."""

    async def preflight(self) -> None:
        """Run one-time preflight (capability detection, warm-up, etc.)."""
        ...


# -----------------------------------------------------------------------------
# Bridge: Metric -> NEL scorer callable
# -----------------------------------------------------------------------------


def metric_as_scorer(metric: Metric) -> Callable[[Any], dict]:
    """Adapt a :class:`Metric` to NEL's function-style scorer protocol.

    NEL's scoring registry (``nemo_evaluator.scoring._SCORER_REGISTRY``)
    holds callables ``(ScorerInput) -> dict``. This helper wraps an object-
    style ``Metric`` so it can register there without any glue code.

    Mapping performed:

    - ``item   = {"reference": scorer_input.target, **scorer_input.metadata}``
    - ``sample = {"output_text": scorer_input.response, "response": scorer_input.response}``
    - ``score  = metric.metric(item, sample)``

    Providers whose metric templates reference these keys (e.g.
    ``reference="{{ reference }}"``) plug in unchanged.
    """
    from nemo_evaluator.scoring.types import ScorerInput  # local: avoid cycles

    def _scorer(scorer_input: ScorerInput) -> dict:
        item: dict = {"reference": scorer_input.target, **(scorer_input.metadata or {})}
        sample: dict = {
            "output_text": scorer_input.response,
            "response": scorer_input.response,
        }
        score = metric.metric(item, sample)
        score_value = float(score) if isinstance(score, (bool, int, float)) else 0.0
        return {
            "score": score_value,
            "metric_type": getattr(metric, "type", type(metric).__name__),
        }

    return _scorer
