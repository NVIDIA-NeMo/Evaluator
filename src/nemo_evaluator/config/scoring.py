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
"""Scoring / metric configuration schemas."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Discriminator, Field, Tag, model_validator


class ScorerMetric(BaseModel):
    """A function-based scorer from the registry."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["scorer"] = "scorer"
    name: str


class _JudgeBase(BaseModel):
    """Shared validation for judge metrics."""

    model_config = ConfigDict(extra="forbid")

    rubric: str | None = None
    rubric_file: str | None = None

    @model_validator(mode="after")
    def _rubric_xor(self) -> _JudgeBase:
        if self.rubric and self.rubric_file:
            raise ValueError("Specify 'rubric' (inline) or 'rubric_file' (path), not both.")
        return self


class JudgeMetric(_JudgeBase):
    """An LLM-as-judge metric."""

    type: Literal["judge"] = "judge"
    name: str
    service: str
    system_prompt: str | None = None
    max_score: float = 5.0
    swap_check: bool = False
    reference_free: bool = False
    temperature: float = 0.0
    timeout: float = 120.0
    max_retries: int = 2
    allow_self_judge: bool = False


class PairwiseJudgeMetric(_JudgeBase):
    """Pairwise comparison judge."""

    type: Literal["pairwise_judge"] = "pairwise_judge"
    name: str
    judge_service: str
    baseline_service: str
    max_score: float = 5.0
    swap_check: bool = True
    temperature: float = 0.0
    timeout: float = 120.0
    max_retries: int = 2


class EnsembleJudgeMetric(_JudgeBase):
    """Ensemble: multiple judge models score independently."""

    type: Literal["ensemble_judge"] = "ensemble_judge"
    name: str
    services: list[str] = Field(min_length=2)
    aggregation: Literal["mean", "median", "majority_vote"] = "mean"
    max_score: float = 5.0
    temperature: float = 0.0
    timeout: float = 120.0
    max_retries: int = 2


class SandboxMetric(BaseModel):
    """Sandbox-based evaluation (run command, check exit code)."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["sandbox"] = "sandbox"
    name: str
    command: str | None = None
    verify_timeout: float = 600.0


class RewardModelMetric(BaseModel):
    """A reward-model-as-judge metric."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["reward_model"] = "reward_model"
    name: str
    service: str
    mode: Literal["single", "multi_aspect", "pairwise"] = "single"
    aspect: str | None = None
    score_range: tuple[float, float] = (0.0, 1.0)
    threshold: float = 0.5
    timeout: float = 120.0
    max_retries: int = 2

    @model_validator(mode="after")
    def _validate_score_range(self) -> RewardModelMetric:
        lo, hi = self.score_range
        if lo >= hi:
            raise ValueError(f"score_range[0] ({lo}) must be less than score_range[1] ({hi})")
        return self


class CustomMetric(BaseModel):
    """Plugin metric — dynamically imported from class_path."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["custom"] = "custom"
    name: str
    class_path: str
    config: dict[str, Any] = Field(default_factory=dict)


def _metric_discriminator(v: Any) -> str:
    if isinstance(v, dict):
        t = v.get("type")
        if t is None:
            raise ValueError("Metric config must include a 'type' field")
        return t
    t = getattr(v, "type", None)
    if t is None:
        raise ValueError(f"Cannot determine metric type from {type(v).__name__}. Expected a dict with a 'type' field.")
    return t


MetricConfig = Annotated[
    Annotated[ScorerMetric, Tag("scorer")]
    | Annotated[JudgeMetric, Tag("judge")]
    | Annotated[PairwiseJudgeMetric, Tag("pairwise_judge")]
    | Annotated[EnsembleJudgeMetric, Tag("ensemble_judge")]
    | Annotated[SandboxMetric, Tag("sandbox")]
    | Annotated[RewardModelMetric, Tag("reward_model")]
    | Annotated[CustomMetric, Tag("custom")],
    Discriminator(_metric_discriminator),
]


class ScoringConfig(BaseModel):
    """Scoring pipeline for a benchmark."""

    model_config = ConfigDict(extra="forbid")

    include_defaults: bool = True
    metrics: list[MetricConfig] = Field(default_factory=list)
    primary: str | None = None

    @model_validator(mode="after")
    def _validate_scoring(self) -> ScoringConfig:
        names = [m.name for m in self.metrics]
        dupes = [n for n in names if names.count(n) > 1]
        if dupes:
            raise ValueError(f"Duplicate metric names: {sorted(set(dupes))}. Each must be unique.")

        if self.primary is not None and self.metrics:
            metric_names = set(names)
            if self.primary not in metric_names:
                raise ValueError(
                    f"scoring.primary={self.primary!r} not found in metrics. Available: {sorted(metric_names)}"
                )
        if len(self.metrics) > 1 and self.primary is None:
            raise ValueError("scoring.primary is required when multiple metrics are defined.")
        return self
