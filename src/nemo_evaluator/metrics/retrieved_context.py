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
"""ATIF trajectory-aware retrieved context metric."""

from __future__ import annotations

from typing import Literal

from jinja2 import Environment, StrictUndefined
from nat.atif import ATIFTrajectory
from nat.atif import ContentPart
from pydantic import BaseModel, ConfigDict, Field

from nemo_evaluator.scoring import MetricInput, MetricOutput, MetricOutputSpec, MetricResult

_JINJA_ENV = Environment(undefined=StrictUndefined, autoescape=False)

__all__ = [
    "RetrievedContextPresenceMetric",
    "RetrievedContextPresenceReasoning",
    "RetrievedContexts",
]


class RetrievedContexts(BaseModel):
    """Text contexts extracted from ATIF observation results."""

    model_config = ConfigDict(extra="forbid")

    contexts: list[str]


class RetrievedContextPresenceReasoning(BaseModel):
    """Explanation for retrieved context presence scoring."""

    model_config = ConfigDict(extra="forbid")

    expected_context: str
    matched_contexts: list[str]
    checked_context_count: int
    case_sensitive: bool


class RetrievedContextPresenceMetric(BaseModel):
    """Score whether an expected context appears in ATIF observation results."""

    model_config = ConfigDict(extra="forbid")

    expected_context: str = Field(description="Jinja template for the expected retrieved context.")
    case_sensitive: bool = Field(default=False, description="Whether context matching should be case-sensitive.")
    type: Literal["retrieved-context-presence"] = Field(
        default="retrieved-context-presence",
        description="Stable metric type identifier.",
    )

    def output_spec(self) -> list[MetricOutputSpec]:
        return [
            MetricOutputSpec.continuous_score("context_found"),
            MetricOutputSpec.model("retrieved_contexts", RetrievedContexts),
            MetricOutputSpec.model("reasoning", RetrievedContextPresenceReasoning),
        ]

    async def compute_scores(self, input: MetricInput) -> MetricResult:
        if input.candidate.trajectory is None:
            raise ValueError("RetrievedContextPresenceMetric requires candidate.trajectory with an ATIF trajectory.")

        trajectory = input.candidate.trajectory
        item, sample = _template_payload_from_metric_input(input)
        expected_context = _render_template(self.expected_context, {**item, **sample, "item": item, "sample": sample})
        retrieved_contexts = _observation_texts_from_trajectory(trajectory)
        matched_contexts = [
            context for context in retrieved_contexts if _contains(context, expected_context, self.case_sensitive)
        ]
        score = 1.0 if matched_contexts else 0.0

        return MetricResult(
            outputs=[
                MetricOutput(name="context_found", value=score),
                MetricOutput(
                    name="retrieved_contexts",
                    value=RetrievedContexts(contexts=retrieved_contexts).model_dump(),
                ),
                MetricOutput(
                    name="reasoning",
                    value=RetrievedContextPresenceReasoning(
                        expected_context=expected_context,
                        matched_contexts=matched_contexts,
                        checked_context_count=len(retrieved_contexts),
                        case_sensitive=self.case_sensitive,
                    ).model_dump(),
                ),
            ]
        )


def _template_payload_from_metric_input(input: MetricInput) -> tuple[dict[str, object], dict[str, object]]:
    item = dict(input.row.data)
    sample = dict(input.candidate.metadata)
    if input.candidate.output_text is not None:
        sample["output_text"] = input.candidate.output_text
    if input.candidate.response is not None:
        sample["response"] = input.candidate.response
    if input.candidate.trajectory is not None:
        sample["trajectory"] = input.candidate.trajectory
    return item, sample


def _render_template(template: str, context: dict[str, object]) -> str:
    return _JINJA_ENV.from_string(template).render(context)


def _contains(context: str, expected_context: str, case_sensitive: bool) -> bool:
    if case_sensitive:
        return expected_context in context
    return expected_context.casefold() in context.casefold()


def _observation_texts_from_trajectory(trajectory: ATIFTrajectory) -> list[str]:
    contexts: list[str] = []
    for step in trajectory.steps:
        if step.observation is None:
            continue
        for result in step.observation.results:
            contexts.extend(_content_to_texts(result.content))
    for subagent_trajectory in trajectory.subagent_trajectories or []:
        contexts.extend(_observation_texts_from_trajectory(subagent_trajectory))
    return contexts


def _content_to_texts(content: str | list[ContentPart] | None) -> list[str]:
    if content is None:
        return []
    if isinstance(content, str):
        return [content] if content else []
    return [part.text for part in content if part.type == "text" and part.text]
