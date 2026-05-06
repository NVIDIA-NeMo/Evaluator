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
"""Tests for the ATIF retrieved context presence metric."""

from __future__ import annotations

import pytest
from nat.atif import (
    ATIFAgentConfig,
    ATIFContentPart,
    ATIFImageSource,
    ATIFObservation,
    ATIFObservationResult,
    ATIFStep,
    ATIFTrajectory,
)
from typing import Any

from pydantic import ValidationError

from nemo_evaluator.metrics import ExactMatchScorer
from nemo_evaluator.metrics import ParameterizedExactMatchConfig
from nemo_evaluator.metrics import ParameterizedExactMatchScorer
from nemo_evaluator.metrics.retrieved_context import RetrievedContextPresenceMetric
from nemo_evaluator.scoring import CandidateOutput, DatasetRow, MetricInput, MetricOutput
from nemo_evaluator.scoring import MetricOutputSpec
from nemo_evaluator.scoring.metric import validate_metric_result


def _trajectory(*contexts: str) -> ATIFTrajectory:
    return ATIFTrajectory(
        session_id="test-session",
        agent=ATIFAgentConfig(name="test-agent", version="0.0.0"),
        steps=[
            ATIFStep(step_id=1, source="user", message="question"),
            ATIFStep(
                step_id=2,
                source="agent",
                message="answer",
                observation=ATIFObservation(
                    results=[ATIFObservationResult(content=context) for context in contexts],
                ),
            ),
        ],
    )


@pytest.mark.asyncio
async def test_retrieved_context_presence_scores_matching_atif_context() -> None:
    metric = RetrievedContextPresenceMetric(expected_context="{{item.expected_context}}")
    metric_input = MetricInput(
        row=DatasetRow(data={"expected_context": "Paris is the capital of France"}),
        candidate=CandidateOutput(trajectory=_trajectory("Paris is the capital of France.")),
    )

    result = validate_metric_result(await metric.compute_scores(metric_input), metric.output_spec())

    assert result.outputs == [
        MetricOutput(name="context_found", value=1.0),
        MetricOutput(name="retrieved_contexts", value={"contexts": ["Paris is the capital of France."]}),
        MetricOutput(
            name="reasoning",
            value={
                "expected_context": "Paris is the capital of France",
                "matched_contexts": ["Paris is the capital of France."],
                "checked_context_count": 1,
                "case_sensitive": False,
            },
        ),
    ]


@pytest.mark.asyncio
async def test_retrieved_context_presence_scores_missing_context() -> None:
    metric = RetrievedContextPresenceMetric(expected_context="Berlin", case_sensitive=True)
    metric_input = MetricInput(
        row=DatasetRow(data={}),
        candidate=CandidateOutput(trajectory=_trajectory("Paris is the capital of France.")),
    )

    result = await metric.compute_scores(metric_input)

    assert result.outputs[0] == MetricOutput(name="context_found", value=0.0)
    assert result.outputs[2] == MetricOutput(
        name="reasoning",
        value={
            "expected_context": "Berlin",
            "matched_contexts": [],
            "checked_context_count": 1,
            "case_sensitive": True,
        },
    )


@pytest.mark.asyncio
async def test_retrieved_context_presence_extracts_text_parts_and_subagent_contexts() -> None:
    subagent = _trajectory("Subagent retrieved context")
    trajectory = ATIFTrajectory(
        session_id="root-session",
        agent=ATIFAgentConfig(name="root-agent", version="0.0.0"),
        steps=[
            ATIFStep(
                step_id=1,
                source="agent",
                message="answer",
                observation=ATIFObservation(
                    results=[
                        ATIFObservationResult(
                            content=[
                                ATIFContentPart(type="text", text="Root retrieved context"),
                                ATIFContentPart(
                                    type="image",
                                    source=ATIFImageSource(media_type="image/png", path="context.png"),
                                ),
                            ]
                        )
                    ]
                ),
            ),
        ],
        subagent_trajectories=[subagent.model_copy(update={"trajectory_id": "subagent-1"})],
    )
    metric = RetrievedContextPresenceMetric(expected_context="subagent retrieved")

    result = await metric.compute_scores(MetricInput(row=DatasetRow(data={}), candidate=CandidateOutput(trajectory=trajectory)))

    assert result.outputs[0] == MetricOutput(name="context_found", value=1.0)
    assert result.outputs[1] == MetricOutput(
        name="retrieved_contexts",
        value={"contexts": ["Root retrieved context", "Subagent retrieved context"]},
    )


@pytest.mark.asyncio
async def test_retrieved_context_presence_requires_atif_trajectory() -> None:
    metric = RetrievedContextPresenceMetric(expected_context="context")

    with pytest.raises(ValueError, match="requires candidate.trajectory"):
        await metric.compute_scores(MetricInput(row=DatasetRow(data={}), candidate=CandidateOutput()))


def test_candidate_output_rejects_non_atif_trajectory() -> None:
    trajectory: Any = [{"step": 1, "action": "not atif"}]
    with pytest.raises(ValidationError):
        CandidateOutput(trajectory=trajectory)


async def test_exact_match_scorer_is_metric_compatible_from_metric_library() -> None:
    metric = ExactMatchScorer.to_metric().bind(target_field="expected_output_obj")

    result = await metric.compute_scores(
        MetricInput(
            row=DatasetRow(data={"expected_output_obj": "The Eiffel Tower"}),
            candidate=CandidateOutput(output_text="the eiffel tower"),
        )
    )

    assert ExactMatchScorer.descriptor.type == "exact_match"
    assert ExactMatchScorer.descriptor.outputs == [MetricOutputSpec.continuous_score("correct")]
    assert result.outputs == [MetricOutput(name="correct", value=True)]


@pytest.mark.asyncio
async def test_parameterized_exact_match_scorer_exposes_and_uses_config_schema() -> None:
    metric = ParameterizedExactMatchScorer.to_metric().bind_raw_config(
        config={"case_sensitive": True},
        target_field="expected_output_obj",
    )

    result = await metric.compute_scores(
        MetricInput(
            row=DatasetRow(data={"expected_output_obj": "The Eiffel Tower"}),
            candidate=CandidateOutput(output_text="the eiffel tower"),
        )
    )

    assert ParameterizedExactMatchScorer.descriptor.type == "parameterized_exact_match"
    assert ParameterizedExactMatchScorer.descriptor.config_schema is ParameterizedExactMatchConfig
    assert ParameterizedExactMatchScorer.descriptor.outputs == [MetricOutputSpec.continuous_score("correct")]
    assert result.outputs == [MetricOutput(name="correct", value=False)]


def test_parameterized_exact_match_scorer_rejects_unknown_config_fields() -> None:
    metric = ParameterizedExactMatchScorer.to_metric()

    with pytest.raises(ValidationError):
        metric.bind_raw_config(config={"case_sensitive": True, "unknown": True})
