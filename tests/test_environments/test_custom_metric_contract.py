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
"""Tests for BYOB scorer compatibility with the shared metric contract."""

from __future__ import annotations

from typing import cast

import pytest
from pydantic import BaseModel

from nemo_evaluator.environments.custom import BenchmarkDefinition, ByobEnvironment, scorer
from nemo_evaluator.scoring import ScorerInput
from nemo_evaluator.scoring.metric import (
    MetricOutputSpec,
)
from nemo_evaluator.sandbox.base import Sandbox


def _dataset() -> list[dict[str, object]]:
    return [{"question": "2+2", "answer": "4", "category": "math"}]


class ThresholdConfig(BaseModel):
    threshold: float


@pytest.mark.asyncio
async def test_plain_scorer_decorator_keeps_current_dict_path() -> None:
    @scorer
    def plain_scorer(sample: ScorerInput) -> dict[str, object]:
        assert sample.response == "4"
        assert sample.target == "4"
        assert sample.metadata["category"] == "math"
        assert sample.config["tolerance"] == "exact"
        return {"correct": True, "extracted": "4", "label": "exact"}

    env = ByobEnvironment(
        BenchmarkDefinition(
            name="plain_contract",
            dataset=_dataset,
            prompt="{question}",
            target_field="answer",
            extra={"tolerance": "exact"},
            scorer_fn=plain_scorer,
        )
    )

    result = await env.verify("4", "4", category="math")

    assert result.reward == 1.0
    assert result.extracted_answer == "4"
    assert result.scoring_details == {
        "method": "byob_plain_contract",
        "correct": True,
        "extracted": "4",
        "label": "exact",
    }


@pytest.mark.asyncio
async def test_typed_scorer_runs_as_metric_through_byob_verify() -> None:
    outputs = [
        MetricOutputSpec.continuous_score("reward"),
        MetricOutputSpec.continuous_score("format"),
        MetricOutputSpec.label("judge_label"),
        MetricOutputSpec.label("extracted"),
    ]
    sandbox = cast(Sandbox, object())

    @scorer(metric_type="tests.typed_byob", outputs=outputs, config_schema=ThresholdConfig)
    async def typed_scorer(sample: ScorerInput[ThresholdConfig]) -> dict[str, object]:
        assert sample.response == "4"
        assert sample.target == "4"
        assert "answer" not in sample.metadata
        assert sample.metadata["category"] == "math"
        assert isinstance(sample.config, ThresholdConfig)
        assert sample.config.threshold == 0.75
        assert sample.sandbox is sandbox
        return {"reward": sample.config.threshold, "format": 1.0, "judge_label": "partial", "extracted": "4"}

    env = ByobEnvironment(
        BenchmarkDefinition(
            name="typed_contract",
            dataset=_dataset,
            prompt="{question}",
            target_field="answer",
            extra={"threshold": "0.75"},
            scorer_fn=typed_scorer,
        )
    )

    result = await env.verify("4", "4", sandbox=sandbox, category="math")

    assert result.reward == 0.75
    assert result.extracted_answer == "4"
    assert result.scoring_details == {
        "method": "byob_typed_contract",
        "metric_type": "tests.typed_byob",
        "outputs": {"reward": 0.75, "format": 1.0, "judge_label": "partial", "extracted": "4"},
        "reward": 0.75,
        "format": 1.0,
        "judge_label": "partial",
        "extracted": "4",
    }


@pytest.mark.parametrize(
    ("score_names", "score_values", "expected_reward"),
    [
        (["reward", "correct"], {"reward": 0.2, "correct": 1.0}, 0.2),
        (["quality", "correct"], {"quality": 0.2, "correct": 1.0}, 1.0),
        (["quality", "format"], {"quality": 0.4, "format": 1.0}, 0.4),
    ],
)
@pytest.mark.asyncio
async def test_typed_scorer_reward_selection(
    score_names: list[str], score_values: dict[str, float], expected_reward: float
) -> None:
    outputs = [MetricOutputSpec.continuous_score(name) for name in score_names]

    @scorer(metric_type=f"tests.reward_selection.{score_names[0]}", outputs=outputs)
    def typed_scorer(sample: ScorerInput) -> dict[str, object]:
        assert sample.response == "4"
        assert sample.target == "4"
        return {name: score_values[name] for name in score_names}

    env = ByobEnvironment(
        BenchmarkDefinition(
            name=f"typed_reward_{score_names[0]}",
            dataset=_dataset,
            prompt="{question}",
            target_field="answer",
            scorer_fn=typed_scorer,
        )
    )

    result = await env.verify("4", "4", category="math")

    assert result.reward == expected_reward
