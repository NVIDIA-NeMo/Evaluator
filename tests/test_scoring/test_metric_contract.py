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
"""Tests for the shared MetricInput -> MetricResult contract."""

from __future__ import annotations

from typing import Protocol, cast

import pytest
from pydantic import BaseModel, ValidationError

from nemo_evaluator.environments.custom import scorer
from nemo_evaluator.metrics import ExactMatchMetric
from nemo_evaluator.scorers import ExactMatchScorer
from nemo_evaluator.scoring import ScorerInput
from nemo_evaluator.scoring.metric import (
    CandidateOutput,
    ContinuousScore,
    DatasetRow,
    Label,
    Metric,
    MetricDescriptor,
    MetricInput,
    MetricOutput,
    MetricOutputSpec,
    MetricResult,
    MetricScorerFunction,
    ScorerFunctionMetric,
    score_names_from_output_spec,
    validate_metric_result,
)
from nemo_evaluator.sandbox.base import Sandbox


class ThresholdConfig(BaseModel):
    threshold: float
    label: str = "pass"


class OtherThresholdConfig(BaseModel):
    threshold: float
    label: str = "other"


class JudgeDetails(BaseModel):
    label: str
    rationale: str
    confidence: float


class _MetricScorerForTest(Protocol):
    @property
    def descriptor(self) -> MetricDescriptor: ...

    def to_metric(self) -> Metric: ...


def test_metric_input_groups_row_and_candidate() -> None:
    metric_input = MetricInput(
        row=DatasetRow(row_index=7, data={"answer": "Paris", "category": "geography"}),
        candidate=CandidateOutput(output_text="Paris", metadata={"model": "mock"}),
    )

    assert metric_input.row.row_index == 7
    assert metric_input.candidate.output_text == "Paris"
    assert metric_input.row.data["answer"] == "Paris"
    assert not hasattr(metric_input, "sandbox")
    assert not hasattr(metric_input, "config")


def test_metric_output_spec_convenience_constructors_and_json_schema() -> None:
    score = MetricOutputSpec.continuous_score("reward", "Reward score")
    label = MetricOutputSpec.label("judge_label")
    details = MetricOutputSpec.model("judge_details", JudgeDetails)

    assert score.name == "reward"
    assert score.description == "Reward score"
    assert score.value_schema is ContinuousScore
    assert score.value_json_schema()["type"] == "number"
    assert label.value_schema is Label
    assert details.value_schema is JudgeDetails
    schema_properties = cast(dict[str, object], details.value_json_schema()["properties"])
    confidence_schema = cast(dict[str, object], schema_properties["confidence"])
    assert confidence_schema["type"] == "number"


def test_metric_output_spec_coerces_values_to_declared_schema() -> None:
    reward = MetricOutputSpec.continuous_score("reward")
    details = MetricOutputSpec.model("judge_details", JudgeDetails)

    coerced_reward = reward.coerce_output(MetricOutput(name="reward", value=1))
    coerced_details = details.coerce_value(
        {"label": "pass", "rationale": "all checks passed", "confidence": 0.9}
    )

    assert isinstance(coerced_reward, ContinuousScore)
    assert coerced_reward.root == 1.0
    assert isinstance(coerced_details, JudgeDetails)
    assert coerced_details.label == "pass"

    with pytest.raises(ValueError, match="Expected metric output"):
        reward.coerce_output(MetricOutput(name="other", value=1))


@pytest.mark.asyncio
async def test_scorer_function_metric_adapts_dict_return_to_metric_outputs() -> None:
    outputs = [
        MetricOutputSpec.boolean("correct"),
        MetricOutputSpec.continuous_score("reward"),
        MetricOutputSpec.discrete_score("attempts"),
        MetricOutputSpec.label("extracted"),
        MetricOutputSpec.model("judge", JudgeDetails),
    ]
    descriptor = MetricDescriptor(type="tests.dict_adapter", outputs=outputs)

    def sync_scorer(sample: ScorerInput) -> dict[str, object]:
        return {
            "correct": True,
            "reward": 0.25,
            "attempts": 2,
            "extracted": "A",
            "judge": {"label": "partial", "rationale": "close", "confidence": 0.5},
        }

    metric = ScorerFunctionMetric(descriptor=descriptor, scorer_fn=sync_scorer)

    result = await metric.compute_scores(
        MetricInput(row=DatasetRow(data={}), candidate=CandidateOutput(output_text="candidate"))
    )

    assert {output.name: output.value for output in result.outputs} == {
        "correct": True,
        "reward": 0.25,
        "attempts": 2,
        "extracted": "A",
        "judge": {"label": "partial", "rationale": "close", "confidence": 0.5},
    }


def test_validate_metric_result_accepts_declared_outputs() -> None:
    outputs = [
        MetricOutputSpec.continuous_score("reward"),
        MetricOutputSpec.boolean("correct"),
        MetricOutputSpec.label("label"),
    ]
    result = MetricResult(
        outputs=[
            MetricOutput(name="reward", value=True),
            MetricOutput(name="correct", value=True),
            MetricOutput(name="label", value="yes"),
        ]
    )

    validated = validate_metric_result(result, outputs)

    assert validated.outputs == [
        MetricOutput(name="reward", value=True),
        MetricOutput(name="correct", value=True),
        MetricOutput(name="label", value="yes"),
    ]


def test_typed_scorer_decorator_exposes_config_schema() -> None:
    outputs = [MetricOutputSpec.continuous_score("reward")]

    @scorer(metric_type="tests.threshold", outputs=outputs, config_schema=ThresholdConfig)
    def threshold_scorer(sample: ScorerInput[ThresholdConfig]) -> dict[str, object]:
        return {"reward": sample.config.threshold >= 0.5}

    typed_scorer: MetricScorerFunction[ThresholdConfig] = threshold_scorer
    metric: ScorerFunctionMetric[ThresholdConfig] = typed_scorer.to_metric()

    assert isinstance(metric, BaseModel)
    assert threshold_scorer.descriptor.config_schema is ThresholdConfig
    assert metric.type == "tests.threshold"


def test_typed_scorer_decorator_requires_metric_type_and_outputs_for_metric_contract_options() -> None:
    with pytest.raises(ValueError, match=r"Metric contract.*outputs=\[MetricOutputSpec"):
        scorer(metric_type="tests.missing_outputs")  # type: ignore[reportArgumentType]  # ty: ignore[invalid-argument-type]

    with pytest.raises(ValueError, match=r"Metric contract.*outputs=\[MetricOutputSpec"):
        scorer(config_schema=ThresholdConfig)  # type: ignore[reportArgumentType]  # ty: ignore[invalid-argument-type]

    with pytest.raises(ValueError, match="no metric_type was declared"):
        scorer(outputs=[MetricOutputSpec.continuous_score("reward")])  # type: ignore[call-overload]  # ty: ignore[invalid-argument-type]


@pytest.mark.asyncio
async def test_scorer_function_metric_prefers_bound_target_over_row_field() -> None:
    outputs = [MetricOutputSpec.boolean("reward")]
    descriptor = MetricDescriptor(type="tests.bound_target", outputs=outputs)

    def sync_scorer(sample: ScorerInput) -> dict[str, object]:
        assert sample.target == "expected-from-verify"
        assert sample.metadata["answer"] == "answer-from-row"
        return {"reward": True}

    metric = ScorerFunctionMetric(descriptor=descriptor, scorer_fn=sync_scorer).bind(
        target="expected-from-verify",
        target_field="answer",
    )

    result = await metric.compute_scores(
        MetricInput(
            row=DatasetRow(data={"answer": "answer-from-row"}),
            candidate=CandidateOutput(output_text="candidate"),
        )
    )

    assert result.outputs == [MetricOutput(name="reward", value=True)]


@pytest.mark.asyncio
async def test_scorer_function_metric_accepts_typed_config_model() -> None:
    outputs = [MetricOutputSpec.continuous_score("reward")]
    descriptor = MetricDescriptor(type="tests.typed_config", outputs=outputs, config_schema=ThresholdConfig)

    def sync_scorer(sample: ScorerInput[ThresholdConfig]) -> dict[str, object]:
        assert isinstance(sample.config, ThresholdConfig)
        assert sample.config.threshold == 0.5
        assert sample.config.label == "typed"
        return {"reward": sample.config.threshold}

    metric = ScorerFunctionMetric(descriptor=descriptor, scorer_fn=sync_scorer).bind(
        config=ThresholdConfig(threshold=0.5, label="typed")
    )

    result = await metric.compute_scores(
        MetricInput(row=DatasetRow(data={}), candidate=CandidateOutput(output_text="candidate"))
    )

    assert result.outputs == [MetricOutput(name="reward", value=0.5)]


@pytest.mark.asyncio
async def test_scorer_function_metric_validates_raw_config_against_typed_schema() -> None:
    outputs = [MetricOutputSpec.continuous_score("reward")]
    descriptor = MetricDescriptor(type="tests.raw_typed_config", outputs=outputs, config_schema=ThresholdConfig)

    def sync_scorer(sample: ScorerInput[ThresholdConfig]) -> dict[str, object]:
        assert isinstance(sample.config, ThresholdConfig)
        assert sample.config.threshold == 0.75
        assert sample.config.label == "pass"
        return {"reward": sample.config.threshold}

    metric = ScorerFunctionMetric(
        descriptor=descriptor,
        scorer_fn=sync_scorer,
    ).bind_raw_config(config={"threshold": "0.75"})

    result = await metric.compute_scores(
        MetricInput(row=DatasetRow(data={}), candidate=CandidateOutput(output_text="candidate"))
    )

    assert result.outputs == [MetricOutput(name="reward", value=0.75)]


def test_scorer_function_metric_rejects_invalid_typed_config() -> None:
    outputs = [MetricOutputSpec.continuous_score("reward")]
    descriptor = MetricDescriptor(type="tests.invalid_config", outputs=outputs, config_schema=ThresholdConfig)
    metric = ScorerFunctionMetric(descriptor=descriptor, scorer_fn=lambda sample: {"reward": True})

    with pytest.raises(ValidationError):
        metric.bind_raw_config(config={"threshold": "not-a-number"})


def test_scorer_function_metric_bind_rejects_wrong_config_model_subtype() -> None:
    outputs = [MetricOutputSpec.continuous_score("reward")]
    descriptor = MetricDescriptor(type="tests.wrong_config_subtype", outputs=outputs, config_schema=ThresholdConfig)
    metric = ScorerFunctionMetric(
        descriptor=descriptor,
        scorer_fn=lambda sample: {"reward": cast(ThresholdConfig, sample.config).threshold},
    )

    with pytest.raises(TypeError, match="ThresholdConfig"):
        metric.bind(config=OtherThresholdConfig(threshold=0.75))


def test_scorer_function_metric_bind_rejects_raw_mapping_for_typed_config() -> None:
    outputs = [MetricOutputSpec.continuous_score("reward")]
    descriptor = MetricDescriptor(type="tests.raw_config_on_typed_bind", outputs=outputs, config_schema=ThresholdConfig)
    metric = ScorerFunctionMetric(
        descriptor=descriptor,
        scorer_fn=lambda sample: {"reward": cast(ThresholdConfig, sample.config).threshold},
    )

    with pytest.raises(TypeError, match="bind_raw_config"):
        metric.bind(config={"threshold": 0.75})


def test_validate_metric_result_rejects_duplicate_output_names() -> None:
    outputs = [MetricOutputSpec.continuous_score("reward")]
    result = MetricResult(
        outputs=[
            MetricOutput(name="reward", value=1.0),
            MetricOutput(name="reward", value=0.0),
        ]
    )

    with pytest.raises(ValueError, match="Duplicate metric output"):
        validate_metric_result(result, outputs)


def test_validate_metric_result_rejects_missing_declared_outputs() -> None:
    outputs = [MetricOutputSpec.continuous_score("reward"), MetricOutputSpec.continuous_score("format")]
    result = MetricResult(outputs=[MetricOutput(name="reward", value=1.0)])

    with pytest.raises(ValueError, match="Missing declared metric outputs"):
        validate_metric_result(result, outputs)


def test_validate_metric_result_rejects_undeclared_outputs() -> None:
    outputs = [MetricOutputSpec.continuous_score("reward")]
    result = MetricResult(
        outputs=[
            MetricOutput(name="reward", value=1.0),
            MetricOutput(name="format", value=1.0),
        ]
    )

    with pytest.raises(ValueError, match="Undeclared metric outputs"):
        validate_metric_result(result, outputs)


def test_validate_metric_result_rejects_value_that_does_not_match_schema() -> None:
    outputs = [MetricOutputSpec.model("judge_details", JudgeDetails)]
    result = MetricResult(outputs=[MetricOutput(name="judge_details", value={"label": "pass"})])

    with pytest.raises(ValidationError):
        validate_metric_result(result, outputs)


def test_scorer_decorator_returns_subclass_without_mutating_original_metric_class() -> None:
    scorer_cls = scorer(ExactMatchMetric)

    assert scorer_cls is not ExactMatchMetric
    assert issubclass(scorer_cls, ExactMatchMetric)
    assert not hasattr(ExactMatchMetric(reference="{{item.answer}}"), "descriptor")
    assert not hasattr(ExactMatchMetric(reference="{{item.answer}}"), "to_metric")

    scorer_metric = cast(_MetricScorerForTest, scorer_cls(reference="{{item.answer}}"))
    assert scorer_metric.descriptor == MetricDescriptor(
        type="exact-match",
        outputs=[MetricOutputSpec.continuous_score("correct")],
    )
    assert scorer_metric.to_metric() is scorer_metric


@pytest.mark.asyncio
async def test_exact_match_metric_is_undecorated_reusable_metric() -> None:
    metric = ExactMatchMetric(reference="{{item.answer}}")

    assert not hasattr(metric, "descriptor")
    assert not hasattr(metric, "to_metric")
    assert isinstance(metric, BaseModel)
    assert metric.type == "exact-match"
    assert metric.model_dump()["type"] == "exact-match"
    assert metric.output_spec() == [MetricOutputSpec.continuous_score("correct")]

    result = await metric.compute_scores(
        MetricInput(
            row=DatasetRow(data={"answer": "Paris"}),
            candidate=CandidateOutput(output_text="Paris"),
        )
    )

    assert result.outputs == [MetricOutput(name="correct", value=1.0)]


@pytest.mark.asyncio
async def test_exact_match_scorer_exposes_descriptor_and_to_metric() -> None:
    scorer_metric = cast(_MetricScorerForTest, ExactMatchScorer(reference="{{item.answer}}"))

    assert scorer_metric.descriptor == MetricDescriptor(
        type="exact-match",
        outputs=[MetricOutputSpec.continuous_score("correct")],
    )

    metric = scorer_metric.to_metric()
    result = await metric.compute_scores(
        MetricInput(
            row=DatasetRow(data={"answer": "Paris"}),
            candidate=CandidateOutput(output_text="Paris"),
        )
    )

    assert metric.type == "exact-match"
    assert result.outputs == [MetricOutput(name="correct", value=1.0)]


@pytest.mark.asyncio
async def test_scorer_decorator_adapts_exact_match_metric_instances() -> None:
    scorer_metric = scorer(ExactMatchMetric(reference="{{item.answer}}"))

    assert scorer_metric.descriptor == MetricDescriptor(
        type="exact-match",
        outputs=[MetricOutputSpec.continuous_score("correct")],
    )

    metric = scorer_metric.to_metric()
    result = await metric.compute_scores(
        MetricInput(
            row=DatasetRow(data={"answer": "Paris"}),
            candidate=CandidateOutput(output_text="Paris"),
        )
    )

    assert metric.type == "exact-match"
    assert result.outputs == [MetricOutput(name="correct", value=1.0)]


@pytest.mark.asyncio
async def test_exact_match_metric_supports_top_level_and_sample_template_aliases() -> None:
    metric = ExactMatchMetric(reference="{{answer}}", candidate="{{sample.prediction}}")

    result = await metric.compute_scores(
        MetricInput(
            row=DatasetRow(data={"answer": "New York"}),
            candidate=CandidateOutput(metadata={"prediction": "new york"}),
        )
    )

    assert result.outputs == [MetricOutput(name="correct", value=1.0)]


@pytest.mark.asyncio
async def test_class_based_scorer_to_metric_returns_configured_instance() -> None:
    @scorer
    class MissingOutputMetric:
        type = "tests.missing_output"

        def output_spec(self) -> list[MetricOutputSpec]:
            return [MetricOutputSpec.continuous_score("correct")]

        async def compute_scores(self, input: MetricInput) -> MetricResult:
            return MetricResult(outputs=[])

    instance = cast(_MetricScorerForTest, MissingOutputMetric())
    metric = instance.to_metric()
    result = await metric.compute_scores(
        MetricInput(row=DatasetRow(data={}), candidate=CandidateOutput(output_text="candidate"))
    )

    assert metric is instance
    assert result == MetricResult(outputs=[])


def test_scorer_decorator_can_adapt_metric_instances() -> None:
    class UndecoratedMetric:
        type = "tests.undecorated"

        def output_spec(self) -> list[MetricOutputSpec]:
            return [MetricOutputSpec.continuous_score("correct")]

        async def compute_scores(self, input: MetricInput) -> MetricResult:
            return MetricResult(outputs=[MetricOutput(name="correct", value=1.0)])

    metric = scorer(UndecoratedMetric())

    assert metric.descriptor == MetricDescriptor(
        type="tests.undecorated",
        outputs=[MetricOutputSpec.continuous_score("correct")],
    )
    assert metric.to_metric() is metric


@pytest.mark.asyncio
async def test_scorer_function_metric_executes_sync_scorers() -> None:
    outputs = [MetricOutputSpec.boolean("reward"), MetricOutputSpec.label("label")]
    descriptor = MetricDescriptor(type="tests.sync_metric", outputs=outputs)
    sandbox = cast(Sandbox, object())

    def sync_scorer(sample: ScorerInput) -> dict[str, object]:
        assert sample.response == "yes"
        assert sample.target == "yes"
        assert sample.metadata["category"] == "boolean"
        assert sample.config == {"mode": "strict"}
        assert sample.sandbox is sandbox
        return {"reward": True, "label": "matched"}

    metric = ScorerFunctionMetric(descriptor=descriptor, scorer_fn=sync_scorer).bind(
        config={"mode": "strict"},
        sandbox=sandbox,
        target_field="answer",
    )

    result = await metric.compute_scores(
        MetricInput(
            row=DatasetRow(data={"answer": "yes", "category": "boolean"}),
            candidate=CandidateOutput(output_text="yes"),
        )
    )

    assert metric.type == "tests.sync_metric"
    assert score_names_from_output_spec(metric.output_spec()) == ["reward"]
    assert result.outputs == [MetricOutput(name="reward", value=True), MetricOutput(name="label", value="matched")]


@pytest.mark.asyncio
async def test_scorer_function_metric_executes_async_scorers() -> None:
    outputs = [MetricOutputSpec.continuous_score("reward"), MetricOutputSpec.label("seen")]
    descriptor = MetricDescriptor(type="tests.async_metric", outputs=outputs)

    async def async_scorer(sample: ScorerInput) -> dict[str, object]:
        return {"reward": 0.5, "seen": sample.response}

    metric = ScorerFunctionMetric(descriptor=descriptor, scorer_fn=async_scorer)

    result = await metric.compute_scores(
        MetricInput(row=DatasetRow(data={"answer": "yes"}), candidate=CandidateOutput(output_text="maybe"))
    )

    assert metric.type == "tests.async_metric"
    assert score_names_from_output_spec(metric.output_spec()) == ["reward"]
    assert result.outputs == [MetricOutput(name="reward", value=0.5), MetricOutput(name="seen", value="maybe")]


def test_typed_scorer_decorator_exposes_descriptor_and_to_metric() -> None:
    outputs = [MetricOutputSpec.boolean("truthful"), MetricOutputSpec.label("judge_grade")]

    @scorer(metric_type="truthfulqa", outputs=outputs)
    def truthfulqa_scorer(sample: ScorerInput) -> dict[str, object]:
        return {"truthful": bool(sample.response), "judge_grade": "C"}

    metric = truthfulqa_scorer.to_metric()

    assert truthfulqa_scorer.descriptor == MetricDescriptor(type="truthfulqa", outputs=outputs)
    assert metric.type == "truthfulqa"
    assert score_names_from_output_spec(metric.output_spec()) == ["truthful"]
