# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Example BYOB benchmark using a class-based Metric as its scorer."""

from nemo_evaluator.environments.custom import benchmark, scorer
from nemo_evaluator.metrics import ExactMatchMetric
from nemo_evaluator.scorers import ExactMatchScorer
from nemo_evaluator.scoring import MetricInput, MetricOutput, MetricOutputSpec, MetricResult


def _dataset() -> list[dict[str, str]]:
    return [
        {"question": "What is the capital of France?", "answer": "Paris"},
        {"question": "What is 2 + 2?", "answer": "4"},
    ]


# Mode 1: use a preannotated scorer wrapper exported by the OSS scorer layer.
benchmark(
    name="exact-match-preannotated-scorer-poc",
    dataset=_dataset,
    prompt="{question}",
    target_field="answer",
)(ExactMatchScorer(reference="{{item.answer}}"))


class InlineExactMatchMetric:
    type = "inline-exact-match"

    def __init__(self, *, reference: str, candidate: str | None = None) -> None:
        self.reference = reference
        self.candidate = candidate

    def output_spec(self) -> list[MetricOutputSpec]:
        return [MetricOutputSpec.continuous_score("correct")]

    async def compute_scores(self, input: MetricInput) -> MetricResult:
        reference = input.row.data.get("answer") if self.reference == "{{item.answer}}" else self.reference
        candidate = input.candidate.output_text if self.candidate is None else self.candidate
        correct = 1.0 if candidate == reference else 0.0
        return MetricResult(outputs=[MetricOutput(name="correct", value=correct)])


# Mode 2: annotate a local class at the benchmark call site, then configure it.
benchmark(
    name="exact-match-inline-class-scorer-poc",
    dataset=_dataset,
    prompt="{question}",
    target_field="answer",
)(scorer(InlineExactMatchMetric)(reference="{{item.answer}}"))


# Mode 3: adapt an already-configured reusable metric instance at the call site.
benchmark(
    name="exact-match-metric-instance-poc",
    dataset=_dataset,
    prompt="{question}",
    target_field="answer",
)(scorer(ExactMatchMetric(reference="{{item.answer}}")))
