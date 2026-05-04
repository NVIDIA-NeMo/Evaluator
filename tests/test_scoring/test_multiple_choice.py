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
"""Tests for multiple-choice scorers on the shared metric contract.

These tests are the proof that Kanishk PR #1's MC functionality fits Sandy's
shared metric contract (#950) **with zero protocol-type changes**. The
scorers expose ``descriptor`` + ``to_metric()`` via ``@scorer(outputs=...)``
exactly like any other metric, the per-row payload (choices, logprobs)
flows through ``MetricInput.candidate.metadata``, and the runtime invariants
(declared vs returned, value-schema coercion) all hold.
"""

from __future__ import annotations

import pytest

from nemo_evaluator.scoring.metric import (
    BooleanValue,
    CandidateOutput,
    ContinuousScore,
    DatasetRow,
    MetricInput,
    MetricOutputSpec,
)
from nemo_evaluator.scoring.multiple_choice import (
    _resolve_gold_index,
    mcq_letter_extract,
    multiple_choice_acc,
)


def _mc_input(*, target: object, choices: list[str], logprobs: list[float], is_greedy: list[bool]) -> MetricInput:
    """Build a MetricInput shaped the way LogprobRankingSolver populates it."""
    argmax_idx = max(range(len(logprobs)), key=lambda i: logprobs[i]) if logprobs else 0
    return MetricInput(
        row=DatasetRow(data={"target": target}),
        candidate=CandidateOutput(
            output_text=choices[argmax_idx] if choices else "",
            metadata={
                "_mc_choices": choices,
                "_mc_choices_logprobs": logprobs,
                "_mc_choices_is_greedy": is_greedy,
            },
        ),
    )


# ── Decorator + descriptor surface ──────────────────────────────────────────


def test_multiple_choice_acc_exposes_descriptor_with_three_continuous_scores() -> None:
    desc = multiple_choice_acc.descriptor
    assert desc.type == "multiple_choice_acc"
    assert [o.name for o in desc.outputs] == ["acc", "acc_norm", "acc_greedy"]
    for output in desc.outputs:
        assert output.value_schema is ContinuousScore


def test_mcq_letter_extract_exposes_descriptor_with_correct_and_parsed() -> None:
    desc = mcq_letter_extract.descriptor
    assert desc.type == "mcq_letter_extract"
    names = {o.name: o.value_schema for o in desc.outputs}
    assert names == {"correct": ContinuousScore, "parsed": BooleanValue}


def test_to_metric_materializes_satisfying_metric_protocol() -> None:
    metric = multiple_choice_acc.to_metric()
    assert metric.type == "multiple_choice_acc"
    assert [o.name for o in metric.output_spec()] == ["acc", "acc_norm", "acc_greedy"]


# ── multiple_choice_acc semantics ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_mc_acc_perfect_match() -> None:
    metric = multiple_choice_acc.to_metric()
    result = await metric.compute_scores(
        _mc_input(
            target=2,
            choices=["A", "B", "Paris", "London"],
            logprobs=[-5.1, -4.8, -1.2, -3.7],
            is_greedy=[False, False, True, False],
        )
    )
    outputs = {o.name: o.value for o in result.outputs}
    assert outputs == {"acc": 1.0, "acc_norm": 1.0, "acc_greedy": 1.0}


@pytest.mark.asyncio
async def test_mc_acc_wrong_answer() -> None:
    metric = multiple_choice_acc.to_metric()
    result = await metric.compute_scores(
        _mc_input(
            target=0,
            choices=["A", "B", "Paris", "London"],
            logprobs=[-5.1, -4.8, -1.2, -3.7],
            is_greedy=[False, False, True, False],
        )
    )
    outputs = {o.name: o.value for o in result.outputs}
    assert outputs == {"acc": 0.0, "acc_norm": 0.0, "acc_greedy": 0.0}


@pytest.mark.asyncio
async def test_mc_acc_norm_diverges_from_acc_when_choices_differ_in_length() -> None:
    """Length normalization can flip argmax: a long choice may have lower raw
    logprob (more tokens to multiply through) but a higher per-character one."""
    metric = multiple_choice_acc.to_metric()
    result = await metric.compute_scores(
        _mc_input(
            target=1,
            choices=["A", "ABCDEFGHIJ"],
            logprobs=[-1.5, -3.0],  # raw argmax = 0
            is_greedy=[False, False],
        )
    )
    # raw: -1.5 > -3.0 → argmax = 0, so acc = 0
    # norm: -1.5/1 = -1.5 vs -3.0/10 = -0.3 → argmax = 1, so acc_norm = 1
    outputs = {o.name: o.value for o in result.outputs}
    assert outputs["acc"] == 0.0
    assert outputs["acc_norm"] == 1.0


@pytest.mark.asyncio
async def test_mc_acc_greedy_zero_when_no_choice_is_greedy() -> None:
    metric = multiple_choice_acc.to_metric()
    result = await metric.compute_scores(
        _mc_input(
            target=2,
            choices=["A", "B", "Paris", "London"],
            logprobs=[-5.1, -4.8, -1.2, -3.7],
            is_greedy=[False, False, False, False],
        )
    )
    outputs = {o.name: o.value for o in result.outputs}
    assert outputs["acc"] == 1.0
    assert outputs["acc_greedy"] == 0.0


@pytest.mark.asyncio
async def test_mc_acc_returns_zeros_when_payload_missing() -> None:
    """No choices in metadata → all metrics 0.0 (graceful degrade)."""
    metric = multiple_choice_acc.to_metric()
    result = await metric.compute_scores(
        MetricInput(
            row=DatasetRow(data={"target": 1}),
            candidate=CandidateOutput(output_text="anything"),
        )
    )
    outputs = {o.name: o.value for o in result.outputs}
    assert outputs == {"acc": 0.0, "acc_norm": 0.0, "acc_greedy": 0.0}


# ── _resolve_gold_index heterogeneous targets ───────────────────────────────


@pytest.mark.parametrize(
    ("target", "choices", "expected"),
    [
        (2, ["A", "B", "C", "D"], 2),
        (-1, ["A", "B"], None),
        (5, ["A", "B"], None),
        ("C", ["A", "B", "C", "D"], 2),
        ("c", ["A", "B", "C", "D"], 2),
        ("A", ["A", "B"], 0),
        ("Z", ["A", "B"], None),
        ("0", ["A", "B"], 0),
        ("3", ["A", "B"], None),
        ("Paris", ["London", "Paris"], 1),
        ("paris", ["London", "Paris"], 1),
        ("Berlin", ["London", "Paris"], None),
        ("", ["A", "B"], None),
        (None, ["A", "B"], None),
    ],
)
def test_resolve_gold_index_heterogeneous(
    target: object, choices: list[str], expected: int | None
) -> None:
    assert _resolve_gold_index(target, choices) == expected


# ── End-to-end: integration with shared contract ────────────────────────────


@pytest.mark.asyncio
async def test_candidate_metadata_flows_to_legacy_scorer_through_translator() -> None:
    """Proof: ``MetricInput.candidate.metadata`` reaches the legacy scorer.

    Without the merge fix in ``ScorerFunctionMetric.compute_scores``, this
    test fails because the translator builds ``ScorerInput.metadata`` from
    ``row.data`` only and the scorer reads zeros.
    """
    metric = multiple_choice_acc.to_metric()
    result = await metric.compute_scores(
        MetricInput(
            row=DatasetRow(data={"target": 0, "question": "What's hot?"}),
            candidate=CandidateOutput(
                output_text="lava",
                metadata={
                    "_mc_choices": ["lava", "ice"],
                    "_mc_choices_logprobs": [-0.1, -3.0],
                    "_mc_choices_is_greedy": [True, False],
                },
            ),
        )
    )
    outputs = {o.name: o.value for o in result.outputs}
    assert outputs == {"acc": 1.0, "acc_norm": 1.0, "acc_greedy": 1.0}


@pytest.mark.asyncio
async def test_mcq_letter_extract_basic_letter_match() -> None:
    metric = mcq_letter_extract.to_metric()
    result = await metric.compute_scores(
        MetricInput(
            row=DatasetRow(data={"target": "B"}),
            candidate=CandidateOutput(output_text="The answer is B."),
        )
    )
    outputs = {o.name: o.value for o in result.outputs}
    assert outputs == {"correct": 1.0, "parsed": True}


@pytest.mark.asyncio
async def test_mcq_letter_extract_no_letter_in_response() -> None:
    metric = mcq_letter_extract.to_metric()
    result = await metric.compute_scores(
        MetricInput(
            row=DatasetRow(data={"target": "C"}),
            candidate=CandidateOutput(output_text="????"),
        )
    )
    outputs = {o.name: o.value for o in result.outputs}
    assert outputs == {"correct": 0.0, "parsed": False}


@pytest.mark.asyncio
async def test_mcq_letter_extract_int_target() -> None:
    metric = mcq_letter_extract.to_metric()
    result = await metric.compute_scores(
        MetricInput(
            row=DatasetRow(data={"target": 1}),
            candidate=CandidateOutput(output_text="Answer: B"),
        )
    )
    outputs = {o.name: o.value for o in result.outputs}
    assert outputs == {"correct": 1.0, "parsed": True}


@pytest.mark.asyncio
async def test_validate_metric_result_enforces_declared_outputs() -> None:
    """Sandy's validate_metric_result rejects extra/missing keys at runtime."""
    from nemo_evaluator.scoring.metric import (
        MetricDescriptor,
        MetricOutput,
        MetricResult,
        validate_metric_result,
    )

    desc = MetricDescriptor(
        type="multiple_choice_acc",
        outputs=[
            MetricOutputSpec.continuous_score("acc"),
            MetricOutputSpec.continuous_score("acc_norm"),
            MetricOutputSpec.continuous_score("acc_greedy"),
        ],
    )
    bad = MetricResult(outputs=[MetricOutput(name="acc", value=1.0)])
    with pytest.raises(ValueError, match="Missing declared metric outputs"):
        validate_metric_result(bad, desc.outputs)
