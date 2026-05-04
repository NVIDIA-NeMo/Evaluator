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
"""End-to-end BYOB integration tests for multiple-choice + few-shot.

These tests exercise the full ``ByobEnvironment.seed → verify`` path with
synthetic datasets and a fake LogprobRankingSolver. No model server is
required — the solver is mocked, the dataset is in-memory, and the eval
loop is sidestepped (we call seed/verify directly to keep the test
hermetic and fast).
"""

from __future__ import annotations

import pytest

from nemo_evaluator.environments.custom import (
    BenchmarkDefinition,
    ByobEnvironment,
    _metric_input_from_verify,
    _resolve_mc_choices,
    benchmark,
    scorer,
)
from nemo_evaluator.scoring.metric import MetricOutputSpec
from nemo_evaluator.scoring.multiple_choice import multiple_choice_acc


# ── _resolve_mc_choices ─────────────────────────────────────────────────────


def test_resolve_mc_choices_static_list() -> None:
    defn = BenchmarkDefinition(
        name="t", dataset=[], prompt="", choices=["A", "B", "C", "D"]
    )
    assert _resolve_mc_choices({"answer": 1}, defn) == ["A", "B", "C", "D"]


def test_resolve_mc_choices_per_row_field() -> None:
    defn = BenchmarkDefinition(
        name="t", dataset=[], prompt="", choices_field="options"
    )
    assert _resolve_mc_choices({"options": ["X", "Y"]}, defn) == ["X", "Y"]


def test_resolve_mc_choices_dotted_path() -> None:
    """ARC-style nested choices: row['choices']['text'] = [...]."""
    defn = BenchmarkDefinition(
        name="t", dataset=[], prompt="", choices_field="choices.text"
    )
    row = {"choices": {"text": ["foo", "bar"], "label": ["A", "B"]}}
    assert _resolve_mc_choices(row, defn) == ["foo", "bar"]


def test_resolve_mc_choices_field_takes_precedence_over_static() -> None:
    defn = BenchmarkDefinition(
        name="t",
        dataset=[],
        prompt="",
        choices=["fallback1", "fallback2"],
        choices_field="options",
    )
    assert _resolve_mc_choices({"options": ["x", "y"]}, defn) == ["x", "y"]


def test_resolve_mc_choices_falls_back_when_field_missing() -> None:
    defn = BenchmarkDefinition(
        name="t",
        dataset=[],
        prompt="",
        choices=["fallback1", "fallback2"],
        choices_field="missing",
    )
    assert _resolve_mc_choices({}, defn) == ["fallback1", "fallback2"]


def test_resolve_mc_choices_returns_none_when_neither_set() -> None:
    defn = BenchmarkDefinition(name="t", dataset=[], prompt="")
    assert _resolve_mc_choices({}, defn) is None


# ── _metric_input_from_verify lifts _mc_* into candidate.metadata ───────────


def test_metric_input_from_verify_separates_mc_keys() -> None:
    metric_input = _metric_input_from_verify(
        response="Paris",
        metadata={
            "question": "Capital of France?",
            "answer_idx": 2,
            "_mc_choices": ["A", "B", "Paris", "London"],
            "_mc_choices_logprobs": [-5.1, -4.8, -1.2, -3.7],
            "_mc_choices_is_greedy": [False, False, True, False],
            "_solver_attempts": 1,
        },
    )
    # Dataset row keys go to row.data
    assert metric_input.row.data == {"question": "Capital of France?", "answer_idx": 2}
    # _mc_* and _solver_* keys go to candidate.metadata
    assert metric_input.candidate.metadata == {
        "_mc_choices": ["A", "B", "Paris", "London"],
        "_mc_choices_logprobs": [-5.1, -4.8, -1.2, -3.7],
        "_mc_choices_is_greedy": [False, False, True, False],
        "_solver_attempts": 1,
    }
    assert metric_input.candidate.output_text == "Paris"


# ── End-to-end BYOB seed + verify with multiple_choice_acc ──────────────────


@pytest.mark.asyncio
async def test_byob_mmlu_style_end_to_end() -> None:
    """MMLU-style 4-way multiple-choice with static choices, in-memory dataset."""
    dataset = [
        {"question": "Capital of France?", "answer": 2},
        {"question": "Capital of UK?", "answer": 3},
    ]
    defn = BenchmarkDefinition(
        name="mmlu_synth",
        dataset=lambda: dataset,
        prompt="Q: {question}\nA: ",
        target_field="answer",
        choices=["Berlin", "Madrid", "Paris", "London"],
        scorer_fn=multiple_choice_acc,
    )
    env = ByobEnvironment(defn)

    # Seed populates _mc_choices in metadata
    seed = await env.seed(0)
    assert seed.metadata["_mc_choices"] == ["Berlin", "Madrid", "Paris", "London"]
    assert seed.expected_answer == "2"
    assert seed.prompt == "Q: Capital of France?\nA: "

    # Simulate the LogprobRankingSolver having already ranked the choices —
    # answer index 2 ("Paris") wins.
    solver_meta = {
        "_mc_choices": ["Berlin", "Madrid", "Paris", "London"],
        "_mc_choices_logprobs": [-5.1, -4.8, -1.2, -3.7],
        "_mc_choices_is_greedy": [False, False, True, False],
    }
    merged = {**seed.metadata, **solver_meta}
    vr = await env.verify("Paris", "2", **merged)

    assert vr.reward == 1.0  # acc=1, the default reward picks first declared "acc"
    assert vr.scoring_details["metric_type"] == "multiple_choice_acc"
    assert vr.scoring_details["outputs"] == {"acc": 1.0, "acc_norm": 1.0, "acc_greedy": 1.0}


@pytest.mark.asyncio
async def test_byob_per_row_choices_arc_style() -> None:
    """ARC-style: variable-length per-row choices via dotted path."""
    dataset = [
        {
            "question": "What's hot?",
            "choices": {"text": ["lava", "ice", "water"], "label": ["A", "B", "C"]},
            "answer": "A",
        },
        {
            "question": "Smallest planet?",
            "choices": {"text": ["Mercury", "Venus"], "label": ["A", "B"]},
            "answer": "A",
        },
    ]
    defn = BenchmarkDefinition(
        name="arc_synth",
        dataset=lambda: dataset,
        prompt="Q: {question}\nA: ",
        target_field="answer",
        choices_field="choices.text",
        scorer_fn=multiple_choice_acc,
    )
    env = ByobEnvironment(defn)

    seed = await env.seed(0)
    assert seed.metadata["_mc_choices"] == ["lava", "ice", "water"]

    seed1 = await env.seed(1)
    assert seed1.metadata["_mc_choices"] == ["Mercury", "Venus"]


@pytest.mark.asyncio
async def test_byob_few_shot_prefix_prepended() -> None:
    dataset = [
        {"q": "1+1", "a": "2"},
        {"q": "2+2", "a": "4"},
        {"q": "3+3", "a": "6"},
    ]
    defn = BenchmarkDefinition(
        name="fewshot_synth",
        dataset=lambda: dataset,
        prompt="Q: {q} A: ",
        target_field="a",
        num_fewshot=2,
        fewshot_seed=0,
    )
    env = ByobEnvironment(defn)

    seed = await env.seed(0)
    # Prompt should have 2 few-shot examples + separator + the test prompt
    assert seed.prompt.count("Q: ") == 3  # 2 fewshot + 1 test
    assert seed.prompt.endswith("Q: 1+1 A: ")


@pytest.mark.asyncio
async def test_byob_few_shot_zero_means_no_prefix() -> None:
    dataset = [{"q": "1+1", "a": "2"}, {"q": "2+2", "a": "4"}]
    defn = BenchmarkDefinition(
        name="nofewshot",
        dataset=lambda: dataset,
        prompt="Q: {q} A: ",
        target_field="a",
        num_fewshot=0,
    )
    env = ByobEnvironment(defn)
    seed = await env.seed(0)
    assert seed.prompt == "Q: 1+1 A: "


@pytest.mark.asyncio
async def test_byob_mc_payload_threads_to_metric_via_verify_meta() -> None:
    """End-to-end: seed → fake solver injects logprobs → verify → metric scores.

    This is the integration proof: the solver's output reaches the metric
    through ``MetricInput.candidate.metadata`` without protocol changes.
    """
    dataset = [
        {"question": "Capital of France?", "answer": 2},
    ]
    defn = BenchmarkDefinition(
        name="proof",
        dataset=lambda: dataset,
        prompt="Q: {question}\nA: ",
        target_field="answer",
        choices=["Berlin", "Madrid", "Paris", "London"],
        scorer_fn=multiple_choice_acc,
    )
    env = ByobEnvironment(defn)

    seed = await env.seed(0)
    # Pretend a LogprobRankingSolver ran and wrote scoring_details
    solver_scoring_details = {
        "_mc_choices": ["Berlin", "Madrid", "Paris", "London"],
        "_mc_choices_logprobs": [-5.1, -4.8, -1.2, -3.7],
        "_mc_choices_is_greedy": [False, False, True, False],
    }
    merged_meta = {**seed.metadata, **solver_scoring_details}
    vr = await env.verify("Paris", seed.expected_answer, **merged_meta)

    assert vr.reward == 1.0
    assert vr.scoring_details["outputs"] == {
        "acc": 1.0,
        "acc_norm": 1.0,
        "acc_greedy": 1.0,
    }


# ── @benchmark decorator integration ────────────────────────────────────────


def test_benchmark_decorator_wires_mc_kwargs_to_definition() -> None:
    """Sanity: the new @benchmark kwargs land on BenchmarkDefinition."""
    from nemo_evaluator.environments.custom import _BYOB_REGISTRY

    @benchmark(
        name="test_mc_decorator",
        dataset=[{"q": "x", "answer": 0}],
        prompt="{q}",
        target_field="answer",
        choices=["a", "b"],
        num_fewshot=1,
        fewshot_separator="||",
    )
    @scorer(
        metric_type="test_score",
        outputs=[MetricOutputSpec.continuous_score("ok")],
    )
    def _score(sample) -> dict:
        return {"ok": 1.0}

    defn = _BYOB_REGISTRY["test_mc_decorator"]
    assert defn.choices == ["a", "b"]
    assert defn.num_fewshot == 1
    assert defn.fewshot_separator == "||"
