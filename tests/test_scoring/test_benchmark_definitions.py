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
"""Golden tests: verify benchmark scorers produce correct results on known data."""

import sys
import types

import pytest

from nemo_evaluator.scoring import (
    ScorerInput,
    answer_line,
    exact_match,
    fuzzy_match,
    multichoice_regex,
    numeric_match,
)


class TestMultichoiceRegex:
    def test_correct_answer(self):
        s = ScorerInput(response="I think the answer is B.\n\nAnswer: B", target="B")
        assert multichoice_regex(s)["correct"] is True

    def test_wrong_answer(self):
        s = ScorerInput(response="Answer: C", target="A")
        assert multichoice_regex(s)["correct"] is False

    def test_case_insensitive(self):
        s = ScorerInput(response="answer: d", target="D")
        assert multichoice_regex(s)["correct"] is True

    def test_no_answer_found(self):
        s = ScorerInput(response="I'm not sure about this one", target="A")
        assert multichoice_regex(s)["correct"] is False

    def test_10_choice(self):
        s = ScorerInput(response="Answer: H", target="H")
        assert multichoice_regex(s, pattern=r"(?i)Answer\s*:\s*([A-J])")["correct"] is True


class TestAnswerLine:
    def test_correct_math_answer(self):
        s = ScorerInput(response="Let me solve...\n\nAnswer: 42", target="42")
        assert answer_line(s)["correct"] is True

    def test_whitespace_tolerance(self):
        s = ScorerInput(response="Answer:  42 ", target="42")
        assert answer_line(s)["correct"] is True

    def test_wrong_answer(self):
        s = ScorerInput(response="Answer: 43", target="42")
        assert answer_line(s)["correct"] is False

    def test_extracts_from_last_line_if_no_pattern(self):
        s = ScorerInput(response="The result is\n42", target="42")
        result = answer_line(s)
        assert result["extracted"] == "42"


class TestNumericMatch:
    def test_integer(self):
        s = ScorerInput(response="The answer is 42", target="42")
        assert numeric_match(s)["correct"] is True

    def test_strips_trailing_zeros(self):
        s = ScorerInput(response="Answer: 5.00", target="5")
        assert numeric_match(s)["correct"] is True

    def test_commas_in_number(self):
        s = ScorerInput(response="1,234 people", target="1234")
        assert numeric_match(s)["correct"] is True

    def test_picks_last_number(self):
        s = ScorerInput(response="Step 1: 10, Step 2: 20, Final: 30", target="30")
        assert numeric_match(s)["correct"] is True


class TestFuzzyMatch:
    def test_substring_match(self):
        s = ScorerInput(
            response="The Battle of Gettysburg", target="Gettysburg", metadata={"correct_answers": ["Gettysburg"]}
        )
        assert fuzzy_match(s)["correct"] is True

    def test_no_match(self):
        s = ScorerInput(response="Paris", target="London", metadata={"correct_answers": ["London"]})
        assert fuzzy_match(s)["correct"] is False


class TestExactMatch:
    def test_exact(self):
        s = ScorerInput(response="  Paris  ", target="paris")
        assert exact_match(s)["correct"] is True

    def test_article_removal(self):
        s = ScorerInput(response="The Eiffel Tower", target="Eiffel Tower")
        assert exact_match(s)["correct"] is True


class TestMgsmAnswerNumber:
    """MGSM's gold answers come out of the loader as strings; they must stay whole."""

    @pytest.mark.parametrize(
        ("answer_number", "expected"),
        [
            pytest.param(100, "100", id="hundred-keeps-its-zeros"),
            pytest.param(70000, "70000", id="seventy-thousand-keeps-its-zeros"),
            pytest.param(540, "540", id="trailing-zero-only"),
            pytest.param(20, "20", id="two-digits"),
            pytest.param(42, "42", id="no-trailing-zero"),
            pytest.param(5.0, "5", id="whole-float-loses-the-point"),
            pytest.param(2.5, "2.5", id="fractional-float-kept"),
        ],
    )
    def test_normalization_keeps_the_number(self, answer_number, expected):
        from nemo_evaluator.benchmarks.mgsm import _normalize_answer_number

        assert _normalize_answer_number(answer_number) == expected

    def test_the_loader_keeps_the_whole_answer(self, monkeypatch):
        """The call site in ``_load_mgsm``, with ``datasets`` replaced by a fake."""
        from nemo_evaluator.benchmarks.mgsm import _load_mgsm

        fake = types.ModuleType("datasets")
        fake.load_dataset = lambda *args, **kwargs: [
            {"question": "q", "answer_number": 100},
            {"question": "q", "answer_number": 70000},
            {"question": "q", "answer_number": 540},
        ]
        monkeypatch.setitem(sys.modules, "datasets", fake)

        assert {row["answer"] for row in _load_mgsm()} == {"100", "70000", "540"}

    @pytest.mark.parametrize("gold", [pytest.param(100, id="hundred"), pytest.param(70000, id="seventy-thousand")])
    def test_the_right_answer_still_scores(self, gold):
        from nemo_evaluator.benchmarks.mgsm import _normalize_answer_number, mgsm_scorer

        s = ScorerInput(response=f"Answer: {gold}", target=_normalize_answer_number(gold))
        assert mgsm_scorer(s)["correct"] is True


class TestBenchmarkScorerImport:
    """Verify each built-in benchmark's scorer is importable and callable."""

    def test_mmlu_scorer(self):
        from nemo_evaluator.benchmarks.mmlu import mmlu_scorer

        s = ScorerInput(response="Answer: C", target="C")
        assert mmlu_scorer(s)["correct"] is True

    def test_mmlu_pro_scorer(self):
        from nemo_evaluator.benchmarks.mmlu_pro import mmlu_pro_scorer

        s = ScorerInput(response="Answer: F", target="F")
        assert mmlu_pro_scorer(s)["correct"] is True

    def test_math500_scorer(self):
        from nemo_evaluator.benchmarks.math500 import math500_scorer

        s = ScorerInput(response="Answer: 42", target="42")
        assert math500_scorer(s)["correct"] is True

    def test_gpqa_scorer(self):
        from nemo_evaluator.benchmarks.gpqa import gpqa_scorer

        s = ScorerInput(response="Answer: A", target="A")
        assert gpqa_scorer(s)["correct"] is True

    def test_gsm8k_scorer(self):
        from nemo_evaluator.benchmarks.gsm8k import gsm8k_scorer

        s = ScorerInput(response="The answer is 42", target="42")
        assert gsm8k_scorer(s)["correct"] is True

    def test_mgsm_scorer(self):
        from nemo_evaluator.benchmarks.mgsm import mgsm_scorer

        s = ScorerInput(response="The result is 15", target="15")
        assert mgsm_scorer(s)["correct"] is True
