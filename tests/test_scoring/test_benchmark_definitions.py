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
    @pytest.mark.parametrize(
        "response,target,letters,correct,expect_none",
        [
            # basic extraction with the default A-D pattern
            ("I think the answer is B.\n\nAnswer: B", "B", "A-D", True, False),
            ("Answer: C", "A", "A-D", False, False),
            ("answer: d", "D", "A-D", True, False),
            ("I'm not sure about this one", "A", "A-D", False, True),
            # LaTeX/markdown wrapper tolerance
            ("Final reasoning.\nAnswer: $B$", "B", "A-D", True, False),
            ("Answer: **C**", "C", "A-D", True, False),
            ("Answer: (A)", "A", "A-D", True, False),
            (r"Answer: \boxed{D}", "D", "A-D", True, False),
            # the final answer wins over an intermediate mention in the reasoning trace
            ("At first Answer: A, but on reflection Answer: D", "D", "A-D", True, False),
            # an unfilled template placeholder must not extract a letter
            ("Answer: $LETTER", "B", "A-D", False, True),
            # 10-choice via the generic letters arg still gets wrapper tolerance
            ("...corresponds to option **H**.\nAnswer: $H$", "H", "A-J", True, False),
            # a letter outside the configured range is not recognized
            ("Answer: H", "H", "A-D", False, True),
        ],
        ids=[
            "plain-correct",
            "wrong-letter",
            "case-insensitive",
            "no-answer",
            "latex",
            "markdown",
            "paren",
            "boxed",
            "last-match",
            "placeholder",
            "ten-choice",
            "out-of-range",
        ],
    )
    def test_multichoice_extraction(self, response, target, letters, correct, expect_none):
        r = multichoice_regex(ScorerInput(response=response, target=target), letters=letters)
        assert r["correct"] is correct
        if expect_none:
            assert r["extracted"] is None

    @pytest.mark.parametrize(
        "pattern",
        [
            r"(?i)Answer\s*:\s*([A-J])",  # single-group override (e.g. explicit 10-choice)
            r"(?i)Answer\s*:\s*([A-J])(\b)?",  # extra capture group: findall would yield a tuple
        ],
        ids=["single-group", "multi-group"],
    )
    def test_pattern_override_supports_extra_groups(self, pattern):
        # An explicit `pattern` overrides `letters`. With more than one capturing group,
        # finditer + group(1) must keep working where findall+`[-1].upper()` would crash
        # on a tuple. group(1) is the answer letter in both patterns.
        s = ScorerInput(response="Answer: H", target="H")
        assert multichoice_regex(s, pattern=pattern)["correct"] is True


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
