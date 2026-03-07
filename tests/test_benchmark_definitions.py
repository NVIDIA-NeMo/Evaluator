"""Golden tests: verify benchmark scorers produce correct results on known data."""
import pytest
from nemo_evaluator.environments.definitions import (
    ScorerInput, answer_line, exact_match, fuzzy_match, multichoice_regex, numeric_match,
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
        s = ScorerInput(response="The Battle of Gettysburg", target="Gettysburg",
                        metadata={"correct_answers": ["Gettysburg"]})
        assert fuzzy_match(s)["correct"] is True

    def test_no_match(self):
        s = ScorerInput(response="Paris", target="London",
                        metadata={"correct_answers": ["London"]})
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
