"""Tests for MCQ extraction and scoring."""
import pytest
from nemo_evaluator.scoring.mcq import extract_mcq_answer, mcq_score


class TestExtractMCQ:
    def test_boxed_letter(self):
        assert extract_mcq_answer(r"The answer is \boxed{C}") == "C"

    def test_answer_is_pattern(self):
        assert extract_mcq_answer("After analysis, the answer is B.") == "B"

    def test_last_standalone_letter_fallback(self):
        assert extract_mcq_answer("I think it's between A and D, so D") == "D"

    def test_no_letter_returns_none(self):
        assert extract_mcq_answer("not sure about this one") is None

    def test_case_insensitive_answer_pattern(self):
        assert extract_mcq_answer("The ANSWER IS: a") == "A"

    def test_boxed_takes_priority_over_answer_is(self):
        text = r"The answer is A, but actually \boxed{B}"
        assert extract_mcq_answer(text) == "B"


class TestMCQScore:
    def test_correct(self):
        assert mcq_score("The answer is B", "B") is True

    def test_incorrect(self):
        assert mcq_score("The answer is A", "B") is False

    def test_no_extraction(self):
        assert mcq_score("I don't know", "A") is False

    def test_whitespace_tolerance(self):
        assert mcq_score("C", "  C  ") is True
