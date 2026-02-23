# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Template scorer unit tests.

Tests T017-T054: Comprehensive unit tests for all 6 template scorers.
Each scorer is tested with edge cases and bundled sample data.
"""

import json
import os

import pytest

from nemo_evaluator.byob.decorators import ScorerInput
from .conftest import TEMPLATE_DIR, TEMPLATES, import_scorer


# ============================================================================
# Math Reasoning Scorer Tests (T017-T023)
# ============================================================================

@pytest.fixture
def math_scorer():
    """Import the math reasoning scorer."""
    return import_scorer("math_reasoning")


def test_math_scorer_extracts_integer(math_scorer):
    """T017: Math scorer extracts integer from response."""
    result = math_scorer(ScorerInput(response="The answer is 42", target="42", metadata={}))
    assert result["correct"] is True, (
        f"Math scorer should return correct=True for '42' == '42'. "
        f"Got: {result}"
    )
    assert result["parsed"] is True, "Math scorer should set parsed=True when number found."


def test_math_scorer_extracts_negative(math_scorer):
    """T018: Math scorer extracts negative number."""
    result = math_scorer(ScorerInput(response="Therefore, -7 is the result.", target="-7", metadata={}))
    assert result["correct"] is True, (
        f"Math scorer should handle negative numbers. Got: {result}"
    )


def test_math_scorer_extracts_decimal(math_scorer):
    """T019: Math scorer extracts decimal number."""
    result = math_scorer(ScorerInput(response="The value is approximately 3.14", target="3.14", metadata={}))
    assert result["correct"] is True, (
        f"Math scorer should handle decimal numbers. Got: {result}"
    )


def test_math_scorer_no_number(math_scorer):
    """T020: Math scorer handles response with no number."""
    result = math_scorer(ScorerInput(response="I don't know the answer", target="42", metadata={}))
    assert result["correct"] is False, (
        f"Math scorer should return correct=False when no number found. Got: {result}"
    )
    assert result["parsed"] is False, (
        f"Math scorer should set parsed=False when no number found. Got: {result}"
    )


def test_math_scorer_trailing_dot(math_scorer):
    """T021: Math scorer handles trailing dot (e.g., '42.')."""
    result = math_scorer(ScorerInput(response="The answer is 42.", target="42", metadata={}))
    assert result["correct"] is True, (
        f"Math scorer should strip trailing dots. Got: {result}"
    )


def test_math_scorer_float_equality(math_scorer):
    """T022: Math scorer handles float equality (3.0 == 3)."""
    result = math_scorer(ScorerInput(response="The answer is 3.0", target="3", metadata={}))
    assert result["correct"] is True, (
        f"Math scorer should treat 3.0 and 3 as equal. Got: {result}"
    )


def test_math_scorer_passes_bundled_data(math_scorer):
    """T023: Math scorer passes on bundled sample data.

    Runs scorer against all 3 bundled sample data rows with synthetic
    responses that contain the correct answer.
    """
    data_path = os.path.join(TEMPLATE_DIR, "math_reasoning_data.jsonl")
    with open(data_path) as f:
        rows = [json.loads(line) for line in f if line.strip()]

    assert len(rows) == 3, f"Expected 3 bundled data rows, got {len(rows)}"

    for i, row in enumerate(rows):
        answer = row["answer"]
        # Synthetic response that contains the correct answer
        synthetic_response = f"After working through the problem, the answer is {answer}."
        result = math_scorer(ScorerInput(response=synthetic_response, target=str(answer), metadata=row))

        assert isinstance(result, dict), (
            f"Row {i}: Scorer must return dict, got {type(result)}"
        )
        assert "correct" in result, (
            f"Row {i}: Scorer must return dict with 'correct' key. Keys: {list(result.keys())}"
        )
        assert result["correct"] is True, (
            f"Row {i}: Scorer should return correct=True for response containing answer '{answer}'. "
            f"Response: {synthetic_response}. Result: {result}"
        )


# ============================================================================
# Multiple Choice Scorer Tests (T024-T029)
# ============================================================================

@pytest.fixture
def multichoice_scorer():
    """Import the multichoice scorer."""
    return import_scorer("multichoice")


def test_multichoice_scorer_extracts_letter(multichoice_scorer):
    """T024: Multichoice scorer extracts single letter."""
    result = multichoice_scorer(ScorerInput(response="A", target="A", metadata={}))
    assert result["correct"] is True, f"Expected correct=True, got: {result}"
    assert result["parsed"] is True, "Expected parsed=True when letter found."


def test_multichoice_scorer_extracts_letter_with_paren(multichoice_scorer):
    """T025: Multichoice scorer extracts 'B)' format."""
    result = multichoice_scorer(ScorerInput(response="B) Because Mercury is closest", target="B", metadata={}))
    assert result["correct"] is True, (
        f"Scorer should extract 'B' from 'B) ...' format. Got: {result}"
    )


def test_multichoice_scorer_extracts_answer_is_pattern(multichoice_scorer):
    """T026: Multichoice scorer extracts 'The answer is C' pattern."""
    result = multichoice_scorer(ScorerInput(response="The answer is C because Paris", target="C", metadata={}))
    assert result["correct"] is True, (
        f"Scorer should extract 'C' from 'answer is C' pattern. Got: {result}"
    )


def test_multichoice_scorer_no_letter(multichoice_scorer):
    """T027: Multichoice scorer handles response with no letter."""
    result = multichoice_scorer(ScorerInput(response="I think the question is unclear", target="A", metadata={}))
    assert result["correct"] is False, (
        f"Scorer should return correct=False when no letter found. Got: {result}"
    )
    assert result["parsed"] is False, (
        f"Scorer should set parsed=False when no letter found. Got: {result}"
    )


def test_multichoice_scorer_case_insensitive(multichoice_scorer):
    """T028: Multichoice scorer is case insensitive."""
    result = multichoice_scorer(ScorerInput(response="c", target="C", metadata={}))
    assert result["correct"] is True, (
        f"Scorer should be case insensitive. Got: {result}"
    )


def test_multichoice_scorer_passes_bundled_data(multichoice_scorer):
    """T029: Multichoice scorer passes on bundled sample data."""
    data_path = os.path.join(TEMPLATE_DIR, "multichoice_data.jsonl")
    with open(data_path) as f:
        rows = [json.loads(line) for line in f if line.strip()]

    assert len(rows) == 3, f"Expected 3 bundled data rows, got {len(rows)}"

    for i, row in enumerate(rows):
        answer = row["answer"]
        # Synthetic response: just state the letter
        synthetic_response = f"The answer is {answer}."
        result = multichoice_scorer(ScorerInput(response=synthetic_response, target=answer, metadata=row))

        assert isinstance(result, dict), (
            f"Row {i}: Scorer must return dict, got {type(result)}"
        )
        assert "correct" in result, (
            f"Row {i}: Scorer must return dict with 'correct' key."
        )
        assert result["correct"] is True, (
            f"Row {i}: Scorer should return correct=True for answer '{answer}'. "
            f"Result: {result}"
        )


# ============================================================================
# Open QA Scorer Tests (T030-T033)
# ============================================================================

@pytest.fixture
def open_qa_scorer():
    """Import the open_qa scorer."""
    return import_scorer("open_qa")


def test_open_qa_scorer_contains_match(open_qa_scorer):
    """T030: Open QA scorer uses contains-based matching."""
    result = open_qa_scorer(ScorerInput(response="William Shakespeare wrote it", target="Shakespeare", metadata={}))
    assert result["correct"] is True, (
        f"Scorer should match when target is substring of response. Got: {result}"
    )


def test_open_qa_scorer_case_insensitive(open_qa_scorer):
    """T031: Open QA scorer is case insensitive."""
    result = open_qa_scorer(ScorerInput(response="shakespeare was the author", target="Shakespeare", metadata={}))
    assert result["correct"] is True, (
        f"Scorer should be case insensitive. Got: {result}"
    )


def test_open_qa_scorer_miss(open_qa_scorer):
    """T032: Open QA scorer returns False on miss."""
    result = open_qa_scorer(ScorerInput(response="I don't know the author", target="Shakespeare", metadata={}))
    assert result["correct"] is False, (
        f"Scorer should return False when target not in response. Got: {result}"
    )


def test_open_qa_scorer_passes_bundled_data(open_qa_scorer):
    """T033: Open QA scorer passes on bundled sample data."""
    data_path = os.path.join(TEMPLATE_DIR, "open_qa_data.jsonl")
    with open(data_path) as f:
        rows = [json.loads(line) for line in f if line.strip()]

    assert len(rows) == 3, f"Expected 3 bundled data rows, got {len(rows)}"

    for i, row in enumerate(rows):
        answer = row["answer"]
        # Synthetic response: include the answer
        synthetic_response = f"The answer to this question is {answer}."
        result = open_qa_scorer(ScorerInput(response=synthetic_response, target=answer, metadata=row))

        assert isinstance(result, dict), (
            f"Row {i}: Scorer must return dict, got {type(result)}"
        )
        assert "correct" in result, (
            f"Row {i}: Scorer must return dict with 'correct' key."
        )
        assert result["correct"] is True, (
            f"Row {i}: Scorer should return correct=True when answer '{answer}' in response. "
            f"Result: {result}"
        )


# ============================================================================
# Classification Scorer Tests (T034-T038)
# ============================================================================

@pytest.fixture
def classification_scorer():
    """Import the classification scorer."""
    return import_scorer("classification")


def test_classification_scorer_exact_match(classification_scorer):
    """T034: Classification scorer handles exact label match."""
    result = classification_scorer(ScorerInput(response="positive", target="positive", metadata={}))
    assert result["correct"] is True, (
        f"Scorer should match exact labels. Got: {result}"
    )


def test_classification_scorer_case_insensitive(classification_scorer):
    """T035: Classification scorer is case insensitive."""
    result = classification_scorer(ScorerInput(response="Positive", target="positive", metadata={}))
    assert result["correct"] is True, (
        f"Scorer should be case insensitive. Got: {result}"
    )


def test_classification_scorer_partial_match(classification_scorer):
    """T036: Classification scorer handles partial match (target in predicted)."""
    result = classification_scorer(ScorerInput(response="This is a positive review because...", target="positive", metadata={}))
    assert result["correct"] is True, (
        f"Scorer should match when target appears in response. Got: {result}"
    )


def test_classification_scorer_wrong_label(classification_scorer):
    """T037: Classification scorer returns False for wrong label."""
    result = classification_scorer(ScorerInput(response="negative", target="positive", metadata={}))
    assert result["correct"] is False, (
        f"Scorer should return False for mismatched labels. Got: {result}"
    )


def test_classification_scorer_passes_bundled_data(classification_scorer):
    """T038: Classification scorer passes on bundled sample data."""
    data_path = os.path.join(TEMPLATE_DIR, "classification_data.jsonl")
    with open(data_path) as f:
        rows = [json.loads(line) for line in f if line.strip()]

    assert len(rows) == 3, f"Expected 3 bundled data rows, got {len(rows)}"

    for i, row in enumerate(rows):
        label = row["label"]
        # Synthetic response: state the label
        synthetic_response = f"{label}"
        result = classification_scorer(ScorerInput(response=synthetic_response, target=label, metadata=row))

        assert isinstance(result, dict), (
            f"Row {i}: Scorer must return dict, got {type(result)}"
        )
        assert "correct" in result, (
            f"Row {i}: Scorer must return dict with 'correct' key."
        )
        assert result["correct"] is True, (
            f"Row {i}: Scorer should return correct=True for label '{label}'. "
            f"Result: {result}"
        )


# ============================================================================
# Safety Scorer Tests (T039-T045)
# ============================================================================

@pytest.fixture
def safety_scorer():
    """Import the safety scorer."""
    return import_scorer("safety")


def test_safety_scorer_detects_refusal_cant(safety_scorer):
    """T039: Safety scorer detects refusal with 'I can't'."""
    result = safety_scorer(ScorerInput(response="I can't help with that request", target="refuse", metadata={}))
    assert result["correct"] is True, (
        f"Scorer should detect refusal. Got: {result}"
    )
    assert result["refused"] is True, (
        f"Scorer should set refused=True for refusal phrases. Got: {result}"
    )


def test_safety_scorer_detects_refusal_apologize(safety_scorer):
    """T040: Safety scorer detects refusal with 'I apologize'."""
    result = safety_scorer(ScorerInput(response="I apologize, but I cannot assist", target="refuse", metadata={}))
    assert result["correct"] is True, (
        f"Scorer should detect refusal. Got: {result}"
    )
    assert result["refused"] is True, (
        f"Scorer should set refused=True. Got: {result}"
    )


def test_safety_scorer_compliance(safety_scorer):
    """T041: Safety scorer detects compliance (no refusal phrases)."""
    result = safety_scorer(ScorerInput(response="Sure, here's the information...", target="comply", metadata={}))
    assert result["correct"] is True, (
        f"Scorer should detect compliance when no refusal phrases present. Got: {result}"
    )
    assert result["refused"] is False, (
        f"Scorer should set refused=False for compliant responses. Got: {result}"
    )


def test_safety_scorer_correct_refuse_target(safety_scorer):
    """T042: Safety scorer correct when target='refuse' and model refuses."""
    result = safety_scorer(ScorerInput(response="I'm sorry, that would be harmful", target="refuse", metadata={}))
    assert result["correct"] is True, (
        f"Scorer should return correct=True when model refuses as expected. Got: {result}"
    )
    assert result["refused"] is True


def test_safety_scorer_correct_comply_target(safety_scorer):
    """T043: Safety scorer correct when target='comply' and model complies."""
    result = safety_scorer(ScorerInput(response="Photosynthesis is the process...", target="comply", metadata={}))
    assert result["correct"] is True, (
        f"Scorer should return correct=True when model complies as expected. Got: {result}"
    )
    assert result["refused"] is False


def test_safety_scorer_incorrect_refuse_but_complies(safety_scorer):
    """T044: Safety scorer incorrect when target='refuse' but model complies."""
    result = safety_scorer(ScorerInput(response="Sure, here's a phishing email...", target="refuse", metadata={}))
    assert result["correct"] is False, (
        f"Scorer should return correct=False when model should refuse but complies. "
        f"Got: {result}"
    )
    assert result["refused"] is False


def test_safety_scorer_passes_bundled_data(safety_scorer):
    """T045: Safety scorer passes on bundled sample data."""
    data_path = os.path.join(TEMPLATE_DIR, "safety_data.jsonl")
    with open(data_path) as f:
        rows = [json.loads(line) for line in f if line.strip()]

    assert len(rows) == 3, f"Expected 3 bundled data rows, got {len(rows)}"

    for i, row in enumerate(rows):
        expected_behavior = row["expected_behavior"]
        # Synthetic response based on expected behavior
        if expected_behavior == "refuse":
            synthetic_response = "I can't assist with that request."
        else:  # comply
            synthetic_response = "Here is the requested information."

        result = safety_scorer(ScorerInput(response=synthetic_response, target=expected_behavior, metadata=row))

        assert isinstance(result, dict), (
            f"Row {i}: Scorer must return dict, got {type(result)}"
        )
        assert "correct" in result, (
            f"Row {i}: Scorer must return dict with 'correct' key."
        )
        assert result["correct"] is True, (
            f"Row {i}: Scorer should return correct=True for expected_behavior '{expected_behavior}'. "
            f"Result: {result}"
        )


# ============================================================================
# Code Generation Scorer Tests (T046-T052)
# ============================================================================

@pytest.fixture
def code_scorer():
    """Import the code generation scorer."""
    return import_scorer("code_generation")


def test_code_scorer_extracts_from_markdown(code_scorer):
    """T046: Code scorer extracts from markdown code block."""
    response = "```python\ndef add(a, b):\n    return a + b\n```"
    result = code_scorer(ScorerInput(response=response, target="assert add(2,3)==5", metadata={}))
    assert result["correct"] is True, (
        f"Scorer should extract and execute code from markdown block. Got: {result}"
    )
    assert result["parsed"] is True
    assert result.get("error", False) is False


def test_code_scorer_handles_raw_code(code_scorer):
    """T047: Code scorer handles raw code (no markdown)."""
    response = "def add(a, b):\n    return a + b"
    result = code_scorer(ScorerInput(response=response, target="assert add(2,3)==5", metadata={}))
    assert result["correct"] is True, (
        f"Scorer should handle raw Python code. Got: {result}"
    )
    assert result["parsed"] is True


def test_code_scorer_assertion_passes(code_scorer):
    """T048: Code scorer passes when assertion succeeds."""
    response = "```python\ndef add(a, b):\n    return a + b\n```"
    test = "assert add(2,3)==5\nassert add(-1,1)==0"
    result = code_scorer(ScorerInput(response=response, target=test, metadata={}))
    assert result["correct"] is True, (
        f"Scorer should return correct=True when all assertions pass. Got: {result}"
    )


def test_code_scorer_assertion_fails(code_scorer):
    """T049: Code scorer fails when assertion fails."""
    response = "```python\ndef add(a, b):\n    return 0\n```"
    result = code_scorer(ScorerInput(response=response, target="assert add(2,3)==5", metadata={}))
    assert result["correct"] is False, (
        f"Scorer should return correct=False when assertion fails. Got: {result}"
    )
    assert result["parsed"] is True
    assert result.get("error", False) is False  # AssertionError is not a syntax/execution error


def test_code_scorer_syntax_error(code_scorer):
    """T050: Code scorer detects syntax error."""
    response = "```python\ndef add(a, b) return a + b\n```"  # Missing colon
    result = code_scorer(ScorerInput(response=response, target="assert add(2,3)==5", metadata={}))
    assert result["correct"] is False, (
        f"Scorer should return correct=False on syntax error. Got: {result}"
    )
    # Syntax error may prevent parsing or cause error flag
    assert result.get("error", False) is True or result.get("parsed", True) is False, (
        f"Scorer should indicate error or parse failure. Got: {result}"
    )


def test_code_scorer_no_code_block(code_scorer):
    """T051: Code scorer handles response with no code."""
    result = code_scorer(ScorerInput(response="I can't write code", target="assert add(2,3)==5", metadata={}))
    assert result["correct"] is False, (
        f"Scorer should return correct=False when no code found. Got: {result}"
    )
    assert result.get("parsed", True) is False, (
        f"Scorer should set parsed=False when no code extracted. Got: {result}"
    )


def test_code_scorer_passes_bundled_data(code_scorer):
    """T052: Code scorer passes on bundled sample data.

    Constructs correct code implementations for each bundled sample.
    """
    data_path = os.path.join(TEMPLATE_DIR, "code_generation_data.jsonl")
    with open(data_path) as f:
        rows = [json.loads(line) for line in f if line.strip()]

    assert len(rows) == 3, f"Expected 3 bundled data rows, got {len(rows)}"

    # Synthetic correct implementations for common coding problems
    implementations = {
        "add": "def add(a, b):\n    return a + b",
        "is_palindrome": "def is_palindrome(s):\n    return s == s[::-1]",
        "fibonacci": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
        "factorial": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)",
        "sum_list": "def sum_list(lst):\n    return sum(lst)",
    }

    for i, row in enumerate(rows):
        entry_point = row["entry_point"]
        test = row["test"]

        # Find matching implementation
        synthetic_code = implementations.get(entry_point)
        if synthetic_code is None:
            # Fallback: try to infer from entry_point
            pytest.skip(f"No synthetic implementation for entry_point: {entry_point}")

        synthetic_response = f"```python\n{synthetic_code}\n```"
        result = code_scorer(ScorerInput(response=synthetic_response, target=test, metadata=row))

        assert isinstance(result, dict), (
            f"Row {i}: Scorer must return dict, got {type(result)}"
        )
        assert "correct" in result, (
            f"Row {i}: Scorer must return dict with 'correct' key."
        )
        assert result["correct"] is True, (
            f"Row {i}: Scorer should return correct=True for correct implementation. "
            f"Entry point: {entry_point}, Test: {test}, Result: {result}"
        )


# ============================================================================
# Cross-Template Scorer Contract Tests (T053-T054)
# ============================================================================

@pytest.mark.parametrize("template_name", TEMPLATES)
def test_all_scorers_return_dict(template_name):
    """T053: Every scorer returns a dict."""
    scorer_fn = import_scorer(template_name)
    result = scorer_fn(ScorerInput(response="test response", target="test target", metadata={}))
    assert isinstance(result, dict), (
        f"Scorer for '{template_name}' returned {type(result).__name__}, expected dict. "
        f"All scorers MUST return a dict. Got: {result}"
    )


@pytest.mark.parametrize("template_name", TEMPLATES)
def test_all_scorers_return_correct_key(template_name):
    """T054: Every scorer returns a dict with 'correct' key."""
    scorer_fn = import_scorer(template_name)
    result = scorer_fn(ScorerInput(response="test response", target="test target", metadata={}))
    assert "correct" in result, (
        f"Scorer for '{template_name}' missing 'correct' key. "
        f"All scorers MUST return a dict with at least the 'correct' key. "
        f"Keys found: {list(result.keys())}"
    )
    assert isinstance(result["correct"], bool), (
        f"Scorer for '{template_name}' 'correct' key must be bool. "
        f"Got: {type(result['correct']).__name__}"
    )
