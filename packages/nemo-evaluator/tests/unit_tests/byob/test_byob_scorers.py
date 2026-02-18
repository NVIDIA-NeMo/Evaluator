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

"""Unit tests for BYOB built-in scorer functions."""

from nemo_evaluator.byob.decorators import ScorerInput
from nemo_evaluator.byob.scorers import (
    all_of,
    any_of,
    contains,
    exact_match,
    f1_token,
    regex_match,
)


def _make_input(response: str = "", target: str = "", metadata: dict | None = None) -> ScorerInput:
    """Helper to construct a ScorerInput with sensible defaults."""
    return ScorerInput(response=response, target=target, metadata=metadata or {})


class TestContainsScorer:
    """Tests for the contains scorer."""

    def test_contains_case_insensitive(self):
        """Validate case-insensitive substring matching."""
        result = contains(_make_input(response="The Answer is YES", target="yes"))
        assert result == {"correct": True}, (
            f"Expected contains to return True for case-insensitive match, got {result}"
        )

    def test_contains_substring(self):
        """Validate substring detection with multiple words."""
        result = contains(_make_input(response="The cat sat on the mat", target="cat sat"))
        assert result == {"correct": True}, (
            f"Expected contains to find 'cat sat' substring, got {result}"
        )

    def test_contains_miss(self):
        """Validate contains returns False when target not found."""
        result = contains(_make_input(response="Hello world", target="xyz"))
        assert result == {"correct": False}, (
            f"Expected contains to return False for missing substring, got {result}"
        )

    def test_contains_empty_string(self):
        """Validate empty string is always a substring (Python behavior)."""
        result = contains(_make_input(response="anything", target=""))
        assert result == {"correct": True}, (
            f"Expected contains to return True for empty target string, got {result}"
        )


class TestExactMatchScorer:
    """Tests for the exact_match scorer."""

    def test_exact_match_case_insensitive(self):
        """Validate case-insensitive exact matching."""
        result = exact_match(_make_input(response="Hello World", target="hello world"))
        assert result == {"correct": True}, (
            f"Expected exact_match to be case-insensitive, got {result}"
        )

    def test_exact_match_whitespace(self):
        """Validate whitespace stripping in exact match."""
        result = exact_match(_make_input(response="  hello  ", target="hello"))
        assert result == {"correct": True}, (
            f"Expected exact_match to strip whitespace, got {result}"
        )

    def test_exact_match_partial_mismatch(self):
        """Validate partial matches fail exact_match."""
        result = exact_match(_make_input(response="hello", target="hello world"))
        assert result == {"correct": False}, (
            f"Expected exact_match to return False for partial match, got {result}"
        )

    def test_exact_match_both_empty(self):
        """Validate both empty strings match exactly."""
        result = exact_match(_make_input(response="", target=""))
        assert result == {"correct": True}, (
            f"Expected exact_match to return True for both empty strings, got {result}"
        )


class TestF1TokenScorer:
    """Tests for the f1_token scorer."""

    def test_f1_token_partial_overlap(self):
        """Validate token-level F1 computation with partial overlap.

        Hand-computed values:
        - pred_tokens = ["the", "big", "cat", "sat"] (4 tokens)
        - ref_tokens = ["the", "cat", "sat", "down"] (4 tokens)
        - common = {"the": 1, "cat": 1, "sat": 1} -> num_common = 3
        - precision = 3/4 = 0.75
        - recall = 3/4 = 0.75
        - f1 = 2 * 0.75 * 0.75 / (0.75 + 0.75) = 0.75
        """
        result = f1_token(_make_input(response="the big cat sat", target="the cat sat down"))
        assert abs(result["f1"] - 0.75) < 0.0001, (
            f"Expected f1=0.75, got {result['f1']}"
        )
        assert abs(result["precision"] - 0.75) < 0.0001, (
            f"Expected precision=0.75, got {result['precision']}"
        )
        assert abs(result["recall"] - 0.75) < 0.0001, (
            f"Expected recall=0.75, got {result['recall']}"
        )

    def test_f1_token_empty_input(self):
        """Validate f1_token returns zeroed metrics for empty inputs."""
        result = f1_token(_make_input(response="", target="hello world"))
        assert result == {"f1": 0.0, "precision": 0.0, "recall": 0.0}, (
            f"Expected zeroed metrics for empty response, got {result}"
        )

        result = f1_token(_make_input(response="hello", target=""))
        assert result == {"f1": 0.0, "precision": 0.0, "recall": 0.0}, (
            f"Expected zeroed metrics for empty target, got {result}"
        )

    def test_f1_token_perfect_match(self):
        """Validate f1_token returns 1.0 for perfect match.

        Hand-computed:
        - pred = ["hello", "world"], ref = ["hello", "world"]
        - common = {"hello": 1, "world": 1} -> 2
        - precision = 2/2 = 1.0, recall = 2/2 = 1.0, f1 = 1.0
        """
        result = f1_token(_make_input(response="hello world", target="hello world"))
        assert result == {"f1": 1.0, "precision": 1.0, "recall": 1.0}, (
            f"Expected perfect f1=1.0 for identical inputs, got {result}"
        )


class TestRegexMatchScorer:
    """Tests for the regex_match scorer."""

    def test_regex_match_digit_pattern(self):
        """Validate regex pattern matching for digits."""
        result = regex_match(_make_input(response="The answer is 42", target=r"\d+"))
        assert result == {"correct": True}, (
            f"Expected regex_match to find digit pattern, got {result}"
        )

    def test_regex_match_no_match(self):
        """Validate regex returns False when pattern not found."""
        result = regex_match(_make_input(response="no digits here", target=r"\d+"))
        assert result == {"correct": False}, (
            f"Expected regex_match to return False when no match, got {result}"
        )

    def test_regex_match_invalid_regex(self):
        """Validate invalid regex patterns are handled gracefully."""
        result = regex_match(_make_input(response="anything", target=r"[invalid"))
        assert result == {"correct": False}, (
            f"Expected regex_match to return False for invalid regex, got {result}"
        )


class TestAnyOfCombinator:
    """Tests for the any_of scorer combinator."""

    def test_any_of_first_matches(self):
        """Validate any_of returns correct=True when first scorer matches."""
        combined = any_of(contains, exact_match)
        result = combined(_make_input(response="The cat sat on the mat", target="cat sat"))
        assert result["correct"] is True, (
            f"Expected any_of to return correct=True when contains matches, got {result}"
        )

    def test_any_of_second_matches(self):
        """Validate any_of returns correct=True when only second scorer matches."""
        combined = any_of(exact_match, contains)
        # "hello world" does not exact-match "hello", but contains matches
        result = combined(_make_input(response="hello world", target="hello"))
        assert result["correct"] is True, (
            f"Expected any_of to return correct=True when contains matches, got {result}"
        )

    def test_any_of_none_match(self):
        """Validate any_of returns correct=False when no scorer matches."""
        combined = any_of(contains, exact_match)
        result = combined(_make_input(response="apples", target="oranges"))
        assert result["correct"] is False, (
            f"Expected any_of to return correct=False when no scorer matches, got {result}"
        )

    def test_any_of_includes_sub_results(self):
        """Validate any_of result dict includes namespaced sub-results."""
        combined = any_of(contains, exact_match)
        result = combined(_make_input(response="hello", target="hello"))
        assert "contains_correct" in result, (
            f"Expected 'contains_correct' key in any_of result, got keys {list(result.keys())}"
        )
        assert "exact_match_correct" in result, (
            f"Expected 'exact_match_correct' key in any_of result, got keys {list(result.keys())}"
        )

    def test_any_of_name(self):
        """Validate the combined function has a descriptive __name__."""
        combined = any_of(contains, exact_match)
        assert combined.__name__ == "any_of(contains, exact_match)", (
            f"Expected descriptive __name__, got {combined.__name__}"
        )


class TestAllOfCombinator:
    """Tests for the all_of scorer combinator."""

    def test_all_of_both_match(self):
        """Validate all_of returns correct=True when all scorers match."""
        combined = all_of(contains, exact_match)
        result = combined(_make_input(response="hello", target="hello"))
        assert result["correct"] is True, (
            f"Expected all_of to return correct=True when both match, got {result}"
        )

    def test_all_of_partial_match(self):
        """Validate all_of returns correct=False when only one scorer matches."""
        combined = all_of(contains, exact_match)
        # "hello world" contains "hello" but does not exact-match it
        result = combined(_make_input(response="hello world", target="hello"))
        assert result["correct"] is False, (
            f"Expected all_of to return correct=False when only contains matches, got {result}"
        )

    def test_all_of_none_match(self):
        """Validate all_of returns correct=False when no scorer matches."""
        combined = all_of(contains, exact_match)
        result = combined(_make_input(response="apples", target="oranges"))
        assert result["correct"] is False, (
            f"Expected all_of to return correct=False when no scorer matches, got {result}"
        )

    def test_all_of_includes_sub_results(self):
        """Validate all_of result dict includes namespaced sub-results."""
        combined = all_of(contains, exact_match)
        result = combined(_make_input(response="hello", target="hello"))
        assert "contains_correct" in result, (
            f"Expected 'contains_correct' key in all_of result, got keys {list(result.keys())}"
        )
        assert "exact_match_correct" in result, (
            f"Expected 'exact_match_correct' key in all_of result, got keys {list(result.keys())}"
        )

    def test_all_of_name(self):
        """Validate the combined function has a descriptive __name__."""
        combined = all_of(contains, exact_match)
        assert combined.__name__ == "all_of(contains, exact_match)", (
            f"Expected descriptive __name__, got {combined.__name__}"
        )


class TestScorerInputDefaults:
    """Tests for ScorerInput dataclass optional field defaults."""

    def test_required_fields_only(self):
        """Validate ScorerInput can be created with only required fields."""
        sample = ScorerInput(response="answer", target="answer", metadata={})
        assert sample.response == "answer"
        assert sample.target == "answer"
        assert sample.metadata == {}

    def test_optional_fields_default_to_none_or_empty(self):
        """Validate optional fields have correct defaults."""
        sample = ScorerInput(response="a", target="b", metadata={})
        assert sample.model_call_fn is None, (
            f"Expected model_call_fn default to be None, got {sample.model_call_fn}"
        )
        assert sample.config == {}, (
            f"Expected config default to be empty dict, got {sample.config}"
        )
        assert sample.conversation is None, (
            f"Expected conversation default to be None, got {sample.conversation}"
        )
        assert sample.turn_index is None, (
            f"Expected turn_index default to be None, got {sample.turn_index}"
        )

    def test_optional_fields_can_be_set(self):
        """Validate optional fields can be explicitly provided."""
        mock_fn = lambda prompt: "response"
        sample = ScorerInput(
            response="a",
            target="b",
            metadata={"key": "value"},
            model_call_fn=mock_fn,
            config={"temperature": 0.7},
            conversation=[{"role": "user", "content": "hi"}],
            turn_index=2,
        )
        assert sample.model_call_fn is mock_fn, (
            f"Expected model_call_fn to be set, got {sample.model_call_fn}"
        )
        assert sample.config == {"temperature": 0.7}, (
            f"Expected config to be set, got {sample.config}"
        )
        assert sample.conversation == [{"role": "user", "content": "hi"}], (
            f"Expected conversation to be set, got {sample.conversation}"
        )
        assert sample.turn_index == 2, (
            f"Expected turn_index to be 2, got {sample.turn_index}"
        )
