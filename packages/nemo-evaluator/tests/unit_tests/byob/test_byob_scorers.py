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

from nemo_evaluator.byob.scorers import contains, exact_match, f1_token, regex_match


class TestContainsScorer:
    """Tests for the contains scorer."""

    def test_contains_case_insensitive(self):
        """Validate case-insensitive substring matching."""
        result = contains("The Answer is YES", "yes", {})
        assert result == {"correct": True}, (
            f"Expected contains to return True for case-insensitive match, got {result}"
        )

    def test_contains_substring(self):
        """Validate substring detection with multiple words."""
        result = contains("The cat sat on the mat", "cat sat", {})
        assert result == {"correct": True}, (
            f"Expected contains to find 'cat sat' substring, got {result}"
        )

    def test_contains_miss(self):
        """Validate contains returns False when target not found."""
        result = contains("Hello world", "xyz", {})
        assert result == {"correct": False}, (
            f"Expected contains to return False for missing substring, got {result}"
        )

    def test_contains_empty_string(self):
        """Validate empty string is always a substring (Python behavior)."""
        result = contains("anything", "", {})
        assert result == {"correct": True}, (
            f"Expected contains to return True for empty target string, got {result}"
        )


class TestExactMatchScorer:
    """Tests for the exact_match scorer."""

    def test_exact_match_case_insensitive(self):
        """Validate case-insensitive exact matching."""
        result = exact_match("Hello World", "hello world", {})
        assert result == {"correct": True}, (
            f"Expected exact_match to be case-insensitive, got {result}"
        )

    def test_exact_match_whitespace(self):
        """Validate whitespace stripping in exact match."""
        result = exact_match("  hello  ", "hello", {})
        assert result == {"correct": True}, (
            f"Expected exact_match to strip whitespace, got {result}"
        )

    def test_exact_match_partial_mismatch(self):
        """Validate partial matches fail exact_match."""
        result = exact_match("hello", "hello world", {})
        assert result == {"correct": False}, (
            f"Expected exact_match to return False for partial match, got {result}"
        )

    def test_exact_match_both_empty(self):
        """Validate both empty strings match exactly."""
        result = exact_match("", "", {})
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
        result = f1_token("the big cat sat", "the cat sat down", {})
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
        result = f1_token("", "hello world", {})
        assert result == {"f1": 0.0, "precision": 0.0, "recall": 0.0}, (
            f"Expected zeroed metrics for empty response, got {result}"
        )

        result = f1_token("hello", "", {})
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
        result = f1_token("hello world", "hello world", {})
        assert result == {"f1": 1.0, "precision": 1.0, "recall": 1.0}, (
            f"Expected perfect f1=1.0 for identical inputs, got {result}"
        )


class TestRegexMatchScorer:
    """Tests for the regex_match scorer."""

    def test_regex_match_digit_pattern(self):
        """Validate regex pattern matching for digits."""
        result = regex_match("The answer is 42", r"\d+", {})
        assert result == {"correct": True}, (
            f"Expected regex_match to find digit pattern, got {result}"
        )

    def test_regex_match_no_match(self):
        """Validate regex returns False when pattern not found."""
        result = regex_match("no digits here", r"\d+", {})
        assert result == {"correct": False}, (
            f"Expected regex_match to return False when no match, got {result}"
        )

    def test_regex_match_invalid_regex(self):
        """Validate invalid regex patterns are handled gracefully."""
        result = regex_match("anything", r"[invalid", {})
        assert result == {"correct": False}, (
            f"Expected regex_match to return False for invalid regex, got {result}"
        )
