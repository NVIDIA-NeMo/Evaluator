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

from nemo_evaluator.contrib.byob.decorators import ScorerInput
from nemo_evaluator.contrib.byob.scorers import (
    BUILTIN_SCORERS,
    all_of,
    any_of,
    bleu,
    contains,
    exact_match,
    f1_token,
    regex_match,
    retrieval_metrics,
    rouge,
)


def _make_input(
    response: str = "", target: str = "", metadata: dict | None = None
) -> ScorerInput:
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
        result = contains(
            _make_input(response="The cat sat on the mat", target="cat sat")
        )
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
        result = f1_token(
            _make_input(response="the big cat sat", target="the cat sat down")
        )
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
        result = combined(
            _make_input(response="The cat sat on the mat", target="cat sat")
        )
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

        def mock_fn(prompt):
            return "response"

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


# ---------------------------------------------------------------------------
# BLEU scorer tests
# ---------------------------------------------------------------------------


class TestBleuScorer:
    """Tests for the bleu scorer function."""

    def test_bleu_perfect_match(self):
        """Validate identical response and target produce bleu_1 close to 1.0."""
        result = bleu(
            _make_input(
                response="the cat sat on the mat", target="the cat sat on the mat"
            )
        )
        assert abs(result["bleu_1"] - 1.0) < 0.01, (
            f"Expected bleu_1 close to 1.0 for perfect match, got {result['bleu_1']}"
        )

    def test_bleu_no_overlap(self):
        """Validate completely different texts produce all bleu_n close to 0 (but smoothed > 0)."""
        result = bleu(_make_input(response="alpha beta gamma", target="one two three"))
        for key in ("bleu_1", "bleu_2", "bleu_3", "bleu_4"):
            assert result[key] < 0.5, (
                f"Expected {key} < 0.5 for no overlap, got {result[key]}"
            )
            # Smoothed BLEU should still be > 0 due to add-1 smoothing
            assert result[key] > 0.0, (
                f"Expected {key} > 0.0 due to Laplace smoothing, got {result[key]}"
            )

    def test_bleu_partial_overlap(self):
        """Validate partial overlap produces bleu values strictly between 0 and 1."""
        result = bleu(
            _make_input(
                response="the cat sat on the mat",
                target="the cat played on the rug",
            )
        )
        for key in ("bleu_1", "bleu_2", "bleu_3", "bleu_4"):
            assert 0.0 < result[key] < 1.0, (
                f"Expected 0 < {key} < 1 for partial overlap, got {result[key]}"
            )

    def test_bleu_empty_inputs(self):
        """Validate empty response or target returns all 0.0."""
        result_empty_response = bleu(_make_input(response="", target="hello world"))
        result_empty_target = bleu(_make_input(response="hello world", target=""))
        result_both_empty = bleu(_make_input(response="", target=""))

        for result in (result_empty_response, result_empty_target, result_both_empty):
            for key in ("bleu_1", "bleu_2", "bleu_3", "bleu_4"):
                assert result[key] == 0.0, (
                    f"Expected {key} == 0.0 for empty input, got {result[key]}"
                )

    def test_bleu_returns_four_keys(self):
        """Validate bleu result dict has exactly bleu_1, bleu_2, bleu_3, bleu_4."""
        result = bleu(_make_input(response="hello world", target="hello world"))
        expected_keys = {"bleu_1", "bleu_2", "bleu_3", "bleu_4"}
        assert set(result.keys()) == expected_keys, (
            f"Expected keys {expected_keys}, got {set(result.keys())}"
        )

    def test_bleu_registered(self):
        """Validate 'bleu' is registered in BUILTIN_SCORERS."""
        assert "bleu" in BUILTIN_SCORERS, (
            f"Expected 'bleu' in BUILTIN_SCORERS, got keys {list(BUILTIN_SCORERS.keys())}"
        )


# ---------------------------------------------------------------------------
# ROUGE scorer tests
# ---------------------------------------------------------------------------


class TestRougeScorer:
    """Tests for the rouge scorer function."""

    def test_rouge_perfect_match(self):
        """Validate identical texts produce rouge_1 == rouge_2 == rouge_l == 1.0."""
        result = rouge(
            _make_input(
                response="the cat sat on the mat", target="the cat sat on the mat"
            )
        )
        assert result["rouge_1"] == 1.0, (
            f"Expected rouge_1 == 1.0, got {result['rouge_1']}"
        )
        assert result["rouge_2"] == 1.0, (
            f"Expected rouge_2 == 1.0, got {result['rouge_2']}"
        )
        assert result["rouge_l"] == 1.0, (
            f"Expected rouge_l == 1.0, got {result['rouge_l']}"
        )

    def test_rouge_no_overlap(self):
        """Validate completely different texts produce all 0.0."""
        result = rouge(_make_input(response="alpha beta gamma", target="one two three"))
        assert result["rouge_1"] == 0.0, (
            f"Expected rouge_1 == 0.0, got {result['rouge_1']}"
        )
        assert result["rouge_2"] == 0.0, (
            f"Expected rouge_2 == 0.0, got {result['rouge_2']}"
        )
        assert result["rouge_l"] == 0.0, (
            f"Expected rouge_l == 0.0, got {result['rouge_l']}"
        )

    def test_rouge_partial_overlap(self):
        """Validate some shared tokens produce values between 0 and 1."""
        result = rouge(
            _make_input(
                response="the cat sat on the mat",
                target="the cat played on the rug",
            )
        )
        for key in ("rouge_1", "rouge_2", "rouge_l"):
            assert 0.0 < result[key] < 1.0, (
                f"Expected 0 < {key} < 1 for partial overlap, got {result[key]}"
            )

    def test_rouge_lcs_correctness(self):
        """Validate ROUGE-L differs from n-gram overlap when LCS is shorter.

        pred: "A B C D"  ref: "A C B D"
        ROUGE-1 tokens overlap: A, B, C, D (4/4 each) => F1 = 1.0
        LCS: "A C D" or "A B D" (length 3), precision = 3/4, recall = 3/4
        ROUGE-L F1 = 2 * 0.75 * 0.75 / (0.75 + 0.75) = 0.75
        """
        result = rouge(_make_input(response="A B C D", target="A C B D"))
        assert result["rouge_1"] == 1.0, (
            f"Expected rouge_1 == 1.0 (all unigrams match), got {result['rouge_1']}"
        )
        assert abs(result["rouge_l"] - 0.75) < 0.01, (
            f"Expected rouge_l close to 0.75, got {result['rouge_l']}"
        )

    def test_rouge_empty_inputs(self):
        """Validate empty response or target returns all 0.0."""
        for resp, tgt in [("", "hello"), ("hello", ""), ("", "")]:
            result = rouge(_make_input(response=resp, target=tgt))
            for key in ("rouge_1", "rouge_2", "rouge_l"):
                assert result[key] == 0.0, (
                    f"Expected {key} == 0.0 for empty input ({resp!r}, {tgt!r}), "
                    f"got {result[key]}"
                )


# ---------------------------------------------------------------------------
# Retrieval metrics tests
# ---------------------------------------------------------------------------


class TestRetrievalMetrics:
    """Tests for the retrieval_metrics scorer function."""

    def test_retrieval_perfect(self):
        """Validate all retrieved are relevant produces perfect scores."""
        sample = _make_input(
            metadata={
                "retrieved": ["a", "b", "c"],
                "relevant": ["a", "b", "c"],
            }
        )
        result = retrieval_metrics(sample)
        assert result["precision_at_k"] == 1.0, (
            f"Expected precision_at_k == 1.0, got {result['precision_at_k']}"
        )
        assert result["recall_at_k"] == 1.0, (
            f"Expected recall_at_k == 1.0, got {result['recall_at_k']}"
        )
        assert result["mrr"] == 1.0, f"Expected mrr == 1.0, got {result['mrr']}"
        assert result["ndcg"] == 1.0, f"Expected ndcg == 1.0, got {result['ndcg']}"

    def test_retrieval_none_relevant(self):
        """Validate no retrieved item is relevant produces all 0.0."""
        sample = _make_input(
            metadata={
                "retrieved": ["x", "y", "z"],
                "relevant": ["a", "b", "c"],
            }
        )
        result = retrieval_metrics(sample)
        assert result["precision_at_k"] == 0.0, (
            f"Expected precision_at_k == 0.0, got {result['precision_at_k']}"
        )
        assert result["recall_at_k"] == 0.0, (
            f"Expected recall_at_k == 0.0, got {result['recall_at_k']}"
        )
        assert result["mrr"] == 0.0, f"Expected mrr == 0.0, got {result['mrr']}"
        assert result["ndcg"] == 0.0, f"Expected ndcg == 0.0, got {result['ndcg']}"

    def test_retrieval_partial(self):
        """Validate some hits produce expected precision and recall.

        retrieved = ["a", "x", "b"], relevant = ["a", "b"]
        k = 3 (default), hits = 2
        precision@k = 2/3, recall@k = 2/2 = 1.0
        """
        sample = _make_input(
            metadata={
                "retrieved": ["a", "x", "b"],
                "relevant": ["a", "b"],
            }
        )
        result = retrieval_metrics(sample)
        assert abs(result["precision_at_k"] - 2.0 / 3.0) < 0.001, (
            f"Expected precision_at_k close to 0.667, got {result['precision_at_k']}"
        )
        assert result["recall_at_k"] == 1.0, (
            f"Expected recall_at_k == 1.0, got {result['recall_at_k']}"
        )

    def test_retrieval_mrr_position(self):
        """Validate first relevant item at position 3 produces mrr == 1/3."""
        sample = _make_input(
            metadata={
                "retrieved": ["x", "y", "a", "b"],
                "relevant": ["a", "b"],
            }
        )
        result = retrieval_metrics(sample)
        assert abs(result["mrr"] - 1.0 / 3.0) < 0.001, (
            f"Expected mrr close to 0.333, got {result['mrr']}"
        )

    def test_retrieval_empty_lists(self):
        """Validate missing metadata keys produce all 0.0."""
        # Empty metadata dict -- no "retrieved" or "relevant" keys
        sample = _make_input(metadata={})
        result = retrieval_metrics(sample)
        for key in ("precision_at_k", "recall_at_k", "mrr", "ndcg"):
            assert result[key] == 0.0, (
                f"Expected {key} == 0.0 for empty metadata, got {result[key]}"
            )


# ---------------------------------------------------------------------------
# Multi-turn conversation tests (B3)
# ---------------------------------------------------------------------------


class TestMultiTurnConversation:
    """Tests for ScorerInput multi-turn conversation fields (B3)."""

    def test_scorer_input_with_conversation(self):
        """Validate ScorerInput with conversation list is accessible."""
        conversation = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]
        inp = ScorerInput(
            response="I'm doing well",
            target="fine",
            metadata={},
            conversation=conversation,
        )
        assert inp.conversation == conversation, (
            f"Expected conversation to be accessible, got {inp.conversation}"
        )
        assert len(inp.conversation) == 3, (
            f"Expected 3 turns, got {len(inp.conversation)}"
        )

    def test_scorer_input_with_turn_index(self):
        """Validate turn_index field works correctly."""
        inp = ScorerInput(
            response="resp",
            target="tgt",
            metadata={},
            turn_index=5,
        )
        assert inp.turn_index == 5, f"Expected turn_index == 5, got {inp.turn_index}"

    def test_scorer_receives_conversation_in_eval(self):
        """Validate a scorer that reads conversation receives it through ScorerInput."""
        captured = {}

        def conversation_scorer(sample):
            captured["conversation"] = sample.conversation
            captured["turn_index"] = sample.turn_index
            return {"correct": True}

        conversation = [{"role": "user", "content": "test"}]
        inp = ScorerInput(
            response="resp",
            target="tgt",
            metadata={},
            conversation=conversation,
            turn_index=0,
        )
        result = conversation_scorer(inp)

        assert result == {"correct": True}, (
            f"Expected scorer to return correct result, got {result}"
        )
        assert captured["conversation"] == conversation, (
            f"Expected scorer to receive conversation, got {captured['conversation']}"
        )
        assert captured["turn_index"] == 0, (
            f"Expected scorer to receive turn_index=0, got {captured['turn_index']}"
        )

    def test_conversation_default_is_none(self):
        """Validate conversation defaults to None when not provided."""
        inp = ScorerInput(response="r", target="t", metadata={})
        assert inp.conversation is None, (
            f"Expected conversation default to be None, got {inp.conversation}"
        )
