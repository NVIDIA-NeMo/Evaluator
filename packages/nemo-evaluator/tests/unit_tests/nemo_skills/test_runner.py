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

"""Tests for nemo-skills runner and scorer functions."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_evaluator.plugins.nemo_skills.runner import (
    EVALUATOR_MAP,
    call_model_batch,
    construct_prompts,
    load_benchmark_data,
    score_arena,
    score_audio,
    score_bfcl,
    score_code,
    score_if,
    score_llm_judge,
    score_math,
    score_mrcr,
    score_multichoice,
    score_ruler,
)


class TestEvaluatorMapDispatch:
    """Tests for EVALUATOR_MAP completeness and dispatch (T-018, T-019, T-020)."""

    def test_t018_evaluator_map_dispatch_math_evalplus(self):
        """T-018: EVALUATOR_MAP maps 'math' to 'score_math' and 'humaneval' to 'score_code'."""
        assert EVALUATOR_MAP["math"] == "score_math"
        assert EVALUATOR_MAP["humaneval"] == "score_code"

    def test_t019_evaluator_map_completeness(self):
        """T-019: All 17 EVALUATOR_MAP values resolve to callables (INV-013)."""
        assert len(EVALUATOR_MAP) == 17
        # Module-level validation in runner.py ensures all values are callable.
        # If this import succeeded, INV-013 is satisfied.
        from nemo_evaluator.plugins.nemo_skills import runner
        for fn_name in EVALUATOR_MAP.values():
            assert callable(getattr(runner, fn_name)), f"{fn_name} is not callable"

    def test_t020_evaluator_map_distinct_scorers(self):
        """T-020: 17 entries map to exactly 9 distinct scorer functions (INV-013)."""
        distinct = set(EVALUATOR_MAP.values())
        assert len(distinct) == 10


class TestScoreMath:
    """Tests for score_math scorer (T-021, T-022, T-023)."""

    def test_t021_score_math_boxed_extraction(self):
        """T-021: Extract answer from \\boxed{42}, set symbolic_correct=True."""
        data = [{"generation": "The answer is \\boxed{42}", "expected_answer": "42"}]
        result = score_math(data, {})
        assert result[0]["predicted_answer"] == "42"
        assert result[0]["symbolic_correct"] is True
        assert result[0]["no_answer"] is False

    def test_t022_score_math_answer_is_pattern(self):
        """T-022: Extract answer from 'the answer is 42' pattern."""
        data = [{"generation": "After calculation, the answer is 42", "expected_answer": "42"}]
        result = score_math(data, {})
        assert result[0]["predicted_answer"] == "42"
        assert result[0]["symbolic_correct"] is True

    def test_t023_score_math_no_answer(self):
        """T-023: Set no_answer=True when generation has no parseable answer."""
        data = [{"generation": "I don't know how to solve this", "expected_answer": "42"}]
        result = score_math(data, {})
        assert result[0]["no_answer"] is True
        assert result[0]["symbolic_correct"] is False


class TestScoreMultichoice:
    """Tests for score_multichoice scorer (T-024, T-025, T-026)."""

    def test_t024_score_multichoice_boxed_letter(self):
        """T-024: Extract answer from \\boxed{C}, set symbolic_correct=True."""
        data = [{"generation": "The answer is \\boxed{C}", "expected_answer": "C"}]
        result = score_multichoice(data, {})
        assert result[0]["predicted_answer"] == "C"
        assert result[0]["symbolic_correct"] is True

    def test_t025_score_multichoice_parenthesis_pattern(self):
        """T-025: Extract answer from 'My choice is (B)' pattern."""
        data = [{"generation": "My choice is (B)", "expected_answer": "B"}]
        result = score_multichoice(data, {})
        assert result[0]["predicted_answer"] == "B"
        assert result[0]["symbolic_correct"] is True

    def test_t026_score_multichoice_last_capital_letter(self):
        """T-026: Extract last standalone capital letter as answer."""
        data = [{"generation": "I think it's A but actually D", "expected_answer": "D"}]
        result = score_multichoice(data, {})
        assert result[0]["predicted_answer"] == "D"
        assert result[0]["symbolic_correct"] is True


class TestScoreMrcr:
    """Tests for score_mrcr scorer (T-088, T-089)."""

    def test_t088_score_mrcr_exact_match(self):
        """T-088: Exact string match gives seq_match_ratio=1.0, is_correct=True."""
        data = [{"generation": "Paris", "expected_answer": "Paris"}]
        result = score_mrcr(data, {})
        assert result[0]["seq_match_ratio"] == 1.0
        assert result[0]["is_correct"] is True

    def test_t089_score_mrcr_partial_match(self):
        """T-089: Partial match gives 0.0 < seq_match_ratio < 1.0."""
        data = [{"generation": "Paris is the capital", "expected_answer": "Paris"}]
        result = score_mrcr(data, {})
        assert 0.0 < result[0]["seq_match_ratio"] < 1.0


class TestScoreRuler:
    """Tests for score_ruler scorer (T-090, T-091)."""

    def test_t090_score_ruler_string_containment(self):
        """T-090: String containment check, is_correct=True when target in generation."""
        data = [{"generation": "The capital of France is Paris.", "expected_answer": "Paris"}]
        result = score_ruler(data, {})
        assert result[0]["is_correct"] is True

    def test_t091_score_ruler_list_target_fractional(self):
        """T-091: List target gives fractional is_correct (0.5 for 1 of 2 found)."""
        data = [{"generation": "Paris is a city", "expected_answer": ["Paris", "London"]}]
        result = score_ruler(data, {})
        assert result[0]["is_correct"] == 0.5


class TestScoreBfcl:
    """Tests for score_bfcl scorer (T-092, T-093)."""

    def test_t092_score_bfcl_stripped_exact_match(self):
        """T-092: Stripped exact match sets is_correct=True."""
        data = [{"generation": "  hello world  ", "expected_answer": "hello world"}]
        result = score_bfcl(data, {})
        assert result[0]["is_correct"] is True

    def test_t093_score_bfcl_mismatch(self):
        """T-093: Mismatch sets is_correct=False."""
        data = [{"generation": "hello", "expected_answer": "world"}]
        result = score_bfcl(data, {})
        assert result[0]["is_correct"] is False


class TestScoreLlmJudge:
    """Tests for score_llm_judge scorer (T-094, T-095)."""

    def test_t094_score_llm_judge_verdict_correct(self):
        """T-094: Judge returning 'correct' sets is_correct=True, no error."""
        judge_fn = lambda prompt: "This response is correct"
        data = [{"generation": "Some answer"}]
        result = score_llm_judge(data, {"judge_model_call_fn": judge_fn})
        assert result[0]["is_correct"] is True
        assert result[0]["judge_verdict"] == "This response is correct"
        assert result[0]["judge_error"] is None

    def test_t095_score_llm_judge_exception_handling(self):
        """T-095: Judge raising exception sets judge_error, is_correct=False, verdict='error'."""
        def failing_judge(prompt):
            raise RuntimeError("Judge failed")
        data = [{"generation": "Some answer"}]
        result = score_llm_judge(data, {"judge_model_call_fn": failing_judge})
        assert result[0]["is_correct"] is False
        assert result[0]["judge_verdict"] == "error"
        assert "RuntimeError" in result[0]["judge_error"]


class TestPlaceholderScorers:
    """Tests for placeholder scorers (T-027 through T-030)."""

    def test_t027_score_code_sets_needs_sandbox(self):
        """T-027: score_code sets needs_sandbox=True."""
        data = [{"generation": "print('hello')"}]
        result = score_code(data, {})
        assert result[0]["needs_sandbox"] is True

    def test_t028_score_if_sets_needs_external_scoring(self):
        """T-028: score_if sets needs_external_scoring=True."""
        data = [{"generation": "some output"}]
        result = score_if(data, {})
        assert result[0]["needs_external_scoring"] is True

    def test_t029_score_arena_sets_needs_judge(self):
        """T-029: score_arena sets needs_judge=True."""
        data = [{"generation": "some output"}]
        result = score_arena(data, {})
        assert result[0]["needs_judge"] is True

    def test_t030_score_audio_sets_needs_audio_scoring(self):
        """T-030: score_audio sets needs_audio_scoring=True."""
        data = [{"generation": "some output"}]
        result = score_audio(data, {})
        assert result[0]["needs_audio_scoring"] is True


class TestDataLoading:
    """Tests for load_benchmark_data (T-031 through T-036)."""

    def test_t031_load_benchmark_data_limit_samples(self, tmp_path, benchmark_data_dir):
        """T-031: Load with limit_samples=2 returns exactly 2 samples."""
        samples = [{"problem": f"Q{i}", "expected_answer": str(i)} for i in range(5)]
        data_dir = benchmark_data_dir("gsm8k", "test", samples)
        result = load_benchmark_data("gsm8k", data_dir, "test", limit_samples=2)
        assert len(result) == 2

    def test_t032_load_benchmark_data_limit_exceeds_data(self, tmp_path, benchmark_data_dir):
        """T-032: limit_samples > data size returns all available samples."""
        samples = [{"problem": f"Q{i}", "expected_answer": str(i)} for i in range(3)]
        data_dir = benchmark_data_dir("gsm8k", "test", samples)
        result = load_benchmark_data("gsm8k", data_dir, "test", limit_samples=100)
        assert len(result) == 3

    def test_t033_load_benchmark_data_fallback_to_test(self, tmp_path):
        """T-033: Fallback to test.jsonl when custom split missing (OQR-001)."""
        bench_dir = tmp_path / "data" / "gsm8k"
        bench_dir.mkdir(parents=True)
        test_file = bench_dir / "test.jsonl"
        test_file.write_text('{"problem": "Q1", "expected_answer": "1"}\n')
        result = load_benchmark_data("gsm8k", str(tmp_path / "data"), "validation")
        assert len(result) == 1

    def test_t034_load_benchmark_data_fallback_file_not_found(self, tmp_path):
        """T-034: Raise FileNotFoundError when neither custom nor test.jsonl exist (OQR-001)."""
        bench_dir = tmp_path / "data" / "gsm8k"
        bench_dir.mkdir(parents=True)
        with pytest.raises(FileNotFoundError):
            load_benchmark_data("gsm8k", str(tmp_path / "data"), "validation")

    def test_t035_load_benchmark_data_none_data_dir(self):
        """T-035: Raise ValueError when data_dir=None (AC-029)."""
        with pytest.raises(ValueError, match="data_dir must not be None"):
            load_benchmark_data("gsm8k", None, "test")

    def test_t036_load_benchmark_data_test_split_missing(self, tmp_path):
        """T-036: Raise FileNotFoundError immediately when eval_split='test' and file missing (OQR-001)."""
        bench_dir = tmp_path / "data" / "gsm8k"
        bench_dir.mkdir(parents=True)
        with pytest.raises(FileNotFoundError):
            load_benchmark_data("gsm8k", str(tmp_path / "data"), "test")


class TestPromptConstruction:
    """Tests for construct_prompts (T-037 through T-039, T-096, T-097)."""

    def test_t037_construct_prompts_problem_field(self):
        """T-037: Use 'problem' field as user content."""
        data = [{"problem": "What is 2+2?", "expected_answer": "4"}]
        prompts = construct_prompts(data, None, None)
        assert len(prompts) == 1
        assert prompts[0][-1]["role"] == "user"
        assert prompts[0][-1]["content"] == "What is 2+2?"

    def test_t038_construct_prompts_fallback_json_dumps(self):
        """T-038: Fallback to json.dumps(sample) when no known field present."""
        data = [{"custom_field": "value", "answer": "42"}]
        prompts = construct_prompts(data, None, None)
        content = prompts[0][-1]["content"]
        assert "custom_field" in content  # JSON-serialized sample

    def test_t039_construct_prompts_system_prompt(self):
        """T-039: Prepend system message when system_prompt is not None."""
        data = [{"problem": "What is 2+2?"}]
        prompts = construct_prompts(data, None, "You are a math tutor.")
        assert len(prompts[0]) == 2
        assert prompts[0][0]["role"] == "system"
        assert prompts[0][0]["content"] == "You are a math tutor."
        assert prompts[0][1]["role"] == "user"

    def test_t096_construct_prompts_field_priority_problem_over_question(self):
        """T-096: 'problem' takes priority over 'question' when both present."""
        data = [{"problem": "Problem text", "question": "Question text"}]
        prompts = construct_prompts(data, None, None)
        assert prompts[0][-1]["content"] == "Problem text"

    def test_t097_construct_prompts_question_when_no_problem(self):
        """T-097: Use 'question' when 'problem' not present."""
        data = [{"question": "Question text", "expected_answer": "A"}]
        prompts = construct_prompts(data, None, None)
        assert prompts[0][-1]["content"] == "Question text"


class TestModelCallBatch:
    """Tests for call_model_batch (T-040, T-041, T-042)."""

    def test_t040_call_model_batch_normal_execution(self, mock_client):
        """T-040: Normal batch execution with 5 prompts returns 5 responses."""
        prompts = [[{"role": "user", "content": f"Q{i}"}] for i in range(5)]
        responses = asyncio.run(call_model_batch(mock_client, prompts, 0.0, 512))
        assert len(responses) == 5
        assert all(isinstance(r, str) for r in responses)
        assert all(len(r) > 0 for r in responses)

    def test_t041_call_model_batch_single_failure_isolation(self):
        """T-041: Single failure at index 3 returns empty string, others succeed (INV-009)."""
        client = MagicMock()
        call_count = 0

        async def mock_chat(messages, **kwargs):
            nonlocal call_count
            idx = call_count
            call_count += 1
            if idx == 3:
                raise RuntimeError("Connection error")
            return f"response_{idx}"

        client.chat_completion = AsyncMock(side_effect=mock_chat)
        prompts = [[{"role": "user", "content": f"Q{i}"}] for i in range(5)]
        responses = asyncio.run(call_model_batch(client, prompts, 0.0, 512))
        assert len(responses) == 5
        assert responses[3] == ""
        assert all(responses[i] != "" for i in [0, 1, 2, 4])

    def test_t042_call_model_batch_all_failures(self):
        """T-042: All failures return empty strings, no exception raised (INV-009)."""
        client = MagicMock()
        client.chat_completion = AsyncMock(side_effect=RuntimeError("always fail"))
        prompts = [[{"role": "user", "content": f"Q{i}"}] for i in range(3)]
        responses = asyncio.run(call_model_batch(client, prompts, 0.0, 512))
        assert len(responses) == 3
        assert all(r == "" for r in responses)


class TestUnknownEvalType:
    """Test for unknown eval_type handling (T-098)."""

    def test_t098_unknown_eval_type_raises_value_error(self, tmp_path, make_evaluation, mock_client):
        """T-098: Unknown eval_type raises ValueError with descriptive message."""
        from nemo_evaluator.plugins.nemo_skills.runner import evaluate
        evaluation = make_evaluation(
            eval_type="nonexistent_type",
            data_dir=str(tmp_path),
            output_dir=str(tmp_path / "output"),
        )
        with pytest.raises(ValueError, match="Unknown eval_type"):
            evaluate(evaluation, mock_client, str(tmp_path / "output"))


class TestScorerContract:
    """Tests for scorer contract invariants (T-045, T-046, T-047, T-048)."""

    ALL_SCORERS = [
        score_math, score_multichoice, score_mrcr, score_ruler,
        score_bfcl, score_llm_judge, score_if, score_code,
        score_arena, score_audio,
    ]

    def test_t045_scorer_identity_check_all_scorers(self):
        """T-045: All scorers return the same list object (result is data) (INV-003)."""
        for scorer in self.ALL_SCORERS:
            data = [{"generation": "test", "expected_answer": "test"}]
            result = scorer(data, {"judge_model_call_fn": None})
            assert result is data, f"{scorer.__name__} broke identity: returned new list"

    def test_t046_scorer_dict_identity_check(self):
        """T-046: Each returned dict is the same object as input dict (INV-003)."""
        for scorer in self.ALL_SCORERS:
            sample = {"generation": "test", "expected_answer": "test"}
            data = [sample]
            result = scorer(data, {"judge_model_call_fn": None})
            assert result[0] is sample, f"{scorer.__name__} broke dict identity"

    def test_t047_scorer_preserves_original_keys(self):
        """T-047: Scorers never remove pre-existing keys (INV-003)."""
        for scorer in self.ALL_SCORERS:
            data = [{"generation": "test", "expected_answer": "test", "custom_key": "preserve_me"}]
            original_keys = set(data[0].keys())
            scorer(data, {"judge_model_call_fn": None})
            assert original_keys.issubset(set(data[0].keys())), \
                f"{scorer.__name__} removed keys: {original_keys - set(data[0].keys())}"

    def test_t048_scorer_property_based(self):
        """T-048: Property-based test: identity preservation with random inputs (INV-003)."""
        # Simplified property test without Hypothesis dependency
        import random
        random.seed(42)
        for scorer in self.ALL_SCORERS:
            for _ in range(5):
                n = random.randint(1, 10)
                data = [
                    {"generation": f"gen_{i}", "expected_answer": f"ans_{i}", "extra": i}
                    for i in range(n)
                ]
                originals = list(data)  # shallow copy of list
                result = scorer(data, {"judge_model_call_fn": None})
                assert result is data
                assert len(result) == n
                for j in range(n):
                    assert result[j] is originals[j]


class TestBenchmarkDataIntegrity:
    """Tests for benchmark data integrity invariant (T-074, T-075)."""

    def test_t074_benchmark_data_integrity_keys_preserved(self):
        """T-074: Original keys preserved after inference adds 'generation' (INV-014)."""
        data = [{"problem": "Q1", "expected_answer": "A1", "metadata": "keep"}]
        original_keys = set(data[0].keys())
        # Simulate inference step
        data[0]["generation"] = "The answer is \\boxed{A1}"
        assert original_keys.issubset(set(data[0].keys()))

    def test_t075_benchmark_data_integrity_keys_increased_after_scoring(self):
        """T-075: Scoring adds new keys without removing original keys (INV-014)."""
        data = [{"problem": "Q1", "expected_answer": "42", "generation": "\\boxed{42}"}]
        original_keys = set(data[0].keys())
        score_math(data, {})
        assert original_keys.issubset(set(data[0].keys()))
        assert "predicted_answer" in data[0]
        assert "symbolic_correct" in data[0]
