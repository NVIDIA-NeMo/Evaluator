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

"""Unit tests for BYOB parallel evaluation."""

import time

import pytest
from unittest.mock import MagicMock

from nemo_evaluator.byob.decorators import BenchmarkDefinition, ScorerInput
from nemo_evaluator.byob.eval_logic import SampleResult, run_eval_loop


class TestParallelism:
    """Tests for parallel evaluation via ThreadPoolExecutor."""

    @pytest.fixture
    def simple_benchmark(self):
        def scorer_fn(sample: ScorerInput):
            return {"correct": sample.target.lower() in sample.response.lower()}

        return BenchmarkDefinition(
            name="test-parallel",
            normalized_name="test_parallel",
            dataset="unused",
            prompt="Q: {question}\nA:",
            target_field="answer",
            scorer_fn=scorer_fn,
        )

    @pytest.fixture
    def dataset(self):
        return [
            {"question": f"Question {i}", "answer": "yes"}
            for i in range(10)
        ]

    @pytest.fixture
    def mock_model_call_fn(self):
        return MagicMock(return_value="Yes, that is correct.")

    def test_sequential_default(self, simple_benchmark, dataset, mock_model_call_fn):
        """Test that parallelism=1 produces identical results to default behavior."""
        scores_seq, _ = run_eval_loop(
            bench=simple_benchmark,
            dataset=dataset,
            model_call_fn=mock_model_call_fn,
            endpoint_type="chat",
            parallelism=1,
        )

        scores_default, _ = run_eval_loop(
            bench=simple_benchmark,
            dataset=dataset,
            model_call_fn=mock_model_call_fn,
            endpoint_type="chat",
        )

        assert len(scores_seq) == len(scores_default)
        for s_seq, s_def in zip(scores_seq, scores_default):
            assert s_seq == s_def

    def test_parallel_all_samples_scored(
        self, simple_benchmark, dataset, mock_model_call_fn
    ):
        """Test that parallelism=4 scores all samples."""
        scores, _ = run_eval_loop(
            bench=simple_benchmark,
            dataset=dataset,
            model_call_fn=mock_model_call_fn,
            endpoint_type="chat",
            parallelism=4,
        )

        assert len(scores) == 10, \
            f"Expected 10 scores, got {len(scores)}"
        assert mock_model_call_fn.call_count == 10

    def test_parallel_results_match_sequential(
        self, simple_benchmark, dataset, mock_model_call_fn
    ):
        """Test that parallel and sequential produce same scores."""
        scores_seq, _ = run_eval_loop(
            bench=simple_benchmark,
            dataset=dataset,
            model_call_fn=mock_model_call_fn,
            endpoint_type="chat",
            parallelism=1,
        )

        scores_par, _ = run_eval_loop(
            bench=simple_benchmark,
            dataset=dataset,
            model_call_fn=mock_model_call_fn,
            endpoint_type="chat",
            parallelism=4,
        )

        assert len(scores_seq) == len(scores_par)
        for s1, s2 in zip(scores_seq, scores_par):
            assert s1 == s2

    def test_parallel_ordering(self, simple_benchmark):
        """Test that results maintain sample order regardless of completion order."""
        # Model with variable response times to cause out-of-order completion
        responses = [f"Answer_{i}" for i in range(5)]
        call_idx = {"n": 0}

        def model_fn(prompt, endpoint_type):
            idx = call_idx["n"]
            call_idx["n"] += 1
            return responses[min(idx, len(responses) - 1)]

        scorer_calls = []

        def ordering_scorer(sample: ScorerInput):
            scorer_calls.append(sample.response)
            return {"idx": float(sample.response.split("_")[-1]) if "_" in sample.response else 0.0}

        bench = BenchmarkDefinition(
            name="test-order",
            normalized_name="test_order",
            dataset="unused",
            prompt="Q: {question}",
            target_field="answer",
            scorer_fn=ordering_scorer,
        )

        dataset = [{"question": f"Q{i}", "answer": "a"} for i in range(5)]

        scores, predictions = run_eval_loop(
            bench=bench,
            dataset=dataset,
            model_call_fn=model_fn,
            endpoint_type="chat",
            parallelism=4,
            save_predictions=True,
        )

        # All samples should be scored
        assert len(scores) == 5
        assert len(predictions) == 5

        # Predictions should be in sample_id order
        for i, pred in enumerate(predictions):
            assert pred.sample_id == i, \
                f"Expected prediction {i} to have sample_id={i}, got {pred.sample_id}"

    def test_parallel_errors_dont_corrupt_others(self, simple_benchmark):
        """Test that model errors in parallel don't corrupt other samples."""
        call_count = {"n": 0}

        def flaky_model(prompt, endpoint_type):
            idx = call_count["n"]
            call_count["n"] += 1
            if idx % 3 == 1:  # Fail every 3rd sample starting at index 1
                raise RuntimeError("Transient error")
            return "Yes, correct."

        dataset = [
            {"question": f"Q{i}", "answer": "yes"}
            for i in range(9)
        ]

        scores, predictions = run_eval_loop(
            bench=simple_benchmark,
            dataset=dataset,
            model_call_fn=flaky_model,
            endpoint_type="chat",
            parallelism=3,
            save_predictions=True,
        )

        # 3 out of 9 should fail (indices 1, 4, 7)
        assert len(scores) == 6, \
            f"Expected 6 scores (3 failed), got {len(scores)}"
        assert len(predictions) == 9, \
            f"Expected 9 predictions (incl. failures), got {len(predictions)}"

    def test_parallel_fail_on_skip(self, simple_benchmark):
        """Test that fail_on_skip raises RuntimeError in parallel mode."""
        def failing_model(prompt, endpoint_type):
            raise RuntimeError("Model down")

        dataset = [
            {"question": "Q1", "answer": "yes"},
            {"question": "Q2", "answer": "no"},
        ]

        with pytest.raises(RuntimeError):
            run_eval_loop(
                bench=simple_benchmark,
                dataset=dataset,
                model_call_fn=failing_model,
                endpoint_type="chat",
                parallelism=2,
                fail_on_skip=True,
            )

    def test_parallel_max_consecutive_errors(self, simple_benchmark):
        """Test that max_consecutive_errors is tracked in parallel mode."""
        def always_failing_model(prompt, endpoint_type):
            raise RuntimeError("Always fails")

        dataset = [
            {"question": f"Q{i}", "answer": "yes"}
            for i in range(10)
        ]

        with pytest.raises(RuntimeError, match="consecutive errors"):
            run_eval_loop(
                bench=simple_benchmark,
                dataset=dataset,
                model_call_fn=always_failing_model,
                endpoint_type="chat",
                parallelism=4,
                max_consecutive_errors=3,
            )

    def test_parallel_save_predictions(
        self, simple_benchmark, dataset, mock_model_call_fn
    ):
        """Test that save_predictions works correctly in parallel mode."""
        scores, predictions = run_eval_loop(
            bench=simple_benchmark,
            dataset=dataset,
            model_call_fn=mock_model_call_fn,
            endpoint_type="chat",
            parallelism=4,
            save_predictions=True,
        )

        assert len(predictions) == 10
        for pred in predictions:
            assert isinstance(pred, SampleResult)
            assert pred.status == "scored"
            assert pred.scores is not None

    def test_parallel_empty_dataset(
        self, simple_benchmark, mock_model_call_fn
    ):
        """Test that empty dataset works with parallelism > 1."""
        scores, predictions = run_eval_loop(
            bench=simple_benchmark,
            dataset=[],
            model_call_fn=mock_model_call_fn,
            endpoint_type="chat",
            parallelism=4,
        )

        assert scores == []
        assert predictions == []

    def test_parallel_single_sample_uses_sequential(
        self, simple_benchmark, mock_model_call_fn
    ):
        """Test that a single-sample dataset uses sequential path even with parallelism > 1."""
        dataset = [{"question": "Q1", "answer": "yes"}]
        scores, _ = run_eval_loop(
            bench=simple_benchmark,
            dataset=dataset,
            model_call_fn=mock_model_call_fn,
            endpoint_type="chat",
            parallelism=4,
        )

        assert len(scores) == 1
