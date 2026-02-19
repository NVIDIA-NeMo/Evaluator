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

"""Unit tests for BYOB native runner (ByobNativeHarness)."""

import pytest
from unittest.mock import MagicMock, patch

from nemo_evaluator.api.api_dataclasses import (
    ApiEndpoint,
    Evaluation,
    EvaluationConfig,
    EvaluationResult,
    ConfigParams,
    EvaluationTarget,
)
from nemo_evaluator.byob.decorators import get_registered_benchmarks
from nemo_evaluator.byob.native_runner import ByobNativeHarness


class TestByobNativeHarness:
    """Tests for ByobNativeHarness.execute()."""

    @pytest.fixture
    def temp_benchmark_file(self, tmp_path):
        """Create a temporary benchmark .py file for testing."""
        code = '''
from nemo_evaluator.byob import benchmark, scorer

@benchmark(name="test-native", dataset="unused", prompt="Q: {question}\\nA:", target_field="answer")
@scorer
def simple_scorer(response, target, metadata):
    return {"correct": target.lower() in response.lower()}
'''
        benchmark_file = tmp_path / "test_benchmark.py"
        benchmark_file.write_text(code)
        return benchmark_file

    @pytest.fixture
    def temp_dataset_file(self, tmp_path):
        """Create a temporary JSONL dataset for testing."""
        lines = [
            '{"question": "Is the sky blue?", "answer": "yes"}',
            '{"question": "Is water dry?", "answer": "no"}',
            '{"question": "Do cats meow?", "answer": "yes"}',
        ]
        dataset_file = tmp_path / "test_data.jsonl"
        dataset_file.write_text('\n'.join(lines))
        return dataset_file

    @pytest.fixture
    def mock_evaluation(self, temp_benchmark_file, temp_dataset_file):
        """Create a mock Evaluation object for native mode."""
        return Evaluation(
            name="test_native",
            config=EvaluationConfig(
                type="byob_test.test-native",
                params=ConfigParams(
                    limit_samples=None,
                    max_new_tokens=4096,
                    temperature=0.0,
                    extra={
                        "benchmark_module": str(temp_benchmark_file),
                        "benchmark_name": "test-native",
                        "dataset": str(temp_dataset_file),
                    },
                ),
            ),
            target=EvaluationTarget(
                api_endpoint=ApiEndpoint(type="chat"),
            ),
        )

    @pytest.fixture
    def mock_model_call_fn(self):
        """Mock model call function that returns 'Yes' for all prompts."""
        return MagicMock(return_value="Yes")

    def test_native_run_basic_evaluation(
        self, mock_evaluation, mock_model_call_fn
    ):
        """Test that native harness executes the full evaluation pipeline.

        Validates:
        - Benchmark is imported and loaded
        - Dataset is loaded and limited correctly
        - Model call function is invoked
        - Scorer is executed
        - Results are aggregated correctly
        """
        harness = ByobNativeHarness()
        result = harness.execute(mock_evaluation, mock_model_call_fn)

        assert isinstance(result, EvaluationResult), \
            f"Expected EvaluationResult, got {type(result)}"
        assert "test-native" in result.tasks, \
            f"Expected 'test-native' in tasks, got {list(result.tasks.keys())}"

        task = result.tasks["test-native"]
        assert "pass@1" in task.metrics, \
            f"Expected 'pass@1' in metrics, got {list(task.metrics.keys())}"

        scores = task.metrics["pass@1"].scores
        assert "correct" in scores, \
            f"Expected 'correct' in scores, got {list(scores.keys())}"

        # All 3 samples should be scored (model returns "Yes" which contains "yes" and "no")
        assert scores["correct"].stats.count == 3, \
            f"Expected count=3, got {scores['correct'].stats.count}"

        # Verify model was called 3 times (once per sample)
        assert mock_model_call_fn.call_count == 3, \
            f"Expected 3 model calls, got {mock_model_call_fn.call_count}"

    def test_native_result_structure_matches_contract(
        self, mock_evaluation, mock_model_call_fn
    ):
        """Test that native harness output matches EvaluationResult schema exactly.

        Contract validation:
        - tasks: Dict[str, TaskResult]
        - TaskResult has metrics: Dict[str, MetricResult]
        - MetricResult has scores: Dict[str, Score]
        - Score has value: float, stats: ScoreStats
        - ScoreStats has count, mean, stderr, stddev
        """
        harness = ByobNativeHarness()
        result = harness.execute(mock_evaluation, mock_model_call_fn)

        # Validate top-level structure
        assert isinstance(result, EvaluationResult), \
            "Result must be EvaluationResult instance"
        assert hasattr(result, "tasks"), "Missing 'tasks' attribute"

        # Validate task structure
        for task_name, task_data in result.tasks.items():
            assert hasattr(task_data, "metrics"), \
                f"Task '{task_name}' missing 'metrics' attribute"

            # Validate metric structure
            for metric_name, metric_data in task_data.metrics.items():
                assert hasattr(metric_data, "scores"), \
                    f"Metric '{metric_name}' missing 'scores' attribute"

                # Validate score structure
                for score_name, score_data in metric_data.scores.items():
                    assert hasattr(score_data, "value"), \
                        f"Score '{score_name}' missing 'value' attribute"
                    assert hasattr(score_data, "stats"), \
                        f"Score '{score_name}' missing 'stats' attribute"
                    assert isinstance(score_data.value, (int, float)), \
                        f"Score value must be numeric, got {type(score_data.value)}"

                    # Validate stats structure
                    stats = score_data.stats
                    assert hasattr(stats, "count"), \
                        f"Stats for '{score_name}' missing 'count'"
                    assert hasattr(stats, "mean"), \
                        f"Stats for '{score_name}' missing 'mean'"
                    assert hasattr(stats, "stderr"), \
                        f"Stats for '{score_name}' missing 'stderr'"
                    assert hasattr(stats, "stddev"), \
                        f"Stats for '{score_name}' missing 'stddev'"
                    assert stats.count > 0, \
                        f"Score count must be > 0, got {stats.count}"

    def test_native_runner_limit_samples(
        self, temp_benchmark_file, temp_dataset_file, mock_model_call_fn
    ):
        """Test that limit_samples parameter is respected.

        With limit_samples=2, only first 2 samples should be processed.
        """
        evaluation = Evaluation(
            name="test_limit",
            config=EvaluationConfig(
                type="byob_test.test-limit",
                params=ConfigParams(
                    limit_samples=2,  # Limit to 2 samples
                    max_new_tokens=4096,
                    temperature=0.0,
                    extra={
                        "benchmark_module": str(temp_benchmark_file),
                        "benchmark_name": "test-native",
                        "dataset": str(temp_dataset_file),
                    },
                ),
            ),
            target=EvaluationTarget(api_endpoint=ApiEndpoint(type="chat")),
        )

        harness = ByobNativeHarness()
        result = harness.execute(evaluation, mock_model_call_fn)

        scores = result.tasks["test-native"].metrics["pass@1"].scores
        assert scores["correct"].stats.count == 2, \
            f"Expected count=2 with limit_samples=2, got {scores['correct'].stats.count}"

        # Verify model was called exactly 2 times
        assert mock_model_call_fn.call_count == 2, \
            f"Expected 2 model calls with limit_samples=2, got {mock_model_call_fn.call_count}"

    def test_native_runner_missing_prompt_field(
        self, temp_benchmark_file, tmp_path, mock_model_call_fn
    ):
        """Test that samples missing prompt fields are skipped, not crashed.

        Sample 2 has wrong field name -> should be skipped.
        Only samples 1 and 3 should be scored.
        """
        # Dataset with middle sample missing 'question' field
        lines = [
            '{"question": "q1", "answer": "yes"}',
            '{"WRONG_FIELD": "q2", "answer": "no"}',  # Missing 'question'
            '{"question": "q3", "answer": "yes"}',
        ]
        dataset_file = tmp_path / "bad_data.jsonl"
        dataset_file.write_text('\n'.join(lines))

        evaluation = Evaluation(
            name="test_skip",
            config=EvaluationConfig(
                type="byob_test.test-skip",
                params=ConfigParams(
                    limit_samples=None,
                    max_new_tokens=4096,
                    temperature=0.0,
                    extra={
                        "benchmark_module": str(temp_benchmark_file),
                        "benchmark_name": "test-native",
                        "dataset": str(dataset_file),
                    },
                ),
            ),
            target=EvaluationTarget(api_endpoint=ApiEndpoint(type="chat")),
        )

        harness = ByobNativeHarness()
        result = harness.execute(evaluation, mock_model_call_fn)

        scores = result.tasks["test-native"].metrics["pass@1"].scores
        assert scores["correct"].stats.count == 2, \
            f"Expected count=2 (sample with missing field skipped), got {scores['correct'].stats.count}"

    def test_native_runner_model_error_handled(
        self, mock_evaluation
    ):
        """Test that model call errors cause sample to be skipped, not crash.

        Model succeeds for sample 0, fails for sample 1, succeeds for sample 2.
        Only 2 samples should be scored.
        """
        mock_model_call_fn = MagicMock(side_effect=[
            "Yes",  # sample 0 succeeds
            Exception("500 Server Error"),  # sample 1 fails
            "No",  # sample 2 succeeds
        ])

        harness = ByobNativeHarness()
        result = harness.execute(mock_evaluation, mock_model_call_fn)

        scores = result.tasks["test-native"].metrics["pass@1"].scores
        assert scores["correct"].stats.count == 2, \
            f"Expected count=2 (HTTP error should skip sample), got {scores['correct'].stats.count}"

    def test_native_runner_empty_dataset(
        self, temp_benchmark_file, tmp_path, mock_model_call_fn
    ):
        """Test that empty dataset produces empty result without calling model."""
        # Empty JSONL file
        empty_dataset = tmp_path / "empty.jsonl"
        empty_dataset.write_text("")

        evaluation = Evaluation(
            name="test_empty",
            config=EvaluationConfig(
                type="byob_test.test-empty",
                params=ConfigParams(
                    limit_samples=None,
                    max_new_tokens=4096,
                    temperature=0.0,
                    extra={
                        "benchmark_module": str(temp_benchmark_file),
                        "benchmark_name": "test-native",
                        "dataset": str(empty_dataset),
                    },
                ),
            ),
            target=EvaluationTarget(api_endpoint=ApiEndpoint(type="chat")),
        )

        harness = ByobNativeHarness()
        result = harness.execute(evaluation, mock_model_call_fn)

        assert len(result.tasks) == 0, \
            f"Expected empty tasks for empty dataset, got {list(result.tasks.keys())}"
        assert mock_model_call_fn.call_count == 0, \
            f"Expected 0 model calls for empty dataset, got {mock_model_call_fn.call_count}"

    def test_native_runner_clears_registry(
        self, mock_evaluation, mock_model_call_fn
    ):
        """Test that native harness clears registry after execution.

        Registry pollution is a critical bug - must verify cleanup happens.
        """
        # Pre-condition: registry should be empty (autouse fixture)
        assert len(get_registered_benchmarks()) == 0, \
            "Registry should be empty before test"

        harness = ByobNativeHarness()
        result = harness.execute(mock_evaluation, mock_model_call_fn)

        # Post-condition: registry should be empty after execution
        assert len(get_registered_benchmarks()) == 0, \
            "Native harness must clear registry after execution to prevent engine pollution"

    def test_native_runner_cleanup_on_error(
        self, temp_benchmark_file, temp_dataset_file
    ):
        """Test that registry is cleaned up even when native runner encounters error.

        This validates the finally block in ByobNativeHarness.execute().
        """
        # Create mock that always raises
        mock_model_call_fn = MagicMock(side_effect=Exception("Connection refused"))

        evaluation = Evaluation(
            name="test_error",
            config=EvaluationConfig(
                type="byob_test.test-error",
                params=ConfigParams(
                    limit_samples=None,
                    max_new_tokens=4096,
                    temperature=0.0,
                    extra={
                        "benchmark_module": str(temp_benchmark_file),
                        "benchmark_name": "test-native",
                        "dataset": str(temp_dataset_file),
                    },
                ),
            ),
            target=EvaluationTarget(api_endpoint=ApiEndpoint(type="chat")),
        )

        harness = ByobNativeHarness()

        # Runner should handle gracefully (all samples skipped on error)
        # What matters is registry cleanup happens
        try:
            result = harness.execute(evaluation, mock_model_call_fn)
            # If no exception, result should have no scores (all samples skipped)
            # This is acceptable behavior
        except Exception:
            # If exception propagates, that's also acceptable
            pass

        # The critical assertion: registry must be clean even after error
        assert len(get_registered_benchmarks()) == 0, \
            "Registry must be clean even after error (finally block must execute)"

    def test_native_runner_missing_config_fields(self, mock_model_call_fn):
        """Test that missing required config fields raise clear error."""
        # Evaluation missing benchmark_module
        evaluation = Evaluation(
            name="test_missing",
            config=EvaluationConfig(
                type="byob_test.test-missing",
                params=ConfigParams(
                    limit_samples=None,
                    max_new_tokens=4096,
                    temperature=0.0,
                    extra={
                        "benchmark_name": "test-native",
                        "dataset": "/path/to/data.jsonl",
                        # Missing benchmark_module
                    },
                ),
            ),
            target=EvaluationTarget(api_endpoint=ApiEndpoint(type="chat")),
        )

        harness = ByobNativeHarness()

        with pytest.raises(ValueError, match="benchmark_module"):
            harness.execute(evaluation, mock_model_call_fn)

    def test_native_runner_chat_endpoint(
        self, mock_evaluation, mock_model_call_fn
    ):
        """Test that chat endpoint type is passed correctly to model_call_fn.

        model_call_fn receives (prompt, endpoint_type).
        For chat endpoints, endpoint_type should be "chat".
        """
        harness = ByobNativeHarness()
        result = harness.execute(mock_evaluation, mock_model_call_fn)

        # Verify all calls used "chat" endpoint type
        for call in mock_model_call_fn.call_args_list:
            args, kwargs = call
            # Second positional arg is endpoint_type
            assert len(args) == 2, \
                f"Expected 2 args (prompt, endpoint_type), got {len(args)}"
            assert args[1] == "chat", \
                f"Expected endpoint_type='chat', got '{args[1]}'"

    def test_native_runner_completions_endpoint(
        self, temp_benchmark_file, temp_dataset_file, mock_model_call_fn
    ):
        """Test that completions endpoint type is passed correctly to model_call_fn.

        For completions endpoints, endpoint_type should be "completions".
        """
        evaluation = Evaluation(
            name="test_completions",
            config=EvaluationConfig(
                type="byob_test.test-completions",
                params=ConfigParams(
                    limit_samples=None,
                    max_new_tokens=4096,
                    temperature=0.0,
                    extra={
                        "benchmark_module": str(temp_benchmark_file),
                        "benchmark_name": "test-native",
                        "dataset": str(temp_dataset_file),
                    },
                ),
            ),
            target=EvaluationTarget(
                api_endpoint=ApiEndpoint(type="completions"),
            ),
        )

        harness = ByobNativeHarness()
        result = harness.execute(evaluation, mock_model_call_fn)

        # Verify all calls used "completions" endpoint type
        for call in mock_model_call_fn.call_args_list:
            args, kwargs = call
            assert args[1] == "completions", \
                f"Expected endpoint_type='completions', got '{args[1]}'"

    def test_native_runner_float_limit_samples(
        self, temp_benchmark_file, temp_dataset_file, mock_model_call_fn
    ):
        """Test that float limit_samples is converted to int.

        Engine may provide limit_samples as float (e.g., from YAML parsing).
        """
        evaluation = Evaluation(
            name="test_float_limit",
            config=EvaluationConfig(
                type="byob_test.test-float-limit",
                params=ConfigParams(
                    limit_samples=2.0,  # Float instead of int
                    max_new_tokens=4096,
                    temperature=0.0,
                    extra={
                        "benchmark_module": str(temp_benchmark_file),
                        "benchmark_name": "test-native",
                        "dataset": str(temp_dataset_file),
                    },
                ),
            ),
            target=EvaluationTarget(api_endpoint=ApiEndpoint(type="chat")),
        )

        harness = ByobNativeHarness()
        result = harness.execute(evaluation, mock_model_call_fn)

        scores = result.tasks["test-native"].metrics["pass@1"].scores
        assert scores["correct"].stats.count == 2, \
            f"Expected count=2 with limit_samples=2.0, got {scores['correct'].stats.count}"
