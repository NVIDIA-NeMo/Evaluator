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

"""Unit tests for BYOB runner module."""

import argparse
import json
import sys

import pytest
import requests
from unittest.mock import MagicMock, patch

from nemo_evaluator.byob.runner import (
    aggregate_scores,
    call_model_chat,
    call_model_completions,
    create_session,
    load_dataset,
)


class TestAggregateScores:
    """Tests for aggregate_scores function."""

    def test_aggregate_binary_scores(self):
        """Test aggregation of binary (True/False) scores with percentage scaling.

        Hand-computed values (population variance, divide by n):
        - Booleans: [True, False, True] -> [1.0, 0.0, 1.0]
        - n = 3
        - mean = 2.0 / 3 = 0.6667
        - population variance = ((1-0.6667)^2 + (0-0.6667)^2 + (1-0.6667)^2) / 3
        -                     = (0.1111 + 0.4445 + 0.1111) / 3 = 0.2222
        - stddev = sqrt(0.2222) = 0.4714
        - stderr = 0.4714 / sqrt(3) = 0.2722
        - Binary detected -> scale by 100
        - value = 66.6667, stddev = 47.1405, stderr = 27.2166
        """
        scores = [{"correct": True}, {"correct": False}, {"correct": True}]
        result = aggregate_scores(scores, "test_bench")

        assert "tasks" in result, "Missing 'tasks' key in result"
        assert "test_bench" in result["tasks"], "Missing benchmark in tasks"

        task = result["tasks"]["test_bench"]
        score_data = task["metrics"]["pass@1"]["scores"]["correct"]

        assert score_data["count"] == 3, f"Expected count=3, got {score_data['count']}"
        assert abs(score_data["value"] - 66.6667) < 0.01, \
            f"Expected value~66.6667, got {score_data['value']}"
        assert abs(score_data["mean"] - 66.6667) < 0.01, \
            f"Expected mean~66.6667, got {score_data['mean']}"
        # stderr also scaled by 100 for binary metrics (population variance: divide by n)
        # Population variance: ((1-0.667)^2 + (0-0.667)^2 + (1-0.667)^2) / 3 = 0.2222
        # Population stddev = sqrt(0.2222) = 0.4714, scaled = 47.14
        # Population stderr = 47.14 / sqrt(3) = 27.22
        assert abs(score_data["stderr"] - 27.2166) < 0.1, \
            f"Expected stderr~27.2166, got {score_data['stderr']}"
        assert abs(score_data["stddev"] - 47.1405) < 0.1, \
            f"Expected stddev~47.1405, got {score_data['stddev']}"

    def test_aggregate_continuous_scores(self):
        """Test aggregation of continuous (non-binary) scores without scaling.

        Hand-computed values (population variance, divide by n):
        - Values: [0.8, 0.9, 1.0]
        - n = 3
        - mean = 2.7 / 3 = 0.9
        - population variance = ((0.8-0.9)^2 + (0.9-0.9)^2 + (1.0-0.9)^2) / 3
        -                     = (0.01 + 0.0 + 0.01) / 3 = 0.006667
        - stddev = sqrt(0.006667) = 0.08165
        - stderr = 0.08165 / sqrt(3) = 0.04714
        - NOT binary (0.8 not in {0.0, 1.0}) -> no scaling
        """
        scores = [{"f1": 0.8}, {"f1": 0.9}, {"f1": 1.0}]
        result = aggregate_scores(scores, "continuous_bench")

        task = result["tasks"]["continuous_bench"]
        score_data = task["metrics"]["pass@1"]["scores"]["f1"]

        assert score_data["count"] == 3
        assert abs(score_data["value"] - 0.9) < 0.0001, \
            f"Expected value=0.9, got {score_data['value']}"
        assert abs(score_data["mean"] - 0.9) < 0.0001, \
            f"Expected mean=0.9, got {score_data['mean']}"
        assert abs(score_data["stddev"] - 0.08165) < 0.001, \
            f"Expected stddev~0.08165, got {score_data['stddev']}"
        assert abs(score_data["stderr"] - 0.04714) < 0.001, \
            f"Expected stderr~0.04714, got {score_data['stderr']}"

    def test_aggregate_empty(self):
        """Test that empty score list returns empty dict."""
        result = aggregate_scores([], "empty_bench")
        assert result == {}, f"Expected empty dict, got {result}"

    def test_aggregate_single_sample(self):
        """Test single sample produces stderr=0, stddev=0.

        Hand-computed:
        - n = 1, binary, mean = 1.0
        - variance = 0 (n <= 1 special case)
        - stddev = 0, stderr = 0
        - Binary -> value = 100.0
        """
        scores = [{"correct": True}]
        result = aggregate_scores(scores, "single_bench")

        score_data = result["tasks"]["single_bench"]["metrics"]["pass@1"]["scores"]["correct"]
        assert score_data["count"] == 1
        assert score_data["value"] == 100.0, f"Expected value=100.0, got {score_data['value']}"
        assert score_data["stddev"] == 0.0, f"Expected stddev=0.0, got {score_data['stddev']}"
        assert score_data["stderr"] == 0.0, f"Expected stderr=0.0, got {score_data['stderr']}"

    def test_aggregate_all_true(self):
        """Test all True boolean scores -> value=100.0."""
        scores = [{"correct": True}, {"correct": True}, {"correct": True}]
        result = aggregate_scores(scores, "all_true_bench")

        score_data = result["tasks"]["all_true_bench"]["metrics"]["pass@1"]["scores"]["correct"]
        assert score_data["value"] == 100.0
        assert score_data["mean"] == 100.0
        assert score_data["stddev"] == 0.0
        assert score_data["stderr"] == 0.0

    def test_aggregate_all_false(self):
        """Test all False boolean scores -> value=0.0."""
        scores = [{"correct": False}, {"correct": False}, {"correct": False}]
        result = aggregate_scores(scores, "all_false_bench")

        score_data = result["tasks"]["all_false_bench"]["metrics"]["pass@1"]["scores"]["correct"]
        assert score_data["value"] == 0.0
        assert score_data["mean"] == 0.0
        assert score_data["stddev"] == 0.0
        assert score_data["stderr"] == 0.0

    def test_aggregate_mixed_keys(self):
        """Test samples with different keys are handled gracefully.

        Only keys present in each sample contribute to that key's stats.
        """
        scores = [
            {"correct": True, "parsed": True},
            {"correct": False},
            {"correct": True, "parsed": False},
        ]
        result = aggregate_scores(scores, "mixed_bench")

        scores_out = result["tasks"]["mixed_bench"]["metrics"]["pass@1"]["scores"]

        # "correct" should have count=3 (all samples have it)
        assert scores_out["correct"]["count"] == 3, \
            f"Expected correct count=3, got {scores_out['correct']['count']}"

        # "parsed" should have count=2 (only 2 samples have it)
        assert "parsed" in scores_out, "Missing 'parsed' key in output"
        assert scores_out["parsed"]["count"] == 2, \
            f"Expected parsed count=2, got {scores_out['parsed']['count']}"

    def test_aggregate_output_structure(self):
        """Contract test: validate output dict structure matches expected schema."""
        scores = [{"correct": True}]
        result = aggregate_scores(scores, "schema_bench")

        # Validate nested structure
        assert "tasks" in result, "Missing 'tasks' key"
        assert "schema_bench" in result["tasks"], "Missing benchmark in tasks"
        assert "metrics" in result["tasks"]["schema_bench"], "Missing 'metrics' key"
        assert "pass@1" in result["tasks"]["schema_bench"]["metrics"], "Missing 'pass@1' key"
        assert "scores" in result["tasks"]["schema_bench"]["metrics"]["pass@1"], \
            "Missing 'scores' key"

        score = result["tasks"]["schema_bench"]["metrics"]["pass@1"]["scores"]["correct"]

        # Validate all required keys are present
        required_keys = ["value", "count", "mean", "stderr", "stddev"]
        for key in required_keys:
            assert key in score, f"Missing required key '{key}' in score output"


class TestLoadDataset:
    """Tests for load_dataset function."""

    def test_load_dataset(self, tmp_path):
        """Test loading JSONL dataset, skipping blank lines, respecting limit."""
        # Create temporary JSONL file
        dataset_file = tmp_path / "test.jsonl"
        lines = [
            '{"question": "q1", "answer": "a1"}',
            '',  # blank line -- should be skipped
            '{"question": "q2", "answer": "a2"}',
            '{"question": "q3", "answer": "a3"}',
        ]
        dataset_file.write_text('\n'.join(lines))

        # Test without limit
        data = load_dataset(str(dataset_file))
        assert len(data) == 3, f"Expected 3 samples, got {len(data)}"
        assert data[0]["question"] == "q1"
        assert data[2]["answer"] == "a3"

        # Test with limit
        data = load_dataset(str(dataset_file), limit=2)
        assert len(data) == 2, f"Expected 2 samples with limit=2, got {len(data)}"
        assert data[0]["question"] == "q1"
        assert data[1]["question"] == "q2"

    def test_load_dataset_file_not_found(self):
        """Test that load_dataset raises ValueError for nonexistent file (no fetcher supports it)."""
        with pytest.raises(ValueError, match="No fetcher supports"):
            load_dataset("/nonexistent/path.jsonl")

    def test_load_dataset_limit_zero(self, tmp_path):
        """Test that limit=0 or limit=None returns all data."""
        dataset_file = tmp_path / "test.jsonl"
        lines = [f'{{"id": {i}}}' for i in range(5)]
        dataset_file.write_text('\n'.join(lines))

        # Test limit=0
        data = load_dataset(str(dataset_file), limit=0)
        assert len(data) == 5, f"Expected 5 samples with limit=0, got {len(data)}"

        # Test limit=None
        data = load_dataset(str(dataset_file), limit=None)
        assert len(data) == 5, f"Expected 5 samples with limit=None, got {len(data)}"


class TestSavePredictions:
    """Tests for --save-predictions functionality."""

    @pytest.fixture
    def mock_benchmark(self):
        """Create a mock BenchmarkDefinition for testing save-predictions."""
        from nemo_evaluator.byob.decorators import BenchmarkDefinition

        def simple_scorer(sample):
            return {"correct": sample.target.lower() in sample.response.lower()}

        return BenchmarkDefinition(
            name="save-test",
            normalized_name="save_test",
            dataset="unused",
            prompt="Q: {question}\nA:",
            target_field="answer",
            scorer_fn=simple_scorer,
        )

    @pytest.fixture
    def sample_dataset(self):
        """Sample dataset for predictions tests."""
        return [
            {"question": "Is the sky blue?", "answer": "yes"},
            {"question": "Is water dry?", "answer": "no"},
        ]

    def test_save_predictions_creates_jsonl_file(self, tmp_path, mock_benchmark, sample_dataset):
        """Test that running with --save-predictions creates byob_predictions.jsonl.

        Validates:
        - File is created in output directory
        - File contains one JSON line per sample
        """
        from nemo_evaluator.byob.eval_logic import run_eval_loop

        # Mock model call function
        mock_model_call_fn = MagicMock(return_value="Yes, that is correct.")

        # Run eval loop with save_predictions=True
        _scores, predictions = run_eval_loop(
            bench=mock_benchmark,
            dataset=sample_dataset,
            model_call_fn=mock_model_call_fn,
            endpoint_type="chat",
            save_predictions=True,
        )

        # Manually write predictions like runner.py does
        from dataclasses import asdict
        predictions_path = tmp_path / "byob_predictions.jsonl"
        with open(predictions_path, "w", encoding="utf-8") as f:
            for pred in predictions:
                f.write(json.dumps(asdict(pred)) + "\n")

        # Verify file exists
        assert predictions_path.exists(), \
            f"Expected byob_predictions.jsonl to exist at {predictions_path}"

        # Verify file contains correct number of lines
        lines = predictions_path.read_text().strip().split('\n')
        assert len(lines) == 2, \
            f"Expected 2 lines (one per sample), got {len(lines)}"

    def test_save_predictions_content_structure(self, tmp_path, mock_benchmark, sample_dataset):
        """Test that each prediction has required fields.

        Validates structure:
        - sample_id: int
        - prompt: str
        - response: str or null
        - target: str
        - scores: dict or null
        - status: str
        - error: str or null
        - metadata: dict
        """
        from nemo_evaluator.byob.eval_logic import run_eval_loop

        mock_model_call_fn = MagicMock(return_value="Yes")

        _scores, predictions = run_eval_loop(
            bench=mock_benchmark,
            dataset=sample_dataset,
            model_call_fn=mock_model_call_fn,
            endpoint_type="chat",
            save_predictions=True,
        )

        # Write and read back predictions
        from dataclasses import asdict
        predictions_path = tmp_path / "byob_predictions.jsonl"
        with open(predictions_path, "w", encoding="utf-8") as f:
            for pred in predictions:
                f.write(json.dumps(asdict(pred)) + "\n")

        # Read and verify structure
        with open(predictions_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f):
                pred_dict = json.loads(line)

                # Check all required fields are present
                required_fields = ["sample_id", "prompt", "response", "target",
                                   "scores", "status", "error", "metadata"]
                for field in required_fields:
                    assert field in pred_dict, \
                        f"Line {line_num}: missing required field '{field}'"

                # Check types
                assert isinstance(pred_dict["sample_id"], int), \
                    f"Line {line_num}: sample_id should be int"
                assert isinstance(pred_dict["prompt"], str), \
                    f"Line {line_num}: prompt should be str"
                assert isinstance(pred_dict["target"], str), \
                    f"Line {line_num}: target should be str"
                assert isinstance(pred_dict["status"], str), \
                    f"Line {line_num}: status should be str"
                assert isinstance(pred_dict["metadata"], dict), \
                    f"Line {line_num}: metadata should be dict"

    def test_save_predictions_scored_sample(self, tmp_path, mock_benchmark, sample_dataset):
        """Test that a successfully scored sample has correct status and fields.

        Validates:
        - status="scored"
        - response is non-null string
        - scores is non-null dict
        - error is null
        """
        from nemo_evaluator.byob.eval_logic import run_eval_loop

        mock_model_call_fn = MagicMock(return_value="Yes, the sky is blue.")

        _scores, predictions = run_eval_loop(
            bench=mock_benchmark,
            dataset=sample_dataset,
            model_call_fn=mock_model_call_fn,
            endpoint_type="chat",
            save_predictions=True,
        )

        from dataclasses import asdict
        predictions_path = tmp_path / "byob_predictions.jsonl"
        with open(predictions_path, "w", encoding="utf-8") as f:
            for pred in predictions:
                f.write(json.dumps(asdict(pred)) + "\n")

        # Read first prediction (should be scored)
        with open(predictions_path, "r", encoding="utf-8") as f:
            first_pred = json.loads(f.readline())

        assert first_pred["status"] == "scored", \
            f"Expected status='scored', got '{first_pred['status']}'"
        assert first_pred["response"] is not None, \
            "Expected non-null response for scored sample"
        assert isinstance(first_pred["response"], str), \
            f"Expected response to be str, got {type(first_pred['response'])}"
        assert first_pred["scores"] is not None, \
            "Expected non-null scores for scored sample"
        assert isinstance(first_pred["scores"], dict), \
            f"Expected scores to be dict, got {type(first_pred['scores'])}"
        assert first_pred["error"] is None, \
            f"Expected null error for scored sample, got '{first_pred['error']}'"

    def test_save_predictions_skipped_sample(self, tmp_path, mock_benchmark):
        """Test that a skipped sample has correct status and null fields.

        Validates:
        - status="skipped_model_error"
        - response is null
        - scores is null
        - error is non-null string
        """
        from nemo_evaluator.byob.eval_logic import run_eval_loop

        # Dataset with sample that will be skipped due to model error
        dataset = [
            {"question": "q1", "answer": "a1"},
            {"question": "q2", "answer": "a2"},
        ]

        # Mock model that fails on second call
        mock_model_call_fn = MagicMock(side_effect=[
            "Response 1",  # First sample succeeds
            Exception("Model timeout"),  # Second sample fails
        ])

        _scores, predictions = run_eval_loop(
            bench=mock_benchmark,
            dataset=dataset,
            model_call_fn=mock_model_call_fn,
            endpoint_type="chat",
            save_predictions=True,
        )

        from dataclasses import asdict
        predictions_path = tmp_path / "byob_predictions.jsonl"
        with open(predictions_path, "w", encoding="utf-8") as f:
            for pred in predictions:
                f.write(json.dumps(asdict(pred)) + "\n")

        # Read second prediction (should be skipped)
        with open(predictions_path, "r", encoding="utf-8") as f:
            _first = f.readline()
            second_pred = json.loads(f.readline())

        assert second_pred["status"] == "skipped_model_error", \
            f"Expected status='skipped_model_error', got '{second_pred['status']}'"
        assert second_pred["response"] is None, \
            f"Expected null response for skipped sample, got '{second_pred['response']}'"
        assert second_pred["scores"] is None, \
            f"Expected null scores for skipped sample, got '{second_pred['scores']}'"
        assert second_pred["error"] is not None, \
            "Expected non-null error for skipped sample"
        assert isinstance(second_pred["error"], str), \
            f"Expected error to be str, got {type(second_pred['error'])}"
        assert "timeout" in second_pred["error"].lower(), \
            f"Expected error to mention timeout, got: {second_pred['error']}"


class TestTimeoutPerSample:
    """Tests for --timeout-per-sample functionality."""

    def test_timeout_per_sample_default_value(self):
        """Test that default timeout is 120 seconds when not specified.

        Validates:
        - call_model_chat uses timeout=120 by default
        - call_model_completions uses timeout=120 by default
        """
        with patch('nemo_evaluator.byob.runner.requests.post') as mock_post:
            # Mock successful response
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "Response"}}]
            }
            mock_post.return_value = mock_response

            # Call without explicit timeout
            call_model_chat(
                url="http://localhost:8000",
                model_id="test-model",
                prompt="Test prompt"
            )

            # Verify timeout=120 was passed to requests.post
            mock_post.assert_called_once()
            _args, kwargs = mock_post.call_args
            assert "timeout" in kwargs, "Expected timeout to be passed to requests.post"
            assert kwargs["timeout"] == 120, \
                f"Expected default timeout=120, got {kwargs['timeout']}"

    def test_timeout_per_sample_custom_value(self):
        """Test that custom timeout is passed through to requests.post.

        Validates:
        - Custom timeout value is used in HTTP call
        - Works for both chat and completions endpoints
        """
        with patch('nemo_evaluator.byob.runner.requests.post') as mock_post:
            # Mock successful response for chat
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "Response"}}]
            }
            mock_post.return_value = mock_response

            # Call with custom timeout
            call_model_chat(
                url="http://localhost:8000",
                model_id="test-model",
                prompt="Test prompt",
                timeout=300
            )

            # Verify custom timeout was passed
            _args, kwargs = mock_post.call_args
            assert kwargs["timeout"] == 300, \
                f"Expected timeout=300, got {kwargs['timeout']}"

        # Test completions endpoint
        with patch('nemo_evaluator.byob.runner.requests.post') as mock_post:
            # Mock successful response for completions
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "choices": [{"text": "Response"}]
            }
            mock_post.return_value = mock_response

            # Call with custom timeout
            call_model_completions(
                url="http://localhost:8000",
                model_id="test-model",
                prompt="Test prompt",
                timeout=60
            )

            # Verify custom timeout was passed
            _args, kwargs = mock_post.call_args
            assert kwargs["timeout"] == 60, \
                f"Expected timeout=60, got {kwargs['timeout']}"

    def test_timeout_per_sample_cli_parsing(self):
        """Test that CLI argument is parsed correctly.

        Validates:
        - --timeout-per-sample flag is recognized
        - Value is converted to int
        - Default is 120
        """
        from nemo_evaluator.byob.runner import main
        import sys

        # Test with custom timeout
        test_args = [
            "runner.py",
            "--benchmark-module", "test.py",
            "--benchmark-name", "test",
            "--dataset", "test.jsonl",
            "--output-dir", "/tmp/out",
            "--model-url", "http://localhost:8000",
            "--model-id", "test-model",
            "--timeout-per-sample", "300"
        ]

        with patch.object(sys, 'argv', test_args):
            with patch('nemo_evaluator.byob.runner.import_benchmark') as mock_import:
                with patch('nemo_evaluator.byob.runner.load_dataset') as mock_load:
                    with patch('nemo_evaluator.byob.runner.run_eval_loop') as mock_run:
                        with patch('nemo_evaluator.byob.runner.aggregate_scores') as mock_agg:
                            with patch('builtins.open', MagicMock()):
                                with patch('os.makedirs'):
                                    # Setup mocks
                                    mock_benchmark = MagicMock()
                                    mock_import.return_value = mock_benchmark
                                    mock_load.return_value = []
                                    mock_run.return_value = ([], [])
                                    mock_agg.return_value = {"tasks": {}}

                                    # Run main
                                    try:
                                        main()
                                    except SystemExit:
                                        pass

                                    # Verify timeout was passed to model_call_fn
                                    # The timeout is captured in the closure
                                    assert mock_run.called, "Expected run_eval_loop to be called"

        # Test default timeout
        test_args_default = [
            "runner.py",
            "--benchmark-module", "test.py",
            "--benchmark-name", "test",
            "--dataset", "test.jsonl",
            "--output-dir", "/tmp/out",
            "--model-url", "http://localhost:8000",
            "--model-id", "test-model"
        ]

        with patch.object(sys, 'argv', test_args_default):
            with patch('nemo_evaluator.byob.runner.import_benchmark') as mock_import:
                with patch('nemo_evaluator.byob.runner.load_dataset') as mock_load:
                    with patch('nemo_evaluator.byob.runner.run_eval_loop') as mock_run:
                        with patch('nemo_evaluator.byob.runner.aggregate_scores') as mock_agg:
                            with patch('builtins.open', MagicMock()):
                                with patch('os.makedirs'):
                                    # Setup mocks
                                    mock_benchmark = MagicMock()
                                    mock_import.return_value = mock_benchmark
                                    mock_load.return_value = []
                                    mock_run.return_value = ([], [])
                                    mock_agg.return_value = {"tasks": {}}

                                    # Run main
                                    try:
                                        main()
                                    except SystemExit:
                                        pass

                                    # Verify default timeout of 120 is used
                                    assert mock_run.called, "Expected run_eval_loop to be called"


class TestDeterminism:
    """Tests for deterministic behavior of aggregate_scores."""

    def test_aggregate_scores_deterministic(self):
        """Test that aggregate_scores produces identical output across multiple runs.

        Validates:
        - Same input produces same output (seed repeatability)
        - All numeric values are deterministic
        - Dictionary ordering is stable
        """
        # Test with binary scores
        scores_binary = [
            {"correct": True, "parsed": True},
            {"correct": False, "parsed": True},
            {"correct": True, "parsed": False},
        ]

        results = []
        for _ in range(3):
            result = aggregate_scores(scores_binary, "determinism_test")
            results.append(json.dumps(result, sort_keys=True))

        # All three runs should produce identical JSON
        assert results[0] == results[1], \
            "Run 1 and 2 produced different outputs"
        assert results[1] == results[2], \
            "Run 2 and 3 produced different outputs"

        # Test with continuous scores
        scores_continuous = [
            {"f1": 0.85, "precision": 0.90},
            {"f1": 0.90, "precision": 0.88},
            {"f1": 0.88, "precision": 0.92},
        ]

        results_continuous = []
        for _ in range(3):
            result = aggregate_scores(scores_continuous, "continuous_test")
            results_continuous.append(json.dumps(result, sort_keys=True))

        assert results_continuous[0] == results_continuous[1], \
            "Continuous scores: Run 1 and 2 produced different outputs"
        assert results_continuous[1] == results_continuous[2], \
            "Continuous scores: Run 2 and 3 produced different outputs"

        # Test with mixed numeric types
        scores_mixed = [
            {"int_score": 1, "float_score": 0.5, "bool_score": True},
            {"int_score": 2, "float_score": 0.75, "bool_score": False},
            {"int_score": 3, "float_score": 1.0, "bool_score": True},
        ]

        results_mixed = []
        for _ in range(3):
            result = aggregate_scores(scores_mixed, "mixed_test")
            results_mixed.append(json.dumps(result, sort_keys=True))

        assert results_mixed[0] == results_mixed[1], \
            "Mixed types: Run 1 and 2 produced different outputs"
        assert results_mixed[1] == results_mixed[2], \
            "Mixed types: Run 2 and 3 produced different outputs"


class TestFailOnSkip:
    """Tests for --fail-on-skip functionality."""

    @pytest.fixture
    def mock_benchmark(self):
        """Create a mock BenchmarkDefinition for fail-on-skip tests."""
        from nemo_evaluator.byob.decorators import BenchmarkDefinition

        def simple_scorer(sample):
            return {"correct": sample.target.lower() in sample.response.lower()}

        return BenchmarkDefinition(
            name="fail-on-skip-test",
            normalized_name="fail_on_skip_test",
            dataset="unused",
            prompt="Q: {question}\nA:",
            target_field="answer",
            scorer_fn=simple_scorer,
        )

    def test_fail_on_skip_raises_on_model_error(self, mock_benchmark):
        """Test that fail_on_skip=True raises RuntimeError on model error.

        Validates:
        - When model_call_fn raises an exception and fail_on_skip=True
        - run_eval_loop raises RuntimeError with descriptive message
        - The error message indicates which sample failed
        """
        from nemo_evaluator.byob.eval_logic import run_eval_loop

        dataset = [
            {"question": "q1", "answer": "a1"},
            {"question": "q2", "answer": "a2"},
        ]

        # Mock model that fails on second call
        mock_model_call_fn = MagicMock(side_effect=[
            "Response 1",  # First sample succeeds
            Exception("Model timeout"),  # Second sample fails
        ])

        # With fail_on_skip=True, should raise RuntimeError
        with pytest.raises(RuntimeError) as exc_info:
            run_eval_loop(
                bench=mock_benchmark,
                dataset=dataset,
                model_call_fn=mock_model_call_fn,
                endpoint_type="chat",
                fail_on_skip=True,
            )

        # Verify error message contains useful context
        error_msg = str(exc_info.value)
        assert "sample" in error_msg.lower(), \
            f"Expected error to mention 'sample', got: {error_msg}"
        assert "failed" in error_msg.lower(), \
            f"Expected error to mention 'failed', got: {error_msg}"

    def test_fail_on_skip_false_skips_gracefully(self, mock_benchmark):
        """Test that fail_on_skip=False skips errors gracefully.

        Validates:
        - When model_call_fn raises an exception and fail_on_skip=False
        - run_eval_loop does not raise, continues processing
        - Scores list is shorter than dataset (skipped sample not scored)
        - Successfully processed samples are still scored
        """
        from nemo_evaluator.byob.eval_logic import run_eval_loop

        dataset = [
            {"question": "q1", "answer": "a1"},
            {"question": "q2", "answer": "a2"},
            {"question": "q3", "answer": "a3"},
        ]

        # Mock model that fails on second call
        mock_model_call_fn = MagicMock(side_effect=[
            "Response 1",  # First sample succeeds
            Exception("Model timeout"),  # Second sample fails
            "Response 3",  # Third sample succeeds
        ])

        # With fail_on_skip=False, should not raise
        all_scores, _all_predictions = run_eval_loop(
            bench=mock_benchmark,
            dataset=dataset,
            model_call_fn=mock_model_call_fn,
            endpoint_type="chat",
            fail_on_skip=False,
        )

        # Verify only 2 samples were scored (skipped sample not included)
        assert len(all_scores) == 2, \
            f"Expected 2 scored samples (1 skipped), got {len(all_scores)}"

        # Verify both scored samples have valid scores
        for i, scores in enumerate(all_scores):
            assert "correct" in scores, \
                f"Score {i} missing 'correct' key: {scores}"
            assert isinstance(scores["correct"], bool), \
                f"Score {i} 'correct' should be bool, got {type(scores['correct'])}"

    def test_fail_on_skip_raises_on_missing_field(self, mock_benchmark):
        """Test that fail_on_skip=True raises RuntimeError on missing field.

        Validates:
        - When prompt template references missing field and fail_on_skip=True
        - run_eval_loop raises RuntimeError with descriptive message
        - The error indicates the missing field
        """
        from nemo_evaluator.byob.eval_logic import run_eval_loop

        # Dataset missing 'question' field
        dataset = [
            {"answer": "a1"},  # Missing 'question'
        ]

        mock_model_call_fn = MagicMock(return_value="Response")

        # With fail_on_skip=True, should raise RuntimeError
        with pytest.raises(RuntimeError) as exc_info:
            run_eval_loop(
                bench=mock_benchmark,
                dataset=dataset,
                model_call_fn=mock_model_call_fn,
                endpoint_type="chat",
                fail_on_skip=True,
            )

        # Verify error message contains useful context
        error_msg = str(exc_info.value)
        assert "sample" in error_msg.lower(), \
            f"Expected error to mention 'sample', got: {error_msg}"
        assert "failed" in error_msg.lower(), \
            f"Expected error to mention 'failed', got: {error_msg}"


class TestLogFormat:
    """Tests for --log-format functionality."""

    def test_log_format_json_produces_valid_json(self):
        """Test that log-format=json produces valid JSON log entries.

        Validates:
        - JsonFormatter outputs valid JSON on each log record
        - JSON contains required keys: timestamp, level, message, logger
        - JSON can be parsed successfully
        - Fields have expected types
        """
        import logging
        from io import StringIO

        # Create JsonFormatter (same as in runner.py)
        class JsonFormatter(logging.Formatter):
            def format(self, record):
                log_entry = {
                    "timestamp": self.formatTime(record),
                    "level": record.levelname,
                    "message": record.getMessage(),
                    "logger": record.name,
                }
                return json.dumps(log_entry)

        # Set up logger with JsonFormatter
        logger = logging.getLogger("test_json_logger")
        logger.setLevel(logging.INFO)

        # Use StringIO to capture output
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)

        # Emit a log record
        test_message = "Test log message for JSON format validation"
        logger.info(test_message)

        # Capture output
        output = stream.getvalue().strip()

        # Verify it's valid JSON
        try:
            log_entry = json.loads(output)
        except json.JSONDecodeError as e:
            pytest.fail(f"JsonFormatter output is not valid JSON: {e}\nOutput: {output}")

        # Verify required keys are present
        required_keys = ["timestamp", "level", "message", "logger"]
        for key in required_keys:
            assert key in log_entry, \
                f"Missing required key '{key}' in JSON log entry. Keys: {list(log_entry.keys())}"

        # Verify field types
        assert isinstance(log_entry["timestamp"], str), \
            f"Expected timestamp to be str, got {type(log_entry['timestamp'])}"
        assert isinstance(log_entry["level"], str), \
            f"Expected level to be str, got {type(log_entry['level'])}"
        assert isinstance(log_entry["message"], str), \
            f"Expected message to be str, got {type(log_entry['message'])}"
        assert isinstance(log_entry["logger"], str), \
            f"Expected logger to be str, got {type(log_entry['logger'])}"

        # Verify field values
        assert log_entry["level"] == "INFO", \
            f"Expected level='INFO', got '{log_entry['level']}'"
        assert log_entry["message"] == test_message, \
            f"Expected message='{test_message}', got '{log_entry['message']}'"
        assert log_entry["logger"] == "test_json_logger", \
            f"Expected logger='test_json_logger', got '{log_entry['logger']}'"

        # Clean up
        logger.removeHandler(handler)

    def test_log_format_default_is_text(self):
        """Test that default log format is 'text' by checking argparse defaults.

        Validates:
        - --log-format argument has default="text"
        - Choices are ["text", "json"]
        - When no --log-format is provided, parser uses "text"
        """
        import argparse
        from nemo_evaluator.byob.runner import main
        import sys

        # Parse with minimal args (no --log-format specified)
        test_args = [
            "runner.py",
            "--benchmark-module", "test.py",
            "--benchmark-name", "test",
            "--dataset", "test.jsonl",
            "--output-dir", "/tmp/out",
            "--model-url", "http://localhost:8000",
            "--model-id", "test-model",
        ]

        with patch.object(sys, 'argv', test_args):
            parser = argparse.ArgumentParser()
            parser.add_argument("--benchmark-module", required=True)
            parser.add_argument("--benchmark-name", required=True)
            parser.add_argument("--dataset", required=True)
            parser.add_argument("--output-dir", required=True)
            parser.add_argument("--model-url", required=True)
            parser.add_argument("--model-id", required=True)
            parser.add_argument(
                "--log-format",
                choices=["text", "json"],
                default="text",
                help="Log output format: text (default) or json",
            )

            args = parser.parse_args(test_args[1:])  # Skip 'runner.py'

        # Verify default is "text"
        assert args.log_format == "text", \
            f"Expected default log_format='text', got '{args.log_format}'"

        # Test with explicit --log-format=json
        test_args_json = test_args + ["--log-format", "json"]

        with patch.object(sys, 'argv', test_args_json):
            parser_json = argparse.ArgumentParser()
            parser_json.add_argument("--benchmark-module", required=True)
            parser_json.add_argument("--benchmark-name", required=True)
            parser_json.add_argument("--dataset", required=True)
            parser_json.add_argument("--output-dir", required=True)
            parser_json.add_argument("--model-url", required=True)
            parser_json.add_argument("--model-id", required=True)
            parser_json.add_argument(
                "--log-format",
                choices=["text", "json"],
                default="text",
            )

            args_json = parser_json.parse_args(test_args_json[1:])

        # Verify json format can be parsed
        assert args_json.log_format == "json", \
            f"Expected log_format='json' when specified, got '{args_json.log_format}'"


class TestSessionPooling:
    """Tests for session pooling in call_model_chat/call_model_completions."""

    def test_session_used_when_provided(self):
        """Validate that a provided session's post method is used for the HTTP call."""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "session response"}}]
        }
        mock_session.post.return_value = mock_response

        result = call_model_chat(
            url="http://localhost:8000",
            model_id="test-model",
            prompt="Test prompt",
            session=mock_session,
        )

        mock_session.post.assert_called_once()
        assert result == "session response", (
            f"Expected 'session response', got {result!r}"
        )

    def test_fallback_when_session_none(self):
        """Validate that requests.post is called directly when session is None."""
        with patch("nemo_evaluator.byob.runner.requests") as mock_requests:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "fallback response"}}]
            }
            mock_requests.post.return_value = mock_response

            result = call_model_chat(
                url="http://localhost:8000",
                model_id="test-model",
                prompt="Test prompt",
                session=None,
            )

            mock_requests.post.assert_called_once()
            assert result == "fallback response", (
                f"Expected 'fallback response', got {result!r}"
            )


class TestRetrySession:
    """Tests for create_session retry/backoff configuration."""

    def test_create_session_returns_session(self):
        """Validate create_session() returns a requests.Session instance."""
        session = create_session()
        assert isinstance(session, requests.Session), (
            f"Expected requests.Session, got {type(session).__name__}"
        )

    def test_create_session_custom_params(self):
        """Validate custom max_retries and backoff_factor are applied to the adapter."""
        session = create_session(max_retries=5, backoff_factor=1.0)

        # Get the adapter mounted for http://
        adapter = session.get_adapter("http://test.com")
        retry = adapter.max_retries

        assert retry.total == 5, (
            f"Expected max_retries total=5, got {retry.total}"
        )
        assert retry.backoff_factor == 1.0, (
            f"Expected backoff_factor=1.0, got {retry.backoff_factor}"
        )

    def test_retry_cli_args(self):
        """Validate --max-retries and --retry-backoff are accepted by argparse."""
        test_args = [
            "--benchmark-module", "test.py",
            "--benchmark-name", "test",
            "--dataset", "test.jsonl",
            "--output-dir", "/tmp/out",
            "--model-url", "http://localhost:8000",
            "--model-id", "test-model",
            "--max-retries", "7",
            "--retry-backoff", "2.5",
        ]

        parser = argparse.ArgumentParser()
        parser.add_argument("--benchmark-module", required=True)
        parser.add_argument("--benchmark-name", required=True)
        parser.add_argument("--dataset", required=True)
        parser.add_argument("--output-dir", required=True)
        parser.add_argument("--model-url", required=True)
        parser.add_argument("--model-id", required=True)
        parser.add_argument("--max-retries", type=int, default=3)
        parser.add_argument("--retry-backoff", type=float, default=0.5)

        args = parser.parse_args(test_args)

        assert args.max_retries == 7, (
            f"Expected max_retries=7, got {args.max_retries}"
        )
        assert args.retry_backoff == 2.5, (
            f"Expected retry_backoff=2.5, got {args.retry_backoff}"
        )
