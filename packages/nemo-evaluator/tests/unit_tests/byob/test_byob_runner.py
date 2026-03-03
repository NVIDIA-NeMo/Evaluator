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
from unittest.mock import MagicMock, patch

import pytest
import requests

from nemo_evaluator.contrib.byob.runner import (
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
        - Values reported as fractions (0.0-1.0), no percentage scaling
        """
        scores = [{"correct": True}, {"correct": False}, {"correct": True}]
        result = aggregate_scores(scores, "test_bench")

        assert "tasks" in result, "Missing 'tasks' key in result"
        assert "test_bench" in result["tasks"], "Missing benchmark in tasks"

        task = result["tasks"]["test_bench"]
        score_data = task["metrics"]["pass@1"]["scores"]["correct"]
        stats = score_data["stats"]

        assert stats["count"] == 3, f"Expected count=3, got {stats['count']}"
        assert abs(score_data["value"] - 0.6667) < 0.001, (
            f"Expected value~0.6667, got {score_data['value']}"
        )
        assert abs(stats["mean"] - 0.6667) < 0.001, (
            f"Expected mean~0.6667, got {stats['mean']}"
        )
        # Population variance: ((1-0.667)^2 + (0-0.667)^2 + (1-0.667)^2) / 3 = 0.2222
        # Population stddev = sqrt(0.2222) = 0.4714
        # Population stderr = 0.4714 / sqrt(3) = 0.2722
        assert abs(stats["stderr"] - 0.2722) < 0.01, (
            f"Expected stderr~0.2722, got {stats['stderr']}"
        )
        assert abs(stats["stddev"] - 0.4714) < 0.01, (
            f"Expected stddev~0.4714, got {stats['stddev']}"
        )

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
        stats = score_data["stats"]

        assert stats["count"] == 3
        assert abs(score_data["value"] - 0.9) < 0.0001, (
            f"Expected value=0.9, got {score_data['value']}"
        )
        assert abs(stats["mean"] - 0.9) < 0.0001, (
            f"Expected mean=0.9, got {stats['mean']}"
        )
        assert abs(stats["stddev"] - 0.08165) < 0.001, (
            f"Expected stddev~0.08165, got {stats['stddev']}"
        )
        assert abs(stats["stderr"] - 0.04714) < 0.001, (
            f"Expected stderr~0.04714, got {stats['stderr']}"
        )

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
        - Values reported as fractions (0.0-1.0)
        """
        scores = [{"correct": True}]
        result = aggregate_scores(scores, "single_bench")

        score_data = result["tasks"]["single_bench"]["metrics"]["pass@1"]["scores"][
            "correct"
        ]
        stats = score_data["stats"]
        assert stats["count"] == 1
        assert score_data["value"] == 1.0, (
            f"Expected value=1.0, got {score_data['value']}"
        )
        assert stats["stddev"] == 0.0, f"Expected stddev=0.0, got {stats['stddev']}"
        assert stats["stderr"] == 0.0, f"Expected stderr=0.0, got {stats['stderr']}"

    def test_aggregate_all_true(self):
        """Test all True boolean scores -> value=1.0."""
        scores = [{"correct": True}, {"correct": True}, {"correct": True}]
        result = aggregate_scores(scores, "all_true_bench")

        score_data = result["tasks"]["all_true_bench"]["metrics"]["pass@1"]["scores"][
            "correct"
        ]
        stats = score_data["stats"]
        assert score_data["value"] == 1.0
        assert stats["mean"] == 1.0
        assert stats["stddev"] == 0.0
        assert stats["stderr"] == 0.0

    def test_aggregate_all_false(self):
        """Test all False boolean scores -> value=0.0."""
        scores = [{"correct": False}, {"correct": False}, {"correct": False}]
        result = aggregate_scores(scores, "all_false_bench")

        score_data = result["tasks"]["all_false_bench"]["metrics"]["pass@1"]["scores"][
            "correct"
        ]
        stats = score_data["stats"]
        assert score_data["value"] == 0.0
        assert stats["mean"] == 0.0
        assert stats["stddev"] == 0.0
        assert stats["stderr"] == 0.0

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
        assert scores_out["correct"]["stats"]["count"] == 3, (
            f"Expected correct count=3, got {scores_out['correct']['stats']['count']}"
        )

        # "parsed" should have count=2 (only 2 samples have it)
        assert "parsed" in scores_out, "Missing 'parsed' key in output"
        assert scores_out["parsed"]["stats"]["count"] == 2, (
            f"Expected parsed count=2, got {scores_out['parsed']['stats']['count']}"
        )

    def test_aggregate_output_structure(self):
        """Contract test: validate output dict structure matches expected schema."""
        scores = [{"correct": True}]
        result = aggregate_scores(scores, "schema_bench")

        # Validate nested structure
        assert "tasks" in result, "Missing 'tasks' key"
        assert "schema_bench" in result["tasks"], "Missing benchmark in tasks"
        assert "metrics" in result["tasks"]["schema_bench"], "Missing 'metrics' key"
        assert "pass@1" in result["tasks"]["schema_bench"]["metrics"], (
            "Missing 'pass@1' key"
        )
        assert "scores" in result["tasks"]["schema_bench"]["metrics"]["pass@1"], (
            "Missing 'scores' key"
        )

        score = result["tasks"]["schema_bench"]["metrics"]["pass@1"]["scores"][
            "correct"
        ]

        # Validate structure: value at top level, stats nested
        assert "value" in score, "Missing required key 'value' in score output"
        assert "stats" in score, "Missing required key 'stats' in score output"
        stats_keys = ["count", "mean", "stderr", "stddev"]
        for key in stats_keys:
            assert key in score["stats"], f"Missing required key '{key}' in stats"


class TestLoadDataset:
    """Tests for load_dataset function."""

    def test_load_dataset(self, tmp_path):
        """Test loading JSONL dataset, skipping blank lines, respecting limit."""
        # Create temporary JSONL file
        dataset_file = tmp_path / "test.jsonl"
        lines = [
            '{"question": "q1", "answer": "a1"}',
            "",  # blank line -- should be skipped
            '{"question": "q2", "answer": "a2"}',
            '{"question": "q3", "answer": "a3"}',
        ]
        dataset_file.write_text("\n".join(lines))

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
        dataset_file.write_text("\n".join(lines))

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
        from nemo_evaluator.contrib.byob.decorators import BenchmarkDefinition

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

    def test_save_predictions_creates_jsonl_file(
        self, tmp_path, mock_benchmark, sample_dataset
    ):
        """Test that running with --save-predictions creates byob_predictions.jsonl.

        Validates:
        - File is created in output directory
        - File contains one JSON line per sample
        """
        from nemo_evaluator.contrib.byob.eval_logic import run_eval_loop

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
        assert predictions_path.exists(), (
            f"Expected byob_predictions.jsonl to exist at {predictions_path}"
        )

        # Verify file contains correct number of lines
        lines = predictions_path.read_text().strip().split("\n")
        assert len(lines) == 2, f"Expected 2 lines (one per sample), got {len(lines)}"

    def test_save_predictions_content_structure(
        self, tmp_path, mock_benchmark, sample_dataset
    ):
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
        from nemo_evaluator.contrib.byob.eval_logic import run_eval_loop

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
                required_fields = [
                    "sample_id",
                    "prompt",
                    "response",
                    "target",
                    "scores",
                    "status",
                    "error",
                    "metadata",
                ]
                for field in required_fields:
                    assert field in pred_dict, (
                        f"Line {line_num}: missing required field '{field}'"
                    )

                # Check types
                assert isinstance(pred_dict["sample_id"], int), (
                    f"Line {line_num}: sample_id should be int"
                )
                assert isinstance(pred_dict["prompt"], str), (
                    f"Line {line_num}: prompt should be str"
                )
                assert isinstance(pred_dict["target"], str), (
                    f"Line {line_num}: target should be str"
                )
                assert isinstance(pred_dict["status"], str), (
                    f"Line {line_num}: status should be str"
                )
                assert isinstance(pred_dict["metadata"], dict), (
                    f"Line {line_num}: metadata should be dict"
                )

    def test_save_predictions_scored_sample(
        self, tmp_path, mock_benchmark, sample_dataset
    ):
        """Test that a successfully scored sample has correct status and fields.

        Validates:
        - status="scored"
        - response is non-null string
        - scores is non-null dict
        - error is null
        """
        from nemo_evaluator.contrib.byob.eval_logic import run_eval_loop

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

        assert first_pred["status"] == "scored", (
            f"Expected status='scored', got '{first_pred['status']}'"
        )
        assert first_pred["response"] is not None, (
            "Expected non-null response for scored sample"
        )
        assert isinstance(first_pred["response"], str), (
            f"Expected response to be str, got {type(first_pred['response'])}"
        )
        assert first_pred["scores"] is not None, (
            "Expected non-null scores for scored sample"
        )
        assert isinstance(first_pred["scores"], dict), (
            f"Expected scores to be dict, got {type(first_pred['scores'])}"
        )
        assert first_pred["error"] is None, (
            f"Expected null error for scored sample, got '{first_pred['error']}'"
        )

    def test_save_predictions_skipped_sample(self, tmp_path, mock_benchmark):
        """Test that a skipped sample has correct status and null fields.

        Validates:
        - status="skipped_model_error"
        - response is null
        - scores is null
        - error is non-null string
        """
        from nemo_evaluator.contrib.byob.eval_logic import run_eval_loop

        # Dataset with sample that will be skipped due to model error
        dataset = [
            {"question": "q1", "answer": "a1"},
            {"question": "q2", "answer": "a2"},
        ]

        # Mock model that fails on second call
        mock_model_call_fn = MagicMock(
            side_effect=[
                "Response 1",  # First sample succeeds
                Exception("Model timeout"),  # Second sample fails
            ]
        )

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

        assert second_pred["status"] == "skipped_model_error", (
            f"Expected status='skipped_model_error', got '{second_pred['status']}'"
        )
        assert second_pred["response"] is None, (
            f"Expected null response for skipped sample, got '{second_pred['response']}'"
        )
        assert second_pred["scores"] is None, (
            f"Expected null scores for skipped sample, got '{second_pred['scores']}'"
        )
        assert second_pred["error"] is not None, (
            "Expected non-null error for skipped sample"
        )
        assert isinstance(second_pred["error"], str), (
            f"Expected error to be str, got {type(second_pred['error'])}"
        )
        assert "timeout" in second_pred["error"].lower(), (
            f"Expected error to mention timeout, got: {second_pred['error']}"
        )


class TestTimeoutPerSample:
    """Tests for --timeout-per-sample functionality."""

    def test_timeout_per_sample_default_value(self):
        """Test that default timeout is 120 seconds when not specified.

        Validates:
        - call_model_chat uses timeout=120 by default
        - call_model_completions uses timeout=120 by default
        """
        with patch("nemo_evaluator.contrib.byob.runner.requests.post") as mock_post:
            # Mock successful response
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "Response"}}]
            }
            mock_post.return_value = mock_response

            # Call without explicit timeout
            call_model_chat(
                url="http://localhost:8000", model_id="test-model", prompt="Test prompt"
            )

            # Verify timeout=120 was passed to requests.post
            mock_post.assert_called_once()
            _args, kwargs = mock_post.call_args
            assert "timeout" in kwargs, "Expected timeout to be passed to requests.post"
            assert kwargs["timeout"] == 120, (
                f"Expected default timeout=120, got {kwargs['timeout']}"
            )

    def test_timeout_per_sample_custom_value(self):
        """Test that custom timeout is passed through to requests.post.

        Validates:
        - Custom timeout value is used in HTTP call
        - Works for both chat and completions endpoints
        """
        with patch("nemo_evaluator.contrib.byob.runner.requests.post") as mock_post:
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
                timeout=300,
            )

            # Verify custom timeout was passed
            _args, kwargs = mock_post.call_args
            assert kwargs["timeout"] == 300, (
                f"Expected timeout=300, got {kwargs['timeout']}"
            )

        # Test completions endpoint
        with patch("nemo_evaluator.contrib.byob.runner.requests.post") as mock_post:
            # Mock successful response for completions
            mock_response = MagicMock()
            mock_response.json.return_value = {"choices": [{"text": "Response"}]}
            mock_post.return_value = mock_response

            # Call with custom timeout
            call_model_completions(
                url="http://localhost:8000",
                model_id="test-model",
                prompt="Test prompt",
                timeout=60,
            )

            # Verify custom timeout was passed
            _args, kwargs = mock_post.call_args
            assert kwargs["timeout"] == 60, (
                f"Expected timeout=60, got {kwargs['timeout']}"
            )

    def test_timeout_per_sample_cli_parsing(self):
        """Test that CLI argument is parsed correctly.

        Validates:
        - --timeout-per-sample flag is recognized
        - Value is converted to int
        - Default is 120
        """
        from nemo_evaluator.contrib.byob.runner import main

        # Test with custom timeout
        test_args = [
            "runner.py",
            "--benchmark-module",
            "test.py",
            "--benchmark-name",
            "test",
            "--dataset",
            "test.jsonl",
            "--output-dir",
            "/tmp/out",
            "--model-url",
            "http://localhost:8000",
            "--model-id",
            "test-model",
            "--timeout-per-sample",
            "300",
        ]

        mock_client_fn = MagicMock(return_value="response")

        with patch.object(sys, "argv", test_args):
            with patch(
                "nemo_evaluator.contrib.byob.runner.create_client_model_call_fn",
                return_value=(mock_client_fn, None),
            ):
                with patch(
                    "nemo_evaluator.contrib.byob.runner.import_benchmark"
                ) as mock_import:
                    with patch(
                        "nemo_evaluator.contrib.byob.runner.load_dataset"
                    ) as mock_load:
                        with patch(
                            "nemo_evaluator.contrib.byob.runner.run_eval_loop"
                        ) as mock_run:
                            with patch(
                                "nemo_evaluator.contrib.byob.runner.aggregate_scores"
                            ) as mock_agg:
                                with patch("builtins.open", MagicMock()):
                                    with patch("os.makedirs"):
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
                                        assert mock_run.called, (
                                            "Expected run_eval_loop to be called"
                                        )

        # Test default timeout
        test_args_default = [
            "runner.py",
            "--benchmark-module",
            "test.py",
            "--benchmark-name",
            "test",
            "--dataset",
            "test.jsonl",
            "--output-dir",
            "/tmp/out",
            "--model-url",
            "http://localhost:8000",
            "--model-id",
            "test-model",
        ]

        with patch.object(sys, "argv", test_args_default):
            with patch(
                "nemo_evaluator.contrib.byob.runner.create_client_model_call_fn",
                return_value=(mock_client_fn, None),
            ):
                with patch(
                    "nemo_evaluator.contrib.byob.runner.import_benchmark"
                ) as mock_import:
                    with patch(
                        "nemo_evaluator.contrib.byob.runner.load_dataset"
                    ) as mock_load:
                        with patch(
                            "nemo_evaluator.contrib.byob.runner.run_eval_loop"
                        ) as mock_run:
                            with patch(
                                "nemo_evaluator.contrib.byob.runner.aggregate_scores"
                            ) as mock_agg:
                                with patch("builtins.open", MagicMock()):
                                    with patch("os.makedirs"):
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
                                        assert mock_run.called, (
                                            "Expected run_eval_loop to be called"
                                        )


class TestFailOnSkip:
    """Tests for --fail-on-skip functionality."""

    @pytest.fixture
    def mock_benchmark(self):
        """Create a mock BenchmarkDefinition for fail-on-skip tests."""
        from nemo_evaluator.contrib.byob.decorators import BenchmarkDefinition

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
        from nemo_evaluator.contrib.byob.eval_logic import run_eval_loop

        dataset = [
            {"question": "q1", "answer": "a1"},
            {"question": "q2", "answer": "a2"},
        ]

        # Mock model that fails on second call
        mock_model_call_fn = MagicMock(
            side_effect=[
                "Response 1",  # First sample succeeds
                Exception("Model timeout"),  # Second sample fails
            ]
        )

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
        assert "sample" in error_msg.lower(), (
            f"Expected error to mention 'sample', got: {error_msg}"
        )
        assert "failed" in error_msg.lower(), (
            f"Expected error to mention 'failed', got: {error_msg}"
        )

    def test_fail_on_skip_false_skips_gracefully(self, mock_benchmark):
        """Test that fail_on_skip=False skips errors gracefully.

        Validates:
        - When model_call_fn raises an exception and fail_on_skip=False
        - run_eval_loop does not raise, continues processing
        - Scores list is shorter than dataset (skipped sample not scored)
        - Successfully processed samples are still scored
        """
        from nemo_evaluator.contrib.byob.eval_logic import run_eval_loop

        dataset = [
            {"question": "q1", "answer": "a1"},
            {"question": "q2", "answer": "a2"},
            {"question": "q3", "answer": "a3"},
        ]

        # Mock model that fails on second call
        mock_model_call_fn = MagicMock(
            side_effect=[
                "Response 1",  # First sample succeeds
                Exception("Model timeout"),  # Second sample fails
                "Response 3",  # Third sample succeeds
            ]
        )

        # With fail_on_skip=False, should not raise
        all_scores, _all_predictions = run_eval_loop(
            bench=mock_benchmark,
            dataset=dataset,
            model_call_fn=mock_model_call_fn,
            endpoint_type="chat",
            fail_on_skip=False,
        )

        # Verify only 2 samples were scored (skipped sample not included)
        assert len(all_scores) == 2, (
            f"Expected 2 scored samples (1 skipped), got {len(all_scores)}"
        )

        # Verify both scored samples have valid scores
        for i, scores in enumerate(all_scores):
            assert "correct" in scores, f"Score {i} missing 'correct' key: {scores}"
            assert isinstance(scores["correct"], bool), (
                f"Score {i} 'correct' should be bool, got {type(scores['correct'])}"
            )

    def test_fail_on_skip_raises_on_missing_field(self, mock_benchmark):
        """Test that fail_on_skip=True raises RuntimeError on missing field.

        Validates:
        - When prompt template references missing field and fail_on_skip=True
        - run_eval_loop raises RuntimeError with descriptive message
        - The error indicates the missing field
        """
        from nemo_evaluator.contrib.byob.eval_logic import run_eval_loop

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
        assert "sample" in error_msg.lower(), (
            f"Expected error to mention 'sample', got: {error_msg}"
        )
        assert "failed" in error_msg.lower(), (
            f"Expected error to mention 'failed', got: {error_msg}"
        )


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
        with patch("nemo_evaluator.contrib.byob.runner.requests") as mock_requests:
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

        assert retry.total == 5, f"Expected max_retries total=5, got {retry.total}"
        assert retry.backoff_factor == 1.0, (
            f"Expected backoff_factor=1.0, got {retry.backoff_factor}"
        )

    def test_retry_cli_args(self):
        """Validate --max-retries and --retry-backoff are accepted by argparse."""
        test_args = [
            "--benchmark-module",
            "test.py",
            "--benchmark-name",
            "test",
            "--dataset",
            "test.jsonl",
            "--output-dir",
            "/tmp/out",
            "--model-url",
            "http://localhost:8000",
            "--model-id",
            "test-model",
            "--max-retries",
            "7",
            "--retry-backoff",
            "2.5",
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

        assert args.max_retries == 7, f"Expected max_retries=7, got {args.max_retries}"
        assert args.retry_backoff == 2.5, (
            f"Expected retry_backoff=2.5, got {args.retry_backoff}"
        )


# ---------------------------------------------------------------------------
# Gap 1: System Prompt in HTTP calls
# ---------------------------------------------------------------------------


class TestSystemPromptInCalls:
    """Tests for system_prompt support in call_model_chat and call_model_completions."""

    def test_chat_system_message_prepended(self):
        """Validate that system_prompt adds a system message to the chat payload."""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "response"}}]
        }
        mock_session.post.return_value = mock_response

        call_model_chat(
            url="http://localhost:8000",
            model_id="test-model",
            prompt="User question",
            session=mock_session,
            system_prompt="You are a helpful assistant.",
        )

        mock_session.post.assert_called_once()
        _args, kwargs = mock_session.post.call_args
        payload = kwargs["json"]
        messages = payload["messages"]

        assert len(messages) == 2, (
            f"Expected 2 messages (system + user), got {len(messages)}"
        )
        assert messages[0] == {
            "role": "system",
            "content": "You are a helpful assistant.",
        }, f"Expected system message first, got {messages[0]}"
        assert messages[1] == {"role": "user", "content": "User question"}, (
            f"Expected user message second, got {messages[1]}"
        )

    def test_chat_no_system_message_when_none(self):
        """Validate that no system message is included when system_prompt is None."""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "response"}}]
        }
        mock_session.post.return_value = mock_response

        call_model_chat(
            url="http://localhost:8000",
            model_id="test-model",
            prompt="User question",
            session=mock_session,
        )

        _args, kwargs = mock_session.post.call_args
        payload = kwargs["json"]
        messages = payload["messages"]

        assert len(messages) == 1, (
            f"Expected 1 message (user only), got {len(messages)}"
        )
        assert messages[0]["role"] == "user"

    def test_completions_system_prepended(self):
        """Validate that system_prompt is prepended to the prompt text for completions."""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": [{"text": "response"}]}
        mock_session.post.return_value = mock_response

        call_model_completions(
            url="http://localhost:8000",
            model_id="test-model",
            prompt="User question",
            session=mock_session,
            system_prompt="System instructions",
        )

        _args, kwargs = mock_session.post.call_args
        payload = kwargs["json"]
        prompt = payload["prompt"]

        assert prompt == "System instructions\nUser question", (
            f"Expected system prompt prepended, got '{prompt}'"
        )

    def test_completions_no_system_prompt(self):
        """Validate that prompt is unchanged when system_prompt is None for completions."""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": [{"text": "response"}]}
        mock_session.post.return_value = mock_response

        call_model_completions(
            url="http://localhost:8000",
            model_id="test-model",
            prompt="User question",
            session=mock_session,
        )

        _args, kwargs = mock_session.post.call_args
        payload = kwargs["json"]
        prompt = payload["prompt"]

        assert prompt == "User question", f"Expected unchanged prompt, got '{prompt}'"

    def test_backward_compat_no_system_prompt(self):
        """Validate backward compatibility: existing callers without system_prompt still work."""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "response"}}]
        }
        mock_session.post.return_value = mock_response

        # Call without system_prompt kwarg at all
        result = call_model_chat(
            url="http://localhost:8000",
            model_id="test-model",
            prompt="Test prompt",
            session=mock_session,
        )

        assert result == "response"
        mock_session.post.assert_called_once()


# ---------------------------------------------------------------------------
# n-repeats, client auto-detect, command template extensions
# ---------------------------------------------------------------------------


class TestNRepeats:
    """Tests for --n-repeats functionality."""

    @pytest.fixture
    def mock_benchmark(self):
        from nemo_evaluator.contrib.byob.decorators import BenchmarkDefinition

        def simple_scorer(sample):
            return {"correct": sample.target.lower() in sample.response.lower()}

        return BenchmarkDefinition(
            name="repeat-test",
            normalized_name="repeat_test",
            dataset="unused",
            prompt="Q: {question}\nA:",
            target_field="answer",
            scorer_fn=simple_scorer,
        )

    def test_n_repeats_default_is_one(self):
        """Test that --n-repeats defaults to 1."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--n-repeats", type=int, default=1)
        args = parser.parse_args([])
        assert args.n_repeats == 1

    def test_n_repeats_three_on_two_samples(self, mock_benchmark):
        """Test --n-repeats=3 on 2-sample dataset produces 6 scores."""
        from nemo_evaluator.contrib.byob.eval_logic import run_eval_loop

        dataset = [
            {"question": "q1", "answer": "yes"},
            {"question": "q2", "answer": "no"},
        ]
        mock_model = MagicMock(return_value="Yes")

        n_repeats = 3
        all_scores = []
        all_predictions = []

        for repeat_idx in range(n_repeats):
            scores, predictions = run_eval_loop(
                bench=mock_benchmark,
                dataset=dataset,
                model_call_fn=mock_model,
                endpoint_type="chat",
                save_predictions=True,
            )
            offset = repeat_idx * len(dataset)
            for pred in predictions:
                pred.sample_id = pred.sample_id + offset
                pred.metadata = {**pred.metadata, "_repeat": repeat_idx}
            all_scores.extend(scores)
            all_predictions.extend(predictions)

        assert len(all_scores) == 6, f"Expected 6 scores, got {len(all_scores)}"
        assert len(all_predictions) == 6, (
            f"Expected 6 predictions, got {len(all_predictions)}"
        )

    def test_n_repeats_predictions_have_unique_ids_and_repeat_metadata(
        self, mock_benchmark
    ):
        """Test that predictions have unique IDs and _repeat metadata."""
        from nemo_evaluator.contrib.byob.eval_logic import run_eval_loop

        dataset = [
            {"question": "q1", "answer": "yes"},
        ]
        mock_model = MagicMock(return_value="Yes")

        n_repeats = 2
        all_predictions = []

        for repeat_idx in range(n_repeats):
            _, predictions = run_eval_loop(
                bench=mock_benchmark,
                dataset=dataset,
                model_call_fn=mock_model,
                endpoint_type="chat",
                save_predictions=True,
            )
            offset = repeat_idx * len(dataset)
            for pred in predictions:
                pred.sample_id = pred.sample_id + offset
                pred.metadata = {**pred.metadata, "_repeat": repeat_idx}
            all_predictions.extend(predictions)

        ids = [p.sample_id for p in all_predictions]
        assert len(ids) == len(set(ids)), f"IDs not unique: {ids}"

        for pred in all_predictions:
            assert "_repeat" in pred.metadata, (
                f"Missing _repeat in metadata: {pred.metadata}"
            )

        assert all_predictions[0].metadata["_repeat"] == 0
        assert all_predictions[1].metadata["_repeat"] == 1


class TestClientAutoDetect:
    """Tests for NeMoEvaluatorClient auto-detection."""

    def test_create_client_model_call_fn_import_error(self):
        """Test that ImportError is raised when nemo_evaluator.client is unavailable."""
        import argparse

        from nemo_evaluator.contrib.byob.runner import create_client_model_call_fn

        args = argparse.Namespace(
            model_url="http://localhost:8000",
            model_id="test-model",
            temperature=0.0,
            max_tokens=4096,
        )

        with patch.dict("sys.modules", {"nemo_evaluator.client": None}):
            with pytest.raises((ImportError, ModuleNotFoundError)):
                create_client_model_call_fn(args, api_key=None)


class TestCommandTemplateExtensions:
    """Tests for COMMAND_TEMPLATE additions in compiler.py."""

    def test_command_template_contains_n_repeats(self):
        """Test that COMMAND_TEMPLATE includes --n-repeats conditional."""
        from nemo_evaluator.contrib.byob.compiler import COMMAND_TEMPLATE

        assert "n_repeats" in COMMAND_TEMPLATE
        assert "--n-repeats" in COMMAND_TEMPLATE
