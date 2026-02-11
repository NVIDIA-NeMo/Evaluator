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

import pytest

from nemo_evaluator.byob.runner import aggregate_scores, load_dataset


class TestAggregateScores:
    """Tests for aggregate_scores function."""

    def test_aggregate_binary_scores(self):
        """Test aggregation of binary (True/False) scores with percentage scaling.

        Hand-computed values:
        - Booleans: [True, False, True] -> [1.0, 0.0, 1.0]
        - n = 3
        - mean = 2.0 / 3 = 0.6667
        - variance = ((1-0.6667)^2 + (0-0.6667)^2 + (1-0.6667)^2) / 3
        -          = (0.1111 + 0.4445 + 0.1111) / 3 = 0.2222
        - stddev = sqrt(0.2222) = 0.4714
        - stderr = 0.4714 / sqrt(3) = 0.2722
        - Binary detected -> scale by 100
        - value = 66.6667, stderr = 27.2166
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
        # stderr also scaled by 100 for binary metrics
        assert abs(score_data["stderr"] - 27.2166) < 0.1, \
            f"Expected stderr~27.2166, got {score_data['stderr']}"
        assert abs(score_data["stddev"] - 47.14) < 0.1, \
            f"Expected stddev~47.14, got {score_data['stddev']}"

    def test_aggregate_continuous_scores(self):
        """Test aggregation of continuous (non-binary) scores without scaling.

        Hand-computed values:
        - Values: [0.8, 0.9, 1.0]
        - n = 3
        - mean = 2.7 / 3 = 0.9
        - variance = ((0.8-0.9)^2 + (0.9-0.9)^2 + (1.0-0.9)^2) / 3
        -          = (0.01 + 0.0 + 0.01) / 3 = 0.0067
        - stddev = sqrt(0.0067) = 0.0816
        - stderr = 0.0816 / sqrt(3) = 0.0471
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
        assert abs(score_data["stddev"] - 0.0816) < 0.001, \
            f"Expected stddev~0.0816, got {score_data['stddev']}"
        assert abs(score_data["stderr"] - 0.0471) < 0.001, \
            f"Expected stderr~0.0471, got {score_data['stderr']}"

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
        """Test that load_dataset raises FileNotFoundError for nonexistent file."""
        with pytest.raises(FileNotFoundError):
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
