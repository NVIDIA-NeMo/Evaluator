# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Tests for the CLI ls tasks command."""

import json
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

import pytest

from nemo_evaluator_launcher.cli.ls_tasks import Cmd as LsTasksCmd


class TestLsTasksCommand:
    """Test the ls tasks CLI command."""

    @pytest.fixture
    def sample_tasks_data(self):
        """Sample task data for testing."""
        return [
            ["garak", "chat", "garak", "nvcr.io/nvidia/eval-factory/garak:25.08.1"],
            ["mmlu", "chat", "lm-eval", "nvcr.io/nvidia/eval-factory/lm-eval:25.08.1"],
            [
                "arc",
                "completions",
                "lm-eval",
                "nvcr.io/nvidia/eval-factory/lm-eval:25.08.1",
            ],
            [
                "truthfulqa",
                "chat",
                "lm-eval",
                "nvcr.io/nvidia/eval-factory/lm-eval:25.08.1",
            ],
            ["ifeval", "chat", "helm", "nvcr.io/nvidia/eval-factory/helm:25.08.1"],
        ]

    @pytest.fixture
    def single_task_data(self):
        """Single task data for testing edge cases."""
        return [
            ["garak", "chat", "garak", "nvcr.io/nvidia/eval-factory/garak:25.08.1"],
        ]

    @pytest.fixture
    def empty_tasks_data(self):
        """Empty task data for testing edge cases."""
        return []

    def test_json_output_basic(self, sample_tasks_data):
        """Test basic JSON output format."""
        cmd = LsTasksCmd(json=True)

        with patch(
            "nemo_evaluator_launcher.api.functional.get_tasks_list",
            return_value=sample_tasks_data,
        ):
            output = StringIO()
            with redirect_stdout(output):
                cmd.execute()

            result = output.getvalue().strip()
            parsed_json = json.loads(result)

            # Verify structure
            assert "tasks" in parsed_json
            assert isinstance(parsed_json["tasks"], list)
            assert len(parsed_json["tasks"]) == 5

            # Verify first task content
            first_task = parsed_json["tasks"][0]
            expected_keys = ["task", "endpoint_type", "harness", "container"]
            assert all(key in first_task for key in expected_keys)
            assert first_task["task"] == "garak"
            assert first_task["endpoint_type"] == "chat"
            assert first_task["harness"] == "garak"
            assert (
                first_task["container"] == "nvcr.io/nvidia/eval-factory/garak:25.08.1"
            )

    def test_json_output_empty_data(self, empty_tasks_data):
        """Test JSON output with empty data."""
        cmd = LsTasksCmd(json=True)

        with patch(
            "nemo_evaluator_launcher.api.functional.get_tasks_list",
            return_value=empty_tasks_data,
        ):
            output = StringIO()
            with redirect_stdout(output):
                cmd.execute()

            result = output.getvalue().strip()
            parsed_json = json.loads(result)

            assert "tasks" in parsed_json
            assert parsed_json["tasks"] == []

    def test_json_output_single_task(self, single_task_data):
        """Test JSON output with single task."""
        cmd = LsTasksCmd(json=True)

        with patch(
            "nemo_evaluator_launcher.api.functional.get_tasks_list",
            return_value=single_task_data,
        ):
            output = StringIO()
            with redirect_stdout(output):
                cmd.execute()

            result = output.getvalue().strip()
            parsed_json = json.loads(result)

            assert "tasks" in parsed_json
            assert len(parsed_json["tasks"]) == 1
            assert parsed_json["tasks"][0]["task"] == "garak"

    def test_table_output_basic(self, sample_tasks_data):
        """Test basic table output format."""
        cmd = LsTasksCmd(json=False)

        with patch(
            "nemo_evaluator_launcher.api.functional.get_tasks_list",
            return_value=sample_tasks_data,
        ):
            output = StringIO()
            with redirect_stdout(output):
                cmd.execute()

            result = output.getvalue()

            # Verify harness and container info is present
            assert "harness: garak" in result
            assert "harness: lm-eval" in result
            assert "harness: helm" in result
            assert "container: nvcr.io/nvidia/eval-factory/garak:25.08.1" in result
            assert "container: nvcr.io/nvidia/eval-factory/lm-eval:25.08.1" in result
            assert "container: nvcr.io/nvidia/eval-factory/helm:25.08.1" in result

            # Verify task names are present
            assert "garak" in result
            assert "mmlu" in result
            assert "arc" in result
            assert "truthfulqa" in result
            assert "ifeval" in result

            # Verify endpoint types are present
            assert "chat" in result
            assert "completions" in result

            # Verify task count messages
            assert "task available" in result or "tasks available" in result

    def test_table_output_formatting(self, sample_tasks_data):
        """Test table formatting details."""
        cmd = LsTasksCmd(json=False)

        with patch(
            "nemo_evaluator_launcher.api.functional.get_tasks_list",
            return_value=sample_tasks_data,
        ):
            output = StringIO()
            with redirect_stdout(output):
                cmd.execute()

            result = output.getvalue()
            lines = result.split("\n")

            # Check for equals signs (top and bottom borders)
            equals_lines = [
                line
                for line in lines
                if line.strip() and all(c in "=" for c in line.strip())
            ]
            assert len(equals_lines) >= 2  # At least top and bottom borders

            # Check for header structure
            assert any("task" in line and "endpoint_type" in line for line in lines)

            # Check that tasks are properly sorted alphabetically within each group
            # (this depends on the implementation but we can verify basic structure)

    def test_table_output_empty_data(self, empty_tasks_data):
        """Test table output with empty data."""
        cmd = LsTasksCmd(json=False)

        with patch(
            "nemo_evaluator_launcher.api.functional.get_tasks_list",
            return_value=empty_tasks_data,
        ):
            output = StringIO()
            with redirect_stdout(output):
                cmd.execute()

            result = output.getvalue().strip()
            assert "No tasks found." in result

    def test_table_output_single_task(self, single_task_data):
        """Test table output with single task."""
        cmd = LsTasksCmd(json=False)

        with patch(
            "nemo_evaluator_launcher.api.functional.get_tasks_list",
            return_value=single_task_data,
        ):
            output = StringIO()
            with redirect_stdout(output):
                cmd.execute()

            result = output.getvalue()

            # Should contain single task info
            assert "harness: garak" in result
            assert "container: nvcr.io/nvidia/eval-factory/garak:25.08.1" in result
            assert "garak" in result
            assert "chat" in result
            assert "1 task available" in result

    def test_table_grouping_by_harness_and_container(self):
        """Test that tasks are properly grouped by harness and container."""
        # Create data with multiple tasks for same harness/container and different ones
        test_data = [
            ["task1", "chat", "harness1", "container1"],
            ["task2", "completions", "harness1", "container1"],
            ["task3", "chat", "harness1", "container2"],
            ["task4", "chat", "harness2", "container3"],
        ]

        cmd = LsTasksCmd(json=False)

        with patch(
            "nemo_evaluator_launcher.api.functional.get_tasks_list",
            return_value=test_data,
        ):
            output = StringIO()
            with redirect_stdout(output):
                cmd.execute()

            result = output.getvalue()

            # Should have separate sections for different harness/container combinations
            assert "harness: harness1" in result
            assert "harness: harness2" in result
            assert "container: container1" in result
            assert "container: container2" in result
            assert "container: container3" in result

            # All tasks should be present
            for i in range(1, 5):
                assert f"task{i}" in result

    def test_table_width_calculation(self):
        """Test that table width is calculated correctly for long container names."""
        test_data = [
            [
                "short",
                "chat",
                "harness",
                "very-long-container-name-that-should-determine-table-width",
            ],
        ]

        cmd = LsTasksCmd(json=False)

        with patch(
            "nemo_evaluator_launcher.api.functional.get_tasks_list",
            return_value=test_data,
        ):
            output = StringIO()
            with redirect_stdout(output):
                cmd.execute()

            result = output.getvalue()
            lines = result.split("\n")

            # Find the equals lines (borders)
            equals_lines = [
                line
                for line in lines
                if line.strip() and all(c in "=" for c in line.strip())
            ]

            if equals_lines:
                # All equals lines should have the same length
                first_length = len(equals_lines[0])
                assert all(len(line) == first_length for line in equals_lines)

                # Width should be at least as long as the container name
                assert first_length >= len(
                    "container: very-long-container-name-that-should-determine-table-width"
                )

    def test_cmd_default_values(self):
        """Test that Cmd has correct default values."""
        cmd = LsTasksCmd()
        assert cmd.json is False

    def test_cmd_with_json_flag(self):
        """Test that Cmd can be created with json=True."""
        cmd = LsTasksCmd(json=True)
        assert cmd.json is True

    def test_data_format_validation(self, sample_tasks_data):
        """Test that the command validates data format correctly."""
        cmd = LsTasksCmd(json=False)

        # Test with malformed data (wrong number of fields)
        malformed_data = [
            ["task1", "chat", "harness1"],  # Missing container field
        ]

        with patch(
            "nemo_evaluator_launcher.api.functional.get_tasks_list",
            return_value=malformed_data,
        ):
            with pytest.raises(AssertionError):
                cmd.execute()

    def test_column_width_distribution(self):
        """Test that column widths are distributed correctly."""
        test_data = [
            ["very-long-task-name-here", "chat", "harness", "container"],
            ["short", "completions", "harness", "container"],
        ]

        cmd = LsTasksCmd(json=False)

        with patch(
            "nemo_evaluator_launcher.api.functional.get_tasks_list",
            return_value=test_data,
        ):
            output = StringIO()
            with redirect_stdout(output):
                cmd.execute()

            result = output.getvalue()
            lines = result.split("\n")

            # Find lines with task data
            task_lines = [
                line
                for line in lines
                if "very-long-task-name-here" in line
                or ("short" in line and "completions" in line)
            ]

            # Should have proper spacing and alignment
            assert len(task_lines) >= 2

    @patch("nemo_evaluator_launcher.api.functional.get_tasks_list")
    def test_api_function_called(self, mock_get_tasks):
        """Test that the API function is called correctly."""
        mock_get_tasks.return_value = []

        cmd = LsTasksCmd(json=True)

        output = StringIO()
        with redirect_stdout(output):
            cmd.execute()

        mock_get_tasks.assert_called_once()

    def test_task_count_messages(self):
        """Test task count messages for different scenarios."""
        # Test single task
        single_data = [["task1", "chat", "harness", "container"]]
        cmd = LsTasksCmd(json=False)

        with patch(
            "nemo_evaluator_launcher.api.functional.get_tasks_list",
            return_value=single_data,
        ):
            output = StringIO()
            with redirect_stdout(output):
                cmd.execute()

            result = output.getvalue()
            assert "1 task available" in result

        # Test multiple tasks
        multi_data = [
            ["task1", "chat", "harness", "container"],
            ["task2", "completions", "harness", "container"],
        ]

        with patch(
            "nemo_evaluator_launcher.api.functional.get_tasks_list",
            return_value=multi_data,
        ):
            output = StringIO()
            with redirect_stdout(output):
                cmd.execute()

            result = output.getvalue()
            assert "2 tasks available" in result
