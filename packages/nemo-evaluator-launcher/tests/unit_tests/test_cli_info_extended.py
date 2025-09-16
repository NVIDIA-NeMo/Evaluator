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
"""Extended tests for info CLI command to increase coverage."""

from unittest.mock import patch

import pytest

from nemo_evaluator_launcher.cli.info import Cmd


class TestInfoFiltering:
    """Test filtering functionality in info command."""

    def test_executor_filtering(self, capsys):
        """Test filtering by executor."""
        mock_data = [
            {
                "invocation_id": "inv1",
                "executor": "local",
                "num_jobs": 5,
                "earliest_job_ts": 1640995200.0,
            },
            {
                "invocation_id": "inv2",
                "executor": "slurm",
                "num_jobs": 3,
                "earliest_job_ts": 1640995300.0,
            },
            {
                "invocation_id": "inv3",
                "executor": "LOCAL",  # Test case insensitive
                "num_jobs": 2,
                "earliest_job_ts": 1640995400.0,
            },
        ]

        with patch(
            "nemo_evaluator_launcher.cli.info.list_all_invocations_summary",
            return_value=mock_data,
        ):
            with patch(
                "nemo_evaluator_launcher.cli.info.get_invocation_benchmarks",
                return_value=["test_benchmark"],
            ):
                cmd = Cmd(executor="local")
                cmd.execute()

        captured = capsys.readouterr()
        assert "inv1" in captured.out
        assert "inv3" in captured.out  # Case insensitive match
        assert "inv2" not in captured.out

    def test_executor_filtering_empty_executor(self, capsys):
        """Test filtering by executor when some entries have empty executor."""
        mock_data = [
            {
                "invocation_id": "inv1",
                "executor": "local",
                "num_jobs": 5,
                "earliest_job_ts": 1640995200.0,
            },
            {
                "invocation_id": "inv2",
                "executor": "",  # Empty executor
                "num_jobs": 3,
                "earliest_job_ts": 1640995300.0,
            },
            {
                "invocation_id": "inv3",
                "num_jobs": 2,  # Missing executor key
                "earliest_job_ts": 1640995400.0,
            },
        ]

        with patch(
            "nemo_evaluator_launcher.cli.info.list_all_invocations_summary",
            return_value=mock_data,
        ):
            with patch(
                "nemo_evaluator_launcher.cli.info.get_invocation_benchmarks",
                return_value=["test_benchmark"],
            ):
                cmd = Cmd(executor="local")
                cmd.execute()

        captured = capsys.readouterr()
        assert "inv1" in captured.out
        assert "inv2" not in captured.out
        assert "inv3" not in captured.out

    def test_since_filtering_with_time(self, capsys):
        """Test filtering by since parameter with full datetime."""
        mock_data = [
            {
                "invocation_id": "inv1",
                "executor": "local",
                "num_jobs": 5,
                "earliest_job_ts": 1640995200.0,  # 2022-01-01 00:00:00
            },
            {
                "invocation_id": "inv2",
                "executor": "local",
                "num_jobs": 3,
                "earliest_job_ts": 1641081600.0,  # 2022-01-02 00:00:00
            },
        ]

        with patch(
            "nemo_evaluator_launcher.cli.info.list_all_invocations_summary",
            return_value=mock_data,
        ):
            with patch(
                "nemo_evaluator_launcher.cli.info.get_invocation_benchmarks",
                return_value=["test_benchmark"],
            ):
                cmd = Cmd(since="2022-01-01T12:00:00")
                cmd.execute()

        captured = capsys.readouterr()
        assert "inv1" not in captured.out  # Before cutoff
        assert "inv2" in captured.out  # After cutoff

    def test_since_filtering_date_only(self, capsys):
        """Test filtering by since parameter with date only."""
        mock_data = [
            {
                "invocation_id": "inv1",
                "executor": "local",
                "num_jobs": 5,
                "earliest_job_ts": 1640908800.0,  # 2021-12-31 00:00:00
            },
            {
                "invocation_id": "inv2",
                "executor": "local",
                "num_jobs": 3,
                "earliest_job_ts": 1640995200.0,  # 2022-01-01 00:00:00
            },
        ]

        with patch(
            "nemo_evaluator_launcher.cli.info.list_all_invocations_summary",
            return_value=mock_data,
        ):
            with patch(
                "nemo_evaluator_launcher.cli.info.get_invocation_benchmarks",
                return_value=["test_benchmark"],
            ):
                cmd = Cmd(since="2022-01-01")
                cmd.execute()

        captured = capsys.readouterr()
        assert "inv1" not in captured.out
        assert "inv2" in captured.out

    def test_since_filtering_missing_timestamp(self, capsys):
        """Test filtering with missing or zero timestamp."""
        mock_data = [
            {
                "invocation_id": "inv1",
                "executor": "local",
                "num_jobs": 5,
                "earliest_job_ts": 0,  # Zero timestamp
            },
            {
                "invocation_id": "inv2",
                "executor": "local",
                "num_jobs": 3,
                # Missing timestamp
            },
            {
                "invocation_id": "inv3",
                "executor": "local",
                "num_jobs": 2,
                "earliest_job_ts": None,  # None timestamp
            },
        ]

        with patch(
            "nemo_evaluator_launcher.cli.info.list_all_invocations_summary",
            return_value=mock_data,
        ):
            with patch(
                "nemo_evaluator_launcher.cli.info.get_invocation_benchmarks",
                return_value=["test_benchmark"],
            ):
                cmd = Cmd(since="2022-01-01")
                cmd.execute()

        captured = capsys.readouterr()
        # All should be filtered out due to timestamp < since
        assert "inv1" not in captured.out
        assert "inv2" not in captured.out
        assert "inv3" not in captured.out

    def test_since_invalid_format(self, capsys):
        """Test since parameter with invalid date format."""
        mock_data = [{"invocation_id": "inv1", "executor": "local", "num_jobs": 5}]

        with patch(
            "nemo_evaluator_launcher.cli.info.list_all_invocations_summary",
            return_value=mock_data,
        ):
            cmd = Cmd(since="invalid-date-format")

            with pytest.raises(SystemExit) as exc_info:
                cmd.execute()

            assert exc_info.value.code == 2

        captured = capsys.readouterr()
        assert "Invalid --since value" in captured.err

    def test_limit_filtering(self, capsys):
        """Test limit parameter."""
        mock_data = [
            {
                "invocation_id": f"inv{i}",
                "executor": "local",
                "num_jobs": 1,
                "earliest_job_ts": 1640995200.0 + i,
            }
            for i in range(10)
        ]

        with patch(
            "nemo_evaluator_launcher.cli.info.list_all_invocations_summary",
            return_value=mock_data,
        ):
            with patch(
                "nemo_evaluator_launcher.cli.info.get_invocation_benchmarks",
                return_value=["test_benchmark"],
            ):
                cmd = Cmd(limit=3)
                cmd.execute()

        captured = capsys.readouterr()
        # Should only show first 3 invocations
        assert "inv0" in captured.out
        assert "inv1" in captured.out
        assert "inv2" in captured.out
        assert "inv3" not in captured.out

    def test_limit_zero(self, capsys):
        """Test limit parameter with zero value."""
        mock_data = [
            {
                "invocation_id": "inv1",
                "executor": "local",
                "num_jobs": 1,
                "earliest_job_ts": 1640995200.0,
            },
        ]

        with patch(
            "nemo_evaluator_launcher.cli.info.list_all_invocations_summary",
            return_value=mock_data,
        ):
            with patch(
                "nemo_evaluator_launcher.cli.info.get_invocation_benchmarks",
                return_value=["test_benchmark"],
            ):
                cmd = Cmd(limit=0)
                cmd.execute()

        captured = capsys.readouterr()
        # Should show no rows (empty table except header)
        lines = captured.out.strip().split("\n")
        assert len(lines) == 2  # Header + separator line only


class TestInfoTableFormatting:
    """Test table formatting functionality."""

    def test_timestamp_formatting_success(self, capsys):
        """Test successful timestamp formatting."""
        test_timestamp = 1640995200.0  # 2022-01-01 00:00:00 UTC
        mock_data = [
            {
                "invocation_id": "inv1",
                "executor": "local",
                "num_jobs": 5,
                "earliest_job_ts": test_timestamp,
            },
        ]

        with patch(
            "nemo_evaluator_launcher.cli.info.list_all_invocations_summary",
            return_value=mock_data,
        ):
            with patch(
                "nemo_evaluator_launcher.cli.info.get_invocation_benchmarks",
                return_value=["test_benchmark"],
            ):
                cmd = Cmd()
                cmd.execute()

        captured = capsys.readouterr()
        # Dynamically compute expected timestamp format (same as the actual code)
        import datetime as dt

        expected_timestamp = (
            dt.datetime.fromtimestamp(test_timestamp).replace(microsecond=0).isoformat()
        )
        assert expected_timestamp in captured.out

    def test_timestamp_formatting_error(self, capsys):
        """Test timestamp formatting with invalid timestamp."""
        mock_data = [
            {
                "invocation_id": "inv1",
                "executor": "local",
                "num_jobs": 5,
                "earliest_job_ts": float("inf"),  # Invalid timestamp
            },
        ]

        with patch(
            "nemo_evaluator_launcher.cli.info.list_all_invocations_summary",
            return_value=mock_data,
        ):
            with patch(
                "nemo_evaluator_launcher.cli.info.get_invocation_benchmarks",
                return_value=["test_benchmark"],
            ):
                cmd = Cmd()
                cmd.execute()

        captured = capsys.readouterr()
        # Should have empty timestamp field
        lines = captured.out.strip().split("\n")
        data_line = lines[2]  # Skip header and separator
        fields = data_line.split()
        assert len(fields) >= 2
        # The timestamp field should be empty or minimal

    def test_benchmarks_retrieval_success(self, capsys):
        """Test successful benchmark retrieval."""
        mock_data = [
            {
                "invocation_id": "inv1",
                "executor": "local",
                "num_jobs": 5,
                "earliest_job_ts": 1640995200.0,
            },
        ]

        with patch(
            "nemo_evaluator_launcher.cli.info.list_all_invocations_summary",
            return_value=mock_data,
        ):
            with patch(
                "nemo_evaluator_launcher.cli.info.get_invocation_benchmarks",
                return_value=["bench1", "bench2"],
            ):
                cmd = Cmd()
                cmd.execute()

        captured = capsys.readouterr()
        assert "bench1,bench2" in captured.out

    def test_benchmarks_retrieval_empty(self, capsys):
        """Test benchmark retrieval with empty result."""
        mock_data = [
            {
                "invocation_id": "inv1",
                "executor": "local",
                "num_jobs": 5,
                "earliest_job_ts": 1640995200.0,
            },
        ]

        with patch(
            "nemo_evaluator_launcher.cli.info.list_all_invocations_summary",
            return_value=mock_data,
        ):
            with patch(
                "nemo_evaluator_launcher.cli.info.get_invocation_benchmarks",
                return_value=[],
            ):
                cmd = Cmd()
                cmd.execute()

        captured = capsys.readouterr()
        assert "unknown" in captured.out

    def test_benchmarks_retrieval_error(self, capsys):
        """Test benchmark retrieval with exception."""
        mock_data = [
            {
                "invocation_id": "inv1",
                "executor": "local",
                "num_jobs": 5,
                "earliest_job_ts": 1640995200.0,
            },
        ]

        with patch(
            "nemo_evaluator_launcher.cli.info.list_all_invocations_summary",
            return_value=mock_data,
        ):
            with patch(
                "nemo_evaluator_launcher.cli.info.get_invocation_benchmarks",
                side_effect=Exception("DB error"),
            ):
                cmd = Cmd()
                cmd.execute()

        captured = capsys.readouterr()
        assert "unknown" in captured.out

    def test_missing_fields_handling(self, capsys):
        """Test handling of missing fields in data."""
        mock_data = [
            {
                "invocation_id": "inv1",
                # Missing other fields
            },
        ]

        with patch(
            "nemo_evaluator_launcher.cli.info.list_all_invocations_summary",
            return_value=mock_data,
        ):
            with patch(
                "nemo_evaluator_launcher.cli.info.get_invocation_benchmarks",
                return_value=["test_benchmark"],
            ):
                cmd = Cmd()
                cmd.execute()

        captured = capsys.readouterr()
        assert "inv1" in captured.out
        # Should handle missing fields gracefully with empty/default values

    def test_table_column_width_calculation(self, capsys):
        """Test dynamic column width calculation."""
        mock_data = [
            {
                "invocation_id": "very_long_invocation_id_that_should_expand_column",
                "executor": "local",
                "num_jobs": 5,
                "earliest_job_ts": 1640995200.0,
            },
            {
                "invocation_id": "short",
                "executor": "very_long_executor_name_here",
                "num_jobs": 12345,
                "earliest_job_ts": 1640995200.0,
            },
        ]

        with patch(
            "nemo_evaluator_launcher.cli.info.list_all_invocations_summary",
            return_value=mock_data,
        ):
            with patch(
                "nemo_evaluator_launcher.cli.info.get_invocation_benchmarks",
                return_value=["benchmark"],
            ):
                cmd = Cmd()
                cmd.execute()

        captured = capsys.readouterr()
        lines = captured.out.strip().split("\n")

        # Check that the header line and data lines have consistent formatting
        header_line = lines[0]
        data_line1 = lines[2]  # Skip separator
        data_line2 = lines[3]

        # All lines should have similar structure (accounting for spacing)
        assert len(header_line) > 50  # Should be reasonably wide
        assert "very_long_invocation_id_that_should_expand_column" in data_line1
        assert "very_long_executor_name_here" in data_line2


class TestInfoCombinedFilters:
    """Test combinations of filters."""

    def test_multiple_filters_combined(self, capsys):
        """Test multiple filters applied together."""
        mock_data = [
            {
                "invocation_id": "inv1",
                "executor": "local",
                "num_jobs": 5,
                "earliest_job_ts": 1640995200.0,  # 2022-01-01 00:00:00
            },
            {
                "invocation_id": "inv2",
                "executor": "slurm",
                "num_jobs": 3,
                "earliest_job_ts": 1641081600.0,  # 2022-01-02 00:00:00
            },
            {
                "invocation_id": "inv3",
                "executor": "local",
                "num_jobs": 2,
                "earliest_job_ts": 1641168000.0,  # 2022-01-03 00:00:00
            },
        ]

        with patch(
            "nemo_evaluator_launcher.cli.info.list_all_invocations_summary",
            return_value=mock_data,
        ):
            with patch(
                "nemo_evaluator_launcher.cli.info.get_invocation_benchmarks",
                return_value=["test_benchmark"],
            ):
                cmd = Cmd(executor="local", since="2022-01-01T12:00:00", limit=1)
                cmd.execute()

        captured = capsys.readouterr()
        # Should only show inv3 (local executor, after cutoff time, limited to 1)
        assert "inv1" not in captured.out  # Before time cutoff
        assert "inv2" not in captured.out  # Wrong executor
        assert "inv3" in captured.out  # Matches all criteria

    def test_empty_result_after_filtering(self, capsys):
        """Test when all entries are filtered out."""
        mock_data = [
            {
                "invocation_id": "inv1",
                "executor": "slurm",
                "num_jobs": 5,
                "earliest_job_ts": 1640995200.0,
            },
        ]

        with patch(
            "nemo_evaluator_launcher.cli.info.list_all_invocations_summary",
            return_value=mock_data,
        ):
            cmd = Cmd(executor="local")  # No matches
            cmd.execute()

        captured = capsys.readouterr()
        lines = captured.out.strip().split("\n")
        # Should only have header and separator lines
        assert len(lines) == 2
        assert "invocation_id" in lines[0]


class TestInfoSpecialCases:
    """Test special cases unique to info command."""

    def test_always_includes_benchmarks_column(self, capsys):
        """Test that benchmarks column is always included in info output."""
        mock_data = [
            {
                "invocation_id": "inv1",
                "executor": "local",
                "num_jobs": 5,
                "earliest_job_ts": 1640995200.0,
            },
        ]

        with patch(
            "nemo_evaluator_launcher.cli.info.list_all_invocations_summary",
            return_value=mock_data,
        ):
            with patch(
                "nemo_evaluator_launcher.cli.info.get_invocation_benchmarks",
                return_value=["test_benchmark"],
            ):
                cmd = Cmd()
                cmd.execute()

        captured = capsys.readouterr()
        lines = captured.out.strip().split("\n")
        header_line = lines[0]
        assert "benchmarks" in header_line

    def test_concise_table_description(self, capsys):
        """Test that info command provides concise invocations table."""
        # The class docstring mentions "concise invocations table"
        # This test ensures the functionality matches the description
        mock_data = [
            {
                "invocation_id": "inv1",
                "executor": "local",
                "num_jobs": 5,
                "earliest_job_ts": 1640995200.0,
            },
        ]

        with patch(
            "nemo_evaluator_launcher.cli.info.list_all_invocations_summary",
            return_value=mock_data,
        ):
            with patch(
                "nemo_evaluator_launcher.cli.info.get_invocation_benchmarks",
                return_value=["test_benchmark"],
            ):
                cmd = Cmd()
                cmd.execute()

        captured = capsys.readouterr()
        lines = captured.out.strip().split("\n")

        # Should have exactly the expected columns and be concise
        assert len(lines) == 3  # Header, separator, one data row
        expected_headers = [
            "invocation_id",
            "earliest_job_ts",
            "num_jobs",
            "executor",
            "benchmarks",
        ]
        header_line = lines[0]
        for header in expected_headers:
            assert header in header_line
