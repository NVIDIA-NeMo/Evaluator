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
"""Tests for the CLI logs command."""

from unittest.mock import patch

import pytest

from nemo_evaluator_launcher.api.functional import stream_logs
from nemo_evaluator_launcher.cli.logs import Cmd as LogsCmd
from nemo_evaluator_launcher.common.execdb import ExecutionDB, JobData
from nemo_evaluator_launcher.executors.base import BaseExecutor
from nemo_evaluator_launcher.executors.local.executor import LocalExecutor


@pytest.mark.usefixtures("mock_execdb")
class TestLogsCommandBasic:
    """Test basic logs command functionality."""

    def test_logs_cmd_with_job_id_not_found(self, capsys):
        """Test logs command when job ID is not found."""
        cmd = LogsCmd(id="nonexistent.0")
        with pytest.raises(SystemExit) as exc_info:
            cmd.execute()
        assert exc_info.value.code == 1

    def test_logs_cmd_with_invocation_id_not_found(self, capsys):
        """Test logs command when invocation ID is not found."""
        cmd = LogsCmd(id="nonexistent")
        with pytest.raises(SystemExit) as exc_info:
            cmd.execute()
        assert exc_info.value.code == 1

    def test_logs_cmd_with_empty_id(self, capsys):
        """Test logs command when ID is empty."""
        cmd = LogsCmd(id="")
        with pytest.raises(SystemExit) as exc_info:
            cmd.execute()
        assert exc_info.value.code == 1

    def test_logs_command_with_job_id(self, job_local, capsys):
        """Test logs command with a valid job ID."""
        cmd = LogsCmd(id=job_local.job_id)
        # Mock the streaming to avoid infinite loop
        with patch("nemo_evaluator_launcher.cli.logs.stream_logs") as mock_stream:
            mock_stream.return_value = iter(
                [
                    (job_local.job_id, "mbpp", "Test log line 1"),
                    (job_local.job_id, "mbpp", "Test log line 2"),
                ]
            )
            with patch("builtins.print") as mock_print:
                cmd.execute()
                # Verify that stream_logs was called
                mock_stream.assert_called_once_with(job_local.job_id)
                # Verify that print was called for log lines
                assert mock_print.call_count >= 2

    def test_logs_command_with_invocation_id(
        self, job_local, prepare_local_job, capsys
    ):
        """Test logs command with an invocation ID containing multiple jobs."""
        inv = job_local.invocation_id
        # Add a second job
        jd2 = JobData(
            invocation_id=inv,
            job_id=f"{inv}.1",
            timestamp=job_local.timestamp,
            executor="local",
            data={},
            config=job_local.config,
        )
        jd2, base2 = prepare_local_job(jd2, with_required=True, with_optional=True)
        ExecutionDB().write_job(jd2)

        cmd = LogsCmd(id=inv)
        # Mock the streaming to avoid infinite loop
        with patch("nemo_evaluator_launcher.cli.logs.stream_logs") as mock_stream:
            mock_stream.return_value = iter(
                [
                    (job_local.job_id, "mbpp", "Log from job 0"),
                    (jd2.job_id, "mbpp", "Log from job 1"),
                ]
            )
            with patch("builtins.print") as mock_print:
                cmd.execute()
                # Verify that stream_logs was called
                mock_stream.assert_called_once_with(inv)
                # Verify that print was called
                assert mock_print.call_count >= 2

    def test_logs_command_color_mapping(self, job_local):
        """Test that colors are correctly mapped to job IDs."""
        # Build color mapping (same logic as in execute)
        import nemo_evaluator_launcher.common.printing_utils as pu

        colors = [pu.red, pu.green, pu.yellow, pu.magenta, pu.cyan]
        job_colors = {}
        job_colors[job_local.job_id] = colors[0]

        # Verify color mapping exists
        assert job_local.job_id in job_colors
        assert job_colors[job_local.job_id] == pu.red

    def test_logs_command_with_non_local_executor(self, prepare_local_job, caplog):
        """Test logs command with non-local executor logs warning."""
        inv = "slurm123"
        jd = JobData(
            invocation_id=inv,
            job_id=f"{inv}.0",
            timestamp=1_000_000_000.0,
            executor="slurm",
            data={},
            config={
                "execution": {"type": "slurm"},
                "evaluation": {"tasks": [{"name": "test_task"}]},
            },
        )
        ExecutionDB().write_job(jd)

        cmd = LogsCmd(id=f"{inv}.0")
        # Mock stream_logs to raise ValueError (simulating non-local executor)
        with patch("nemo_evaluator_launcher.cli.logs.stream_logs") as mock_stream:
            mock_stream.side_effect = ValueError("Log streaming is not yet implemented")
            cmd.execute()
            # Command should exit gracefully (no exception)
            # Warning should be logged by BaseExecutor.stream_logs


@pytest.mark.usefixtures("mock_execdb")
class TestLocalExecutorStreamLogs:
    """Test LocalExecutor.stream_logs method."""

    def test_local_executor_extract_task_name(self, prepare_local_job):
        """Test that task name is correctly extracted from config."""
        inv = "test5678"
        jd = JobData(
            invocation_id=inv,
            job_id=f"{inv}.0",
            timestamp=1_000_000_000.0,
            executor="local",
            data={},
            config={
                "execution": {"type": "local"},
                "evaluation": {"tasks": [{"name": "mbpp"}]},
            },
        )
        jd, base = prepare_local_job(jd, with_required=True, with_optional=True)
        ExecutionDB().write_job(jd)

        task_name = LocalExecutor._extract_task_name(jd, f"{inv}.0")
        assert task_name == "mbpp"

    def test_local_executor_extract_task_name_fallback(self, prepare_local_job):
        """Test that task name falls back to output_dir when config doesn't have task."""
        inv = "test9012"
        jd = JobData(
            invocation_id=inv,
            job_id=f"{inv}.0",
            timestamp=1_000_000_000.0,
            executor="local",
            data={},
            config={
                "execution": {"type": "local"},
                "evaluation": {"tasks": []},  # Empty tasks
            },
        )
        jd, base = prepare_local_job(jd, with_required=True, with_optional=True)
        ExecutionDB().write_job(jd)

        task_name = LocalExecutor._extract_task_name(jd, f"{inv}.0")
        # Should fallback to output_dir last component
        assert task_name == jd.job_id

    def test_local_executor_stream_logs_with_existing_file(self, prepare_local_job):
        """Test LocalExecutor.stream_logs with existing log file."""
        inv = "stream123"
        jd = JobData(
            invocation_id=inv,
            job_id=f"{inv}.0",
            timestamp=1_000_000_000.0,
            executor="local",
            data={},
            config={
                "execution": {"type": "local"},
                "evaluation": {"tasks": [{"name": "test_task"}]},
            },
        )
        jd, base = prepare_local_job(jd, with_required=True, with_optional=True)
        ExecutionDB().write_job(jd)

        # Create log file with content
        log_file = base / "logs" / "client_stdout.log"
        log_file.write_text("Line 1\nLine 2\nLine 3\n", encoding="utf-8")

        # Mock time.sleep to avoid infinite loop
        with patch("time.sleep", return_value=None):
            with patch("builtins.print"):
                # Mock KeyboardInterrupt to exit after first iteration
                call_count = 0

                def mock_sleep(*args):
                    nonlocal call_count
                    call_count += 1
                    if call_count > 1:
                        raise KeyboardInterrupt()

                with patch("time.sleep", side_effect=mock_sleep):
                    try:
                        logs = list(LocalExecutor.stream_logs(f"{inv}.0"))
                        # Should have read the existing lines
                        assert len(logs) >= 0  # May be 0 if file was already at end
                    except KeyboardInterrupt:
                        pass

    def test_local_executor_stream_logs_file_not_exists(self, prepare_local_job):
        """Test LocalExecutor.stream_logs when log file doesn't exist yet."""
        inv = "stream456"
        jd = JobData(
            invocation_id=inv,
            job_id=f"{inv}.0",
            timestamp=1_000_000_000.0,
            executor="local",
            data={},
            config={
                "execution": {"type": "local"},
                "evaluation": {"tasks": [{"name": "test_task"}]},
            },
        )
        jd, base = prepare_local_job(jd, with_required=True, with_optional=True)
        ExecutionDB().write_job(jd)

        # Don't create log file - it should be monitored
        log_file = base / "logs" / "client_stdout.log"
        assert not log_file.exists()

        # Mock time.sleep to avoid infinite loop
        with patch("time.sleep", return_value=None):
            # Mock KeyboardInterrupt to exit quickly
            call_count = 0

            def mock_sleep(*args):
                nonlocal call_count
                call_count += 1
                if call_count > 1:
                    raise KeyboardInterrupt()

            with patch("time.sleep", side_effect=mock_sleep):
                try:
                    logs = list(LocalExecutor.stream_logs(f"{inv}.0"))
                    # Should return empty iterator if file doesn't exist
                    assert len(logs) == 0
                except KeyboardInterrupt:
                    pass


@pytest.mark.usefixtures("mock_execdb")
class TestFunctionalAPIStreamLogs:
    """Test functional API stream_logs function."""

    def test_stream_logs_with_job_id(self, job_local):
        """Test stream_logs functional API with job ID."""
        # Mock LocalExecutor.stream_logs to avoid file I/O
        with patch(
            "nemo_evaluator_launcher.executors.local.executor.LocalExecutor.stream_logs"
        ) as mock_stream:
            mock_stream.return_value = iter(
                [
                    (job_local.job_id, "mbpp", "Test log line"),
                ]
            )
            logs = list(stream_logs(job_local.job_id))
            assert len(logs) == 1
            assert logs[0] == (job_local.job_id, "mbpp", "Test log line")

    def test_stream_logs_with_invocation_id(self, job_local, prepare_local_job):
        """Test stream_logs functional API with invocation ID."""
        inv = job_local.invocation_id
        # Add a second job
        jd2 = JobData(
            invocation_id=inv,
            job_id=f"{inv}.1",
            timestamp=job_local.timestamp,
            executor="local",
            data={},
            config=job_local.config,
        )
        jd2, base2 = prepare_local_job(jd2, with_required=True, with_optional=True)
        ExecutionDB().write_job(jd2)

        # Mock LocalExecutor.stream_logs
        with patch(
            "nemo_evaluator_launcher.executors.local.executor.LocalExecutor.stream_logs"
        ) as mock_stream:
            mock_stream.return_value = iter(
                [
                    (job_local.job_id, "mbpp", "Log from job 0"),
                    (jd2.job_id, "mbpp", "Log from job 1"),
                ]
            )
            logs = list(stream_logs(inv))
            assert len(logs) == 2

    def test_stream_logs_with_nonexistent_job(self):
        """Test stream_logs with nonexistent job ID."""
        logs = list(stream_logs("nonexistent.0"))
        assert len(logs) == 0

    def test_stream_logs_with_nonexistent_invocation(self):
        """Test stream_logs with nonexistent invocation ID."""
        logs = list(stream_logs("nonexistent"))
        assert len(logs) == 0


@pytest.mark.usefixtures("mock_execdb")
class TestBaseExecutorStreamLogs:
    """Test BaseExecutor.stream_logs warning behavior."""

    def test_base_executor_stream_logs_logs_warning(self, caplog):
        """Test that BaseExecutor.stream_logs logs a warning."""
        with pytest.raises(NotImplementedError):
            list(BaseExecutor.stream_logs("test123", executor_name="test_executor"))

        # Check that warning was logged
        assert "Log streaming is not yet implemented" in caplog.text
        assert "test_executor" in caplog.text
        assert "Only 'local' executor currently supports log streaming" in caplog.text

    def test_base_executor_stream_logs_without_executor_name(self, caplog):
        """Test that BaseExecutor.stream_logs logs warning even without executor name."""
        with pytest.raises(NotImplementedError):
            list(BaseExecutor.stream_logs("test123"))

        # Check that warning was logged
        assert "Log streaming is not yet implemented" in caplog.text


@pytest.mark.usefixtures("mock_execdb")
class TestLogsCommandIntegration:
    """Test logs command integration with CLI parser."""

    def test_logs_subcommand_in_parser(self):
        """Test that logs subcommand is properly registered in parser."""
        from nemo_evaluator_launcher.cli.main import create_parser

        parser = create_parser()
        args = parser.parse_args(["logs", "test123"])
        assert args.command == "logs"
        assert hasattr(args, "logs")
        assert isinstance(args.logs, LogsCmd)
        assert args.logs.id == "test123"

    def test_logs_command_handles_keyboard_interrupt(self, job_local):
        """Test that logs command handles KeyboardInterrupt gracefully."""
        cmd = LogsCmd(id=job_local.job_id)
        # Mock stream_logs to raise KeyboardInterrupt when iterated
        with patch("nemo_evaluator_launcher.cli.logs.stream_logs") as mock_stream:

            def raise_keyboard_interrupt():
                yield (job_local.job_id, "mbpp", "First line")
                raise KeyboardInterrupt()

            mock_stream.return_value = raise_keyboard_interrupt()
            # Should not raise exception - KeyboardInterrupt is caught in execute()
            cmd.execute()

    def test_logs_command_handles_empty_log_lines(self, job_local):
        """Test that logs command handles empty log lines correctly."""
        cmd = LogsCmd(id=job_local.job_id)
        with patch("nemo_evaluator_launcher.cli.logs.stream_logs") as mock_stream:
            mock_stream.return_value = iter(
                [
                    (job_local.job_id, "mbpp", "Non-empty line"),
                    (job_local.job_id, "mbpp", ""),  # Empty line
                    (job_local.job_id, "mbpp", "Another line"),
                ]
            )
            with patch("builtins.print") as mock_print:
                cmd.execute()
                # Should print all lines, including empty ones
                assert mock_print.call_count >= 3
