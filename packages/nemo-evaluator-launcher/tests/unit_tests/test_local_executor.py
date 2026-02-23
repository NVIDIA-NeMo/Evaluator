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
"""Tests for LocalExecutor implementation."""

import os
import pathlib
import re
import time
from unittest.mock import patch

import pytest
import yaml
from omegaconf import OmegaConf

from nemo_evaluator_launcher.common.execdb import ExecutionDB, JobData
from nemo_evaluator_launcher.executors.base import ExecutionState
from nemo_evaluator_launcher.executors.local.executor import LocalExecutor


class TestLocalExecutorDryRun:
    """Test LocalExecutor dry run functionality."""

    @pytest.fixture
    def sample_config(self, tmpdir):
        """Create a sample configuration for testing."""
        config_dict = {
            "deployment": {"type": "none"},
            "execution": {
                "type": "local",
                "output_dir": str(tmpdir / "test_output"),
                "auto_export": {"destinations": ["local", "wandb"]},
            },
            "target": {
                "api_endpoint": {
                    "api_key_name": "TEST_API_KEY",
                    "model_id": "test_model",
                    "url": "https://test.api.com/v1/chat/completions",
                }
            },
            "evaluation": {
                "env_vars": {"GLOBAL_ENV": "host:GLOBAL_VALUE"},
                "tasks": [
                    {
                        "name": "test_task_1",
                        "env_vars": {"TASK_ENV": "host:TASK_VALUE"},
                        "nemo_evaluator_config": {
                            "config": {"params": {"param1": "value1"}}
                        },
                    },
                    {
                        "name": "test_task_2",
                        "container": "custom-container:v2.0",
                        "nemo_evaluator_config": {
                            "config": {"params": {"param2": "value2"}}
                        },
                    },
                ],
            },
        }
        return OmegaConf.create(config_dict)

    @pytest.fixture
    def mock_tasks_mapping(self):
        """Mock tasks mapping for testing."""
        return {
            ("lm-eval", "test_task_1"): {
                "task": "test_task_1",
                "endpoint_type": "openai",
                "harness": "lm-eval",
                "container": "test-container:latest",
                "required_env_vars": ["TASK_ENV"],
            },
            ("helm", "test_task_2"): {
                "task": "test_task_2",
                "endpoint_type": "anthropic",
                "harness": "helm",
                "container": "test-container:latest",
            },
        }

    @pytest.fixture
    def test_execute_eval_dry_run_basic(
        self, mock_execdb, sample_config, mock_tasks_mapping, tmpdir
    ):
        """Test basic dry run execution."""
        # Set up environment variable that the config references
        os.environ["TEST_API_KEY"] = "test_key_value"
        os.environ["GLOBAL_VALUE"] = "global_env_value"
        os.environ["TASK_VALUE"] = "task_env_value"

        try:
            with (
                patch(
                    "nemo_evaluator_launcher.executors.local.executor.load_tasks_mapping"
                ) as mock_load_mapping,
                patch(
                    "nemo_evaluator_launcher.executors.local.executor.get_task_definition_for_job"
                ) as mock_get_task_def,
                patch(
                    "nemo_evaluator_launcher.executors.local.executor.get_eval_factory_command"
                ) as mock_get_command,
                patch("builtins.print") as mock_print,
            ):
                # Configure mocks
                mock_load_mapping.return_value = mock_tasks_mapping

                def mock_get_task_def_side_effect(*_args, **kwargs):
                    task_name = kwargs.get("task_query")
                    mapping = kwargs.get("base_mapping", {})
                    for (_harness, name), definition in mapping.items():
                        if name == task_name:
                            return definition
                    raise KeyError(f"Task {task_name} not found")

                mock_get_task_def.side_effect = mock_get_task_def_side_effect
                from nemo_evaluator_launcher.common.helpers import CmdAndReadableComment

                mock_get_command.return_value = CmdAndReadableComment(
                    cmd="nemo-evaluator-launcher --task test_command",
                    debug="# Test command for local executor",
                )

                # Execute dry run
                invocation_id = LocalExecutor.execute_eval(sample_config, dry_run=True)

                # Verify invocation ID format
                assert isinstance(invocation_id, str)
                assert len(invocation_id) == 16
                assert re.match(r"^[a-f0-9]{16}$", invocation_id)

                # Verify output directory structure was created
                output_base = pathlib.Path(sample_config.execution.output_dir)
                assert any(
                    dir_name.endswith(f"-{invocation_id}")
                    for dir_name in os.listdir(output_base)
                )

                # Find the actual output directory
                output_dir = None
                for item in output_base.iterdir():
                    if item.is_dir() and item.name.endswith(f"-{invocation_id}"):
                        output_dir = item
                        break

                assert output_dir is not None

                # Verify task directories were created
                task1_dir = output_dir / "test_task_1"
                task2_dir = output_dir / "test_task_2"
                assert task1_dir.exists()
                assert task2_dir.exists()

                # Verify scripts were created
                run_all_sequential_script = output_dir / "run_all.sequential.sh"
                task1_script = task1_dir / "run.sh"
                task2_script = task2_dir / "run.sh"

                assert run_all_sequential_script.exists()
                assert task1_script.exists()
                assert task2_script.exists()

                # Verify print was called with dry run information
                mock_print.assert_called()
                print_calls = [
                    call.args[0] for call in mock_print.call_args_list if call.args
                ]

                # Check that dry run message was printed
                dry_run_messages = [msg for msg in print_calls if "DRY RUN" in str(msg)]
                assert len(dry_run_messages) > 0

        finally:
            # Clean up environment
            for env_var in ["TEST_API_KEY", "GLOBAL_VALUE", "TASK_VALUE"]:
                if env_var in os.environ:
                    del os.environ[env_var]

    def test_execute_eval_dry_run_env_var_validation(
        self, sample_config, mock_tasks_mapping
    ):
        """Test that missing environment variables are properly validated."""
        # Don't set the required environment variables

        with (
            patch(
                "nemo_evaluator_launcher.executors.local.executor.load_tasks_mapping"
            ) as mock_load_mapping,
            patch(
                "nemo_evaluator_launcher.executors.local.executor.get_task_definition_for_job"
            ) as mock_get_task_def,
        ):
            mock_load_mapping.return_value = mock_tasks_mapping

            def mock_get_task_def_side_effect(*_args, **kwargs):
                task_name = kwargs.get("task_query")
                mapping = kwargs.get("base_mapping", {})
                for (_harness, name), definition in mapping.items():
                    if name == task_name:
                        return definition
                raise KeyError(f"Task {task_name} not found")

            mock_get_task_def.side_effect = mock_get_task_def_side_effect

            # Should raise ValueError for missing API key
            with pytest.raises(ValueError, match="is not set"):
                LocalExecutor.execute_eval(sample_config, dry_run=True)

    def test_execute_eval_dry_run_required_task_env_vars(
        self, sample_config, mock_tasks_mapping
    ):
        """Test validation of required task-specific environment variables."""
        # Set some but not all required env vars
        os.environ["TEST_API_KEY"] = "test_key_value"
        os.environ["GLOBAL_VALUE"] = "global_env_value"
        # Missing TASK_VALUE for test_task_1

        try:
            with (
                patch(
                    "nemo_evaluator_launcher.executors.local.executor.load_tasks_mapping"
                ) as mock_load_mapping,
                patch(
                    "nemo_evaluator_launcher.executors.local.executor.get_task_definition_for_job"
                ) as mock_get_task_def,
            ):
                mock_load_mapping.return_value = mock_tasks_mapping

                def mock_get_task_def_side_effect(*_args, **kwargs):
                    task_name = kwargs.get("task_query")
                    mapping = kwargs.get("base_mapping", {})
                    for (_harness, name), definition in mapping.items():
                        if name == task_name:
                            return definition
                    raise KeyError(f"Task {task_name} not found")

                mock_get_task_def.side_effect = mock_get_task_def_side_effect

                # Should raise ValueError for missing environment variable TASK_VALUE
                # (which is the value of TASK_ENV in the configuration)
                with pytest.raises(
                    ValueError,
                    match="TASK_VALUE.*is not set",
                ):
                    LocalExecutor.execute_eval(sample_config, dry_run=True)

        finally:
            # Clean up environment
            for env_var in ["TEST_API_KEY", "GLOBAL_VALUE"]:
                if env_var in os.environ:
                    del os.environ[env_var]


class TestLocalExecutorGetStatus:
    """Test LocalExecutor get_status functionality."""

    @pytest.fixture
    def sample_job_data(self, tmpdir) -> JobData:
        """Create sample job data for testing."""
        output_dir = tmpdir / "test_job_output"
        output_dir.mkdir()

        return JobData(
            invocation_id="abc12345",
            job_id="abc12345.0",
            timestamp=time.time(),
            executor="local",
            data={
                "output_dir": str(output_dir),
                "container": "test-container-name",
                "eval_image": "test-image:latest",
            },
            config={},
        )

    @pytest.fixture
    def setup_job_filesystem(self, sample_job_data):
        """Set up the job filesystem structure for testing."""

        def _setup(
            stage_files: dict = None,
            progress_value: int = None,
            dataset_size: int = None,
            killed: bool = False,
        ):
            output_dir = pathlib.Path(sample_job_data.data["output_dir"])
            artifacts_dir = output_dir / "artifacts"
            logs_dir = output_dir / "logs"

            artifacts_dir.mkdir(parents=True, exist_ok=True)
            logs_dir.mkdir(parents=True, exist_ok=True)

            # Create stage files if provided
            if stage_files:
                for stage, content in stage_files.items():
                    stage_file = logs_dir / f"stage.{stage}"
                    stage_file.write_text(content)

            # Create progress file if provided
            if progress_value is not None:
                progress_file = artifacts_dir / "progress"
                progress_file.write_text(str(progress_value))

            # Create dataset size file if provided
            if dataset_size is not None:
                run_config_file = artifacts_dir / "run_config.yml"
                config_data = {"config": {"params": {"limit_samples": dataset_size}}}
                run_config_file.write_text(yaml.dump(config_data))

            # Mark as killed if requested
            if killed:
                sample_job_data.data["killed"] = True

            return output_dir, artifacts_dir, logs_dir

        return _setup

    def test_get_status_invocation_id(
        self, mock_execdb, sample_job_data, setup_job_filesystem
    ):
        """Test get_status with invocation ID (multiple jobs)."""
        # Set up filesystem for first job
        setup_job_filesystem(
            stage_files={"exit": "2025-01-01T12:00:00Z 0"},
            progress_value=50,
            dataset_size=100,
        )

        # Create second job data
        job_data2 = JobData(
            invocation_id="abc12345",
            job_id="abc12345.1",
            timestamp=time.time(),
            executor="local",
            data={
                "output_dir": str(
                    pathlib.Path(sample_job_data.data["output_dir"]).parent / "job2"
                ),
                "container": "test-container-2",
                "eval_image": "test-image:latest",
            },
            config={},
        )

        # Set up second job filesystem
        output_dir2 = pathlib.Path(job_data2.data["output_dir"])
        output_dir2.mkdir(parents=True, exist_ok=True)
        (output_dir2 / "artifacts").mkdir(parents=True, exist_ok=True)
        logs_dir2 = output_dir2 / "logs"
        logs_dir2.mkdir(parents=True, exist_ok=True)
        (logs_dir2 / "stage.running").write_text("2025-01-01T12:00:00Z")

        # Mock database calls
        db = ExecutionDB()
        db.write_job(sample_job_data)
        db.write_job(job_data2)

        # Test
        statuses = LocalExecutor.get_status("abc12345")

        assert len(statuses) == 2
        assert statuses[0].id == "abc12345.0"
        assert statuses[0].state == ExecutionState.SUCCESS
        assert statuses[0].progress["progress"] == 0.5

        assert statuses[1].id == "abc12345.1"
        assert statuses[1].state == ExecutionState.RUNNING

    def test_get_status_job_not_found(self):
        """Test get_status with non-existent job ID."""
        statuses = LocalExecutor.get_status("nonexistent.0")
        assert statuses == []

    def test_get_status_wrong_executor(self, mock_execdb, sample_job_data):
        """Test get_status with job from different executor."""
        # Change executor to something else
        sample_job_data.executor = "slurm"

        db = ExecutionDB()
        db.write_job(sample_job_data)

        statuses = LocalExecutor.get_status("abc12345.0")
        assert statuses == []

    def test_get_status_output_dir_missing(self, mock_execdb, sample_job_data):
        """Test get_status when output directory doesn't exist."""
        # Set output_dir to non-existent path
        sample_job_data.data["output_dir"] = "/nonexistent/path"

        db = ExecutionDB()
        db.write_job(sample_job_data)

        statuses = LocalExecutor.get_status("abc12345.0")
        assert len(statuses) == 1
        assert statuses[0].state == ExecutionState.PENDING

    def test_get_status_logs_dir_missing(
        self, mock_execdb, sample_job_data, setup_job_filesystem
    ):
        """Test get_status when logs directory doesn't exist."""
        output_dir, artifacts_dir, _ = setup_job_filesystem(progress_value=25)

        # Remove logs directory
        logs_dir = output_dir / "logs"
        if logs_dir.exists():
            import shutil

            shutil.rmtree(logs_dir)

        db = ExecutionDB()
        db.write_job(sample_job_data)

        statuses = LocalExecutor.get_status("abc12345.0")
        assert len(statuses) == 1
        assert statuses[0].state == ExecutionState.PENDING
        assert statuses[0].progress["progress"] == 25

    def test_get_status_killed_job(
        self, mock_execdb, sample_job_data, setup_job_filesystem
    ):
        """Test get_status for a killed job."""
        setup_job_filesystem(
            stage_files={"running": "2025-01-01T12:00:00Z"},
            progress_value=30,
            killed=True,
        )

        db = ExecutionDB()
        db.write_job(sample_job_data)

        statuses = LocalExecutor.get_status("abc12345.0")
        assert len(statuses) == 1
        assert statuses[0].state == ExecutionState.KILLED
        assert statuses[0].progress["progress"] == 30

    def test_get_status_success_with_exit_code_0(
        self, mock_execdb, sample_job_data, setup_job_filesystem
    ):
        """Test get_status for successfully completed job."""
        setup_job_filesystem(
            stage_files={"exit": "2025-01-01T12:00:00Z 0"},
            progress_value=100,
            dataset_size=100,
        )

        db = ExecutionDB()
        db.write_job(sample_job_data)

        statuses = LocalExecutor.get_status("abc12345.0")
        assert len(statuses) == 1
        assert statuses[0].state == ExecutionState.SUCCESS
        assert statuses[0].progress["progress"] == 1.0

    def test_get_status_failed_with_nonzero_exit_code(
        self, mock_execdb, sample_job_data, setup_job_filesystem
    ):
        """Test get_status for job that failed with non-zero exit code."""
        setup_job_filesystem(
            stage_files={"exit": "2025-01-01T12:00:00Z 1"},
            progress_value=75,
            dataset_size=100,
        )

        db = ExecutionDB()
        db.write_job(sample_job_data)

        statuses = LocalExecutor.get_status("abc12345.0")
        assert len(statuses) == 1
        assert statuses[0].state == ExecutionState.FAILED
        assert statuses[0].progress["progress"] == 0.75

    def test_get_status_malformed_exit_file(
        self, mock_execdb, sample_job_data, setup_job_filesystem
    ):
        """Test get_status with malformed exit file content."""
        setup_job_filesystem(stage_files={"exit": "invalid content"}, progress_value=40)

        db = ExecutionDB()
        db.write_job(sample_job_data)

        statuses = LocalExecutor.get_status("abc12345.0")
        assert len(statuses) == 1
        assert statuses[0].state == ExecutionState.FAILED
        assert statuses[0].progress["progress"] == 40

    def test_get_status_exit_file_no_space(
        self, mock_execdb, sample_job_data, setup_job_filesystem
    ):
        """Test get_status with exit file content without space separator."""
        setup_job_filesystem(stage_files={"exit": "nospace"}, progress_value=60)

        db = ExecutionDB()
        db.write_job(sample_job_data)

        statuses = LocalExecutor.get_status("abc12345.0")
        assert len(statuses) == 1
        assert statuses[0].state == ExecutionState.FAILED
        assert statuses[0].progress["progress"] == 60

    def test_get_status_running(
        self, mock_execdb, sample_job_data, setup_job_filesystem
    ):
        """Test get_status for a running job."""
        setup_job_filesystem(
            stage_files={"running": "2025-01-01T12:00:00Z"},
            progress_value=35,
            dataset_size=200,
        )

        db = ExecutionDB()
        db.write_job(sample_job_data)

        statuses = LocalExecutor.get_status("abc12345.0")
        assert len(statuses) == 1
        assert statuses[0].state == ExecutionState.RUNNING
        assert statuses[0].progress["progress"] == 0.175  # 35/200

    def test_get_status_after_pre_start(
        self, mock_execdb, sample_job_data, setup_job_filesystem
    ):
        """Test get_status for job after pre_start stage and before running stage."""
        setup_job_filesystem(
            stage_files={"pre-start": "2025-01-01T12:00:00Z"}, progress_value=0
        )

        db = ExecutionDB()
        db.write_job(sample_job_data)

        statuses = LocalExecutor.get_status("abc12345.0")
        assert len(statuses) == 1
        assert statuses[0].state == ExecutionState.PENDING
        assert statuses[0].progress["progress"] == 0

    def test_get_status_no_stage_files(
        self, mock_execdb, sample_job_data, setup_job_filesystem
    ):
        """Test get_status when no stage files exist."""
        setup_job_filesystem(progress_value=10)

        db = ExecutionDB()
        db.write_job(sample_job_data)

        statuses = LocalExecutor.get_status("abc12345.0")
        assert len(statuses) == 1
        assert statuses[0].state == ExecutionState.PENDING
        assert statuses[0].progress["progress"] == 10

    def test_get_status_no_progress_file(
        self, mock_execdb, sample_job_data, setup_job_filesystem
    ):
        """Test get_status when progress file doesn't exist."""
        setup_job_filesystem(stage_files={"running": "2025-01-01T12:00:00Z"})

        db = ExecutionDB()
        db.write_job(sample_job_data)

        statuses = LocalExecutor.get_status("abc12345.0")
        assert len(statuses) == 1
        assert statuses[0].state == ExecutionState.RUNNING
        assert statuses[0].progress["progress"] is None

    def test_get_status_invalid_progress_file(
        self, mock_execdb, sample_job_data, setup_job_filesystem
    ):
        """Test get_status with invalid progress file content."""
        output_dir, artifacts_dir, logs_dir = setup_job_filesystem(
            stage_files={"running": "2025-01-01T12:00:00Z"}
        )

        # Create invalid progress file
        progress_file = artifacts_dir / "progress"
        progress_file.write_text("not_a_number")

        db = ExecutionDB()
        db.write_job(sample_job_data)

        statuses = LocalExecutor.get_status("abc12345.0")
        assert len(statuses) == 1
        assert statuses[0].state == ExecutionState.RUNNING
        assert statuses[0].progress["progress"] is None

    def test_get_status_progress_without_dataset_size(
        self, mock_execdb, sample_job_data, setup_job_filesystem
    ):
        """Test get_status with progress but no dataset size."""
        setup_job_filesystem(
            stage_files={"running": "2025-01-01T12:00:00Z"}, progress_value=42
        )

        db = ExecutionDB()
        db.write_job(sample_job_data)

        with patch(
            "nemo_evaluator_launcher.executors.local.executor.get_eval_factory_dataset_size_from_run_config"
        ) as mock_get_size:
            mock_get_size.return_value = None

            statuses = LocalExecutor.get_status("abc12345.0")
            assert len(statuses) == 1
            assert statuses[0].state == ExecutionState.RUNNING
            # When no dataset size, progress should be the raw number of processed samples
            assert statuses[0].progress["progress"] == 42


class TestLocalExecutorStreamLogs:
    """Test LocalExecutor stream_logs functionality."""

    @pytest.fixture
    def sample_job_for_logs(self, tmpdir):
        """Create a sample job for log streaming tests."""
        invocation_id = "test1234"
        job_id = f"{invocation_id}.0"
        output_dir = pathlib.Path(tmpdir) / invocation_id / job_id
        logs_dir = output_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        job_data = JobData(
            invocation_id=invocation_id,
            job_id=job_id,
            timestamp=1_000_000_000.0,
            executor="local",
            data={"output_dir": str(output_dir)},
            config={
                "execution": {"type": "local"},
                "evaluation": {"tasks": [{"name": "test_task"}]},
            },
        )
        ExecutionDB().write_job(job_data)
        return job_data, output_dir, logs_dir

    def test_stream_logs_with_existing_file(self, sample_job_for_logs):
        """Test streaming logs from an existing file."""
        job_data, output_dir, logs_dir = sample_job_for_logs
        log_file = logs_dir / "client_stdout.log"
        log_file.write_text("Line 1\nLine 2\nLine 3\n", encoding="utf-8")

        # Mock time.sleep to avoid infinite loop
        with patch("time.sleep", return_value=None):
            call_count = 0

            def mock_sleep(*args):
                nonlocal call_count
                call_count += 1
                if call_count > 1:
                    raise KeyboardInterrupt()

            with patch("time.sleep", side_effect=mock_sleep):
                try:
                    logs = list(LocalExecutor.stream_logs(job_data.job_id))
                    # Should have read existing lines (if file wasn't already at end)
                    assert isinstance(logs, list)
                except KeyboardInterrupt:
                    pass

    def test_stream_logs_file_not_exists(self, sample_job_for_logs):
        """Test streaming logs when file doesn't exist yet."""
        job_data, output_dir, logs_dir = sample_job_for_logs
        log_file = logs_dir / "client_stdout.log"
        assert not log_file.exists()

        # Mock time.sleep to avoid infinite loop
        with patch("time.sleep", return_value=None):
            call_count = 0

            def mock_sleep(*args):
                nonlocal call_count
                call_count += 1
                if call_count > 1:
                    raise KeyboardInterrupt()

            with patch("time.sleep", side_effect=mock_sleep):
                try:
                    logs = list(LocalExecutor.stream_logs(job_data.job_id))
                    # Should return empty if file doesn't exist
                    assert len(logs) == 0
                except KeyboardInterrupt:
                    pass

    def test_stream_logs_with_invocation_id(
        self, sample_job_for_logs, prepare_local_job
    ):
        """Test streaming logs with invocation ID containing multiple jobs."""
        job_data, output_dir, logs_dir = sample_job_for_logs
        inv = job_data.invocation_id

        # Update first job config to have 2 tasks (for consistency with second job)
        config_with_two_tasks = {
            "execution": {"type": "local"},
            "evaluation": {"tasks": [{"name": "test_task"}, {"name": "test_task2"}]},
        }
        job_data.config = config_with_two_tasks
        ExecutionDB().write_job(job_data)

        # Add a second job with matching config
        jd2 = JobData(
            invocation_id=inv,
            job_id=f"{inv}.1",
            timestamp=job_data.timestamp,
            executor="local",
            data={},
            config=config_with_two_tasks,
        )
        jd2, base2 = prepare_local_job(jd2, with_required=True, with_optional=True)
        ExecutionDB().write_job(jd2)

        # Mock time.sleep to avoid infinite loop
        with patch("time.sleep", return_value=None):
            call_count = 0

            def mock_sleep(*args):
                nonlocal call_count
                call_count += 1
                if call_count > 1:
                    raise KeyboardInterrupt()

            with patch("time.sleep", side_effect=mock_sleep):
                try:
                    logs = list(LocalExecutor.stream_logs(inv))
                    # Should handle multiple jobs
                    assert isinstance(logs, list)
                except KeyboardInterrupt:
                    pass

    def test_stream_logs_nonexistent_job(self):
        """Test streaming logs for nonexistent job."""
        logs = list(LocalExecutor.stream_logs("nonexistent.0"))
        assert len(logs) == 0

    def test_extract_task_name_from_config(self, sample_job_for_logs):
        """Test extracting task name from config."""
        job_data, output_dir, logs_dir = sample_job_for_logs
        task_name = LocalExecutor._extract_task_name(job_data, job_data.job_id)
        assert task_name == "test_task"

    def test_extract_task_name_fallback(self, tmpdir):
        """Test extracting task name falls back to output_dir."""
        invocation_id = "test5678"
        job_id = f"{invocation_id}.0"
        output_dir = pathlib.Path(tmpdir) / invocation_id / "some_task_name"
        output_dir.mkdir(parents=True, exist_ok=True)

        job_data = JobData(
            invocation_id=invocation_id,
            job_id=job_id,
            timestamp=1_000_000_000.0,
            executor="local",
            data={"output_dir": str(output_dir)},
            config={
                "execution": {"type": "local"},
                "evaluation": {"tasks": []},  # Empty tasks
            },
        )

        task_name = LocalExecutor._extract_task_name(job_data, job_id)
        assert task_name == "some_task_name"
