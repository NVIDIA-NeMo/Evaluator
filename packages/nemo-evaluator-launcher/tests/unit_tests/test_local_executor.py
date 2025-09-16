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
from unittest.mock import mock_open, patch

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
                "env_vars": {"GLOBAL_ENV": "GLOBAL_VALUE"},
                "tasks": [
                    {
                        "name": "test_task_1",
                        "env_vars": {"TASK_ENV": "TASK_VALUE"},
                        "overrides": {"param1": "value1"},
                    },
                    {
                        "name": "test_task_2",
                        "container": "custom-container:v2.0",
                        "overrides": {"param2": "value2"},
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
    def mock_run_template(self):
        """Mock run template content."""
        return """
# Mock template for testing
{% for task in evaluation_tasks %}
echo "Running task: {{ task.job_id }}"
echo "Container: {{ task.container_name }}"
echo "Image: {{ task.eval_image }}"
echo "Command: {{ task.eval_factory_command }}"
echo "Output dir: {{ task.output_dir }}"
{% for env_var in task.env_vars %}
echo "Env: {{ env_var }}"
{% endfor %}
{% if auto_export_destinations %}
echo "Auto-export destinations: {{ auto_export_destinations|join(',') }}"
{% endif %}
echo "Invocation ID: {{ invocation_id }}"
{% endfor %}
""".strip()

    def test_execute_eval_dry_run_basic(
        self, mock_execdb, sample_config, mock_tasks_mapping, mock_run_template, tmpdir
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
                    "nemo_evaluator_launcher.executors.local.executor.get_task_from_mapping"
                ) as mock_get_task,
                patch(
                    "nemo_evaluator_launcher.executors.local.executor.get_eval_factory_command"
                ) as mock_get_command,
                patch("builtins.open", mock_open(read_data=mock_run_template)),
                patch("builtins.print") as mock_print,
            ):
                # Configure mocks
                mock_load_mapping.return_value = mock_tasks_mapping

                def mock_get_task_side_effect(task_name, mapping):
                    # Return matching task definition
                    for (harness, name), definition in mapping.items():
                        if name == task_name:
                            return definition
                    raise KeyError(f"Task {task_name} not found")

                mock_get_task.side_effect = mock_get_task_side_effect
                mock_get_command.return_value = "nemo-evaluator-launcher --task test_command"

                # Execute dry run
                invocation_id = LocalExecutor.execute_eval(sample_config, dry_run=True)

                # Verify invocation ID format
                assert isinstance(invocation_id, str)
                assert len(invocation_id) == 8
                assert re.match(r"^[a-f0-9]{8}$", invocation_id)

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

                # Verify database entries were created
                db = ExecutionDB()
                jobs = db.get_jobs(invocation_id)
                assert len(jobs) == 2  # Two tasks

                # Verify job data structure
                for job_id, job_data in jobs.items():
                    assert job_data.invocation_id == invocation_id
                    assert job_data.executor == "local"
                    assert "output_dir" in job_data.data
                    assert "container" in job_data.data
                    assert "eval_image" in job_data.data

        finally:
            # Clean up environment
            for env_var in ["TEST_API_KEY", "GLOBAL_VALUE", "TASK_VALUE"]:
                if env_var in os.environ:
                    del os.environ[env_var]

    def test_execute_eval_dry_run_env_var_validation(
        self, sample_config, mock_tasks_mapping, mock_run_template
    ):
        """Test that missing environment variables are properly validated."""
        # Don't set the required environment variables

        with (
            patch(
                "nemo_evaluator_launcher.executors.local.executor.load_tasks_mapping"
            ) as mock_load_mapping,
            patch(
                "nemo_evaluator_launcher.executors.local.executor.get_task_from_mapping"
            ) as mock_get_task,
            patch("builtins.open", mock_open(read_data=mock_run_template)),
        ):
            mock_load_mapping.return_value = mock_tasks_mapping

            def mock_get_task_side_effect(task_name, mapping):
                for (harness, name), definition in mapping.items():
                    if name == task_name:
                        return definition
                raise KeyError(f"Task {task_name} not found")

            mock_get_task.side_effect = mock_get_task_side_effect

            # Should raise ValueError for missing API key
            with pytest.raises(
                ValueError, match="Trying to pass an unset environment variable"
            ):
                LocalExecutor.execute_eval(sample_config, dry_run=True)

    def test_execute_eval_dry_run_required_task_env_vars(
        self, sample_config, mock_tasks_mapping, mock_run_template
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
                    "nemo_evaluator_launcher.executors.local.executor.get_task_from_mapping"
                ) as mock_get_task,
                patch("builtins.open", mock_open(read_data=mock_run_template)),
            ):
                mock_load_mapping.return_value = mock_tasks_mapping

                def mock_get_task_side_effect(task_name, mapping):
                    for (harness, name), definition in mapping.items():
                        if name == task_name:
                            return definition
                    raise KeyError(f"Task {task_name} not found")

                mock_get_task.side_effect = mock_get_task_side_effect

                # Should raise ValueError for missing environment variable TASK_VALUE
                # (which is the value of TASK_ENV in the configuration)
                with pytest.raises(
                    ValueError,
                    match="Trying to pass an unset environment variable TASK_VALUE",
                ):
                    LocalExecutor.execute_eval(sample_config, dry_run=True)

        finally:
            # Clean up environment
            for env_var in ["TEST_API_KEY", "GLOBAL_VALUE"]:
                if env_var in os.environ:
                    del os.environ[env_var]

    def test_execute_eval_dry_run_custom_container(
        self, mock_execdb, sample_config, mock_tasks_mapping, mock_run_template, tmpdir
    ):
        """Test that custom container images are handled correctly."""
        # Set up all required environment variables
        os.environ["TEST_API_KEY"] = "test_key_value"
        os.environ["GLOBAL_VALUE"] = "global_env_value"
        os.environ["TASK_VALUE"] = "task_env_value"

        try:
            with (
                patch(
                    "nemo_evaluator_launcher.executors.local.executor.load_tasks_mapping"
                ) as mock_load_mapping,
                patch(
                    "nemo_evaluator_launcher.executors.local.executor.get_task_from_mapping"
                ) as mock_get_task,
                patch(
                    "nemo_evaluator_launcher.executors.local.executor.get_eval_factory_command"
                ) as mock_get_command,
                patch("builtins.open", mock_open(read_data=mock_run_template)),
                patch("builtins.print"),
            ):
                mock_load_mapping.return_value = mock_tasks_mapping

                def mock_get_task_side_effect(task_name, mapping):
                    for (harness, name), definition in mapping.items():
                        if name == task_name:
                            return definition
                    raise KeyError(f"Task {task_name} not found")

                mock_get_task.side_effect = mock_get_task_side_effect
                mock_get_command.return_value = "nemo-evaluator-launcher --task test_command"

                # Execute dry run
                invocation_id = LocalExecutor.execute_eval(sample_config, dry_run=True)

                # Verify database entries include correct container images
                db = ExecutionDB()
                jobs = db.get_jobs(invocation_id)

                # Find job data for each task
                task1_job = None
                task2_job = None
                for job_id, job_data in jobs.items():
                    if "test_task_1" in job_data.data["output_dir"]:
                        task1_job = job_data
                    elif "test_task_2" in job_data.data["output_dir"]:
                        task2_job = job_data

                assert task1_job is not None
                assert task2_job is not None

                # test_task_1 should use default container from mapping
                assert "test-container:latest" in task1_job.data["eval_image"]

                # test_task_2 should use custom container from config
                assert task2_job.data["eval_image"] == "custom-container:v2.0"

        finally:
            # Clean up environment
            for env_var in ["TEST_API_KEY", "GLOBAL_VALUE", "TASK_VALUE"]:
                if env_var in os.environ:
                    del os.environ[env_var]

    def test_execute_eval_deployment_type_validation(self, sample_config):
        """Test that non-'none' deployment types raise NotImplementedError."""
        # Change deployment type to something other than 'none'
        sample_config.deployment.type = "slurm"

        with pytest.raises(NotImplementedError, match="type slurm is not implemented"):
            LocalExecutor.execute_eval(sample_config, dry_run=True)

    def test_execute_eval_dry_run_no_auto_export(
        self, sample_config, mock_tasks_mapping, mock_run_template, tmpdir
    ):
        """Test dry run without auto-export configuration."""
        # Remove auto_export from config
        del sample_config.execution.auto_export

        # Set up environment variables
        os.environ["TEST_API_KEY"] = "test_key_value"
        os.environ["GLOBAL_VALUE"] = "global_env_value"
        os.environ["TASK_VALUE"] = "task_env_value"

        try:
            with (
                patch(
                    "nemo_evaluator_launcher.executors.local.executor.load_tasks_mapping"
                ) as mock_load_mapping,
                patch(
                    "nemo_evaluator_launcher.executors.local.executor.get_task_from_mapping"
                ) as mock_get_task,
                patch(
                    "nemo_evaluator_launcher.executors.local.executor.get_eval_factory_command"
                ) as mock_get_command,
                patch("builtins.open", mock_open(read_data=mock_run_template)),
                patch("builtins.print"),
            ):
                mock_load_mapping.return_value = mock_tasks_mapping

                def mock_get_task_side_effect(task_name, mapping):
                    for (harness, name), definition in mapping.items():
                        if name == task_name:
                            return definition
                    raise KeyError(f"Task {task_name} not found")

                mock_get_task.side_effect = mock_get_task_side_effect
                mock_get_command.return_value = "nemo-evaluator-launcher --task test_command"

                # Should execute successfully without auto-export
                invocation_id = LocalExecutor.execute_eval(sample_config, dry_run=True)

                # Verify invocation ID is valid
                assert isinstance(invocation_id, str)
                assert len(invocation_id) == 8

        finally:
            # Clean up environment
            for env_var in ["TEST_API_KEY", "GLOBAL_VALUE", "TASK_VALUE"]:
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
