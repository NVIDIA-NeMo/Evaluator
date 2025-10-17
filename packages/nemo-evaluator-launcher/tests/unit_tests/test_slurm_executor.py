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
"""Tests for the SLURM executor functionality."""

import os
import re
import time
import warnings
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from omegaconf import OmegaConf

from nemo_evaluator_launcher.common.execdb import ExecutionDB, JobData
from nemo_evaluator_launcher.executors.base import ExecutionState, ExecutionStatus
from nemo_evaluator_launcher.executors.slurm.executor import (
    SlurmExecutor,
    _create_slurm_sbatch_script,
)


class TestSlurmExecutorFeatures:
    """Test new SLURM executor functionality added in the recent changes."""

    @pytest.fixture
    def base_config(self):
        """Base configuration for testing."""
        return {
            "deployment": {
                "type": "vllm",
                "image": "test-image:latest",
                "command": "test-command",
                "served_model_name": "test-model",
            },
            "execution": {
                "type": "slurm",
                "output_dir": "/test/output",
                "walltime": "01:00:00",
                "account": "test-account",
                "partition": "test-partition",
                "num_nodes": 1,
                "ntasks_per_node": 1,
                "subproject": "test-subproject",
            },
            "evaluation": {"env_vars": {}},
            "target": {"api_endpoint": {"url": "http://localhost:8000/v1"}},
        }

    @pytest.fixture
    def mock_task(self):
        """Mock task configuration."""
        return OmegaConf.create({"name": "test_task"})

    @pytest.fixture
    def mock_task_definition(self):
        """Mock task definition."""
        return {
            "container": "test-eval-container:latest",
            "required_env_vars": [],
        }

    @pytest.fixture
    def mock_dependencies(self):
        """Mock external dependencies used by _create_slurm_sbatch_script."""
        with (
            patch(
                "nemo_evaluator_launcher.executors.slurm.executor.load_tasks_mapping"
            ) as mock_load_tasks,
            patch(
                "nemo_evaluator_launcher.executors.slurm.executor.get_task_from_mapping"
            ) as mock_get_task,
            patch(
                "nemo_evaluator_launcher.executors.slurm.executor.get_health_url"
            ) as mock_get_health,
            patch(
                "nemo_evaluator_launcher.executors.slurm.executor.get_endpoint_url"
            ) as mock_get_endpoint,
            patch(
                "nemo_evaluator_launcher.common.helpers.get_eval_factory_command"
            ) as mock_get_eval_command,
            patch(
                "nemo_evaluator_launcher.common.helpers.get_served_model_name"
            ) as mock_get_model_name,
        ):
            mock_load_tasks.return_value = {}
            mock_get_task.return_value = {
                "container": "test-eval-container:latest",
                "required_env_vars": [],
                "endpoint_type": "openai",
                "task": "test_task",
            }
            mock_get_health.return_value = "http://localhost:8000/health"
            mock_get_endpoint.return_value = "http://localhost:8000/v1"
            mock_get_eval_command.return_value = "nemo-evaluator run_eval --test"
            mock_get_model_name.return_value = "test-model"

            yield {
                "load_tasks_mapping": mock_load_tasks,
                "get_task_from_mapping": mock_get_task,
                "get_health_url": mock_get_health,
                "get_endpoint_url": mock_get_endpoint,
                "get_eval_factory_command": mock_get_eval_command,
                "get_served_model_name": mock_get_model_name,
            }

    def test_new_execution_env_vars_deployment(
        self, base_config, mock_task, mock_dependencies
    ):
        """Test new execution.env_vars.deployment configuration."""
        # Add new env_vars structure
        base_config["execution"]["env_vars"] = {
            "deployment": {
                "DEPLOY_VAR1": "deploy_value1",
                "DEPLOY_VAR2": "deploy_value2",
            },
            "evaluation": {},
        }

        cfg = OmegaConf.create(base_config)

        script = _create_slurm_sbatch_script(
            cfg=cfg,
            task=mock_task,
            eval_image="test-eval-container:latest",
            remote_task_subdir=Path("/test/remote"),
            invocation_id="test123",
            job_id="test123.0",
        )

        # Check that deployment environment variables are exported
        assert "export DEPLOY_VAR1=deploy_value1" in script
        assert "export DEPLOY_VAR2=deploy_value2" in script

        # Check that deployment env vars are passed to deployment container
        assert "--container-env DEPLOY_VAR1,DEPLOY_VAR2" in script

    def test_new_execution_env_vars_evaluation(
        self, base_config, mock_task, mock_dependencies
    ):
        """Test new execution.env_vars.evaluation configuration."""
        # Add new env_vars structure
        base_config["execution"]["env_vars"] = {
            "deployment": {},
            "evaluation": {
                "EVAL_VAR1": "eval_value1",
                "EVAL_VAR2": "eval_value2",
            },
        }

        cfg = OmegaConf.create(base_config)

        script = _create_slurm_sbatch_script(
            cfg=cfg,
            task=mock_task,
            eval_image="test-eval-container:latest",
            remote_task_subdir=Path("/test/remote"),
            invocation_id="test123",
            job_id="test123.0",
        )

        # Check that evaluation environment variables are exported
        assert "export EVAL_VAR1=eval_value1" in script
        assert "export EVAL_VAR2=eval_value2" in script

        # Check that evaluation env vars are passed to evaluation container
        assert "--container-env EVAL_VAR1,EVAL_VAR2" in script

    def test_new_execution_mounts_deployment(
        self, base_config, mock_task, mock_dependencies
    ):
        """Test new execution.mounts.deployment configuration."""
        base_config["execution"]["mounts"] = {
            "deployment": {
                "/host/path1": "/container/path1",
                "/host/path2": "/container/path2",
            },
            "evaluation": {},
        }

        cfg = OmegaConf.create(base_config)

        script = _create_slurm_sbatch_script(
            cfg=cfg,
            task=mock_task,
            eval_image="test-eval-container:latest",
            remote_task_subdir=Path("/test/remote"),
            invocation_id="test123",
            job_id="test123.0",
        )

        # Check that deployment mounts are added to deployment container
        assert "/host/path1:/container/path1" in script
        assert "/host/path2:/container/path2" in script
        # The mount should appear in the deployment srun command
        assert (
            "--container-mounts" in script and "/host/path1:/container/path1" in script
        )

    def test_new_execution_mounts_evaluation(
        self, base_config, mock_task, mock_dependencies
    ):
        """Test new execution.mounts.evaluation configuration."""
        base_config["execution"]["mounts"] = {
            "deployment": {},
            "evaluation": {
                "/host/eval1": "/container/eval1",
                "/host/eval2": "/container/eval2",
            },
        }

        cfg = OmegaConf.create(base_config)

        script = _create_slurm_sbatch_script(
            cfg=cfg,
            task=mock_task,
            eval_image="test-eval-container:latest",
            remote_task_subdir=Path("/test/remote"),
            invocation_id="test123",
            job_id="test123.0",
        )

        # Check that evaluation mounts are added to evaluation container
        assert "/host/eval1:/container/eval1" in script
        assert "/host/eval2:/container/eval2" in script

    def test_mount_home_flag_enabled(self, base_config, mock_task, mock_dependencies):
        """Test mount_home flag when enabled (default behavior)."""
        base_config["execution"]["mounts"] = {
            "deployment": {},
            "evaluation": {},
            "mount_home": True,
        }

        cfg = OmegaConf.create(base_config)

        script = _create_slurm_sbatch_script(
            cfg=cfg,
            task=mock_task,
            eval_image="test-eval-container:latest",
            remote_task_subdir=Path("/test/remote"),
            invocation_id="test123",
            job_id="test123.0",
        )

        # Should NOT contain --no-container-mount-home when mount_home is True
        assert "--no-container-mount-home" not in script

    def test_mount_home_flag_disabled(self, base_config, mock_task, mock_dependencies):
        """Test mount_home flag when disabled."""
        base_config["execution"]["mounts"] = {
            "deployment": {},
            "evaluation": {},
            "mount_home": False,
        }

        cfg = OmegaConf.create(base_config)

        script = _create_slurm_sbatch_script(
            cfg=cfg,
            task=mock_task,
            eval_image="test-eval-container:latest",
            remote_task_subdir=Path("/test/remote"),
            invocation_id="test123",
            job_id="test123.0",
        )

        # Should contain --no-container-mount-home when mount_home is False
        assert "--no-container-mount-home" in script

    def test_mount_home_default_behavior(
        self, base_config, mock_task, mock_dependencies
    ):
        """Test mount_home default behavior (should be True if not specified)."""
        # Don't set mount_home explicitly - test default behavior
        cfg = OmegaConf.create(base_config)

        script = _create_slurm_sbatch_script(
            cfg=cfg,
            task=mock_task,
            eval_image="test-eval-container:latest",
            remote_task_subdir=Path("/test/remote"),
            invocation_id="test123",
            job_id="test123.0",
        )

        # Should NOT contain --no-container-mount-home by default (mount_home defaults to True)
        assert "--no-container-mount-home" not in script

    def test_deprecation_warning_deployment_env_vars(
        self, base_config, mock_task, mock_dependencies
    ):
        """Test deprecation warning for old deployment.env_vars usage."""
        # Use old-style deployment.env_vars
        base_config["deployment"]["env_vars"] = {
            "OLD_VAR1": "old_value1",
            "OLD_VAR2": "old_value2",
        }

        cfg = OmegaConf.create(base_config)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            script = _create_slurm_sbatch_script(
                cfg=cfg,
                task=mock_task,
                eval_image="test-eval-container:latest",
                remote_task_subdir=Path("/test/remote"),
                invocation_id="test123",
                job_id="test123.0",
            )

            # Check that deprecation warnings were issued
            deprecation_warnings = [
                warning
                for warning in w
                if issubclass(warning.category, DeprecationWarning)
            ]
            assert len(deprecation_warnings) >= 1
            assert any(
                "cfg.deployment.env_vars will be deprecated" in str(warning.message)
                for warning in deprecation_warnings
            )

        # Old env vars should still work
        assert "export OLD_VAR1=old_value1" in script
        assert "export OLD_VAR2=old_value2" in script

    def test_backward_compatibility_mixed_env_vars(
        self, base_config, mock_task, mock_dependencies
    ):
        """Test backward compatibility when both old and new env_vars are present."""
        # Mix old and new style
        base_config["deployment"]["env_vars"] = {
            "OLD_VAR": "old_value",
        }
        base_config["execution"]["env_vars"] = {
            "deployment": {
                "NEW_VAR": "new_value",
            },
            "evaluation": {
                "EVAL_VAR": "eval_value",
            },
        }

        cfg = OmegaConf.create(base_config)

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")

            script = _create_slurm_sbatch_script(
                cfg=cfg,
                task=mock_task,
                eval_image="test-eval-container:latest",
                remote_task_subdir=Path("/test/remote"),
                invocation_id="test123",
                job_id="test123.0",
            )

        # Both old and new env vars should be present
        assert "export OLD_VAR=old_value" in script
        assert "export NEW_VAR=new_value" in script
        assert "export EVAL_VAR=eval_value" in script

        # Check that both old and new deployment vars are passed to deployment container
        assert (
            "--container-env NEW_VAR,OLD_VAR" in script
            or "--container-env OLD_VAR,NEW_VAR" in script
        )

        # Check that evaluation vars are passed to evaluation container
        assert "--container-env EVAL_VAR" in script

    def test_empty_configurations(self, base_config, mock_task, mock_dependencies):
        """Test behavior with empty new configurations."""
        base_config["execution"]["env_vars"] = {"deployment": {}, "evaluation": {}}
        base_config["execution"]["mounts"] = {
            "deployment": {},
            "evaluation": {},
            "mount_home": True,
        }

        cfg = OmegaConf.create(base_config)

        script = _create_slurm_sbatch_script(
            cfg=cfg,
            task=mock_task,
            eval_image="test-eval-container:latest",
            remote_task_subdir=Path("/test/remote"),
            invocation_id="test123",
            job_id="test123.0",
        )

        # Script should be generated successfully without errors
        assert "srun" in script
        assert "--container-image" in script

    def test_no_deployment_type_none(self, base_config, mock_task, mock_dependencies):
        """Test behavior when deployment type is 'none'."""
        base_config["deployment"]["type"] = "none"
        base_config["execution"]["env_vars"] = {
            "deployment": {"DEPLOY_VAR": "deploy_value"},
            "evaluation": {"EVAL_VAR": "eval_value"},
        }

        cfg = OmegaConf.create(base_config)

        script = _create_slurm_sbatch_script(
            cfg=cfg,
            task=mock_task,
            eval_image="test-eval-container:latest",
            remote_task_subdir=Path("/test/remote"),
            invocation_id="test123",
            job_id="test123.0",
        )

        # Environment variables should still be exported
        assert "export DEPLOY_VAR=deploy_value" in script
        assert "export EVAL_VAR=eval_value" in script

        # Should not have deployment server section when type is 'none'

        # Evaluation should still be present
        assert "evaluation client" in script
        assert "--container-env EVAL_VAR" in script

    def test_complex_configuration_integration(
        self, base_config, mock_task, mock_dependencies
    ):
        """Test complex configuration with all new features together."""
        base_config["execution"]["env_vars"] = {
            "deployment": {
                "DEPLOY_VAR1": "deploy_value1",
                "DEPLOY_VAR2": "deploy_value2",
            },
            "evaluation": {
                "EVAL_VAR1": "eval_value1",
                "EVAL_VAR2": "eval_value2",
            },
        }
        base_config["execution"]["mounts"] = {
            "deployment": {
                "/host/deploy1": "/container/deploy1",
                "/host/deploy2": "/container/deploy2:ro",
            },
            "evaluation": {
                "/host/eval1": "/container/eval1",
                "/host/eval2": "/container/eval2:rw",
            },
            "mount_home": False,
        }
        # Also add old-style for compatibility test
        base_config["deployment"]["env_vars"] = {"OLD_DEPLOY_VAR": "old_deploy_value"}

        cfg = OmegaConf.create(base_config)

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")

            script = _create_slurm_sbatch_script(
                cfg=cfg,
                task=mock_task,
                eval_image="test-eval-container:latest",
                remote_task_subdir=Path("/test/remote"),
                invocation_id="test123",
                job_id="test123.0",
            )

        # All environment variables should be exported
        assert "export DEPLOY_VAR1=deploy_value1" in script
        assert "export DEPLOY_VAR2=deploy_value2" in script
        assert "export EVAL_VAR1=eval_value1" in script
        assert "export EVAL_VAR2=eval_value2" in script
        assert "export OLD_DEPLOY_VAR=old_deploy_value" in script

        # All mounts should be present
        assert "/host/deploy1:/container/deploy1" in script
        assert "/host/deploy2:/container/deploy2:ro" in script
        assert "/host/eval1:/container/eval1" in script
        assert "/host/eval2:/container/eval2:rw" in script

        # mount_home=False should add --no-container-mount-home
        assert "--no-container-mount-home" in script


class TestSlurmExecutorDryRun:
    """Test SlurmExecutor dry run functionality."""

    @pytest.fixture
    def sample_config(self, tmpdir):
        """Create a sample configuration for testing."""
        config_dict = {
            "deployment": {
                "type": "vllm",
                "image": "nvcr.io/nvidia/vllm:latest",
                "command": "python -m vllm.entrypoints.openai.api_server --model /model --port 8000",
                "served_model_name": "llama-3.1-8b-instruct",
                "port": 8000,
                "endpoints": {"health": "/health", "openai": "/v1"},
            },
            "execution": {
                "type": "slurm",
                "output_dir": str(tmpdir / "test_output"),
                "walltime": "02:00:00",
                "account": "test-account",
                "partition": "gpu",
                "num_nodes": 1,
                "ntasks_per_node": 8,
                "gpus_per_node": 8,
                "subproject": "eval",
                "username": "testuser",
                "hostname": "slurm.example.com",
                "auto_export": {"destinations": ["local", "wandb"]},
            },
            "target": {
                "api_endpoint": {
                    "api_key_name": "TEST_API_KEY",
                    "model_id": "llama-3.1-8b-instruct",
                    "url": "http://localhost:8000/v1/chat/completions",
                }
            },
            "evaluation": {
                "env_vars": {"GLOBAL_ENV": "GLOBAL_VALUE"},
                "tasks": [
                    {
                        "name": "mmlu_pro",
                        "env_vars": {"TASK_ENV": "TASK_VALUE"},
                        "overrides": {"num_fewshot": 5},
                    },
                    {
                        "name": "gsm8k",
                        "container": "custom-math-container:v2.0",
                        "overrides": {"batch_size": 16},
                    },
                ],
            },
        }
        return OmegaConf.create(config_dict)

    @pytest.fixture
    def mock_tasks_mapping(self):
        """Mock tasks mapping for testing."""
        return {
            ("lm-eval", "mmlu_pro"): {
                "task": "mmlu_pro",
                "endpoint_type": "openai",
                "harness": "lm-eval",
                "container": "nvcr.io/nvidia/nemo:24.01",
                "required_env_vars": ["TASK_ENV"],
            },
            ("lm-eval", "gsm8k"): {
                "task": "gsm8k",
                "endpoint_type": "openai",
                "harness": "lm-eval",
                "container": "nvcr.io/nvidia/nemo:24.01",
            },
        }

    def test_execute_eval_dry_run_basic(
        self, sample_config, mock_tasks_mapping, tmpdir
    ):
        """Test basic dry run execution."""
        # Set up environment variable that the config references
        os.environ["TEST_API_KEY"] = "test_key_value"
        os.environ["GLOBAL_VALUE"] = "global_env_value"
        os.environ["TASK_VALUE"] = "task_env_value"

        try:
            with (
                patch(
                    "nemo_evaluator_launcher.executors.slurm.executor.load_tasks_mapping"
                ) as mock_load_mapping,
                patch(
                    "nemo_evaluator_launcher.executors.slurm.executor.get_task_from_mapping"
                ) as mock_get_task,
                patch(
                    "nemo_evaluator_launcher.executors.slurm.executor.get_eval_factory_command"
                ) as mock_get_command,
                patch(
                    "nemo_evaluator_launcher.executors.slurm.executor.get_health_url"
                ) as mock_get_health,
                patch(
                    "nemo_evaluator_launcher.executors.slurm.executor.get_endpoint_url"
                ) as mock_get_endpoint,
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
                mock_get_command.return_value = "nemo-evaluator-launcher --model llama-3.1-8b-instruct --task {task_name}"
                mock_get_health.return_value = "http://localhost:8000/health"
                mock_get_endpoint.return_value = "http://localhost:8000/v1"

                # Execute dry run
                invocation_id = SlurmExecutor.execute_eval(sample_config, dry_run=True)

                # Verify invocation ID format
                assert isinstance(invocation_id, str)
                assert len(invocation_id) == 16
                assert re.match(r"^[a-f0-9]{16}$", invocation_id)

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
                "nemo_evaluator_launcher.executors.slurm.executor.load_tasks_mapping"
            ) as mock_load_mapping,
            patch(
                "nemo_evaluator_launcher.executors.slurm.executor.get_task_from_mapping"
            ) as mock_get_task,
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
                SlurmExecutor.execute_eval(sample_config, dry_run=True)

    def test_execute_eval_dry_run_required_task_env_vars(
        self, sample_config, mock_tasks_mapping
    ):
        """Test validation of required task-specific environment variables."""
        # Set some but not all required env vars
        os.environ["TEST_API_KEY"] = "test_key_value"
        os.environ["GLOBAL_VALUE"] = "global_env_value"
        # Missing TASK_VALUE for mmlu_pro

        try:
            with (
                patch(
                    "nemo_evaluator_launcher.executors.slurm.executor.load_tasks_mapping"
                ) as mock_load_mapping,
                patch(
                    "nemo_evaluator_launcher.executors.slurm.executor.get_task_from_mapping"
                ) as mock_get_task,
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
                    SlurmExecutor.execute_eval(sample_config, dry_run=True)

        finally:
            # Clean up environment
            for env_var in ["TEST_API_KEY", "GLOBAL_VALUE"]:
                if env_var in os.environ:
                    del os.environ[env_var]

    def test_execute_eval_dry_run_custom_container(
        self, sample_config, mock_tasks_mapping, tmpdir
    ):
        """Test that custom container images are handled correctly."""
        # Set up all required environment variables
        os.environ["TEST_API_KEY"] = "test_key_value"
        os.environ["GLOBAL_VALUE"] = "global_env_value"
        os.environ["TASK_VALUE"] = "task_env_value"

        try:
            with (
                patch(
                    "nemo_evaluator_launcher.executors.slurm.executor.load_tasks_mapping"
                ) as mock_load_mapping,
                patch(
                    "nemo_evaluator_launcher.executors.slurm.executor.get_task_from_mapping"
                ) as mock_get_task,
                patch(
                    "nemo_evaluator_launcher.executors.slurm.executor.get_eval_factory_command"
                ) as mock_get_command,
                patch(
                    "nemo_evaluator_launcher.executors.slurm.executor.get_health_url"
                ) as mock_get_health,
                patch(
                    "nemo_evaluator_launcher.executors.slurm.executor.get_endpoint_url"
                ) as mock_get_endpoint,
                patch("builtins.print"),
            ):
                mock_load_mapping.return_value = mock_tasks_mapping

                def mock_get_task_side_effect(task_name, mapping):
                    for (harness, name), definition in mapping.items():
                        if name == task_name:
                            return definition
                    raise KeyError(f"Task {task_name} not found")

                mock_get_task.side_effect = mock_get_task_side_effect
                mock_get_command.return_value = (
                    "nemo-evaluator-launcher --task test_command"
                )
                mock_get_health.return_value = "http://localhost:8000/health"
                mock_get_endpoint.return_value = "http://localhost:8000/v1"

                # Execute dry run
                invocation_id = SlurmExecutor.execute_eval(sample_config, dry_run=True)

                # Verify invocation ID is valid
                assert isinstance(invocation_id, str)
                assert len(invocation_id) == 16

        finally:
            # Clean up environment
            for env_var in ["TEST_API_KEY", "GLOBAL_VALUE", "TASK_VALUE"]:
                if env_var in os.environ:
                    del os.environ[env_var]

    def test_execute_eval_dry_run_no_auto_export(
        self, sample_config, mock_tasks_mapping, tmpdir
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
                    "nemo_evaluator_launcher.executors.slurm.executor.load_tasks_mapping"
                ) as mock_load_mapping,
                patch(
                    "nemo_evaluator_launcher.executors.slurm.executor.get_task_from_mapping"
                ) as mock_get_task,
                patch(
                    "nemo_evaluator_launcher.executors.slurm.executor.get_eval_factory_command"
                ) as mock_get_command,
                patch(
                    "nemo_evaluator_launcher.executors.slurm.executor.get_health_url"
                ) as mock_get_health,
                patch(
                    "nemo_evaluator_launcher.executors.slurm.executor.get_endpoint_url"
                ) as mock_get_endpoint,
                patch("builtins.print"),
            ):
                mock_load_mapping.return_value = mock_tasks_mapping

                def mock_get_task_side_effect(task_name, mapping):
                    for (harness, name), definition in mapping.items():
                        if name == task_name:
                            return definition
                    raise KeyError(f"Task {task_name} not found")

                mock_get_task.side_effect = mock_get_task_side_effect
                mock_get_command.return_value = (
                    "nemo-evaluator-launcher --task test_command"
                )
                mock_get_health.return_value = "http://localhost:8000/health"
                mock_get_endpoint.return_value = "http://localhost:8000/v1"

                # Should execute successfully without auto-export
                invocation_id = SlurmExecutor.execute_eval(sample_config, dry_run=True)

                # Verify invocation ID is valid
                assert isinstance(invocation_id, str)
                assert len(invocation_id) == 16

        finally:
            # Clean up environment
            for env_var in ["TEST_API_KEY", "GLOBAL_VALUE", "TASK_VALUE"]:
                if env_var in os.environ:
                    del os.environ[env_var]


class TestSlurmExecutorGetStatus:
    """Test SlurmExecutor get_status functionality."""

    @pytest.fixture
    def sample_job_data(self, tmpdir) -> JobData:
        """Create sample job data for testing."""
        return JobData(
            invocation_id="def67890",
            job_id="def67890.0",
            timestamp=time.time(),
            executor="slurm",
            data={
                "slurm_job_id": "123456789",
                "remote_rundir_path": "/remote/output/test_job",
                "hostname": "slurm.example.com",
                "username": "testuser",
                "eval_image": "test-image:latest",
            },
            config={},
        )

    def test_get_status_invocation_id(self, mock_execdb, sample_job_data):
        """Test get_status with invocation ID (multiple jobs)."""
        # Create second job data
        job_data2 = JobData(
            invocation_id="def67890",
            job_id="def67890.1",
            timestamp=time.time(),
            executor="slurm",
            data={
                "slurm_job_id": "123456790",
                "remote_rundir_path": "/remote/output/test_job2",
                "hostname": "slurm.example.com",
                "username": "testuser",
                "eval_image": "test-image:latest",
            },
            config={},
        )

        # Mock database calls
        db = ExecutionDB()
        db.write_job(sample_job_data)
        db.write_job(job_data2)

        with patch.object(
            SlurmExecutor, "_query_slurm_for_status_and_progress"
        ) as mock_query:
            mock_query.return_value = [
                ExecutionStatus(
                    id="def67890.0",
                    state=ExecutionState.SUCCESS,
                    progress=dict(progress=1.0),
                ),
                ExecutionStatus(
                    id="def67890.1",
                    state=ExecutionState.RUNNING,
                    progress=dict(progress=0.6),
                ),
            ]

            # Test
            statuses = SlurmExecutor.get_status("def67890")

            assert len(statuses) == 2
            assert statuses[0].id == "def67890.0"
            assert statuses[0].state == ExecutionState.SUCCESS
            assert statuses[1].id == "def67890.1"
            assert statuses[1].state == ExecutionState.RUNNING

    def test_get_status_job_not_found(self):
        """Test get_status with non-existent job ID."""
        statuses = SlurmExecutor.get_status("nonexistent.0")
        assert statuses == []

    def test_get_status_wrong_executor(self, mock_execdb, sample_job_data):
        """Test get_status with job from different executor."""
        # Change executor to something else
        sample_job_data.executor = "local"

        db = ExecutionDB()
        db.write_job(sample_job_data)

        statuses = SlurmExecutor.get_status("def67890.0")
        assert statuses == []

    def test_get_status_missing_slurm_job_id(self, mock_execdb, sample_job_data):
        """Test get_status when SLURM job ID is missing."""
        # Remove slurm_job_id from data
        del sample_job_data.data["slurm_job_id"]

        db = ExecutionDB()
        db.write_job(sample_job_data)

        statuses = SlurmExecutor.get_status("def67890.0")
        assert len(statuses) == 1
        assert statuses[0].state == ExecutionState.FAILED

    def test_get_status_query_exception(self, mock_execdb, sample_job_data):
        """Test get_status when SLURM query raises exception."""
        db = ExecutionDB()
        db.write_job(sample_job_data)

        with patch.object(
            SlurmExecutor, "_query_slurm_for_status_and_progress"
        ) as mock_query:
            mock_query.side_effect = Exception("SLURM connection failed")

            statuses = SlurmExecutor.get_status("def67890.0")
            assert len(statuses) == 1
            assert statuses[0].state == ExecutionState.FAILED

    def test_get_status_invocation_missing_data(self, mock_execdb):
        """Test get_status for invocation with missing required data."""
        # Create job with missing required fields
        job_data = JobData(
            invocation_id="def67890",
            job_id="def67890.0",
            timestamp=time.time(),
            executor="slurm",
            data={
                # Missing slurm_job_id, hostname, username
                "remote_rundir_path": "/remote/output/test_job",
            },
            config={},
        )

        db = ExecutionDB()
        db.write_job(job_data)

        statuses = SlurmExecutor.get_status("def67890")
        assert len(statuses) == 1
        assert statuses[0].state == ExecutionState.FAILED

    def test_get_status_invocation_empty_jobs(self):
        """Test get_status for invocation with no jobs."""
        statuses = SlurmExecutor.get_status("nonexist.1")
        assert statuses == []

    def test_map_slurm_state_to_execution_state(self):
        """Test SLURM state mapping to ExecutionState."""
        # Test success states
        assert (
            SlurmExecutor._map_slurm_state_to_execution_state("COMPLETED")
            == ExecutionState.SUCCESS
        )

        # Test pending states
        assert (
            SlurmExecutor._map_slurm_state_to_execution_state("PENDING")
            == ExecutionState.PENDING
        )

        # Test running states
        assert (
            SlurmExecutor._map_slurm_state_to_execution_state("RUNNING")
            == ExecutionState.RUNNING
        )
        assert (
            SlurmExecutor._map_slurm_state_to_execution_state("CONFIGURING")
            == ExecutionState.RUNNING
        )
        assert (
            SlurmExecutor._map_slurm_state_to_execution_state("SUSPENDED")
            == ExecutionState.RUNNING
        )

        # Test auto-resume states (mapped to PENDING)
        assert (
            SlurmExecutor._map_slurm_state_to_execution_state("PREEMPTED")
            == ExecutionState.PENDING
        )
        assert (
            SlurmExecutor._map_slurm_state_to_execution_state("TIMEOUT")
            == ExecutionState.PENDING
        )
        assert (
            SlurmExecutor._map_slurm_state_to_execution_state("NODE_FAIL")
            == ExecutionState.PENDING
        )

        # Test killed states
        assert (
            SlurmExecutor._map_slurm_state_to_execution_state("CANCELLED")
            == ExecutionState.KILLED
        )

        # Test failed states
        assert (
            SlurmExecutor._map_slurm_state_to_execution_state("FAILED")
            == ExecutionState.FAILED
        )

        # Test unknown states (should default to FAILED)
        assert (
            SlurmExecutor._map_slurm_state_to_execution_state("UNKNOWN_STATE")
            == ExecutionState.FAILED
        )
        assert (
            SlurmExecutor._map_slurm_state_to_execution_state("")
            == ExecutionState.FAILED
        )

    def test_query_slurm_for_status_and_progress_basic(self):
        """Test basic _query_slurm_for_status_and_progress functionality."""
        slurm_job_ids = ["123456789"]
        remote_rundir_paths = [Path("/remote/output/test_job")]
        username = "testuser"
        hostname = "slurm.example.com"
        job_id_to_execdb_id = {"123456789": "def67890.0"}

        with (
            patch(
                "nemo_evaluator_launcher.executors.slurm.executor._open_master_connection"
            ) as mock_open,
            patch(
                "nemo_evaluator_launcher.executors.slurm.executor._close_master_connection"
            ),
            patch(
                "nemo_evaluator_launcher.executors.slurm.executor._query_slurm_jobs_status"
            ) as mock_query_status,
            patch(
                "nemo_evaluator_launcher.executors.slurm.executor._read_autoresumed_slurm_job_ids"
            ) as mock_autoresume,
            patch(
                "nemo_evaluator_launcher.executors.slurm.executor._get_progress"
            ) as mock_progress,
        ):
            mock_open.return_value = "/tmp/socket"
            mock_query_status.return_value = {"123456789": "COMPLETED"}
            mock_autoresume.return_value = {"123456789": ["123456789"]}
            mock_progress.return_value = [0.8]

            statuses = SlurmExecutor._query_slurm_for_status_and_progress(
                slurm_job_ids=slurm_job_ids,
                remote_rundir_paths=remote_rundir_paths,
                username=username,
                hostname=hostname,
                job_id_to_execdb_id=job_id_to_execdb_id,
            )

            assert len(statuses) == 1
            assert statuses[0].id == "def67890.0"
            assert statuses[0].state == ExecutionState.SUCCESS
            assert statuses[0].progress == 0.8

    def test_query_slurm_for_status_and_progress_autoresumed(self):
        """Test _query_slurm_for_status_and_progress with autoresumed jobs."""
        slurm_job_ids = ["123456789"]
        remote_rundir_paths = [Path("/remote/output/test_job")]
        username = "testuser"
        hostname = "slurm.example.com"
        job_id_to_execdb_id = {"123456789": "def67890.0"}

        with (
            patch(
                "nemo_evaluator_launcher.executors.slurm.executor._open_master_connection"
            ) as mock_open,
            patch(
                "nemo_evaluator_launcher.executors.slurm.executor._close_master_connection"
            ),
            patch(
                "nemo_evaluator_launcher.executors.slurm.executor._query_slurm_jobs_status"
            ) as mock_query_status,
            patch(
                "nemo_evaluator_launcher.executors.slurm.executor._read_autoresumed_slurm_job_ids"
            ) as mock_autoresume,
            patch(
                "nemo_evaluator_launcher.executors.slurm.executor._get_progress"
            ) as mock_progress,
        ):
            mock_open.return_value = "/tmp/socket"
            # Initial job was preempted, latest job is running
            mock_query_status.side_effect = [
                {"123456789": "PREEMPTED"},  # Original job status
                {"123456790": "RUNNING"},  # Latest autoresumed job status
            ]
            # Autoresume shows there's a newer job ID
            mock_autoresume.return_value = {"123456789": ["123456789", "123456790"]}
            mock_progress.return_value = [0.4]

            statuses = SlurmExecutor._query_slurm_for_status_and_progress(
                slurm_job_ids=slurm_job_ids,
                remote_rundir_paths=remote_rundir_paths,
                username=username,
                hostname=hostname,
                job_id_to_execdb_id=job_id_to_execdb_id,
            )

            assert len(statuses) == 1
            assert statuses[0].id == "def67890.0"
            assert statuses[0].state == ExecutionState.RUNNING  # Uses latest job status
            assert statuses[0].progress == 0.4

    def test_query_slurm_for_status_and_progress_unknown_progress(self):
        """Test _query_slurm_for_status_and_progress with unknown progress."""
        slurm_job_ids = ["123456789"]
        remote_rundir_paths = [Path("/remote/output/test_job")]
        username = "testuser"
        hostname = "slurm.example.com"
        job_id_to_execdb_id = {"123456789": "def67890.0"}

        with (
            patch(
                "nemo_evaluator_launcher.executors.slurm.executor._open_master_connection"
            ) as mock_open,
            patch(
                "nemo_evaluator_launcher.executors.slurm.executor._close_master_connection"
            ),
            patch(
                "nemo_evaluator_launcher.executors.slurm.executor._query_slurm_jobs_status"
            ) as mock_query_status,
            patch(
                "nemo_evaluator_launcher.executors.slurm.executor._read_autoresumed_slurm_job_ids"
            ) as mock_autoresume,
            patch(
                "nemo_evaluator_launcher.executors.slurm.executor._get_progress"
            ) as mock_progress,
        ):
            mock_open.return_value = "/tmp/socket"
            mock_query_status.return_value = {"123456789": "RUNNING"}
            mock_autoresume.return_value = {"123456789": ["123456789"]}
            mock_progress.return_value = [None]  # Unknown progress

            statuses = SlurmExecutor._query_slurm_for_status_and_progress(
                slurm_job_ids=slurm_job_ids,
                remote_rundir_paths=remote_rundir_paths,
                username=username,
                hostname=hostname,
                job_id_to_execdb_id=job_id_to_execdb_id,
            )

            assert len(statuses) == 1
            assert statuses[0].id == "def67890.0"
            assert statuses[0].state == ExecutionState.RUNNING
            assert statuses[0].progress == "unknown"  # None converted to "unknown"


class TestSlurmExecutorSystemCalls:
    """Test SLURM executor system calls by patching subprocess.run."""

    @pytest.fixture
    def sample_config(self, tmpdir):
        """Create a sample configuration for testing."""
        config_dict = {
            "deployment": {
                "type": "vllm",
                "image": "nvcr.io/nvidia/vllm:latest",
                "command": "python -m vllm.entrypoints.openai.api_server --model /model --port 8000",
                "served_model_name": "llama-3.1-8b-instruct",
                "port": 8000,
                "endpoints": {"health": "/health", "openai": "/v1"},
            },
            "execution": {
                "type": "slurm",
                "output_dir": "/remote/slurm/output",
                "walltime": "02:00:00",
                "account": "test-account",
                "partition": "gpu",
                "num_nodes": 1,
                "ntasks_per_node": 8,
                "gpus_per_node": 8,
                "subproject": "eval",
                "username": "testuser",
                "hostname": "slurm.example.com",
            },
            "target": {
                "api_endpoint": {
                    "api_key_name": "TEST_API_KEY",
                    "model_id": "llama-3.1-8b-instruct",
                    "url": "http://localhost:8000/v1/chat/completions",
                }
            },
            "evaluation": {
                "env_vars": {"GLOBAL_ENV": "GLOBAL_VALUE"},
                "tasks": [
                    {
                        "name": "mmlu_pro",
                        "env_vars": {"TASK_ENV": "TASK_VALUE"},
                        "overrides": {"num_fewshot": 5},
                    }
                ],
            },
        }
        return OmegaConf.create(config_dict)

    @pytest.fixture
    def mock_tasks_mapping(self):
        """Mock tasks mapping for testing."""
        return {
            ("lm-eval", "mmlu_pro"): {
                "task": "mmlu_pro",
                "endpoint_type": "openai",
                "harness": "lm-eval",
                "container": "nvcr.io/nvidia/nemo:24.01",
                "required_env_vars": ["TASK_ENV"],
            }
        }

    def test_execute_eval_non_dry_run_success(
        self, mock_execdb, sample_config, mock_tasks_mapping, tmpdir
    ):
        """Test successful non-dry-run execution by patching subprocess.run."""
        # Set up environment variables
        os.environ["TEST_API_KEY"] = "test_key_value"
        os.environ["GLOBAL_VALUE"] = "global_env_value"
        os.environ["TASK_VALUE"] = "task_env_value"

        # Mock subprocess.run calls
        def mock_subprocess_run(*args, **kwargs):
            """Mock subprocess.run based on the command being executed."""
            # Extract command from kwargs['args'] if present, otherwise from args
            if "args" in kwargs:
                cmd_list = kwargs["args"]
            elif args:
                cmd_list = args[0]
            else:
                return Mock(returncode=0)

            cmd = " ".join(cmd_list) if isinstance(cmd_list, list) else str(cmd_list)

            # Mock SSH master connection setup
            if "ssh -MNf -S" in cmd:
                return Mock(returncode=0)

            # Mock remote directory creation
            if "mkdir -p" in cmd:
                return Mock(returncode=0)

            # Mock rsync upload
            if "rsync" in cmd:
                return Mock(returncode=0)

            # Mock sbatch submission
            if "sbatch" in cmd:
                return Mock(
                    returncode=0, stdout=b"Submitted batch job 123456789\n", stderr=b""
                )

            # Mock SSH connection close
            if "ssh -O exit" in cmd:
                return Mock(returncode=0, stderr=b"")

            # Default success
            return Mock(returncode=0)

        try:
            with (
                patch(
                    "nemo_evaluator_launcher.executors.slurm.executor.load_tasks_mapping"
                ) as mock_load_mapping,
                patch(
                    "nemo_evaluator_launcher.executors.slurm.executor.get_task_from_mapping"
                ) as mock_get_task,
                patch(
                    "nemo_evaluator_launcher.executors.slurm.executor.get_eval_factory_command"
                ) as mock_get_command,
                patch(
                    "nemo_evaluator_launcher.executors.slurm.executor.get_health_url"
                ) as mock_get_health,
                patch(
                    "nemo_evaluator_launcher.executors.slurm.executor.get_endpoint_url"
                ) as mock_get_endpoint,
                patch("subprocess.run", side_effect=mock_subprocess_run),
            ):
                # Configure mocks
                mock_load_mapping.return_value = mock_tasks_mapping

                def mock_get_task_side_effect(task_name, mapping):
                    for (harness, name), definition in mapping.items():
                        if name == task_name:
                            return definition
                    raise KeyError(f"Task {task_name} not found")

                mock_get_task.side_effect = mock_get_task_side_effect
                mock_get_command.return_value = (
                    "nemo-evaluator-launcher --task mmlu_pro"
                )
                mock_get_health.return_value = "http://127.0.0.1:8000/health"
                mock_get_endpoint.return_value = "http://127.0.0.1:8000/v1"

                # Execute non-dry-run
                invocation_id = SlurmExecutor.execute_eval(sample_config, dry_run=False)

                # Verify invocation ID format
                assert isinstance(invocation_id, str)
                assert len(invocation_id) == 16
                assert re.match(r"^[a-f0-9]{16}$", invocation_id)

                # Verify job was saved to database
                db = ExecutionDB()
                jobs = db.get_jobs(invocation_id)
                assert len(jobs) == 1

                job_id, job_data = next(iter(jobs.items()))
                assert job_data.executor == "slurm"
                assert job_data.data["slurm_job_id"] == "123456789"
                assert job_data.data["hostname"] == "slurm.example.com"
                assert job_data.data["username"] == "testuser"

        finally:
            # Clean up environment
            for env_var in ["TEST_API_KEY", "GLOBAL_VALUE", "TASK_VALUE"]:
                if env_var in os.environ:
                    del os.environ[env_var]

    def test_execute_eval_non_dry_run_ssh_connection_failure(
        self, sample_config, mock_tasks_mapping
    ):
        """Test non-dry-run execution with SSH connection failure."""
        # Set up environment variables
        os.environ["TEST_API_KEY"] = "test_key_value"
        os.environ["GLOBAL_VALUE"] = "global_env_value"
        os.environ["TASK_VALUE"] = "task_env_value"

        def mock_subprocess_run(*args, **kwargs):
            """Mock subprocess.run to simulate SSH connection failure."""
            # Extract command from kwargs['args'] if present, otherwise from args
            if "args" in kwargs:
                cmd_list = kwargs["args"]
            elif args:
                cmd_list = args[0]
            else:
                return Mock(returncode=0)

            cmd = " ".join(cmd_list) if isinstance(cmd_list, list) else str(cmd_list)

            # Mock SSH master connection failure
            if "ssh -MNf -S" in cmd:
                return Mock(returncode=1)  # Connection failed

            # Mock sbatch command (even though SSH failed, we still need to handle sbatch calls)
            if "sbatch" in cmd:
                return Mock(
                    returncode=0, stdout=b"Submitted batch job 123456789\n", stderr=b""
                )

            return Mock(returncode=0)

        try:
            with (
                patch(
                    "nemo_evaluator_launcher.executors.slurm.executor.load_tasks_mapping"
                ) as mock_load_mapping,
                patch(
                    "nemo_evaluator_launcher.executors.slurm.executor.get_task_from_mapping"
                ) as mock_get_task,
                patch(
                    "nemo_evaluator_launcher.executors.slurm.executor.get_eval_factory_command"
                ) as mock_get_command,
                patch(
                    "nemo_evaluator_launcher.executors.slurm.executor.get_health_url"
                ) as mock_get_health,
                patch(
                    "nemo_evaluator_launcher.executors.slurm.executor.get_endpoint_url"
                ) as mock_get_endpoint,
                patch("subprocess.run", side_effect=mock_subprocess_run),
            ):
                # Configure mocks
                mock_load_mapping.return_value = mock_tasks_mapping

                def mock_get_task_side_effect(task_name, mapping):
                    for (harness, name), definition in mapping.items():
                        if name == task_name:
                            return definition
                    raise KeyError(f"Task {task_name} not found")

                mock_get_task.side_effect = mock_get_task_side_effect
                mock_get_command.return_value = (
                    "nemo-evaluator-launcher --task mmlu_pro"
                )
                mock_get_health.return_value = "http://127.0.0.1:8000/health"
                mock_get_endpoint.return_value = "http://127.0.0.1:8000/v1"

                # Should still succeed (SSH connection can be None)
                invocation_id = SlurmExecutor.execute_eval(sample_config, dry_run=False)

                assert isinstance(invocation_id, str)
                assert len(invocation_id) == 16

        finally:
            # Clean up environment
            for env_var in ["TEST_API_KEY", "GLOBAL_VALUE", "TASK_VALUE"]:
                if env_var in os.environ:
                    del os.environ[env_var]

    def test_execute_eval_non_dry_run_sbatch_failure(
        self, sample_config, mock_tasks_mapping
    ):
        """Test non-dry-run execution with sbatch submission failure."""
        # Set up environment variables
        os.environ["TEST_API_KEY"] = "test_key_value"
        os.environ["GLOBAL_VALUE"] = "global_env_value"
        os.environ["TASK_VALUE"] = "task_env_value"

        def mock_subprocess_run(*args, **kwargs):
            """Mock subprocess.run to simulate sbatch failure."""
            # Extract command from kwargs['args'] if present, otherwise from args
            if "args" in kwargs:
                cmd_list = kwargs["args"]
            elif args:
                cmd_list = args[0]
            else:
                return Mock(returncode=0)

            cmd = " ".join(cmd_list) if isinstance(cmd_list, list) else str(cmd_list)

            # Mock SSH master connection
            if "ssh -MNf -S" in cmd:
                return Mock(returncode=0)

            # Mock remote directory creation
            if "mkdir -p" in cmd:
                return Mock(returncode=0)

            # Mock rsync upload
            if "rsync" in cmd:
                return Mock(returncode=0)

            # Mock sbatch submission failure
            if "sbatch" in cmd:
                return Mock(
                    returncode=1,
                    stdout=b"",
                    stderr=b"sbatch: error: invalid account specified\n",
                )

            return Mock(returncode=0)

        try:
            with (
                patch(
                    "nemo_evaluator_launcher.executors.slurm.executor.load_tasks_mapping"
                ) as mock_load_mapping,
                patch(
                    "nemo_evaluator_launcher.executors.slurm.executor.get_task_from_mapping"
                ) as mock_get_task,
                patch(
                    "nemo_evaluator_launcher.executors.slurm.executor.get_eval_factory_command"
                ) as mock_get_command,
                patch(
                    "nemo_evaluator_launcher.executors.slurm.executor.get_health_url"
                ) as mock_get_health,
                patch(
                    "nemo_evaluator_launcher.executors.slurm.executor.get_endpoint_url"
                ) as mock_get_endpoint,
                patch("subprocess.run", side_effect=mock_subprocess_run),
            ):
                # Configure mocks
                mock_load_mapping.return_value = mock_tasks_mapping

                def mock_get_task_side_effect(task_name, mapping):
                    for (harness, name), definition in mapping.items():
                        if name == task_name:
                            return definition
                    raise KeyError(f"Task {task_name} not found")

                mock_get_task.side_effect = mock_get_task_side_effect
                mock_get_command.return_value = (
                    "nemo-evaluator-launcher --task mmlu_pro"
                )
                mock_get_health.return_value = "http://127.0.0.1:8000/health"
                mock_get_endpoint.return_value = "http://127.0.0.1:8000/v1"

                # Should raise RuntimeError for sbatch failure
                with pytest.raises(
                    RuntimeError, match="failed to submit sbatch scripts"
                ):
                    SlurmExecutor.execute_eval(sample_config, dry_run=False)

        finally:
            # Clean up environment
            for env_var in ["TEST_API_KEY", "GLOBAL_VALUE", "TASK_VALUE"]:
                if env_var in os.environ:
                    del os.environ[env_var]

    def test_query_slurm_jobs_status_success(self):
        """Test _query_slurm_jobs_status function with successful subprocess call."""
        from nemo_evaluator_launcher.executors.slurm.executor import (
            _query_slurm_jobs_status,
        )

        def mock_subprocess_run(*args, **kwargs):
            """Mock subprocess.run for sacct command."""
            # Mock sacct output
            return Mock(
                returncode=0,
                stdout=b"123456789|COMPLETED\n123456790|RUNNING\n",
                stderr=b"",
            )

        with patch("subprocess.run", side_effect=mock_subprocess_run):
            result = _query_slurm_jobs_status(
                slurm_job_ids=["123456789", "123456790"],
                username="testuser",
                hostname="slurm.example.com",
                socket="/tmp/socket",
            )

            assert result == {"123456789": "COMPLETED", "123456790": "RUNNING"}

    def test_query_slurm_jobs_status_failure(self):
        """Test _query_slurm_jobs_status function with failed subprocess call."""
        from nemo_evaluator_launcher.executors.slurm.executor import (
            _query_slurm_jobs_status,
        )

        def mock_subprocess_run(*args, **kwargs):
            """Mock subprocess.run for failed sacct command."""
            return Mock(
                returncode=1, stdout=b"", stderr=b"sacct: error: invalid user\n"
            )

        with patch("subprocess.run", side_effect=mock_subprocess_run):
            with pytest.raises(RuntimeError, match="failed to query slurm job status"):
                _query_slurm_jobs_status(
                    slurm_job_ids=["123456789"],
                    username="testuser",
                    hostname="slurm.example.com",
                    socket="/tmp/socket",
                )

    def test_sbatch_remote_runsubs_success(self):
        """Test _sbatch_remote_runsubs function with successful subprocess call."""
        from pathlib import Path

        from nemo_evaluator_launcher.executors.slurm.executor import (
            _sbatch_remote_runsubs,
        )

        def mock_subprocess_run(*args, **kwargs):
            """Mock subprocess.run for sbatch command."""
            return Mock(
                returncode=0,
                stdout=b"Submitted batch job 123456789\nSubmitted batch job 123456790\n",
                stderr=b"",
            )

        with patch("subprocess.run", side_effect=mock_subprocess_run):
            result = _sbatch_remote_runsubs(
                remote_runsub_paths=[
                    Path("/remote/job1/run.sub"),
                    Path("/remote/job2/run.sub"),
                ],
                username="testuser",
                hostname="slurm.example.com",
                socket="/tmp/socket",
            )

            assert result == ["123456789", "123456790"]

    def test_sbatch_remote_runsubs_failure(self):
        """Test _sbatch_remote_runsubs function with failed subprocess call."""
        from pathlib import Path

        from nemo_evaluator_launcher.executors.slurm.executor import (
            _sbatch_remote_runsubs,
        )

        def mock_subprocess_run(*args, **kwargs):
            """Mock subprocess.run for failed sbatch command."""
            return Mock(
                returncode=1, stdout=b"", stderr=b"sbatch: error: invalid account\n"
            )

        with patch("subprocess.run", side_effect=mock_subprocess_run):
            with pytest.raises(RuntimeError, match="failed to submit sbatch scripts"):
                _sbatch_remote_runsubs(
                    remote_runsub_paths=[Path("/remote/job1/run.sub")],
                    username="testuser",
                    hostname="slurm.example.com",
                    socket="/tmp/socket",
                )

    def test_open_master_connection_success(self):
        """Test _open_master_connection with successful SSH connection."""
        from nemo_evaluator_launcher.executors.slurm.executor import (
            _open_master_connection,
        )

        def mock_subprocess_run(*args, **kwargs):
            """Mock subprocess.run for successful SSH master connection."""
            return Mock(returncode=0)

        with patch("subprocess.run", side_effect=mock_subprocess_run):
            result = _open_master_connection(
                username="testuser", hostname="slurm.example.com", socket="/tmp/socket"
            )

            assert result == "/tmp/socket"

    def test_open_master_connection_failure(self):
        """Test _open_master_connection with failed SSH connection."""
        from nemo_evaluator_launcher.executors.slurm.executor import (
            _open_master_connection,
        )

        def mock_subprocess_run(*args, **kwargs):
            """Mock subprocess.run for failed SSH master connection."""
            return Mock(returncode=1)

        with patch("subprocess.run", side_effect=mock_subprocess_run):
            result = _open_master_connection(
                username="testuser", hostname="slurm.example.com", socket="/tmp/socket"
            )

            assert result is None

    def test_close_master_connection_success(self):
        """Test _close_master_connection with successful connection close."""
        from nemo_evaluator_launcher.executors.slurm.executor import (
            _close_master_connection,
        )

        def mock_subprocess_run(*args, **kwargs):
            """Mock subprocess.run for successful SSH connection close."""
            return Mock(returncode=0, stderr=b"")

        with patch("subprocess.run", side_effect=mock_subprocess_run):
            # Should not raise an exception
            _close_master_connection(
                username="testuser", hostname="slurm.example.com", socket="/tmp/socket"
            )

    def test_close_master_connection_failure(self):
        """Test _close_master_connection with failed connection close."""
        from nemo_evaluator_launcher.executors.slurm.executor import (
            _close_master_connection,
        )

        def mock_subprocess_run(*args, **kwargs):
            """Mock subprocess.run for failed SSH connection close."""
            return Mock(returncode=1, stderr=b"ssh: connection failed\n")

        with patch("subprocess.run", side_effect=mock_subprocess_run):
            with pytest.raises(
                RuntimeError, match="failed to close the master connection"
            ):
                _close_master_connection(
                    username="testuser",
                    hostname="slurm.example.com",
                    socket="/tmp/socket",
                )

    def test_close_master_connection_none_socket(self):
        """Test _close_master_connection with None socket (should do nothing)."""
        from nemo_evaluator_launcher.executors.slurm.executor import (
            _close_master_connection,
        )

        # Should not call subprocess.run or raise any exception
        with patch("subprocess.run") as mock_run:
            _close_master_connection(
                username="testuser", hostname="slurm.example.com", socket=None
            )
            mock_run.assert_not_called()

    def test_make_remote_execution_output_dir_success(self):
        """Test _make_remote_execution_output_dir with successful directory creation."""
        from nemo_evaluator_launcher.executors.slurm.executor import (
            _make_remote_execution_output_dir,
        )

        def mock_subprocess_run(*args, **kwargs):
            """Mock subprocess.run for successful remote mkdir."""
            return Mock(returncode=0)

        with patch("subprocess.run", side_effect=mock_subprocess_run):
            # Should not raise an exception
            _make_remote_execution_output_dir(
                dirpath="/remote/output",
                username="testuser",
                hostname="slurm.example.com",
                socket="/tmp/socket",
            )

    def test_make_remote_execution_output_dir_failure(self):
        """Test _make_remote_execution_output_dir with failed directory creation."""
        from nemo_evaluator_launcher.executors.slurm.executor import (
            _make_remote_execution_output_dir,
        )

        def mock_subprocess_run(*args, **kwargs):
            """Mock subprocess.run for failed remote mkdir."""
            return Mock(returncode=1, stderr=b"mkdir: permission denied\n")

        with patch("subprocess.run", side_effect=mock_subprocess_run):
            with pytest.raises(
                RuntimeError, match="failed to make a remote execution output dir"
            ):
                _make_remote_execution_output_dir(
                    dirpath="/remote/output",
                    username="testuser",
                    hostname="slurm.example.com",
                    socket="/tmp/socket",
                )

    def test_rsync_upload_rundirs_success(self):
        """Test _rsync_upload_rundirs with successful upload."""
        from pathlib import Path

        from nemo_evaluator_launcher.executors.slurm.executor import (
            _rsync_upload_rundirs,
        )

        def mock_subprocess_run(*args, **kwargs):
            """Mock subprocess.run for successful rsync."""
            return Mock(returncode=0)

        with (
            patch("subprocess.run", side_effect=mock_subprocess_run),
            patch.object(Path, "is_dir", return_value=True),
        ):
            # Should not raise an exception
            _rsync_upload_rundirs(
                local_sources=[Path("/tmp/job1"), Path("/tmp/job2")],
                remote_target="/remote/output",
                username="testuser",
                hostname="slurm.example.com",
            )

    def test_rsync_upload_rundirs_failure(self):
        """Test _rsync_upload_rundirs with failed upload."""
        from pathlib import Path

        from nemo_evaluator_launcher.executors.slurm.executor import (
            _rsync_upload_rundirs,
        )

        def mock_subprocess_run(*args, **kwargs):
            """Mock subprocess.run for failed rsync."""
            return Mock(returncode=1, stderr=b"rsync: connection failed\n")

        with (
            patch("subprocess.run", side_effect=mock_subprocess_run),
            patch.object(Path, "is_dir", return_value=True),
        ):
            with pytest.raises(RuntimeError, match="failed to upload local sources"):
                _rsync_upload_rundirs(
                    local_sources=[Path("/tmp/job1")],
                    remote_target="/remote/output",
                    username="testuser",
                    hostname="slurm.example.com",
                )

    def test_read_autoresumed_slurm_job_ids(self, monkeypatch):
        """Test _read_autoresumed_slurm_job_ids parsing."""
        from nemo_evaluator_launcher.executors.slurm.executor import (
            _read_autoresumed_slurm_job_ids,
        )

        # Mock _read_files_from_remote to return job ID lists
        monkeypatch.setattr(
            "nemo_evaluator_launcher.executors.slurm.executor._read_files_from_remote",
            lambda paths, user, host, sock, suffix="": ["123 456 789", "111 222"],
            raising=True,
        )

        result = _read_autoresumed_slurm_job_ids(
            slurm_job_ids=["123", "111"],
            remote_rundir_paths=[Path("/job1"), Path("/job2")],
            username="user",
            hostname="host",
            socket=None,
        )

        assert result == {"123": ["123", "456", "789"], "111": ["111", "222"]}

    def test_read_files_from_remote_success(self, monkeypatch):
        """Test _read_files_from_remote with successful cat."""
        from nemo_evaluator_launcher.executors.slurm.executor import (
            _read_files_from_remote,
        )

        def mock_subprocess_run(*args, **kwargs):
            return Mock(
                returncode=0,
                stdout=b"_START_OF_FILE_ content1 _END_OF_FILE_ _START_OF_FILE_ content2 _END_OF_FILE_",
            )

        monkeypatch.setattr("subprocess.run", mock_subprocess_run, raising=True)

        result = _read_files_from_remote(
            filepaths=[Path("/file1"), Path("/file2")],
            username="user",
            hostname="host",
            socket="/tmp/sock",
        )

        assert result == ["content1", "content2"]

    def test_read_files_from_remote_failure(self, monkeypatch):
        """Test _read_files_from_remote with failed cat."""
        from nemo_evaluator_launcher.executors.slurm.executor import (
            _read_files_from_remote,
        )

        def mock_subprocess_run(*args, **kwargs):
            return Mock(returncode=1, stderr=b"cat: permission denied")

        monkeypatch.setattr("subprocess.run", mock_subprocess_run, raising=True)

        with pytest.raises(RuntimeError, match="failed to read files from remote"):
            _read_files_from_remote(
                filepaths=[Path("/file1")],
                username="user",
                hostname="host",
                socket=None,
            )

    def test_get_progress_with_dataset_size(self, monkeypatch):
        """Test _get_progress with dataset size calculation."""
        from nemo_evaluator_launcher.executors.slurm.executor import _get_progress

        # Mock file reads
        monkeypatch.setattr(
            "nemo_evaluator_launcher.executors.slurm.executor._read_files_from_remote",
            lambda paths, user, host, sock, suffix="": ["100", "config: test"]
            if "progress" in str(paths[0])
            else ["framework_name: test\nconfig:\n  type: test"],
            raising=True,
        )

        # Mock dataset size calculation
        monkeypatch.setattr(
            "nemo_evaluator_launcher.executors.slurm.executor.get_eval_factory_dataset_size_from_run_config",
            lambda config: 200,
            raising=True,
        )

        result = _get_progress(
            remote_rundir_paths=[Path("/job1")],
            username="user",
            hostname="host",
            socket=None,
        )

        assert result == [0.5]  # 100/200

    def test_get_progress_no_dataset_size(self, monkeypatch):
        """Test _get_progress without dataset size (raw progress)."""
        from nemo_evaluator_launcher.executors.slurm.executor import _get_progress

        # Mock file reads - return progress and valid config, but no dataset size
        def mock_read_files(paths, user, host, sock, suffix=""):
            if "progress" in str(paths[0]):
                return ["150"]
            else:  # run_config paths
                return ["config:\n  type: test\nframework_name: unknown"]

        monkeypatch.setattr(
            "nemo_evaluator_launcher.executors.slurm.executor._read_files_from_remote",
            mock_read_files,
            raising=True,
        )

        # Mock dataset size to return None
        monkeypatch.setattr(
            "nemo_evaluator_launcher.executors.slurm.executor.get_eval_factory_dataset_size_from_run_config",
            lambda config: None,
            raising=True,
        )

        result = _get_progress(
            remote_rundir_paths=[Path("/job1")],
            username="user",
            hostname="host",
            socket=None,
        )

        assert result == [150]  # Raw progress when no dataset size

    def test_get_progress_missing_files(self, monkeypatch):
        """Test _get_progress with missing progress/config files."""
        from nemo_evaluator_launcher.executors.slurm.executor import _get_progress

        # Mock file reads to return empty strings (files not found)
        monkeypatch.setattr(
            "nemo_evaluator_launcher.executors.slurm.executor._read_files_from_remote",
            lambda paths, user, host, sock, suffix="": [""],
            raising=True,
        )

        result = _get_progress(
            remote_rundir_paths=[Path("/job1")],
            username="user",
            hostname="host",
            socket=None,
        )

        assert result == [None]  # None when files missing


class TestSlurmExecutorKillJob:
    def test_kill_job_success(sel, mock_execdb, monkeypatch):
        """Test successful job killing."""
        # Create job in DB
        job_data = JobData(
            invocation_id="kill123",
            job_id="kill123.0",
            timestamp=1234567890.0,
            executor="slurm",
            data={
                "slurm_job_id": "987654321",
                "username": "testuser",
                "hostname": "slurm.example.com",
                "socket": "/tmp/socket",
            },
        )
        db = ExecutionDB()
        db.write_job(job_data)

        # Mock _kill_slurm_job to return success (now returns tuple)
        mock_result = Mock(returncode=0)
        monkeypatch.setattr(
            "nemo_evaluator_launcher.executors.slurm.executor._kill_slurm_job",
            lambda **kwargs: (None, mock_result),
            raising=True,
        )

        # Should not raise
        SlurmExecutor.kill_job("kill123.0")

        # Verify job was marked as killed in DB
        updated_job = db.get_job("kill123.0")
        assert updated_job.data.get("killed") is True

    def test_kill_job_not_found(self):
        """Test kill_job with non-existent job."""
        with pytest.raises(ValueError, match="Job nonexistent.0 not found"):
            SlurmExecutor.kill_job("nonexistent.0")

    def test_kill_job_wrong_executor(sel, mock_execdb, monkeypatch):
        """Test kill_job with job from different executor."""
        job_data = JobData(
            invocation_id="wrong123",
            job_id="wrong123.0",
            timestamp=1234567890.0,
            executor="local",  # Not slurm
            data={},
        )
        db = ExecutionDB()
        db.write_job(job_data)

        with pytest.raises(ValueError, match="Job wrong123.0 is not a slurm job"):
            SlurmExecutor.kill_job("wrong123.0")

    def test_kill_job_kill_command_failed(sel, mock_execdb, monkeypatch):
        """Test kill_job when scancel command fails."""
        job_data = JobData(
            invocation_id="fail123",
            job_id="fail123.0",
            timestamp=1234567890.0,
            executor="slurm",
            data={
                "slurm_job_id": "987654321",
                "username": "testuser",
                "hostname": "slurm.example.com",
            },
        )
        db = ExecutionDB()
        db.write_job(job_data)

        # Mock _kill_slurm_job to return failure (now returns tuple)
        mock_result = Mock(returncode=1)
        monkeypatch.setattr(
            "nemo_evaluator_launcher.executors.slurm.executor._kill_slurm_job",
            lambda **kwargs: ("RUNNING", mock_result),
            raising=True,
        )

        with pytest.raises(RuntimeError, match="Could not find or kill job"):
            SlurmExecutor.kill_job("fail123.0")
