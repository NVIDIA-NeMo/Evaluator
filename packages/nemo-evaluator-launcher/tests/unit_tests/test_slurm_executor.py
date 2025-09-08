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

import warnings
from pathlib import Path
from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

from nemo_evaluator_launcher.executors.slurm.executor import _create_slurm_sbatch_script


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
            mock_get_eval_command.return_value = "nv_eval run_eval --test"
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
