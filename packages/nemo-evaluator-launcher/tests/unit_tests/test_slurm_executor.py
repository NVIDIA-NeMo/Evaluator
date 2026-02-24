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
    _generate_autoresume_handler,
    _generate_deployment_srun_command,
    _resolve_multi_node_config,
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
                "port": 8000,
                "endpoints": {
                    "health": "/health",
                },
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
                "nemo_evaluator_launcher.executors.slurm.executor.get_task_definition_for_job"
            ) as mock_get_task_def,
            patch(
                "nemo_evaluator_launcher.common.helpers.get_eval_factory_command"
            ) as mock_get_eval_command,
            patch(
                "nemo_evaluator_launcher.common.helpers.get_served_model_name"
            ) as mock_get_model_name,
        ):
            mock_load_tasks.return_value = {}
            mock_get_task_def.return_value = {
                "container": "test-eval-container:latest",
                "required_env_vars": [],
                "endpoint_type": "openai",
                "task": "test_task",
            }
            from nemo_evaluator_launcher.common.helpers import CmdAndReadableComment

            mock_get_eval_command.return_value = CmdAndReadableComment(
                cmd="nemo-evaluator run_eval --test", debug="# Test command"
            )
            mock_get_model_name.return_value = "test-model"

            yield {
                "load_tasks_mapping": mock_load_tasks,
                "get_task_definition_for_job": mock_get_task_def,
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
        ).cmd

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
        ).cmd

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
        ).cmd

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
        ).cmd

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
        ).cmd

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
        ).cmd

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
        ).cmd

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
            ).cmd

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
            ).cmd

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
        ).cmd

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
        ).cmd

        # Environment variables should still be exported
        assert "export DEPLOY_VAR=deploy_value" in script
        assert "export EVAL_VAR=eval_value" in script

        # Should not have deployment server section when type is 'none'

        # Evaluation should still be present
        assert "evaluation client" in script
        assert "--container-env EVAL_VAR" in script

        # PRIMARY_NODE should be resolved even without deployment
        assert "Resolve PRIMARY_NODE for single-node sruns" in script
        assert 'export PRIMARY_NODE="${nodes_array[0]}"' in script
        assert '--nodelist "${PRIMARY_NODE}" --nodes 1 --ntasks 1 ' in script

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
            ).cmd

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

    @pytest.mark.parametrize(
        "num_nodes,n_tasks,expected_ntasks,should_have_proxy",
        [
            (1, 1, 1, False),  # Single instance, no proxy
            (4, 4, 4, True),  # Multi-instance with matching n_tasks, needs proxy
            # n_tasks is auto-set to total_nodes for vLLM (overrides user value)
            (2, 1, 2, False),  # 2 nodes, user sets n_tasks=1, but vLLM forces n_tasks=2
            (3, 3, 3, True),  # Multi-instance with 3 nodes, needs proxy
        ],
    )
    def test_deployment_n_tasks_and_proxy_setup(
        self,
        base_config,
        mock_task,
        mock_dependencies,
        num_nodes,
        n_tasks,
        expected_ntasks,
        should_have_proxy,
    ):
        """Test deployment.n_tasks with various configurations and proxy setup."""
        base_config["execution"]["deployment"] = {"n_tasks": n_tasks}
        base_config["execution"]["num_nodes"] = num_nodes
        # Set multiple_instances to trigger proxy setup when needed
        base_config["deployment"]["multiple_instances"] = should_have_proxy

        cfg = OmegaConf.create(base_config)

        script = _create_slurm_sbatch_script(
            cfg=cfg,
            task=mock_task,
            eval_image="test-eval-container:latest",
            remote_task_subdir=Path("/test/remote"),
            invocation_id="test123",
            job_id="test123.0",
        ).cmd

        # Check that deployment srun uses correct --ntasks value
        assert f"--nodes {num_nodes} --ntasks {expected_ntasks}" in script

        # Check proxy setup based on multi-instance or not
        if should_have_proxy:
            assert "proxy" in script.lower()
        else:
            assert "proxy" not in script.lower()

    def test_deployment_n_tasks_default_value(
        self, base_config, mock_task, mock_dependencies
    ):
        """Test deployment.n_tasks auto-set to total_nodes for vLLM when not specified."""
        # Don't set deployment.n_tasks explicitly.
        # For vLLM, n_tasks is auto-set to total_nodes (num_nodes_per_instance × num_instances).
        base_config["execution"]["num_nodes"] = 2

        cfg = OmegaConf.create(base_config)

        script = _create_slurm_sbatch_script(
            cfg=cfg,
            task=mock_task,
            eval_image="test-eval-container:latest",
            remote_task_subdir=Path("/test/remote"),
            invocation_id="test123",
            job_id="test123.0",
        ).cmd

        # vLLM auto-sets n_tasks = total_nodes = 2
        assert "--nodes 2 --ntasks 2" in script


class TestMaxWalltimeFeature:
    """Test maximum wall-clock time feature for preventing infinite job resuming."""

    @pytest.fixture
    def base_config(self):
        """Base configuration for testing."""
        return {
            "deployment": {
                "type": "vllm",
                "image": "test-image:latest",
                "command": "test-command",
                "served_model_name": "test-model",
                "port": 8000,
                "endpoints": {
                    "health": "/health",
                },
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
    def mock_dependencies(self):
        """Mock external dependencies used by _create_slurm_sbatch_script."""
        with (
            patch(
                "nemo_evaluator_launcher.executors.slurm.executor.load_tasks_mapping"
            ) as mock_load_tasks,
            patch(
                "nemo_evaluator_launcher.executors.slurm.executor.get_task_definition_for_job"
            ) as mock_get_task_def,
            patch(
                "nemo_evaluator_launcher.common.helpers.get_eval_factory_command"
            ) as mock_get_eval_command,
            patch(
                "nemo_evaluator_launcher.common.helpers.get_served_model_name"
            ) as mock_get_model_name,
        ):
            mock_load_tasks.return_value = {}
            mock_get_task_def.return_value = {
                "container": "test-eval-container:latest",
                "required_env_vars": [],
                "endpoint_type": "openai",
                "task": "test_task",
            }
            from nemo_evaluator_launcher.common.helpers import CmdAndReadableComment

            mock_get_eval_command.return_value = CmdAndReadableComment(
                cmd="nemo-evaluator run_eval --test", debug="# Test command"
            )
            mock_get_model_name.return_value = "test-model"

            yield {
                "load_tasks_mapping": mock_load_tasks,
                "get_task_definition_for_job": mock_get_task_def,
                "get_eval_factory_command": mock_get_eval_command,
                "get_served_model_name": mock_get_model_name,
            }

    def test_generate_autoresume_handler_without_max_walltime(self):
        """Test autoresume handler generation without max_walltime."""
        handler = _generate_autoresume_handler(Path("/test/remote"), max_walltime=None)

        # Should have basic autoresume logic
        assert "_this_script=$0" in handler
        assert "_prev_slurm_job_id=$1" in handler
        assert "sbatch --dependency=afternotok:$SLURM_JOB_ID" in handler

        # Should NOT have max_walltime checks
        assert "_max_walltime=" not in handler
        assert "Maximum total walltime" not in handler
        assert "_accumulated_seconds" not in handler

    def test_generate_autoresume_handler_with_max_walltime(self):
        """Test autoresume handler generation with max_walltime."""
        handler = _generate_autoresume_handler(
            Path("/test/remote"), max_walltime="24:00:00"
        )

        # Should have basic autoresume logic
        assert "_this_script=$0" in handler
        assert "_prev_slurm_job_id=$1" in handler
        assert "sbatch --dependency=afternotok:$SLURM_JOB_ID" in handler

        # Should have max_walltime checks
        assert '_max_walltime="24:00:00"' in handler
        assert "/test/remote/.job_start_time" in handler
        assert "/test/remote/.accumulated_walltime" in handler
        assert "_walltime_to_seconds()" in handler
        assert "_accumulated_seconds" in handler
        assert "Maximum total walltime" in handler
        assert "Stopping job chain to prevent infinite resuming" in handler
        # Should use sacct to get actual elapsed time from previous jobs
        assert "sacct -j $_prev_slurm_job_id -P -n -o Elapsed" in handler

    def test_generate_autoresume_handler_max_walltime_formats(self):
        """Test autoresume handler with various max_walltime formats."""
        # Test HH:MM:SS format
        handler = _generate_autoresume_handler(
            Path("/test/remote"), max_walltime="12:30:45"
        )
        assert '_max_walltime="12:30:45"' in handler

        # Test short format
        handler = _generate_autoresume_handler(
            Path("/test/remote"), max_walltime="02:00:00"
        )
        assert '_max_walltime="02:00:00"' in handler

    def test_create_sbatch_script_without_max_walltime(
        self, base_config, mock_task, mock_dependencies
    ):
        """Test sbatch script generation without explicit max_walltime uses default."""
        cfg = OmegaConf.create(base_config)

        script = _create_slurm_sbatch_script(
            cfg=cfg,
            task=mock_task,
            eval_image="test-eval-container:latest",
            remote_task_subdir=Path("/test/remote"),
            invocation_id="test123",
            job_id="test123.0",
        ).cmd

        # Should have autoresume logic WITH default max_walltime (120:00:00 = 5 days)
        assert "_this_script=$0" in script
        assert "sbatch --dependency=afternotok:$SLURM_JOB_ID" in script
        assert '_max_walltime="120:00:00"' in script
        assert "_accumulated_walltime_file" in script
        assert "Maximum total walltime" in script

    def test_create_sbatch_script_with_max_walltime(
        self, base_config, mock_task, mock_dependencies
    ):
        """Test sbatch script generation with max_walltime."""
        base_config["execution"]["max_walltime"] = "24:00:00"
        cfg = OmegaConf.create(base_config)

        script = _create_slurm_sbatch_script(
            cfg=cfg,
            task=mock_task,
            eval_image="test-eval-container:latest",
            remote_task_subdir=Path("/test/remote"),
            invocation_id="test123",
            job_id="test123.0",
        ).cmd

        # Should have autoresume logic WITH max_walltime checks
        assert "_this_script=$0" in script
        assert "sbatch --dependency=afternotok:$SLURM_JOB_ID" in script
        assert '_max_walltime="24:00:00"' in script
        assert "_accumulated_seconds" in script
        assert "Maximum total walltime" in script
        # Should use sacct for accurate walltime tracking
        assert "sacct" in script

    def test_create_sbatch_script_max_walltime_null(
        self, base_config, mock_task, mock_dependencies
    ):
        """Test sbatch script generation with max_walltime explicitly set to null for unlimited."""
        base_config["execution"]["max_walltime"] = None
        cfg = OmegaConf.create(base_config)

        script = _create_slurm_sbatch_script(
            cfg=cfg,
            task=mock_task,
            eval_image="test-eval-container:latest",
            remote_task_subdir=Path("/test/remote"),
            invocation_id="test123",
            job_id="test123.0",
        ).cmd

        # When explicitly set to None, should have autoresume logic but NO max_walltime checks
        assert "_this_script=$0" in script
        assert "sbatch --dependency=afternotok:$SLURM_JOB_ID" in script
        assert "_max_walltime=" not in script
        assert "_accumulated_walltime_file" not in script

    def test_autoresume_handler_creates_start_time_file(self):
        """Test that autoresume handler creates accumulated walltime file on first run."""
        handler = _generate_autoresume_handler(
            Path("/test/remote"), max_walltime="08:00:00"
        )

        # Should create accumulated walltime file on first run or manual resume
        assert "_accumulated_walltime_file" in handler
        assert 'echo "0" > "$_accumulated_walltime_file"' in handler
        assert "Job chain started at" in handler
        # Should still write start time for current job
        assert 'date +%s > "$_start_time_file"' in handler

    def test_autoresume_handler_time_conversion(self):
        """Test that autoresume handler includes time conversion logic."""
        handler = _generate_autoresume_handler(
            Path("/test/remote"), max_walltime="10:30:00"
        )

        # Should have time conversion function
        assert "_walltime_to_seconds()" in handler
        assert "hours * 3600 + minutes * 60 + seconds" in handler

        # Should handle different time formats
        assert "HH:MM:SS" in handler or "BASH_REMATCH" in handler

    def test_autoresume_handler_elapsed_time_formatting(self):
        """Test that autoresume handler formats elapsed time for logging."""
        handler = _generate_autoresume_handler(
            Path("/test/remote"), max_walltime="04:00:00"
        )

        # Should format elapsed time for human-readable output
        assert "_elapsed_formatted" in handler
        assert "printf" in handler


class TestSlurmExecutorHelperFunctions:
    """Test individual helper functions used by SLURM executor."""

    @pytest.mark.parametrize(
        "num_nodes,n_tasks,has_mounts,mount_home,expected_nodes,expected_ntasks,expected_mount_home_flag",
        [
            (1, 1, False, True, 1, 1, False),  # Single node, no mounts, mount home
            (4, 4, False, True, 4, 4, False),  # Multi-node, no mounts, mount home
            # n_tasks is auto-set to total_nodes for vLLM (overrides user value)
            (
                2,
                1,
                True,
                True,
                2,
                2,
                False,
            ),  # 2 nodes, user sets n_tasks=1, but vLLM forces n_tasks=2
            (1, 1, False, False, 1, 1, True),  # Single node, no mount home
            (3, 3, True, False, 3, 3, True),  # Multi-node with mounts, no mount home
        ],
    )
    def test_generate_deployment_srun_command(
        self,
        num_nodes,
        n_tasks,
        has_mounts,
        mount_home,
        expected_nodes,
        expected_ntasks,
        expected_mount_home_flag,
    ):
        """Test _generate_deployment_srun_command with various configurations."""
        from nemo_evaluator_launcher.executors.slurm.executor import (
            _generate_deployment_srun_command,
            _resolve_multi_node_config,
        )

        # Create config
        config = {
            "deployment": {
                "type": "vllm",
                "image": "test-image:latest",
                "command": "python -m vllm.entrypoints.openai.api_server --model /model",
            },
            "execution": {
                "num_nodes": num_nodes,
                "deployment": {"n_tasks": n_tasks},
                "mounts": {"mount_home": mount_home},
            },
        }
        cfg = OmegaConf.create(config)
        _resolve_multi_node_config(cfg)

        # Create mounts list
        mounts_list = ["/host/path:/container/path"] if has_mounts else []

        # Generate command
        command, _, _ = _generate_deployment_srun_command(
            cfg=cfg,
            deployment_mounts_list=mounts_list,
            remote_task_subdir=Path("/test/remote"),
        )

        # Verify nodes and ntasks
        assert f"--nodes {expected_nodes} --ntasks {expected_ntasks}" in command

        # Verify image
        assert "test-image:latest" in command

        # Verify mounts
        if has_mounts:
            assert "/host/path:/container/path" in command

        # Verify mount_home flag
        if expected_mount_home_flag:
            assert "--no-container-mount-home" in command
        else:
            assert "--no-container-mount-home" not in command

        # Verify node IP collection
        assert "NODES_IPS_ARRAY" in command
        assert "MASTER_IP" in command

    @pytest.mark.parametrize(
        "ip_list,port,health_path,service_name,check_pid,expected_in_output",
        [
            (
                '"127.0.0.1"',
                8000,
                "/health",
                "server",
                True,
                ["127.0.0.1", "8000", "/health", "server", "SERVER_PID"],
            ),
            (
                '"${NODES_IPS_ARRAY[@]}"',
                5009,
                "/status",
                "Proxy",
                False,
                ["NODES_IPS_ARRAY", "5009", "/status", "Proxy"],
            ),
            (
                '"10.0.0.1"',
                8080,
                "/ready",
                "service",
                True,
                ["10.0.0.1", "8080", "/ready", "service", "SERVER_PID"],
            ),
            (
                '"${NODES_IPS_ARRAY[@]}"',
                8000,
                "/health",
                "server",
                True,
                ["NODES_IPS_ARRAY", "8000", "/health", "SERVER_PID"],
            ),
        ],
    )
    def test_get_wait_for_server_handler(
        self, ip_list, port, health_path, service_name, check_pid, expected_in_output
    ):
        """Test _get_wait_for_server_handler with various configurations."""
        from nemo_evaluator_launcher.executors.slurm.executor import (
            _get_wait_for_server_handler,
        )

        # Generate handler
        handler = _get_wait_for_server_handler(
            ip_list=ip_list,
            port=port,
            health_check_path=health_path,
            service_name=service_name,
            check_pid=check_pid,
        )

        # Verify all expected strings are in output
        for expected in expected_in_output:
            assert expected in handler

        # Verify PID check logic
        if check_pid:
            assert "SERVER_PID" in handler
            assert "kill -0" in handler
        else:
            assert "kill -0" not in handler

        # Verify curl command structure
        assert "curl -s -o /dev/null" in handler
        assert f"http://$ip:{port}{health_path}" in handler

        # Verify loop structure
        assert "for ip in" in handler
        assert "while" in handler
        assert "done" in handler


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
                        "nemo_evaluator_config": {
                            "config": {"params": {"temperature": 0.95}}
                        },
                    },
                    {
                        "name": "gsm8k",
                        "container": "custom-math-container:v2.0",
                        "nemo_evaluator_config": {"config": {"params": {"top_p": 0.1}}},
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
                    "nemo_evaluator_launcher.executors.slurm.executor.get_task_definition_for_job"
                ) as mock_get_task_def,
                patch(
                    "nemo_evaluator_launcher.executors.slurm.executor.get_eval_factory_command"
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
                    cmd="nemo-evaluator-launcher --model llama-3.1-8b-instruct --task {task_name}",
                    debug="# Test command for dry run",
                )

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
                "nemo_evaluator_launcher.executors.slurm.executor.get_task_definition_for_job"
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
                    "nemo_evaluator_launcher.executors.slurm.executor.get_task_definition_for_job"
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
                    "nemo_evaluator_launcher.executors.slurm.executor.get_task_definition_for_job"
                ) as mock_get_task_def,
                patch(
                    "nemo_evaluator_launcher.executors.slurm.executor.get_eval_factory_command"
                ) as mock_get_command,
                patch("builtins.print"),
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
                from nemo_evaluator_launcher.common.helpers import CmdAndReadableComment

                mock_get_command.return_value = CmdAndReadableComment(
                    cmd="nemo-evaluator-launcher --task test_command",
                    debug="# Test command for custom container",
                )

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
                    "nemo_evaluator_launcher.executors.slurm.executor.get_task_definition_for_job"
                ) as mock_get_task_def,
                patch(
                    "nemo_evaluator_launcher.executors.slurm.executor.get_eval_factory_command"
                ) as mock_get_command,
                patch("builtins.print"),
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
                from nemo_evaluator_launcher.common.helpers import CmdAndReadableComment

                mock_get_command.return_value = CmdAndReadableComment(
                    cmd="nemo-evaluator-launcher --task test_command",
                    debug="# Test command for no auto-export",
                )

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
            mock_query_status.return_value = {"123456789": ("COMPLETED", "123456789")}
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
                {"123456789": ("PREEMPTED", "123456789")},  # Original job status
                {
                    "123456790": ("RUNNING", "123456790")
                },  # Latest autoresumed job status
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
            mock_query_status.return_value = {"123456789": ("RUNNING", "123456789")}
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
                        "nemo_evaluator_config": {
                            "config": {"params": {"temperature": 0.95}}
                        },
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
                    "nemo_evaluator_launcher.executors.slurm.executor.get_task_definition_for_job"
                ) as mock_get_task_def,
                patch(
                    "nemo_evaluator_launcher.executors.slurm.executor.get_eval_factory_command"
                ) as mock_get_command,
                patch("subprocess.run", side_effect=mock_subprocess_run),
                patch(
                    "nemo_evaluator_launcher.executors.slurm.executor._open_master_connection"
                ) as mock_open_connection,
            ):
                # Configure mocks
                mock_load_mapping.return_value = mock_tasks_mapping
                mock_open_connection.return_value = "/tmp/socket"

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
                    cmd="nemo-evaluator-launcher --task mmlu_pro",
                    debug="# Test command for mmlu_pro",
                )

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
                    "nemo_evaluator_launcher.executors.slurm.executor.get_task_definition_for_job"
                ) as mock_get_task_def,
                patch(
                    "nemo_evaluator_launcher.executors.slurm.executor.get_eval_factory_command"
                ) as mock_get_command,
                patch("subprocess.run", side_effect=mock_subprocess_run),
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
                    cmd="nemo-evaluator-launcher --task mmlu_pro",
                    debug="# Test command for mmlu_pro SSH failure",
                )

                with pytest.raises(
                    RuntimeError,
                    match="Failed to connect to the cluster slurm.example.com as user testuser. Please check your SSH configuration.",
                ):
                    SlurmExecutor.execute_eval(sample_config, dry_run=False)

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
                    "nemo_evaluator_launcher.executors.slurm.executor.get_task_definition_for_job"
                ) as mock_get_task_def,
                patch(
                    "nemo_evaluator_launcher.executors.slurm.executor.get_eval_factory_command"
                ) as mock_get_command,
                patch("subprocess.run", side_effect=mock_subprocess_run),
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
                    cmd="nemo-evaluator-launcher --task mmlu_pro",
                    debug="# Test command for mmlu_pro sbatch failure",
                )

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
            """Mock subprocess.run for squeue and sacct commands."""
            cmd_args = kwargs.get("args", [])
            if not cmd_args:
                return Mock(returncode=1, stdout=b"", stderr=b"")

            cmd_str = (
                " ".join(cmd_args) if isinstance(cmd_args, list) else str(cmd_args)
            )

            if "squeue" in cmd_str:
                # Mock squeue with no active jobs (empty output)
                return Mock(returncode=0, stdout=b"", stderr=b"")
            elif "sacct" in cmd_str:
                # Mock sacct output
                return Mock(
                    returncode=0,
                    stdout=b"123456789|COMPLETED\n123456790|RUNNING\n",
                    stderr=b"",
                )
            return Mock(returncode=1, stdout=b"", stderr=b"")

        with patch("subprocess.run", side_effect=mock_subprocess_run):
            result = _query_slurm_jobs_status(
                slurm_job_ids=["123456789", "123456790"],
                username="testuser",
                hostname="slurm.example.com",
                socket="/tmp/socket",
            )

            assert result["123456789"] == ("COMPLETED", "123456789")
            assert result["123456790"] == ("RUNNING", "123456790")

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

    def test_query_squeue_for_jobs_success(self):
        """Test _query_squeue_for_jobs function with successful subprocess call."""
        from nemo_evaluator_launcher.executors.slurm.executor import (
            _query_squeue_for_jobs,
        )

        def mock_subprocess_run(*args, **kwargs):
            """Mock subprocess.run for squeue command."""
            # Mock squeue output with various job formats
            return Mock(
                returncode=0,
                stdout=b"123456789|RUNNING|\n123456790_0|PENDING|(null)\n123456791[1-10]|PENDING|\n",
                stderr=b"",
            )

        with patch("subprocess.run", side_effect=mock_subprocess_run):
            result = _query_squeue_for_jobs(
                slurm_job_ids=["123456789", "123456790", "123456791"],
                username="testuser",
                hostname="slurm.example.com",
                socket="/tmp/socket",
            )

            assert result["123456789"] == ("RUNNING", "123456789")
            assert result["123456790"] == ("PENDING", "123456790")
            assert result["123456791"] == ("PENDING", "123456791")

    def test_query_squeue_for_jobs_finds_dependent_jobs(self):
        """Test that _query_squeue_for_jobs finds follow-up jobs that depend on known jobs."""
        from nemo_evaluator_launcher.executors.slurm.executor import (
            _query_squeue_for_jobs,
        )

        def mock_subprocess_run(*args, **kwargs):
            """Mock subprocess.run for squeue command with dependent jobs."""
            # Simulate: job 123456789 has finished (not in squeue),
            # but job 123456790 is PENDING with dependency on 123456789
            return Mock(
                returncode=0,
                stdout=b"123456790|PENDING|afternotok:123456789\n123456791|RUNNING|(null)\n",
                stderr=b"",
            )

        with patch("subprocess.run", side_effect=mock_subprocess_run):
            result = _query_squeue_for_jobs(
                slurm_job_ids=["123456789", "123456791"],
                username="testuser",
                hostname="slurm.example.com",
                socket="/tmp/socket",
            )
            assert result["123456789"] == (
                "PENDING",
                "123456790",
            )  # Should find 123456789's status via its dependent job 123456790
            assert result["123456791"] == (
                "RUNNING",
                "123456791",
            )  # Direct match for 123456791

    def test_query_slurm_jobs_status_combined_approach(self):
        """Test _query_slurm_jobs_status using combined squeue + sacct approach."""
        from nemo_evaluator_launcher.executors.slurm.executor import (
            _query_slurm_jobs_status,
        )

        def mock_subprocess_run(*args, **kwargs):
            """Mock subprocess.run for both squeue and sacct commands."""
            # Get the command from kwargs['args'] since that's how subprocess.run is called
            cmd_args = kwargs.get("args", [])
            if not cmd_args:
                return Mock(returncode=1, stdout=b"", stderr=b"")

            cmd_str = (
                " ".join(cmd_args) if isinstance(cmd_args, list) else str(cmd_args)
            )

            if "squeue" in cmd_str:
                # Mock squeue showing only running jobs
                return Mock(
                    returncode=0,
                    stdout=b"123456789|RUNNING|(null)\n",
                    stderr=b"",
                )
            elif "sacct" in cmd_str:
                # Mock sacct showing completed job that's not in squeue
                return Mock(
                    returncode=0,
                    stdout=b"123456790|COMPLETED\n",
                    stderr=b"",
                )
            return Mock(returncode=1, stdout=b"", stderr=b"")

        with patch("subprocess.run", side_effect=mock_subprocess_run):
            result = _query_slurm_jobs_status(
                slurm_job_ids=["123456789", "123456790"],
                username="testuser",
                hostname="slurm.example.com",
                socket="/tmp/socket",
            )

            # Should get running job from squeue and completed job from sacct
            assert result["123456789"] == ("RUNNING", "123456789")
            assert result["123456790"] == ("COMPLETED", "123456790")

    def test_query_sacct_for_jobs_success(self):
        """Test _query_sacct_for_jobs function with successful subprocess call."""
        from nemo_evaluator_launcher.executors.slurm.executor import (
            _query_sacct_for_jobs,
        )

        def mock_subprocess_run(*args, **kwargs):
            """Mock subprocess.run for sacct command."""
            return Mock(
                returncode=0,
                stdout=b"123456789|COMPLETED\n123456790|FAILED\n",
                stderr=b"",
            )

        with patch("subprocess.run", side_effect=mock_subprocess_run):
            result = _query_sacct_for_jobs(
                slurm_job_ids=["123456789", "123456790"],
                username="testuser",
                hostname="slurm.example.com",
                socket="/tmp/socket",
            )

            assert result == {
                "123456789": ("COMPLETED", "123456789"),
                "123456790": ("FAILED", "123456790"),
            }

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
            lambda paths, user, host, sock: ["123 456 789", "111 222"],
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
            lambda paths, user, host, sock: ["100", "config: test"]
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
        def mock_read_files(paths, user, host, sock):
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
            lambda paths, user, host, sock: [""],
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


class TestMultiNodeConfig:
    """Comprehensive tests for the multi-node config redesign.

    Covers _resolve_multi_node_config, Ray auto-injection,
    HAProxy auto-enablement, and n_tasks auto-set for vLLM.
    """

    @pytest.fixture
    def base_config(self):
        """Minimal config without num_nodes/num_nodes_per_instance/num_instances.

        Each test sets exactly the fields it needs.
        """
        return {
            "deployment": {
                "type": "vllm",
                "image": "test-image:latest",
                "command": "python -m vllm.entrypoints.openai.api_server --model /model",
                "served_model_name": "test-model",
                "port": 8000,
                "endpoints": {"health": "/health"},
            },
            "execution": {
                "type": "slurm",
                "output_dir": "/test/output",
                "walltime": "01:00:00",
                "account": "test-account",
                "partition": "test-partition",
                "ntasks_per_node": 1,
                "subproject": "test-subproject",
            },
            "evaluation": {"env_vars": {}},
            "target": {"api_endpoint": {"url": "http://localhost:8000/v1"}},
        }

    @pytest.fixture
    def mock_task(self):
        return OmegaConf.create({"name": "test_task"})

    @pytest.fixture
    def mock_dependencies(self):
        """Mock external dependencies used by _create_slurm_sbatch_script."""
        with (
            patch(
                "nemo_evaluator_launcher.executors.slurm.executor.load_tasks_mapping"
            ) as mock_load_tasks,
            patch(
                "nemo_evaluator_launcher.executors.slurm.executor.get_task_definition_for_job"
            ) as mock_get_task_def,
            patch(
                "nemo_evaluator_launcher.common.helpers.get_eval_factory_command"
            ) as mock_get_eval_command,
            patch(
                "nemo_evaluator_launcher.common.helpers.get_served_model_name"
            ) as mock_get_model_name,
        ):
            mock_load_tasks.return_value = {}
            mock_get_task_def.return_value = {
                "container": "test-eval-container:latest",
                "required_env_vars": [],
                "endpoint_type": "openai",
                "task": "test_task",
            }
            from nemo_evaluator_launcher.common.helpers import CmdAndReadableComment

            mock_get_eval_command.return_value = CmdAndReadableComment(
                cmd="nemo-evaluator run_eval --test", debug="# Test command"
            )
            mock_get_model_name.return_value = "test-model"
            yield

    # ------------------------------------------------------------------ #
    # Category 1a: New-style configs
    # ------------------------------------------------------------------ #
    @pytest.mark.parametrize(
        "npi_in,ni_in,expected_npi,expected_ni",
        [
            (None, None, 1, 1),  # defaults
            (4, None, 4, 1),  # NPI only
            (None, 3, 1, 3),  # NI only
            (2, 3, 2, 3),  # both
        ],
        ids=["defaults", "npi_only", "ni_only", "both"],
    )
    def test_resolve_new_style(
        self, base_config, npi_in, ni_in, expected_npi, expected_ni
    ):
        if npi_in is not None:
            base_config["execution"]["num_nodes_per_instance"] = npi_in
        if ni_in is not None:
            base_config["execution"]["num_instances"] = ni_in

        cfg = OmegaConf.create(base_config)
        _resolve_multi_node_config(cfg)

        assert cfg.execution.num_nodes_per_instance == expected_npi
        assert cfg.execution.num_instances == expected_ni
        # vLLM: n_tasks should equal total nodes
        assert cfg.execution.deployment.n_tasks == expected_npi * expected_ni

    # ------------------------------------------------------------------ #
    # Category 1b: Deprecated num_nodes resolution
    # ------------------------------------------------------------------ #
    @pytest.mark.parametrize(
        "nn,npi_in,ni_in,expected_npi,expected_ni",
        [
            (2, None, None, 2, 1),  # NN alone → NPI=NN, NI=1
            (4, None, None, 4, 1),  # NN alone (larger)
            (6, 2, None, 2, 3),  # NN+NPI → NI derived
            (6, None, 3, 2, 3),  # NN+NI → NPI derived
            (6, 2, 3, 2, 3),  # NN+NPI+NI → consistency check
        ],
        ids=["nn_alone_2", "nn_alone_4", "nn_npi", "nn_ni", "nn_npi_ni"],
    )
    def test_resolve_deprecated_num_nodes(
        self, base_config, nn, npi_in, ni_in, expected_npi, expected_ni
    ):
        base_config["execution"]["num_nodes"] = nn
        if npi_in is not None:
            base_config["execution"]["num_nodes_per_instance"] = npi_in
        if ni_in is not None:
            base_config["execution"]["num_instances"] = ni_in

        cfg = OmegaConf.create(base_config)
        _resolve_multi_node_config(cfg)

        assert cfg.execution.num_nodes_per_instance == expected_npi
        assert cfg.execution.num_instances == expected_ni
        # num_nodes should be deleted
        assert cfg.execution.get("num_nodes") is None

    # ------------------------------------------------------------------ #
    # Category 1c: Deprecated num_nodes errors
    # ------------------------------------------------------------------ #
    @pytest.mark.parametrize(
        "nn,npi_in,ni_in,match",
        [
            (7, 2, None, "must be divisible by.*num_nodes_per_instance"),
            (7, None, 3, "must be divisible by.*num_instances"),
            (
                6,
                2,
                2,
                r"num_nodes \(6\) != .*num_nodes_per_instance \(2\) \* .*num_instances \(2\)",
            ),
        ],
        ids=["nn_npi_indivisible", "nn_ni_indivisible", "nn_npi_ni_inconsistent"],
    )
    def test_resolve_deprecated_num_nodes_errors(
        self, base_config, nn, npi_in, ni_in, match
    ):
        base_config["execution"]["num_nodes"] = nn
        if npi_in is not None:
            base_config["execution"]["num_nodes_per_instance"] = npi_in
        if ni_in is not None:
            base_config["execution"]["num_instances"] = ni_in

        cfg = OmegaConf.create(base_config)
        with pytest.raises(ValueError, match=match):
            _resolve_multi_node_config(cfg)

    # ------------------------------------------------------------------ #
    # Category 1d: Deprecated multiple_instances
    # ------------------------------------------------------------------ #
    def test_mi_true_with_nn(self, base_config):
        """MI=True + NN=4 → NPI=1, NI=4."""
        base_config["deployment"]["multiple_instances"] = True
        base_config["execution"]["num_nodes"] = 4

        cfg = OmegaConf.create(base_config)
        _resolve_multi_node_config(cfg)

        assert cfg.execution.num_nodes_per_instance == 1
        assert cfg.execution.num_instances == 4
        assert cfg.deployment.get("multiple_instances") is None

    def test_mi_true_no_nn_error(self, base_config):
        """MI=True without NN or NI → ValueError."""
        base_config["deployment"]["multiple_instances"] = True

        cfg = OmegaConf.create(base_config)
        with pytest.raises(ValueError, match="Set execution.num_instances instead"):
            _resolve_multi_node_config(cfg)

    def test_mi_true_ni_set_ignored(self, base_config):
        """MI=True + NI=2 already set → MI ignored, NI=2."""
        base_config["deployment"]["multiple_instances"] = True
        base_config["execution"]["num_instances"] = 2

        cfg = OmegaConf.create(base_config)
        _resolve_multi_node_config(cfg)

        assert cfg.execution.num_instances == 2
        assert cfg.execution.num_nodes_per_instance == 1
        assert cfg.deployment.get("multiple_instances") is None

    def test_mi_true_npi_set_ignored(self, base_config):
        """MI=True + NPI=2 already set → MI ignored, NPI=2."""
        base_config["deployment"]["multiple_instances"] = True
        base_config["execution"]["num_nodes_per_instance"] = 2

        cfg = OmegaConf.create(base_config)
        _resolve_multi_node_config(cfg)

        assert cfg.execution.num_nodes_per_instance == 2
        assert cfg.execution.num_instances == 1
        assert cfg.deployment.get("multiple_instances") is None

    def test_mi_false_warns(self, base_config):
        """MI=False → warns, NPI=1, NI=1."""
        base_config["deployment"]["multiple_instances"] = False

        cfg = OmegaConf.create(base_config)
        with patch(
            "nemo_evaluator_launcher.executors.slurm.executor.logger"
        ) as mock_logger:
            _resolve_multi_node_config(cfg)
            mock_logger.warning.assert_called()

        assert cfg.execution.num_nodes_per_instance == 1
        assert cfg.execution.num_instances == 1
        assert cfg.deployment.get("multiple_instances") is None

    # ------------------------------------------------------------------ #
    # Category 1e: n_tasks auto-set
    # ------------------------------------------------------------------ #
    @pytest.mark.parametrize(
        "deploy_type,n_tasks_in,npi,ni,expected_n_tasks",
        [
            ("vllm", None, 2, 3, 6),  # vLLM, unset → auto-set
            ("vllm", 1, 2, 3, 6),  # vLLM, wrong value → overwritten
            ("vllm", 6, 2, 3, 6),  # vLLM, already correct
            ("trtllm", None, 1, 1, 1),  # non-vLLM, unset → default 1
            ("trtllm", 4, 1, 1, 4),  # non-vLLM, explicit → respected
        ],
        ids=[
            "vllm_unset",
            "vllm_wrong",
            "vllm_correct",
            "nonvllm_unset",
            "nonvllm_explicit",
        ],
    )
    def test_n_tasks_auto_set(
        self, base_config, deploy_type, n_tasks_in, npi, ni, expected_n_tasks
    ):
        base_config["deployment"]["type"] = deploy_type
        base_config["execution"]["num_nodes_per_instance"] = npi
        base_config["execution"]["num_instances"] = ni
        if n_tasks_in is not None:
            base_config["execution"]["deployment"] = {"n_tasks": n_tasks_in}

        cfg = OmegaConf.create(base_config)
        _resolve_multi_node_config(cfg)

        assert cfg.execution.deployment.n_tasks == expected_n_tasks

    # ------------------------------------------------------------------ #
    # Category 1f: Idempotency
    # ------------------------------------------------------------------ #
    def test_idempotency_new_style(self, base_config):
        """Calling _resolve_multi_node_config twice with new-style gives same result."""
        base_config["execution"]["num_nodes_per_instance"] = 2
        base_config["execution"]["num_instances"] = 3

        cfg = OmegaConf.create(base_config)
        _resolve_multi_node_config(cfg)
        first_npi = cfg.execution.num_nodes_per_instance
        first_ni = cfg.execution.num_instances
        first_n_tasks = cfg.execution.deployment.n_tasks

        _resolve_multi_node_config(cfg)
        assert cfg.execution.num_nodes_per_instance == first_npi
        assert cfg.execution.num_instances == first_ni
        assert cfg.execution.deployment.n_tasks == first_n_tasks

    def test_idempotency_after_deprecated_num_nodes(self, base_config):
        """Calling twice after deprecated num_nodes deletion doesn't crash."""
        base_config["execution"]["num_nodes"] = 4

        cfg = OmegaConf.create(base_config)
        _resolve_multi_node_config(cfg)
        # num_nodes is now deleted; second call should work
        _resolve_multi_node_config(cfg)

        assert cfg.execution.num_nodes_per_instance == 4
        assert cfg.execution.num_instances == 1

    # ------------------------------------------------------------------ #
    # Category 2: Deprecation warnings
    # ------------------------------------------------------------------ #
    def test_deprecation_warning_num_nodes(self, base_config):
        base_config["execution"]["num_nodes"] = 2

        cfg = OmegaConf.create(base_config)
        with patch(
            "nemo_evaluator_launcher.executors.slurm.executor.logger"
        ) as mock_logger:
            _resolve_multi_node_config(cfg)
            calls = [str(c) for c in mock_logger.warning.call_args_list]
            assert any("num_nodes is deprecated" in c for c in calls)

    def test_deprecation_warning_multiple_instances(self, base_config):
        base_config["deployment"]["multiple_instances"] = True
        base_config["execution"]["num_nodes"] = 2

        cfg = OmegaConf.create(base_config)
        with patch(
            "nemo_evaluator_launcher.executors.slurm.executor.logger"
        ) as mock_logger:
            _resolve_multi_node_config(cfg)
            calls = [str(c) for c in mock_logger.warning.call_args_list]
            assert any("multiple_instances is deprecated" in c for c in calls)

    def test_deprecation_warning_mi_false(self, base_config):
        base_config["deployment"]["multiple_instances"] = False

        cfg = OmegaConf.create(base_config)
        with patch(
            "nemo_evaluator_launcher.executors.slurm.executor.logger"
        ) as mock_logger:
            _resolve_multi_node_config(cfg)
            calls = [str(c) for c in mock_logger.warning.call_args_list]
            assert any("multiple_instances is deprecated" in c for c in calls)

    def test_no_deprecation_warning_new_style(self, base_config):
        base_config["execution"]["num_nodes_per_instance"] = 2
        base_config["execution"]["num_instances"] = 3

        cfg = OmegaConf.create(base_config)
        with patch(
            "nemo_evaluator_launcher.executors.slurm.executor.logger"
        ) as mock_logger:
            _resolve_multi_node_config(cfg)
            mock_logger.warning.assert_not_called()

    # ------------------------------------------------------------------ #
    # Category 3: Ray injection in _generate_deployment_srun_command
    # ------------------------------------------------------------------ #
    def test_ray_injected_for_multi_node_vllm(self, base_config):
        """NPI>1 + vLLM → Ray injected."""
        base_config["execution"]["num_nodes_per_instance"] = 2
        base_config["execution"]["num_instances"] = 1

        cfg = OmegaConf.create(base_config)
        _resolve_multi_node_config(cfg)

        cmd, _, _ = _generate_deployment_srun_command(cfg, [], Path("/test/remote"))

        assert "ray_setup.sh" in cmd
        assert "--distributed-executor-backend ray" in cfg.deployment.extra_args

    def test_no_ray_for_single_node(self, base_config):
        """NPI=1 → no Ray."""
        base_config["execution"]["num_nodes_per_instance"] = 1
        base_config["execution"]["num_instances"] = 1

        cfg = OmegaConf.create(base_config)
        _resolve_multi_node_config(cfg)

        cmd, _, _ = _generate_deployment_srun_command(cfg, [], Path("/test/remote"))

        assert "ray_setup.sh" not in cmd
        assert "--distributed-executor-backend ray" not in cmd

    def test_ray_error_for_non_vllm_multi_node(self, base_config):
        """NPI>1 + non-vLLM → ValueError."""
        base_config["deployment"]["type"] = "trtllm"
        base_config["execution"]["num_nodes_per_instance"] = 2
        base_config["execution"]["num_instances"] = 1

        cfg = OmegaConf.create(base_config)
        _resolve_multi_node_config(cfg)

        with pytest.raises(ValueError, match="only supported for vLLM"):
            _generate_deployment_srun_command(cfg, [], Path("/test/remote"))

    def test_ray_preserves_existing_extra_args(self, base_config):
        """Existing extra_args preserved when Ray appended."""
        base_config["deployment"]["extra_args"] = "--tensor-parallel-size 4"
        base_config["execution"]["num_nodes_per_instance"] = 2
        base_config["execution"]["num_instances"] = 1

        cfg = OmegaConf.create(base_config)
        _resolve_multi_node_config(cfg)

        _generate_deployment_srun_command(cfg, [], Path("/test/remote"))

        assert "--tensor-parallel-size 4" in cfg.deployment.extra_args
        assert "--distributed-executor-backend ray" in cfg.deployment.extra_args

    def test_srun_nodes_and_ntasks(self, base_config):
        """srun uses correct --nodes {NPI*NI} --ntasks {n_tasks}."""
        base_config["execution"]["num_nodes_per_instance"] = 2
        base_config["execution"]["num_instances"] = 3

        cfg = OmegaConf.create(base_config)
        _resolve_multi_node_config(cfg)

        cmd, _, _ = _generate_deployment_srun_command(cfg, [], Path("/test/remote"))

        assert "--nodes 6 --ntasks 6" in cmd

    # ------------------------------------------------------------------ #
    # Category 4: SBATCH script integration via _create_slurm_sbatch_script
    # ------------------------------------------------------------------ #
    @pytest.mark.parametrize(
        "npi,ni",
        [(1, 1), (2, 1), (1, 3), (2, 3)],
        ids=["1x1", "2x1", "1x3", "2x3"],
    )
    def test_sbatch_nodes_header(
        self, base_config, mock_task, mock_dependencies, npi, ni
    ):
        """SBATCH --nodes = NPI*NI."""
        base_config["execution"]["num_nodes_per_instance"] = npi
        base_config["execution"]["num_instances"] = ni

        cfg = OmegaConf.create(base_config)
        script = _create_slurm_sbatch_script(
            cfg=cfg,
            task=mock_task,
            eval_image="test-eval-container:latest",
            remote_task_subdir=Path("/test/remote"),
            invocation_id="test123",
            job_id="test123.0",
        ).cmd

        assert f"#SBATCH --nodes {npi * ni}\n" in script

    def test_sbatch_multi_instance_has_haproxy(
        self, base_config, mock_task, mock_dependencies
    ):
        """NI>1 → HAProxy in script, HEAD_NODE_IPS used, PROXY_PID killed."""
        base_config["execution"]["num_nodes_per_instance"] = 1
        base_config["execution"]["num_instances"] = 3

        cfg = OmegaConf.create(base_config)
        script = _create_slurm_sbatch_script(
            cfg=cfg,
            task=mock_task,
            eval_image="test-eval-container:latest",
            remote_task_subdir=Path("/test/remote"),
            invocation_id="test123",
            job_id="test123.0",
        ).cmd

        assert "HEAD_NODE_IPS" in script
        assert "PROXY_PID" in script
        assert "haproxy" in script.lower()
        assert "kill $PROXY_PID" in script

    def test_sbatch_single_instance_no_haproxy(
        self, base_config, mock_task, mock_dependencies
    ):
        """NI=1 → no HAProxy, 127.0.0.1 used, no PROXY_PID."""
        base_config["execution"]["num_nodes_per_instance"] = 1
        base_config["execution"]["num_instances"] = 1

        cfg = OmegaConf.create(base_config)
        script = _create_slurm_sbatch_script(
            cfg=cfg,
            task=mock_task,
            eval_image="test-eval-container:latest",
            remote_task_subdir=Path("/test/remote"),
            invocation_id="test123",
            job_id="test123.0",
        ).cmd

        assert '"127.0.0.1"' in script
        assert "PROXY_PID" not in script
        assert "haproxy" not in script.lower()
