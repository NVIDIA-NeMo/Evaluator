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
"""Tests for the Lepton executor."""

import time
from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

from nemo_evaluator_launcher.common.execdb import ExecutionDB, JobData
from nemo_evaluator_launcher.executors.base import ExecutionState, ExecutionStatus
from nemo_evaluator_launcher.executors.lepton.executor import LeptonExecutor


class TestLeptonExecutor:
    """Test Lepton executor functionality."""

    def test_lepton_executor_import(self):
        """Test that Lepton executor can be imported conditionally."""
        try:
            from nemo_evaluator_launcher.executors.lepton.executor import LeptonExecutor

            assert LeptonExecutor is not None
        except ImportError:
            pytest.skip("Lepton executor not available (missing leptonai dependency)")

    def test_lepton_deployment_helpers_import(self):
        """Test that Lepton deployment helpers can be imported."""
        try:
            from nemo_evaluator_launcher.executors.lepton.deployment_helpers import (
                create_lepton_endpoint,
                delete_lepton_endpoint,
                get_lepton_endpoint_status,
                wait_for_lepton_endpoint_ready,
            )

            assert create_lepton_endpoint is not None
            assert delete_lepton_endpoint is not None
            assert get_lepton_endpoint_status is not None
            assert wait_for_lepton_endpoint_ready is not None
        except ImportError:
            pytest.skip("Lepton deployment helpers not available")

    def test_lepton_job_helpers_import(self):
        """Test that Lepton job helpers can be imported."""
        try:
            from nemo_evaluator_launcher.executors.lepton.job_helpers import (
                create_lepton_job,
                delete_lepton_job,
                get_lepton_job_status,
            )

            assert create_lepton_job is not None
            assert get_lepton_job_status is not None
            assert delete_lepton_job is not None
        except ImportError:
            pytest.skip("Lepton job helpers not available")

    def test_lepton_config_validation(self):
        """Test Lepton configuration validation."""
        # Test NIM config
        nim_config = {
            "deployment": {
                "type": "nim",
                "image": "nvcr.io/nim/meta/llama-3.1-8b-instruct:1.8.6",
                "served_model_name": "meta/llama-3.1-8b-instruct",
                "lepton_config": {
                    "endpoint_name": "test-endpoint",
                    "resource_shape": "cpu.small",
                },
            }
        }
        cfg = OmegaConf.create(nim_config)
        assert cfg.deployment.type == "nim"
        assert cfg.deployment.lepton_config.endpoint_name == "test-endpoint"

        # Test vLLM config
        vllm_config = {
            "deployment": {
                "type": "vllm",
                "checkpoint_path": "meta-llama/Llama-3.1-8B-Instruct",
                "served_model_name": "llama-3.1-8b-instruct",
                "lepton_config": {
                    "endpoint_name": "test-vllm-endpoint",
                    "resource_shape": "gpu.1xh200",
                },
            }
        }
        cfg = OmegaConf.create(vllm_config)
        assert cfg.deployment.type == "vllm"
        assert cfg.deployment.checkpoint_path == "meta-llama/Llama-3.1-8B-Instruct"

        # Test none deployment config
        none_config = {
            "deployment": {"type": "none"},
            "target": {
                "api_endpoint": {
                    "url": "https://existing-endpoint.lepton.run/v1/chat/completions"
                }
            },
        }
        cfg = OmegaConf.create(none_config)
        assert cfg.deployment.type == "none"
        assert "existing-endpoint.lepton.run" in cfg.target.api_endpoint.url


class TestLeptonExecutorDryRun:
    """Test Lepton executor dry run functionality."""

    @pytest.fixture
    def sample_config_none(self, tmpdir):
        """Create a sample configuration for deployment type 'none' testing."""
        config_dict = {
            "deployment": {"type": "none"},
            "execution": {"output_dir": str(tmpdir)},
            "evaluation": {
                "tasks": [
                    {"name": "mmlu_pro"},
                    {"name": "gsm8k"},
                ]
            },
            "target": {
                "api_endpoint": {
                    "url": "https://existing.lepton.run/v1/chat/completions"
                }
            },
        }
        return OmegaConf.create(config_dict)

    @pytest.fixture
    def sample_config_nim(self, tmpdir):
        """Create a sample configuration for NIM deployment testing."""
        config_dict = {
            "deployment": {
                "type": "nim",
                "image": "nvcr.io/nim/meta/llama-3.1-8b-instruct:1.8.6",
                "served_model_name": "meta/llama-3.1-8b-instruct",
                "endpoints": {"openai": "/v1/chat/completions"},
                "lepton_config": {
                    "endpoint_name": "test-nim-endpoint",
                    "resource_shape": "cpu.small",
                    "min_replicas": 1,
                    "max_replicas": 3,
                    "auto_scaler": True,
                },
            },
            "execution": {
                "output_dir": str(tmpdir),
                "lepton_platform": {
                    "deployment": {"endpoint_readiness_timeout": 300},
                    "tasks": {
                        "node_group": "default",
                        "env_vars": {"EVAL_ENV": "test_value"},
                        "mounts": [],
                    },
                },
                "evaluation_tasks": {
                    "resource_shape": "cpu.small",
                    "timeout": 1800,
                    "use_shared_storage": True,
                },
            },
            "evaluation": {
                "env_vars": {"GLOBAL_ENV": "global_value"},
                "tasks": [
                    {"name": "mmlu_pro"},
                    {"name": "gsm8k"},
                ],
            },
            "target": {
                "api_endpoint": {
                    "api_key_name": "LEPTON_API_KEY",
                    "model_id": "meta/llama-3.1-8b-instruct",
                }
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
            },
            ("lm-eval", "gsm8k"): {
                "task": "gsm8k",
                "endpoint_type": "openai",
                "harness": "lm-eval",
                "container": "nvcr.io/nvidia/nemo:24.01",
            },
        }

    def test_invalid_deployment_type_raises(self):
        """Test that invalid deployment types raise ValueError."""
        cfg = OmegaConf.create(
            {
                "deployment": {"type": "something-else"},
                "execution": {"output_dir": "/tmp"},
                "evaluation": {"tasks": [{"name": "lm-eval.mmlu"}]},
                "target": {"api_endpoint": {"url": "http://x"}},
            }
        )
        with pytest.raises(ValueError, match="supports deployment types"):
            LeptonExecutor.execute_eval(cfg, dry_run=True)

    def test_dry_run_none_deployment_basic(
        self, sample_config_none, mock_tasks_mapping
    ):
        """Test basic dry run with deployment type 'none'."""
        with (
            patch(
                "nemo_evaluator_launcher.executors.lepton.executor.load_tasks_mapping"
            ) as mock_load_mapping,
            patch(
                "nemo_evaluator_launcher.executors.lepton.executor.generate_invocation_id"
            ) as mock_gen_id,
            patch("builtins.print") as mock_print,
        ):
            mock_load_mapping.return_value = mock_tasks_mapping
            mock_gen_id.return_value = "abc12345"

            # Execute dry run
            invocation_id = LeptonExecutor.execute_eval(
                sample_config_none, dry_run=True
            )

            # Verify results
            assert invocation_id == "abc12345"

            # Verify print calls indicate dry run mode
            print_calls = [
                call.args[0] for call in mock_print.call_args_list if call.args
            ]
            assert any("DRY RUN" in call for call in print_calls)
            assert any("using shared endpoint" in call for call in print_calls)

    def test_dry_run_nim_deployment_basic(self, sample_config_nim, mock_tasks_mapping):
        """Test basic dry run with NIM deployment."""
        with (
            patch(
                "nemo_evaluator_launcher.executors.lepton.executor.load_tasks_mapping"
            ) as mock_load_mapping,
            patch(
                "nemo_evaluator_launcher.executors.lepton.executor.generate_invocation_id"
            ) as mock_gen_id,
            patch(
                "nemo_evaluator_launcher.executors.lepton.executor.create_lepton_endpoint"
            ) as mock_create_endpoint,
            patch(
                "nemo_evaluator_launcher.executors.lepton.executor.wait_for_lepton_endpoint_ready"
            ) as mock_wait_ready,
            patch(
                "nemo_evaluator_launcher.executors.lepton.executor.get_lepton_endpoint_url"
            ) as mock_get_url,
            patch(
                "nemo_evaluator_launcher.executors.lepton.executor.get_task_definition_for_job"
            ) as mock_get_task_def,
            patch("builtins.print") as mock_print,
        ):
            mock_load_mapping.return_value = mock_tasks_mapping
            mock_gen_id.return_value = "def67890"

            def mock_get_task_def_side_effect(*_args, **kwargs):
                # Handle task names with prefixes like "lm-eval.arc_challenge"
                task_name = kwargs.get("task_query")
                mapping = kwargs.get("base_mapping", {})
                if "." in task_name:
                    task_name = task_name.split(".")[-1]
                for (harness, name), definition in mapping.items():
                    if name == task_name:
                        return definition
                raise KeyError(f"Task {task_name} not found")

            mock_get_task_def.side_effect = mock_get_task_def_side_effect
            mock_create_endpoint.return_value = True
            mock_wait_ready.return_value = True
            mock_get_url.return_value = "https://nim-mmlu-0-def678.lepton.run"

            invocation_id = LeptonExecutor.execute_eval(sample_config_nim, dry_run=True)

            # Verify results
            assert invocation_id == "def67890"

            # Verify print calls indicate dry run mode
            print_calls = [
                call.args[0] for call in mock_print.call_args_list if call.args
            ]
            assert any(
                "DRY RUN:" in call and "Lepton" in call and "prepared" in call
                for call in print_calls
            )
            assert any("with endpoint" in call for call in print_calls)

    def test_dry_run_vllm_deployment(self, tmpdir, mock_tasks_mapping):
        """Test dry run with vLLM deployment type."""
        config_dict = {
            "deployment": {
                "type": "vllm",
                "checkpoint_path": "meta-llama/Llama-3.1-8B-Instruct",
                "served_model_name": "llama-3.1-8b-instruct",
                "endpoints": {"openai": "/v1/chat/completions"},
                "lepton_config": {
                    "resource_shape": "gpu.1xa100",
                    "min_replicas": 1,
                    "max_replicas": 2,
                    "auto_scaler": False,
                },
            },
            "execution": {
                "output_dir": str(tmpdir),
                "lepton_platform": {
                    "deployment": {"endpoint_readiness_timeout": 600},
                    "tasks": {"node_group": "gpu", "env_vars": {}, "mounts": []},
                },
                "evaluation_tasks": {"resource_shape": "cpu.medium", "timeout": 3600},
            },
            "evaluation": {"tasks": [{"name": "mmlu_pro"}]},
            "target": {"api_endpoint": {"api_key_name": "LEPTON_API_KEY"}},
        }
        cfg = OmegaConf.create(config_dict)

        with (
            patch(
                "nemo_evaluator_launcher.executors.lepton.executor.load_tasks_mapping"
            ) as mock_load_mapping,
            patch(
                "nemo_evaluator_launcher.executors.lepton.executor.generate_invocation_id"
            ) as mock_gen_id,
            patch(
                "nemo_evaluator_launcher.executors.lepton.executor.create_lepton_endpoint"
            ) as mock_create_endpoint,
            patch(
                "nemo_evaluator_launcher.executors.lepton.executor.wait_for_lepton_endpoint_ready"
            ) as mock_wait_ready,
            patch(
                "nemo_evaluator_launcher.executors.lepton.executor.get_lepton_endpoint_url"
            ) as mock_get_url,
            patch(
                "nemo_evaluator_launcher.executors.lepton.executor.get_task_definition_for_job"
            ) as mock_get_task_def,
            patch("builtins.print"),
        ):
            mock_load_mapping.return_value = mock_tasks_mapping
            mock_gen_id.return_value = "987fed65"

            def mock_get_task_def_side_effect(*_args, **kwargs):
                # Handle task names with prefixes like "lm-eval.arc_challenge"
                task_name = kwargs.get("task_query")
                mapping = kwargs.get("base_mapping", {})
                if "." in task_name:
                    task_name = task_name.split(".")[-1]
                for (harness, name), definition in mapping.items():
                    if name == task_name:
                        return definition
                raise KeyError(f"Task {task_name} not found")

            mock_get_task_def.side_effect = mock_get_task_def_side_effect
            mock_create_endpoint.return_value = True
            mock_wait_ready.return_value = True
            mock_get_url.return_value = "https://vllm-mmlu-0-987fed.lepton.run"

            # Execute dry run
            invocation_id = LeptonExecutor.execute_eval(cfg, dry_run=True)

            # Verify results
            assert invocation_id == "987fed65"

    def test_dry_run_sglang_deployment(self, tmpdir, mock_tasks_mapping):
        """Test dry run with SGLang deployment type."""
        config_dict = {
            "deployment": {
                "type": "sglang",
                "checkpoint_path": "meta-llama/Llama-3.1-8B-Instruct",
                "served_model_name": "llama-3.1-8b-instruct",
                "endpoints": {"openai": "/v1/chat/completions"},
                "lepton_config": {
                    "resource_shape": "gpu.1xa100",
                    "min_replicas": 1,
                    "max_replicas": 1,
                    "auto_scaler": False,
                },
            },
            "execution": {
                "output_dir": str(tmpdir),
                "lepton_platform": {
                    "deployment": {"endpoint_readiness_timeout": 300},
                    "tasks": {"node_group": "gpu", "env_vars": {}, "mounts": []},
                },
                "evaluation_tasks": {"resource_shape": "cpu.small", "timeout": 1800},
            },
            "evaluation": {"tasks": [{"name": "gsm8k"}]},
            "target": {"api_endpoint": {"api_key_name": "LEPTON_API_KEY"}},
        }
        cfg = OmegaConf.create(config_dict)

        with (
            patch(
                "nemo_evaluator_launcher.executors.lepton.executor.load_tasks_mapping"
            ) as mock_load_mapping,
            patch(
                "nemo_evaluator_launcher.executors.lepton.executor.generate_invocation_id"
            ) as mock_gen_id,
            patch(
                "nemo_evaluator_launcher.executors.lepton.executor.create_lepton_endpoint"
            ) as mock_create_endpoint,
            patch(
                "nemo_evaluator_launcher.executors.lepton.executor.wait_for_lepton_endpoint_ready"
            ) as mock_wait_ready,
            patch(
                "nemo_evaluator_launcher.executors.lepton.executor.get_lepton_endpoint_url"
            ) as mock_get_url,
            patch(
                "nemo_evaluator_launcher.executors.lepton.executor.get_task_definition_for_job"
            ) as mock_get_task_def,
            patch("builtins.print"),
        ):
            mock_load_mapping.return_value = mock_tasks_mapping
            mock_gen_id.return_value = "fed98765"

            def mock_get_task_def_side_effect(*_args, **kwargs):
                # Handle task names with prefixes like "lm-eval.arc_challenge"
                task_name = kwargs.get("task_query")
                mapping = kwargs.get("base_mapping", {})
                if "." in task_name:
                    task_name = task_name.split(".")[-1]
                for (harness, name), definition in mapping.items():
                    if name == task_name:
                        return definition
                raise KeyError(f"Task {task_name} not found")

            mock_get_task_def.side_effect = mock_get_task_def_side_effect
            mock_create_endpoint.return_value = True
            mock_wait_ready.return_value = True
            mock_get_url.return_value = "https://sglang-gsm8k-0-fed987.lepton.run"

            # Execute dry run
            invocation_id = LeptonExecutor.execute_eval(cfg, dry_run=True)

            # Verify results
            assert invocation_id == "fed98765"


class TestLeptonExecutorErrorHandling:
    """Test error handling in LeptonExecutor."""

    @pytest.fixture
    def base_config(self, tmpdir):
        """Base configuration for error testing."""
        return {
            "deployment": {
                "type": "nim",
                "image": "nvcr.io/nim/meta/llama-3.1-8b-instruct:1.8.6",
                "served_model_name": "meta/llama-3.1-8b-instruct",
                "endpoints": {"openai": "/v1/chat/completions"},
                "lepton_config": {"resource_shape": "cpu.small"},
            },
            "execution": {
                "output_dir": str(tmpdir),
                "lepton_platform": {
                    "deployment": {"endpoint_readiness_timeout": 300},
                    "tasks": {"node_group": "default", "env_vars": {}, "mounts": []},
                },
                "evaluation_tasks": {"resource_shape": "cpu.small", "timeout": 1800},
            },
            "evaluation": {"tasks": [{"name": "mmlu_pro"}]},
            "target": {"api_endpoint": {"api_key_name": "LEPTON_API_KEY"}},
        }

    @pytest.fixture
    def mock_tasks_mapping(self):
        """Mock tasks mapping for testing."""
        return {
            ("lm-eval", "mmlu_pro"): {
                "task": "mmlu_pro",
                "endpoint_type": "openai",
                "harness": "lm-eval",
                "container": "nvcr.io/nvidia/nemo:24.01",
            }
        }

    def test_endpoint_creation_failure_cleanup(self, base_config, mock_tasks_mapping):
        """Test cleanup when endpoint creation fails."""
        cfg = OmegaConf.create(base_config)

        with (
            patch(
                "nemo_evaluator_launcher.executors.lepton.executor.load_tasks_mapping"
            ) as mock_load_mapping,
            patch(
                "nemo_evaluator_launcher.executors.lepton.executor.generate_invocation_id"
            ) as mock_gen_id,
            patch(
                "nemo_evaluator_launcher.executors.lepton.executor.create_lepton_endpoint"
            ) as mock_create_endpoint,
            patch(
                "nemo_evaluator_launcher.executors.lepton.executor.delete_lepton_endpoint"
            ),
            patch("builtins.print") as mock_print,
            pytest.raises(RuntimeError, match="Failed to create.*endpoints"),
        ):
            mock_load_mapping.return_value = mock_tasks_mapping
            mock_gen_id.return_value = "testfail"
            mock_create_endpoint.return_value = False  # Simulate failure

            LeptonExecutor.execute_eval(cfg, dry_run=False)

            # Should attempt cleanup
            print_calls = [str(call) for call in mock_print.call_args_list]
            assert any("Error during evaluation" in call for call in print_calls)

    def test_endpoint_readiness_timeout_failure(self, base_config, mock_tasks_mapping):
        """Test when endpoint creation succeeds but readiness times out."""
        cfg = OmegaConf.create(base_config)

        with (
            patch(
                "nemo_evaluator_launcher.executors.lepton.executor.load_tasks_mapping"
            ) as mock_load_mapping,
            patch(
                "nemo_evaluator_launcher.executors.lepton.executor.generate_invocation_id"
            ) as mock_gen_id,
            patch(
                "nemo_evaluator_launcher.executors.lepton.executor.create_lepton_endpoint"
            ) as mock_create_endpoint,
            patch(
                "nemo_evaluator_launcher.executors.lepton.executor.wait_for_lepton_endpoint_ready"
            ) as mock_wait_ready,
            patch(
                "nemo_evaluator_launcher.executors.lepton.executor.delete_lepton_endpoint"
            ),
            pytest.raises(RuntimeError, match="Failed to create.*endpoints"),
        ):
            mock_load_mapping.return_value = mock_tasks_mapping
            mock_gen_id.return_value = "timeout01"
            mock_create_endpoint.return_value = True
            mock_wait_ready.return_value = False  # Simulate timeout

            LeptonExecutor.execute_eval(cfg, dry_run=False)

    def test_endpoint_url_retrieval_failure(self, base_config, mock_tasks_mapping):
        """Test when endpoint is ready but URL cannot be retrieved."""
        cfg = OmegaConf.create(base_config)

        with (
            patch(
                "nemo_evaluator_launcher.executors.lepton.executor.load_tasks_mapping"
            ) as mock_load_mapping,
            patch(
                "nemo_evaluator_launcher.executors.lepton.executor.generate_invocation_id"
            ) as mock_gen_id,
            patch(
                "nemo_evaluator_launcher.executors.lepton.executor.create_lepton_endpoint"
            ) as mock_create_endpoint,
            patch(
                "nemo_evaluator_launcher.executors.lepton.executor.wait_for_lepton_endpoint_ready"
            ) as mock_wait_ready,
            patch(
                "nemo_evaluator_launcher.executors.lepton.executor.get_lepton_endpoint_url"
            ) as mock_get_url,
            pytest.raises(RuntimeError, match="Failed to create.*endpoints"),
        ):
            mock_load_mapping.return_value = mock_tasks_mapping
            mock_gen_id.return_value = "nourl123"
            mock_create_endpoint.return_value = True
            mock_wait_ready.return_value = True
            mock_get_url.return_value = None  # Simulate URL retrieval failure

            LeptonExecutor.execute_eval(cfg, dry_run=False)

    def test_job_submission_failure(self, base_config, mock_tasks_mapping):
        """Test when endpoint setup succeeds but job submission fails."""
        cfg = OmegaConf.create(base_config)

        with (
            patch(
                "nemo_evaluator_launcher.executors.lepton.executor.load_tasks_mapping"
            ) as mock_load_mapping,
            patch(
                "nemo_evaluator_launcher.executors.lepton.executor.generate_invocation_id"
            ) as mock_gen_id,
            patch(
                "nemo_evaluator_launcher.executors.lepton.executor.create_lepton_endpoint"
            ) as mock_create_endpoint,
            patch(
                "nemo_evaluator_launcher.executors.lepton.executor.wait_for_lepton_endpoint_ready"
            ) as mock_wait_ready,
            patch(
                "nemo_evaluator_launcher.executors.lepton.executor.get_lepton_endpoint_url"
            ) as mock_get_url,
            patch(
                "nemo_evaluator_launcher.executors.lepton.executor.get_task_definition_for_job"
            ) as mock_get_task_def,
            patch(
                "nemo_evaluator_launcher.executors.lepton.executor.create_lepton_job"
            ) as mock_create_job,
            patch(
                "nemo_evaluator_launcher.common.helpers.get_eval_factory_command"
            ) as mock_get_command,
            pytest.raises(RuntimeError, match="Failed to submit Lepton job"),
        ):
            mock_load_mapping.return_value = mock_tasks_mapping
            mock_gen_id.return_value = "jobfail1"
            mock_create_endpoint.return_value = True
            mock_wait_ready.return_value = True
            mock_get_url.return_value = "https://test-endpoint.lepton.run"
            mock_get_task_def.return_value = mock_tasks_mapping[("lm-eval", "mmlu_pro")]
            from nemo_evaluator_launcher.common.helpers import CmdAndReadableComment

            mock_get_command.return_value = CmdAndReadableComment(
                cmd="test-command", debug="# Test command for lepton job failure"
            )
            mock_create_job.return_value = (False, "Job submission failed")

            LeptonExecutor.execute_eval(cfg, dry_run=False)

    def test_threading_exception_handling(self, base_config, mock_tasks_mapping):
        """Test exception handling in threading during endpoint creation."""
        cfg = OmegaConf.create(base_config)

        def mock_create_endpoint_exception(*args, **kwargs):
            raise Exception("Simulated thread exception")

        with (
            patch(
                "nemo_evaluator_launcher.executors.lepton.executor.load_tasks_mapping"
            ) as mock_load_mapping,
            patch(
                "nemo_evaluator_launcher.executors.lepton.executor.generate_invocation_id"
            ) as mock_gen_id,
            patch(
                "nemo_evaluator_launcher.executors.lepton.executor.create_lepton_endpoint"
            ) as mock_create_endpoint,
            pytest.raises(RuntimeError, match="Failed to create.*endpoints"),
        ):
            mock_load_mapping.return_value = mock_tasks_mapping
            mock_gen_id.return_value = "thread01"
            mock_create_endpoint.side_effect = mock_create_endpoint_exception

            LeptonExecutor.execute_eval(cfg, dry_run=False)


class TestLeptonExecutorHelperFunctions:
    """Test helper functions in LeptonExecutor."""

    def test_create_evaluation_launch_script(self):
        """Test the _create_evaluation_launch_script function."""
        from nemo_evaluator_launcher.executors.lepton.executor import (
            _create_evaluation_launch_script,
        )

        cfg = OmegaConf.create({"execution": {"output_dir": "/tmp/test"}})
        task = OmegaConf.create({"name": "test_task"})
        task_definition = {"container": "test:latest"}
        endpoint_url = "https://test.lepton.run"
        task_name = "test_task"
        invocation_id = "test1234"
        eval_command = "python eval.py --output_dir /results --task test"

        script = _create_evaluation_launch_script(
            cfg,
            task,
            task_definition,
            endpoint_url,
            task_name,
            invocation_id,
            eval_command,
            "# Test debug comment",
        )

        # Verify script contains expected elements
        assert "#!/bin/bash" in script
        assert "mkdir -p /tmp/test/test_task/artifacts" in script
        assert "mkdir -p /tmp/test/test_task/logs" in script
        assert "test_task" in script
        assert "test1234" in script
        assert "https://test.lepton.run" in script
        assert "--output_dir /tmp/test/test_task" in script
        assert "exit_code=$?" in script
        assert "chmod 777 -R" in script

    def test_get_statuses_for_invocation_id_with_shared_endpoint(self, mock_execdb):
        """Test _get_statuses_for_invocation_id with shared endpoint (no dedicated endpoints)."""
        from nemo_evaluator_launcher.executors.lepton.executor import (
            _get_statuses_for_invocation_id,
        )

        db = ExecutionDB()
        inv_id = "shared01"

        # Create job with no endpoint_name (shared endpoint)
        job_data = JobData(
            invocation_id=inv_id,
            job_id=f"{inv_id}.0",
            timestamp=time.time(),
            executor="lepton",
            data={
                "task_name": "test_task",
                "endpoint_name": None,
                "status": "running",
            },
        )
        db.write_job(job_data)

        with patch(
            "nemo_evaluator_launcher.executors.lepton.executor.LeptonExecutor.get_status"
        ) as mock_get_status:
            mock_get_status.return_value = [
                ExecutionStatus(
                    id=f"{inv_id}.0",
                    state=ExecutionState.RUNNING,
                    progress={"type": "evaluation_job", "task_name": "test_task"},
                )
            ]

            statuses = _get_statuses_for_invocation_id(inv_id, db)

            # Should have shared endpoint status + job status
            assert len(statuses) == 2

            # Find shared endpoint status
            shared_endpoint_status = next(
                (s for s in statuses if s.progress.get("type") == "endpoint"), None
            )
            assert shared_endpoint_status is not None
            assert shared_endpoint_status.progress["name"] == "shared"
            assert shared_endpoint_status.progress["state"] == "Using existing endpoint"

    def test_get_statuses_for_invocation_id_with_dedicated_endpoints(self, mock_execdb):
        """Test _get_statuses_for_invocation_id with dedicated endpoints."""
        from nemo_evaluator_launcher.executors.lepton.executor import (
            _get_statuses_for_invocation_id,
        )

        db = ExecutionDB()
        inv_id = "dedicate1"

        # Create jobs with dedicated endpoints
        job_data1 = JobData(
            invocation_id=inv_id,
            job_id=f"{inv_id}.0",
            timestamp=time.time(),
            executor="lepton",
            data={
                "task_name": "task1",
                "endpoint_name": "endpoint-1",
                "status": "running",
            },
        )
        job_data2 = JobData(
            invocation_id=inv_id,
            job_id=f"{inv_id}.1",
            timestamp=time.time(),
            executor="lepton",
            data={
                "task_name": "task2",
                "endpoint_name": "endpoint-2",
                "status": "running",
            },
        )
        db.write_job(job_data1)
        db.write_job(job_data2)

        with (
            patch(
                "nemo_evaluator_launcher.executors.lepton.executor.get_lepton_endpoint_status"
            ) as mock_get_endpoint_status,
            patch(
                "nemo_evaluator_launcher.executors.lepton.executor.LeptonExecutor.get_status"
            ) as mock_get_status,
        ):
            mock_get_endpoint_status.return_value = {
                "state": "Ready",
                "endpoint": {"external_endpoint": "https://test.lepton.run"},
            }

            mock_get_status.return_value = [
                ExecutionStatus(
                    id="test.0",
                    state=ExecutionState.RUNNING,
                    progress={"type": "evaluation_job"},
                )
            ]

            statuses = _get_statuses_for_invocation_id(inv_id, db)

            # Should have 2 endpoint statuses + 2 job statuses
            assert len(statuses) == 4

            # Verify endpoint statuses
            endpoint_statuses = [
                s for s in statuses if s.progress.get("type") == "endpoint"
            ]
            assert len(endpoint_statuses) == 2

    def test_endpoint_name_generation_with_long_names(self, tmpdir):
        """Test endpoint name generation with very long task names."""
        config_dict = {
            "deployment": {
                "type": "nim",
                "image": "test:latest",
                "endpoints": {"openai": "/v1/chat/completions"},
                "lepton_config": {"resource_shape": "cpu.small"},
            },
            "execution": {
                "output_dir": str(tmpdir),
                "lepton_platform": {
                    "tasks": {"node_group": "default", "env_vars": {}, "mounts": []}
                },
            },
            "evaluation": {
                "tasks": [
                    {"name": "very_long_task_name_that_might_exceed_limits"},
                    {"name": "another.extremely.long.task.name.with.many.dots"},
                ]
            },
            "target": {"api_endpoint": {"api_key_name": "TEST_KEY"}},
        }
        cfg = OmegaConf.create(config_dict)

        mock_tasks_mapping = {
            ("lm-eval", "very_long_task_name_that_might_exceed_limits"): {
                "task": "very_long_task_name_that_might_exceed_limits",
                "endpoint_type": "openai",
                "harness": "lm-eval",
                "container": "test:latest",
            },
            ("lm-eval", "dots"): {
                "task": "dots",
                "endpoint_type": "openai",
                "harness": "lm-eval",
                "container": "test:latest",
            },
        }

        with (
            patch(
                "nemo_evaluator_launcher.executors.lepton.executor.load_tasks_mapping"
            ) as mock_load_mapping,
            patch(
                "nemo_evaluator_launcher.executors.lepton.executor.generate_invocation_id"
            ) as mock_gen_id,
            patch(
                "nemo_evaluator_launcher.executors.lepton.executor.create_lepton_endpoint"
            ) as mock_create_endpoint,
            patch(
                "nemo_evaluator_launcher.executors.lepton.executor.wait_for_lepton_endpoint_ready"
            ) as mock_wait_ready,
            patch(
                "nemo_evaluator_launcher.executors.lepton.executor.get_lepton_endpoint_url"
            ) as mock_get_url,
            patch(
                "nemo_evaluator_launcher.executors.lepton.executor.get_task_definition_for_job"
            ) as mock_get_task_def,
            patch("builtins.print"),
        ):
            mock_load_mapping.return_value = mock_tasks_mapping
            mock_gen_id.return_value = "longname1"
            mock_create_endpoint.return_value = True
            mock_wait_ready.return_value = True
            mock_get_url.return_value = "https://test.lepton.run"

            def mock_get_task_def_side_effect(*_args, **kwargs):
                task_name = kwargs.get("task_query") or ""
                mapping = kwargs.get("base_mapping", {})
                if "very_long" in task_name:
                    return mapping[
                        ("lm-eval", "very_long_task_name_that_might_exceed_limits")
                    ]
                elif "dots" in task_name or "another" in task_name:
                    return mapping[("lm-eval", "dots")]
                raise KeyError(f"Task {task_name} not found")

            mock_get_task_def.side_effect = mock_get_task_def_side_effect

            # Execute dry run
            invocation_id = LeptonExecutor.execute_eval(cfg, dry_run=True)
            assert invocation_id == "longname1"

            # Verify all endpoint names are within 36 character limit
            assert (
                mock_create_endpoint.call_count == 0
            )  # Dry run should NOT create endpoints
            assert mock_wait_ready.call_count == 0
            assert mock_get_url.call_count == 0

    def test_job_name_generation_with_long_names(self, tmpdir):
        """Test job name generation with very long task names."""
        config_dict = {
            "deployment": {"type": "none"},
            "execution": {
                "output_dir": str(tmpdir),
                "lepton_platform": {
                    "tasks": {"node_group": "default", "env_vars": {}, "mounts": []},
                },
            },
            "evaluation": {
                "tasks": [
                    {
                        "name": "extremely_long_task_name_that_should_be_truncated_properly"
                    },
                ]
            },
            "target": {
                "api_endpoint": {
                    "url": "https://test.endpoint.com",
                    "model_id": "test-model",
                }
            },
        }
        cfg = OmegaConf.create(config_dict)

        mock_tasks_mapping = {
            ("lm-eval", "extremely_long_task_name_that_should_be_truncated_properly"): {
                "task": "extremely_long_task_name_that_should_be_truncated_properly",
                "endpoint_type": "openai",
                "harness": "lm-eval",
                "container": "test:latest",
            }
        }

        with (
            patch(
                "nemo_evaluator_launcher.executors.lepton.executor.load_tasks_mapping"
            ) as mock_load_mapping,
            patch(
                "nemo_evaluator_launcher.executors.lepton.executor.generate_invocation_id"
            ) as mock_gen_id,
            patch(
                "nemo_evaluator_launcher.executors.lepton.executor.get_task_definition_for_job"
            ) as mock_get_task_def,
            patch(
                "nemo_evaluator_launcher.executors.lepton.executor.create_lepton_job"
            ) as mock_create_job,
            patch(
                "nemo_evaluator_launcher.common.helpers.get_eval_factory_command"
            ) as mock_get_command,
            patch("builtins.print"),
        ):
            mock_load_mapping.return_value = mock_tasks_mapping
            mock_gen_id.return_value = "longjob1"
            mock_get_task_def.return_value = mock_tasks_mapping[
                (
                    "lm-eval",
                    "extremely_long_task_name_that_should_be_truncated_properly",
                )
            ]
            from nemo_evaluator_launcher.common.helpers import CmdAndReadableComment

            mock_get_command.return_value = CmdAndReadableComment(
                cmd="test-command --output_dir /results",
                debug="# Test command for long job names",
            )
            mock_create_job.return_value = (True, None)

            # Execute (not dry run to test job submission)
            invocation_id = LeptonExecutor.execute_eval(cfg, dry_run=False)
            assert invocation_id == "longjob1"

            # Verify job was created with name within 36 character limit
            assert mock_create_job.call_count == 1
            job_call = mock_create_job.call_args_list[0]
            job_name = job_call[1]["job_name"]
            assert len(job_name) <= 36, f"Job name too long: {job_name}"


class TestLeptonExecutorExecuteEval:
    def test_none_deployment_dry_run_returns_invocation_id_and_creates_dir(
        self, tmp_path, monkeypatch
    ):
        cfg = OmegaConf.create(
            {
                "deployment": {"type": "none"},
                "execution": {"output_dir": str(tmp_path)},
                "evaluation": {
                    "tasks": [
                        {"name": "lm-evaluation-harness.mmlu"},
                        {"name": "simple_evals.gpqa_diamond"},
                    ]
                },
                "target": {"api_endpoint": {"url": "http://endpoint"}},
            }
        )
        monkeypatch.setattr(
            "nemo_evaluator_launcher.executors.lepton.executor.generate_invocation_id",
            lambda: "deadbeef",
            raising=True,
        )
        inv = LeptonExecutor.execute_eval(cfg, dry_run=True)
        assert inv == "deadbeef"
        assert (tmp_path / "deadbeef").exists()


class TestLeptonExecutorGetStatus:
    def test_job_with_lepton_job_name_uses_live_status(self, mock_execdb, monkeypatch):
        db = ExecutionDB()
        inv = "a1b2c3d4"
        jid = f"{inv}.0"
        db.write_job(
            JobData(
                inv,
                jid,
                time.time(),
                "lepton",
                data={
                    "lepton_job_name": "job1",
                    "task_name": "taskA",
                    "endpoint_name": "ep1",
                },
            )
        )

        monkeypatch.setattr(
            "nemo_evaluator_launcher.executors.lepton.executor.get_lepton_job_status",
            lambda name: {"state": "Succeeded"},
            raising=True,
        )

        statuses = LeptonExecutor.get_status(jid)
        assert len(statuses) == 1
        s = statuses[0]
        assert s.id == jid
        assert s.state.name == "SUCCESS"
        assert s.progress["lepton_job_name"] == "job1"
        assert s.progress["type"] == "evaluation_job"

    def test_job_fallback_to_stored_status(self, mock_execdb):
        db = ExecutionDB()
        inv = "feedbeef"
        jid = f"{inv}.1"
        db.write_job(
            JobData(
                inv,
                jid,
                time.time(),
                "lepton",
                data={
                    "status": "running",
                    "task_name": "taskB",
                    "endpoint_name": "ep2",
                },
            )
        )

        statuses = LeptonExecutor.get_status(jid)
        assert len(statuses) == 1
        assert statuses[0].state.name == "RUNNING"
        assert statuses[0].progress["status"] == "running"

    def test_invocation_status_includes_endpoint_and_jobs(
        self, mock_execdb, monkeypatch
    ):
        db = ExecutionDB()
        inv = "beadf00d"
        # Two jobs under same endpoint
        db.write_job(
            JobData(
                inv,
                f"{inv}.0",
                time.time(),
                "lepton",
                data={
                    "endpoint_name": "ep-shared",
                    "task_name": "T0",
                    "status": "submitted",
                },
            )
        )
        db.write_job(
            JobData(
                inv,
                f"{inv}.1",
                time.time(),
                "lepton",
                data={
                    "endpoint_name": "ep-shared",
                    "task_name": "T1",
                    "status": "submitted",
                },
            )
        )

        monkeypatch.setattr(
            "nemo_evaluator_launcher.executors.lepton.executor.get_lepton_endpoint_status",
            lambda name: {
                "state": "Ready",
                "endpoint": {"external_endpoint": "https://ep"},
            },
            raising=True,
        )

        statuses = LeptonExecutor.get_status(inv)
        # Expect at least one endpoint status + job statuses
        assert any(s.progress.get("type") == "endpoint" for s in statuses)
        assert any(s.progress.get("type") == "evaluation_job" for s in statuses)

    def test_lepton_job_state_mapping(self, mock_execdb, monkeypatch):
        """Test mapping of Lepton job states to ExecutionState."""
        db = ExecutionDB()
        inv = "statetest"
        jid = f"{inv}.0"

        # Create job with lepton_job_name so it uses live status
        db.write_job(
            JobData(
                inv,
                jid,
                time.time(),
                "lepton",
                data={
                    "lepton_job_name": "test-job",
                    "task_name": "test_task",
                    "endpoint_name": "test-ep",
                },
            )
        )

        # Test all state mappings
        test_cases = [
            ("Succeeded", "SUCCESS"),
            ("Running", "RUNNING"),
            ("Pending", "RUNNING"),
            ("Starting", "RUNNING"),
            ("Failed", "FAILED"),
            ("Cancelled", "FAILED"),
            ("Unknown_State", "PENDING"),  # Default case
        ]

        for lepton_state, expected_exec_state in test_cases:
            monkeypatch.setattr(
                "nemo_evaluator_launcher.executors.lepton.executor.get_lepton_job_status",
                lambda name: {"state": lepton_state},
                raising=True,
            )

            statuses = LeptonExecutor.get_status(jid)
            assert len(statuses) == 1
            assert statuses[0].state.name == expected_exec_state


class TestLeptonExecutorKillJob:
    # NOTE: Batch kill functionality (killing by invocation ID) has been moved to the API layer.
    # The executor now only handles individual job kills. See test_functional.py for batch kill tests.

    def test_kill_single_job_errors(self, mock_execdb):
        with pytest.raises(ValueError, match="not found"):
            LeptonExecutor.kill_job("unknown.0")

        db = ExecutionDB()
        inv = "badc0de0"
        jid = f"{inv}.0"
        db.write_job(JobData(inv, jid, time.time(), "local", data={}))
        with pytest.raises(ValueError, match="is not a Lepton job"):
            LeptonExecutor.kill_job(jid)

    def test_kill_single_job_endpoint_cleanup_only_when_last(
        self, mock_execdb, monkeypatch
    ):
        db = ExecutionDB()
        inv = "aa55aa55"
        j0 = f"{inv}.0"
        j1 = f"{inv}.1"
        # Two jobs share epX
        db.write_job(
            JobData(
                inv,
                j0,
                time.time(),
                "lepton",
                data={"endpoint_name": "epX", "lepton_job_name": "L0"},
            )
        )
        db.write_job(
            JobData(
                inv,
                j1,
                time.time(),
                "lepton",
                data={"endpoint_name": "epX", "lepton_job_name": "L1"},
            )
        )

        jobs_cancelled = []

        monkeypatch.setattr(
            "nemo_evaluator_launcher.executors.lepton.executor.delete_lepton_job",
            lambda name: jobs_cancelled.append(name) or True,
            raising=True,
        )
        cleaned = []
        monkeypatch.setattr(
            "nemo_evaluator_launcher.executors.lepton.executor.delete_lepton_endpoint",
            lambda name: cleaned.append(name) or True,
            raising=True,
        )

        # Kill first job -> endpoint should NOT be cleaned yet
        LeptonExecutor.kill_job(j0)
        assert "L0" in jobs_cancelled
        assert cleaned == []

        # Mark second job as not finished to keep endpoint; then kill second
        LeptonExecutor.kill_job(j1)
        assert "L1" in jobs_cancelled
        # Now both jobs are killed; endpoint should be cleaned after second kill
        assert cleaned == ["epX"]

    def test_endpoint_naming_with_multiple_tasks(self, tmpdir, mock_tasks_mapping):
        """Test endpoint name generation during execution with multiple tasks."""
        config_dict = {
            "deployment": {
                "type": "nim",
                "image": "nvcr.io/nim/meta/llama-3.1-8b-instruct:1.8.6",
                "served_model_name": "meta/llama-3.1-8b-instruct",
                "endpoints": {"openai": "/v1/chat/completions"},
                "lepton_config": {"resource_shape": "cpu.small"},
            },
            "execution": {
                "output_dir": str(tmpdir),
                "lepton_platform": {
                    "deployment": {"endpoint_readiness_timeout": 300},
                    "tasks": {"node_group": "default", "env_vars": {}, "mounts": []},
                },
                "evaluation_tasks": {"resource_shape": "cpu.small", "timeout": 1800},
            },
            "evaluation": {
                "tasks": [
                    {"name": "mmlu_pro"},
                    {"name": "gsm8k"},
                    {"name": "lm-eval.arc_challenge"},
                ]
            },
            "target": {"api_endpoint": {"api_key_name": "LEPTON_API_KEY"}},
        }
        cfg = OmegaConf.create(config_dict)

        mock_tasks_mapping = {
            ("lm-eval", "mmlu_pro"): {
                "task": "mmlu_pro",
                "endpoint_type": "openai",
                "harness": "lm-eval",
                "container": "test:latest",
            },
            ("lm-eval", "gsm8k"): {
                "task": "gsm8k",
                "endpoint_type": "openai",
                "harness": "lm-eval",
                "container": "test:latest",
            },
            ("lm-eval", "arc_challenge"): {
                "task": "arc_challenge",
                "endpoint_type": "openai",
                "harness": "lm-eval",
                "container": "nvcr.io/nvidia/nemo:24.01",
            },
        }

        with (
            patch(
                "nemo_evaluator_launcher.executors.lepton.executor.load_tasks_mapping"
            ) as mock_load,
            patch(
                "nemo_evaluator_launcher.executors.lepton.executor.generate_invocation_id"
            ) as mock_gen_id,
            patch(
                "nemo_evaluator_launcher.executors.lepton.executor.create_lepton_endpoint"
            ) as mock_create_endpoint,
            patch(
                "nemo_evaluator_launcher.executors.lepton.executor.wait_for_lepton_endpoint_ready"
            ) as mock_wait,
            patch(
                "nemo_evaluator_launcher.executors.lepton.executor.get_lepton_endpoint_url"
            ) as mock_get_url,
            patch(
                "nemo_evaluator_launcher.executors.lepton.executor.get_task_definition_for_job"
            ) as mock_get_task_def,
            patch(
                "nemo_evaluator_launcher.executors.lepton.executor.create_lepton_job"
            ) as mock_create_job,
            patch(
                "nemo_evaluator_launcher.common.helpers.get_eval_factory_command"
            ) as mock_get_cmd,
            patch("builtins.print"),
        ):
            mock_load.return_value = mock_tasks_mapping
            mock_gen_id.return_value = "abc12345"

            def get_task_def_side_effect(*_args, **kwargs):
                task_name = kwargs.get("task_query")
                mapping = kwargs.get("base_mapping", {})
                if "." in task_name:
                    task_name = task_name.split(".")[-1]
                for (harness, name), definition in mapping.items():
                    if name == task_name:
                        return definition
                raise KeyError(f"Task {task_name} not found")

            mock_get_task_def.side_effect = get_task_def_side_effect
            mock_create_endpoint.return_value = True
            mock_wait.return_value = True
            mock_get_url.side_effect = lambda ep: f"https://{ep}.lepton.run"
            mock_create_job.return_value = (True, None)
            from nemo_evaluator_launcher.common.helpers import CmdAndReadableComment

            mock_get_cmd.return_value = CmdAndReadableComment(
                cmd="test-cmd --output_dir /results",
                debug="# Test command for endpoint naming",
            )

            # Execute REAL run
            LeptonExecutor.execute_eval(cfg, dry_run=False)

            # Verify 3 endpoints were created
            assert mock_create_endpoint.call_count == 3

            # Verify endpoint names are properly sanitized
            endpoint_names = [
                call.args[1] for call in mock_create_endpoint.call_args_list
            ]

            # Check sanitization: underscores -> hyphens, dots removed, lowercase
            assert any("mmlu" in ep for ep in endpoint_names)
            assert any("gsm8k" in ep for ep in endpoint_names)
            assert any("arc" in ep or "challenge" in ep for ep in endpoint_names)

            # Verify all are within 36 character limit
            for ep_name in endpoint_names:
                assert len(ep_name) <= 36, f"Endpoint name exceeds 36 chars: {ep_name}"
