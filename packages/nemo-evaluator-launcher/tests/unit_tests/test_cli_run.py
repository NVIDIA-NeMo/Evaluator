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
"""Tests for the CLI run command methods."""

import os
import subprocess
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from omegaconf import OmegaConf

from nemo_evaluator_launcher.cli.run import Cmd


class TestParseTaskArgs:
    """Tests for _parse_task_args() method."""

    def test_single_task(self):
        """Test parsing a single task."""
        cmd = Cmd(task=["ifeval"])
        result = cmd._parse_task_args()
        assert result == ["ifeval"]

    def test_multiple_tasks(self):
        """Test parsing multiple tasks."""
        cmd = Cmd(task=["ifeval", "gsm8k", "mmlu"])
        result = cmd._parse_task_args()
        assert result == ["ifeval", "gsm8k", "mmlu"]

    def test_deduplication(self):
        """Test that duplicate tasks are removed."""
        cmd = Cmd(task=["ifeval", "gsm8k", "ifeval"])
        result = cmd._parse_task_args()
        assert result == ["ifeval", "gsm8k"]

    def test_empty_and_none_values(self):
        """Test handling of empty and None values."""
        cmd = Cmd(task=["ifeval", "", None, "gsm8k"])
        result = cmd._parse_task_args()
        assert result == ["ifeval", "gsm8k"]

    def test_deprecated_tasks_flag_compatibility(self):
        """Test that deprecated --tasks flag is still supported."""
        cmd = Cmd(task=["ifeval"], tasks=["gsm8k"])
        result = cmd._parse_task_args()
        assert "ifeval" in result
        assert "gsm8k" in result

    def test_empty_task_list(self):
        """Test empty task list returns empty list."""
        cmd = Cmd(task=[])
        result = cmd._parse_task_args()
        assert result == []

    def test_whitespace_stripped(self):
        """Test that whitespace is stripped from task names."""
        cmd = Cmd(task=["  ifeval  ", "gsm8k "])
        result = cmd._parse_task_args()
        assert result == ["ifeval", "gsm8k"]


class TestHasAnyDirectFlags:
    """Tests for _has_any_direct_flags() method."""

    def test_no_flags_returns_false(self):
        """Test that default Cmd returns False."""
        cmd = Cmd()
        assert cmd._has_any_direct_flags() is False

    def test_model_flag_returns_true(self):
        """Test that --model flag makes it return True."""
        cmd = Cmd(model="meta/llama-3.2-3b-instruct")
        assert cmd._has_any_direct_flags() is True

    def test_executor_flags_return_true(self):
        """Test that executor-related flags return True."""
        # Test executor flag
        cmd = Cmd(executor="slurm")
        assert cmd._has_any_direct_flags() is True

        # Test output-dir flag
        cmd = Cmd(output_dir="/tmp/results")
        assert cmd._has_any_direct_flags() is True

        # Test SLURM-specific flags
        cmd = Cmd(slurm_hostname="cluster.example.com")
        assert cmd._has_any_direct_flags() is True

        cmd = Cmd(slurm_account="my-account")
        assert cmd._has_any_direct_flags() is True

        cmd = Cmd(slurm_partition="batch")
        assert cmd._has_any_direct_flags() is True

        cmd = Cmd(slurm_walltime="01:00:00")
        assert cmd._has_any_direct_flags() is True

    def test_deployment_flags_return_true(self):
        """Test that deployment-related flags return True."""
        # Test deployment flag
        cmd = Cmd(deployment="vllm")
        assert cmd._has_any_direct_flags() is True

        # Test vLLM/SGLang flags
        cmd = Cmd(checkpoint="/path/to/model")
        assert cmd._has_any_direct_flags() is True

        cmd = Cmd(model_name="my-model")
        assert cmd._has_any_direct_flags() is True

        cmd = Cmd(tensor_parallel=2)
        assert cmd._has_any_direct_flags() is True

        cmd = Cmd(data_parallel=2)
        assert cmd._has_any_direct_flags() is True

        cmd = Cmd(hf_model_handle="meta-llama/Meta-Llama-3-8B")
        assert cmd._has_any_direct_flags() is True

        # Test NIM flags
        cmd = Cmd(nim_model="llama-3-8b")
        assert cmd._has_any_direct_flags() is True

    def test_task_flag_returns_true(self):
        """Test that --task flag returns True."""
        cmd = Cmd(task=["ifeval"])
        assert cmd._has_any_direct_flags() is True

    def test_url_flag_returns_true(self):
        """Test that --url flag returns True."""
        cmd = Cmd(url="https://example.com/v1/chat/completions")
        assert cmd._has_any_direct_flags() is True

    def test_api_key_env_flag_returns_true(self):
        """Test that --api-key-env flag returns True."""
        cmd = Cmd(api_key_env="MY_API_KEY")
        assert cmd._has_any_direct_flags() is True

    def test_limit_samples_flag_returns_true(self):
        """Test that --limit-samples flag returns True."""
        cmd = Cmd(limit_samples=10)
        assert cmd._has_any_direct_flags() is True


class TestBuildConfigFromFlags:
    """Tests for _build_config_from_flags() method (CRITICAL)."""

    def test_defaults_local_executor_none_deployment(self):
        """Test that default executor is 'local' and deployment is 'none'."""
        cmd = Cmd(task=["ifeval"], model="test-model")
        config = cmd._build_config_from_flags()

        assert config["defaults"][0] == {"execution": "local"}
        assert config["defaults"][1] == {"deployment": "none"}

    def test_api_endpoint_with_model_only(self):
        """Test building config with just model flag."""
        cmd = Cmd(model="meta/llama-3.2-3b-instruct", task=["ifeval"])
        config = cmd._build_config_from_flags()

        assert "target" in config
        assert "api_endpoint" in config["target"]
        assert config["target"]["api_endpoint"]["model_id"] == "meta/llama-3.2-3b-instruct"

    def test_api_endpoint_defaults_nvidia_url(self):
        """Test that default URL is NVIDIA API Catalog."""
        cmd = Cmd(model="test-model", task=["ifeval"])
        config = cmd._build_config_from_flags()

        assert config["target"]["api_endpoint"]["url"] == "https://integrate.api.nvidia.com/v1/chat/completions"
        assert config["target"]["api_endpoint"]["api_key_name"] == "NGC_API_KEY"

    def test_custom_url_and_api_key(self):
        """Test custom URL and API key override defaults."""
        cmd = Cmd(
            model="test-model",
            url="https://custom.api.com/v1",
            api_key_env="MY_CUSTOM_KEY",
            task=["ifeval"],
        )
        config = cmd._build_config_from_flags()

        assert config["target"]["api_endpoint"]["url"] == "https://custom.api.com/v1"
        assert config["target"]["api_endpoint"]["api_key_name"] == "MY_CUSTOM_KEY"

    def test_slurm_executor_config(self):
        """Test SLURM executor configuration."""
        cmd = Cmd(
            executor="slurm",
            slurm_hostname="cluster.example.com",
            slurm_account="my-account",
            slurm_partition="gpu",
            slurm_walltime="02:00:00",
            output_dir="/shared/results",
            model="test-model",
            task=["ifeval"],
        )
        config = cmd._build_config_from_flags()

        assert config["defaults"][0] == {"execution": "slurm/default"}
        assert config["execution"]["hostname"] == "cluster.example.com"
        assert config["execution"]["account"] == "my-account"
        assert config["execution"]["partition"] == "gpu"
        assert config["execution"]["walltime"] == "02:00:00"
        assert config["execution"]["output_dir"] == "/shared/results"

    def test_vllm_deployment_config(self):
        """Test vLLM deployment configuration."""
        cmd = Cmd(
            deployment="vllm",
            checkpoint="/path/to/model",
            model_name="my-model",
            tensor_parallel=4,
            data_parallel=2,
            task=["ifeval"],
        )
        config = cmd._build_config_from_flags()

        assert config["defaults"][1] == {"deployment": "vllm"}
        assert config["deployment"]["checkpoint_path"] == "/path/to/model"
        assert config["deployment"]["served_model_name"] == "my-model"
        assert config["deployment"]["tensor_parallel_size"] == 4
        assert config["deployment"]["data_parallel_size"] == 2

    def test_sglang_deployment_config(self):
        """Test SGLang deployment configuration."""
        cmd = Cmd(
            deployment="sglang",
            hf_model_handle="meta-llama/Meta-Llama-3-8B",
            model_name="llama-3-8b",
            task=["ifeval"],
        )
        config = cmd._build_config_from_flags()

        assert config["defaults"][1] == {"deployment": "sglang"}
        assert config["deployment"]["hf_model_handle"] == "meta-llama/Meta-Llama-3-8B"
        assert config["deployment"]["served_model_name"] == "llama-3-8b"

    def test_nim_deployment_config(self):
        """Test NIM deployment configuration."""
        cmd = Cmd(
            deployment="nim",
            nim_model="llama-3-8b-instruct",
            task=["ifeval"],
        )
        config = cmd._build_config_from_flags()

        assert config["defaults"][1] == {"deployment": "nim"}
        assert config["deployment"]["model_name"] == "llama-3-8b-instruct"

    def test_task_list_structure(self):
        """Test that tasks are structured correctly."""
        cmd = Cmd(task=["ifeval", "gsm8k", "mmlu"], model="test-model")
        config = cmd._build_config_from_flags()

        assert "evaluation" in config
        assert "tasks" in config["evaluation"]
        assert config["evaluation"]["tasks"] == [
            {"name": "ifeval"},
            {"name": "gsm8k"},
            {"name": "mmlu"},
        ]

    def test_limit_samples_nested_correctly(self):
        """Test that limit_samples is nested correctly in config."""
        cmd = Cmd(model="test-model", task=["ifeval"], limit_samples=100)
        config = cmd._build_config_from_flags()

        assert "nemo_evaluator_config" in config["evaluation"]
        assert config["evaluation"]["nemo_evaluator_config"]["config"]["params"]["limit_samples"] == 100

    def test_output_dir_in_execution(self):
        """Test that output_dir is added to execution config."""
        cmd = Cmd(model="test-model", task=["ifeval"], output_dir="/custom/output")
        config = cmd._build_config_from_flags()

        assert config["execution"]["output_dir"] == "/custom/output"


class TestApplyFlagOverrides:
    """Tests for _apply_flag_overrides() method (CRITICAL)."""

    def test_model_override_on_existing_config(self):
        """Test model override on existing config."""
        cmd = Cmd(model="new-model")

        existing_config = OmegaConf.create({
            "target": {
                "api_endpoint": {
                    "model_id": "old-model",
                    "url": "https://example.com",
                }
            },
            "evaluation": {"tasks": [{"name": "ifeval"}]},
        })

        result = cmd._apply_flag_overrides(existing_config)

        assert result.target.api_endpoint.model_id == "new-model"
        assert result.target.api_endpoint.url == "https://example.com"  # preserved

    def test_tasks_replace_not_merge(self):
        """Test that task overrides replace, not merge."""
        cmd = Cmd(task=["gsm8k", "mmlu"])

        existing_config = OmegaConf.create({
            "evaluation": {"tasks": [{"name": "ifeval"}, {"name": "arc"}]},
        })

        result = cmd._apply_flag_overrides(existing_config)

        # Tasks should be replaced
        assert len(result.evaluation.tasks) == 2
        assert result.evaluation.tasks[0]["name"] == "gsm8k"
        assert result.evaluation.tasks[1]["name"] == "mmlu"

    def test_no_overrides_returns_unchanged(self):
        """Test that config is unchanged when no flags provided."""
        cmd = Cmd()

        existing_config = OmegaConf.create({
            "target": {"api_endpoint": {"model_id": "test-model"}},
            "evaluation": {"tasks": [{"name": "ifeval"}]},
        })

        result = cmd._apply_flag_overrides(existing_config)

        assert result.target.api_endpoint.model_id == "test-model"
        assert len(result.evaluation.tasks) == 1

    def test_deep_merge_preserves_existing_values(self):
        """Test that deep merge preserves existing nested values."""
        cmd = Cmd(slurm_partition="gpu")

        existing_config = OmegaConf.create({
            "execution": {
                "type": "slurm",
                "hostname": "cluster.example.com",
                "account": "my-account",
            },
        })

        result = cmd._apply_flag_overrides(existing_config)

        # Existing values should be preserved
        assert result.execution.hostname == "cluster.example.com"
        assert result.execution.account == "my-account"
        # New value should be added
        assert result.execution.partition == "gpu"

    def test_checkpoint_override(self):
        """Test checkpoint path override."""
        cmd = Cmd(checkpoint="/new/path/to/model")

        existing_config = OmegaConf.create({
            "deployment": {
                "type": "vllm",
                "checkpoint_path": "/old/path",
                "served_model_name": "my-model",
            },
        })

        result = cmd._apply_flag_overrides(existing_config)

        assert result.deployment.checkpoint_path == "/new/path/to/model"
        assert result.deployment.served_model_name == "my-model"  # preserved

    def test_limit_samples_override(self):
        """Test limit_samples override is deeply nested correctly."""
        cmd = Cmd(limit_samples=50)

        existing_config = OmegaConf.create({
            "evaluation": {
                "tasks": [{"name": "ifeval"}],
            },
        })

        result = cmd._apply_flag_overrides(existing_config)

        assert result.evaluation.nemo_evaluator_config.config.params.limit_samples == 50


class TestValidateConfig:
    """Tests for _validate_config() wrapper method."""

    def test_valid_config_returns_true(self):
        """Test that valid config returns True."""
        cmd = Cmd()

        valid_config = OmegaConf.create({
            "execution": {"type": "local", "output_dir": "/tmp/results"},
            "deployment": {"type": "none"},
            "target": {"api_endpoint": {"model_id": "test-model"}},
            "evaluation": {"tasks": [{"name": "ifeval"}]},
        })

        with patch("nemo_evaluator_launcher.common.mapping.load_tasks_mapping") as mock:
            mock.return_value = {("helm", "ifeval"): {"task": "ifeval"}}
            result = cmd._validate_config(valid_config)

        assert result is True

    def test_invalid_config_returns_false_and_prints(self):
        """Test that invalid config returns False and prints errors."""
        cmd = Cmd()

        invalid_config = OmegaConf.create({
            "execution": {"type": "slurm"},  # Missing required fields
            "deployment": {"type": "none"},
            "evaluation": {"tasks": []},  # No tasks
        })

        output = StringIO()
        with redirect_stdout(output):
            result = cmd._validate_config(invalid_config)

        assert result is False
        printed = output.getvalue()
        # Should have printed validation errors
        assert "hostname" in printed.lower() or "error" in printed.lower()


class TestRunDeepChecks:
    """Tests for _run_deep_checks() method."""

    def test_docker_check_success(self):
        """Test Docker check success path."""
        cmd = Cmd()

        config = OmegaConf.create({
            "execution": {"type": "local"},
        })

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            output = StringIO()
            with redirect_stdout(output):
                result = cmd._run_deep_checks(config)

            assert result is True
            assert "Docker" in output.getvalue()
            mock_run.assert_called_once()

    def test_docker_check_failure(self):
        """Test Docker check failure path."""
        cmd = Cmd()

        config = OmegaConf.create({
            "execution": {"type": "local"},
        })

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1)

            output = StringIO()
            with redirect_stdout(output):
                result = cmd._run_deep_checks(config)

            assert result is False
            printed = output.getvalue()
            assert "Docker" in printed

    def test_docker_not_installed(self):
        """Test when Docker is not installed."""
        cmd = Cmd()

        config = OmegaConf.create({
            "execution": {"type": "local"},
        })

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError()

            output = StringIO()
            with redirect_stdout(output):
                result = cmd._run_deep_checks(config)

            assert result is False
            assert "not installed" in output.getvalue()

    def test_ssh_check_for_slurm(self):
        """Test SSH check for SLURM executor."""
        cmd = Cmd()

        config = OmegaConf.create({
            "execution": {"type": "slurm", "hostname": "cluster.example.com"},
        })

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            output = StringIO()
            with redirect_stdout(output):
                result = cmd._run_deep_checks(config)

            assert result is True
            assert "SSH" in output.getvalue()
            # Verify SSH command was called with correct hostname
            call_args = mock_run.call_args[0][0]
            assert "ssh" in call_args
            assert "cluster.example.com" in call_args

    def test_ssh_check_failure(self):
        """Test SSH check failure."""
        cmd = Cmd()

        config = OmegaConf.create({
            "execution": {"type": "slurm", "hostname": "unreachable.example.com"},
        })

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=255)

            output = StringIO()
            with redirect_stdout(output):
                result = cmd._run_deep_checks(config)

            assert result is False
            assert "SSH" in output.getvalue()

    def test_ssh_check_timeout(self):
        """Test SSH check timeout."""
        cmd = Cmd()

        config = OmegaConf.create({
            "execution": {"type": "slurm", "hostname": "slow.example.com"},
        })

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="ssh", timeout=10)

            output = StringIO()
            with redirect_stdout(output):
                result = cmd._run_deep_checks(config)

            assert result is False
            assert "timed out" in output.getvalue()

    def test_env_var_check_set(self):
        """Test environment variable check when set."""
        cmd = Cmd()

        config = OmegaConf.create({
            "execution": {"type": "local"},
            "target": {"api_endpoint": {"api_key_name": "TEST_API_KEY"}},
        })

        with patch.dict(os.environ, {"TEST_API_KEY": "test-value"}):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0)

                output = StringIO()
                with redirect_stdout(output):
                    result = cmd._run_deep_checks(config)

                assert result is True
                assert "TEST_API_KEY" in output.getvalue()
                assert "is set" in output.getvalue()

    def test_env_var_check_not_set(self):
        """Test environment variable check when not set."""
        cmd = Cmd()

        config = OmegaConf.create({
            "execution": {"type": "local"},
            "target": {"api_endpoint": {"api_key_name": "NONEXISTENT_API_KEY"}},
        })

        # Ensure the env var doesn't exist
        env_copy = os.environ.copy()
        if "NONEXISTENT_API_KEY" in env_copy:
            del env_copy["NONEXISTENT_API_KEY"]

        with patch.dict(os.environ, env_copy, clear=True):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0)

                output = StringIO()
                with redirect_stdout(output):
                    result = cmd._run_deep_checks(config)

                printed = output.getvalue()
                assert "NONEXISTENT_API_KEY" in printed
                assert "not set" in printed


class TestSaveConfig:
    """Tests for --save-config handling."""

    def test_saves_valid_yaml_to_path(self, tmp_path):
        """Test that config is saved as valid YAML."""
        import yaml

        config_path = tmp_path / "test_config.yaml"
        cmd = Cmd(
            model="test-model",
            task=["ifeval"],
            save_config=str(config_path),
        )

        mock_config = OmegaConf.create({
            "execution": {"type": "local", "output_dir": "/tmp"},
            "deployment": {"type": "none"},
            "target": {"api_endpoint": {"model_id": "test-model"}},
            "evaluation": {"tasks": [{"name": "ifeval"}]},
        })

        with patch.object(cmd, "_validate_config", return_value=True):
            with patch("nemo_evaluator_launcher.api.functional.RunConfig") as mock_rc:
                mock_rc.from_hydra.return_value = mock_config
                output = StringIO()
                with redirect_stdout(output):
                    cmd.execute()

        assert config_path.exists()

        # Verify it's valid YAML
        with open(config_path) as f:
            content = yaml.safe_load(f)

        assert content is not None
        assert "evaluation" in content

    def test_creates_parent_directories(self, tmp_path):
        """Test that parent directories are created."""
        nested_path = tmp_path / "nested" / "dir" / "config.yaml"
        cmd = Cmd(
            model="test-model",
            task=["ifeval"],
            save_config=str(nested_path),
        )

        mock_config = OmegaConf.create({
            "execution": {"type": "local"},
            "deployment": {"type": "none"},
            "target": {"api_endpoint": {"model_id": "test-model"}},
            "evaluation": {"tasks": [{"name": "ifeval"}]},
        })

        with patch.object(cmd, "_validate_config", return_value=True):
            with patch("nemo_evaluator_launcher.api.functional.RunConfig") as mock_rc:
                mock_rc.from_hydra.return_value = mock_config
                cmd.execute()

        assert nested_path.parent.exists()
        assert nested_path.exists()

    def test_includes_header_comments(self, tmp_path):
        """Test that saved config includes header comments."""
        config_path = tmp_path / "config.yaml"
        cmd = Cmd(
            model="test-model",
            task=["ifeval"],
            save_config=str(config_path),
        )

        mock_config = OmegaConf.create({
            "execution": {"type": "local"},
            "deployment": {"type": "none"},
            "target": {"api_endpoint": {"model_id": "test-model"}},
            "evaluation": {"tasks": [{"name": "ifeval"}]},
        })

        with patch.object(cmd, "_validate_config", return_value=True):
            with patch("nemo_evaluator_launcher.api.functional.RunConfig") as mock_rc:
                mock_rc.from_hydra.return_value = mock_config
                cmd.execute()

        with open(config_path) as f:
            content = f.read()

        assert "# Generated by nemo-evaluator-launcher" in content
        assert "# Generated at:" in content
        assert "# To run this configuration:" in content

    def test_exits_without_running_eval(self, tmp_path):
        """Test that --save-config exits without running evaluation."""
        config_path = tmp_path / "config.yaml"
        cmd = Cmd(
            model="test-model",
            task=["ifeval"],
            save_config=str(config_path),
        )

        mock_config = OmegaConf.create({
            "execution": {"type": "local"},
            "deployment": {"type": "none"},
            "target": {"api_endpoint": {"model_id": "test-model"}},
            "evaluation": {"tasks": [{"name": "ifeval"}]},
        })

        with patch.object(cmd, "_validate_config", return_value=True):
            with patch("nemo_evaluator_launcher.api.functional.RunConfig") as mock_rc:
                with patch("nemo_evaluator_launcher.api.functional.run_eval") as mock_run:
                    mock_rc.from_hydra.return_value = mock_config
                    cmd.execute()

                    # run_eval should NOT be called when --save-config is used
                    mock_run.assert_not_called()


class TestCheckFlag:
    """Tests for --check flag handling."""

    def test_runs_deep_checks_when_set(self):
        """Test that deep checks run when --check is set."""
        cmd = Cmd(
            model="test-model",
            task=["ifeval"],
            check=True,
            dry_run=True,  # Don't actually run the evaluation
        )

        mock_config = OmegaConf.create({
            "execution": {"type": "local", "output_dir": "/tmp"},
            "deployment": {"type": "none"},
            "target": {"api_endpoint": {"model_id": "test-model"}},
            "evaluation": {"tasks": [{"name": "ifeval"}]},
        })

        with patch.object(cmd, "_validate_config", return_value=True):
            with patch.object(cmd, "_run_deep_checks", return_value=True) as mock_checks:
                with patch("nemo_evaluator_launcher.api.functional.RunConfig") as mock_rc:
                    with patch("nemo_evaluator_launcher.api.functional.run_eval") as mock_run:
                        mock_rc.from_hydra.return_value = mock_config
                        mock_run.return_value = "test-invocation-id"
                        cmd.execute()
                        mock_checks.assert_called_once()

    def test_exits_on_failure_without_dry_run(self):
        """Test that check failure exits when not using --dry-run."""
        cmd = Cmd(
            model="test-model",
            task=["ifeval"],
            check=True,
            dry_run=False,
        )

        mock_config = OmegaConf.create({
            "execution": {"type": "local", "output_dir": "/tmp"},
            "deployment": {"type": "none"},
            "target": {"api_endpoint": {"model_id": "test-model"}},
            "evaluation": {"tasks": [{"name": "ifeval"}]},
        })

        with patch.object(cmd, "_validate_config", return_value=True):
            with patch.object(cmd, "_run_deep_checks", return_value=False):
                with patch("nemo_evaluator_launcher.api.functional.RunConfig") as mock_rc:
                    mock_rc.from_hydra.return_value = mock_config
                    with pytest.raises(SystemExit) as exc_info:
                        output = StringIO()
                        with redirect_stdout(output):
                            cmd.execute()

                    assert exc_info.value.code == 1

    def test_continues_with_dry_run(self):
        """Test that check failure continues when using --dry-run."""
        cmd = Cmd(
            model="test-model",
            task=["ifeval"],
            check=True,
            dry_run=True,
        )

        mock_config = OmegaConf.create({
            "execution": {"type": "local", "output_dir": "/tmp"},
            "deployment": {"type": "none"},
            "target": {"api_endpoint": {"model_id": "test-model"}},
            "evaluation": {"tasks": [{"name": "ifeval"}]},
        })

        with patch.object(cmd, "_validate_config", return_value=True):
            with patch.object(cmd, "_run_deep_checks", return_value=False):
                with patch("nemo_evaluator_launcher.api.functional.RunConfig") as mock_rc:
                    with patch("nemo_evaluator_launcher.api.functional.run_eval") as mock_run:
                        mock_rc.from_hydra.return_value = mock_config
                        mock_run.return_value = None  # dry_run returns None

                        # Should NOT raise SystemExit
                        output = StringIO()
                        with redirect_stdout(output):
                            cmd.execute()

                        # run_eval should be called even after check failure (with dry_run)
                        mock_run.assert_called_once()

    def test_no_deep_checks_by_default(self):
        """Test that deep checks don't run by default."""
        cmd = Cmd(
            model="test-model",
            task=["ifeval"],
            check=False,
            dry_run=True,
        )

        mock_config = OmegaConf.create({
            "execution": {"type": "local", "output_dir": "/tmp"},
            "deployment": {"type": "none"},
            "target": {"api_endpoint": {"model_id": "test-model"}},
            "evaluation": {"tasks": [{"name": "ifeval"}]},
        })

        with patch.object(cmd, "_validate_config", return_value=True):
            with patch.object(cmd, "_run_deep_checks") as mock_checks:
                with patch("nemo_evaluator_launcher.api.functional.RunConfig") as mock_rc:
                    with patch("nemo_evaluator_launcher.api.functional.run_eval") as mock_run:
                        mock_rc.from_hydra.return_value = mock_config
                        mock_run.return_value = None
                        cmd.execute()
                        mock_checks.assert_not_called()


class TestConfigModeHandling:
    """Tests for --config-mode handling."""

    def test_raw_mode_requires_config(self):
        """Test that raw mode requires --config."""
        cmd = Cmd(config_mode="raw", config=None)

        with patch("nemo_evaluator_launcher.api.functional.RunConfig"):
            with patch("nemo_evaluator_launcher.api.functional.run_eval"):
                with pytest.raises(ValueError, match="raw.*requires.*--config"):
                    cmd.execute()

    def test_raw_mode_no_overrides(self, tmp_path):
        """Test that raw mode doesn't allow -o overrides."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("execution:\n  type: local\n")

        cmd = Cmd(config_mode="raw", config=str(config_file), override=["foo=bar"])

        with patch("nemo_evaluator_launcher.api.functional.RunConfig"):
            with patch("nemo_evaluator_launcher.api.functional.run_eval"):
                with pytest.raises(ValueError, match="Cannot use.*raw.*with.*overrides"):
                    cmd.execute()

    def test_raw_mode_no_direct_flags(self, tmp_path):
        """Test that raw mode doesn't allow direct flags."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("execution:\n  type: local\n")

        cmd = Cmd(config_mode="raw", config=str(config_file), model="test-model")

        with patch("nemo_evaluator_launcher.api.functional.RunConfig"):
            with patch("nemo_evaluator_launcher.api.functional.run_eval"):
                with pytest.raises(ValueError, match="Cannot use.*raw.*with direct flags"):
                    cmd.execute()

    def test_invalid_config_mode(self):
        """Test that invalid config mode raises error."""
        cmd = Cmd(config_mode="invalid")

        with patch("nemo_evaluator_launcher.api.functional.RunConfig"):
            with patch("nemo_evaluator_launcher.api.functional.run_eval"):
                with pytest.raises(ValueError, match="Invalid.*config-mode"):
                    cmd.execute()


class TestDryRunOutput:
    """Tests for --dry-run flag behavior."""

    def test_dry_run_does_not_save_config(self, tmp_path):
        """Test that dry run doesn't save config to default location."""
        cmd = Cmd(
            model="test-model",
            task=["ifeval"],
            dry_run=True,
        )

        home_config_dir = tmp_path / ".nemo-evaluator" / "run_configs"

        mock_config = OmegaConf.create({
            "execution": {"type": "local", "output_dir": "/tmp"},
            "deployment": {"type": "none"},
            "target": {"api_endpoint": {"model_id": "test-model"}},
            "evaluation": {"tasks": [{"name": "ifeval"}]},
        })

        with patch.object(cmd, "_validate_config", return_value=True):
            with patch("pathlib.Path.home", return_value=tmp_path):
                with patch("nemo_evaluator_launcher.api.functional.RunConfig") as mock_rc:
                    with patch("nemo_evaluator_launcher.api.functional.run_eval") as mock_run:
                        mock_rc.from_hydra.return_value = mock_config
                        mock_run.return_value = None  # dry_run returns None
                        cmd.execute()

        # Config directory should not be created for dry run
        assert not home_config_dir.exists()
