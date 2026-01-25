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
"""Tests for the CLI wizard command."""

import pytest

from nemo_evaluator_launcher.cli.wizard import (
    Cmd,
    DEFAULT_INTERCEPTORS,
    EXPORTERS,
    INTERCEPTORS,
    NVIDIA_GREEN,
    REASONING_MODES,
    WIZARD_STYLE,
)


class TestWizardConstants:
    """Tests for wizard module constants."""

    def test_default_interceptors(self):
        """Test that default interceptors include caching and response_stats."""
        assert "caching" in DEFAULT_INTERCEPTORS
        assert "response_stats" in DEFAULT_INTERCEPTORS
        assert len(DEFAULT_INTERCEPTORS) == 2

    def test_interceptors_dict_structure(self):
        """Test INTERCEPTORS dict has correct structure."""
        expected_interceptors = ["caching", "response_stats", "request_logging", "system_message"]
        for name in expected_interceptors:
            assert name in INTERCEPTORS
            assert isinstance(INTERCEPTORS[name], str)
            assert len(INTERCEPTORS[name]) > 0

    def test_reasoning_modes_structure(self):
        """Test REASONING_MODES dict has correct structure."""
        expected_modes = ["none", "think", "no_think", "custom"]
        for mode in expected_modes:
            assert mode in REASONING_MODES
            assert isinstance(REASONING_MODES[mode], str)

        # Check default is first
        assert list(REASONING_MODES.keys())[0] == "none"

    def test_exporters_structure(self):
        """Test EXPORTERS dict has correct structure."""
        expected_exporters = ["local", "mlflow", "wandb", "gsheets"]
        for name in expected_exporters:
            assert name in EXPORTERS
            assert isinstance(EXPORTERS[name], str)

    def test_nvidia_green_color(self):
        """Test NVIDIA green color is correctly defined."""
        assert NVIDIA_GREEN == "#76b900"

    def test_wizard_style_exists(self):
        """Test WIZARD_STYLE is properly defined."""
        assert WIZARD_STYLE is not None


class TestWizardCmd:
    """Tests for the wizard Cmd dataclass."""

    def test_default_values(self):
        """Test default values for Cmd."""
        cmd = Cmd()
        assert cmd.save_only is False
        assert cmd.output == ""

    def test_save_only_flag(self):
        """Test save_only flag can be set."""
        cmd = Cmd(save_only=True)
        assert cmd.save_only is True

    def test_output_path(self):
        """Test output path can be set."""
        cmd = Cmd(output="my-config.yaml")
        assert cmd.output == "my-config.yaml"


class TestBuildYamlConfig:
    """Tests for _build_yaml_config method."""

    def test_basic_api_endpoint_config(self):
        """Test building basic API endpoint config."""
        cmd = Cmd()
        config = {
            "executor": "local",
            "deployment": "none",
            "model": "meta/llama-3.2-3b-instruct",
            "url": "https://integrate.api.nvidia.com/v1/chat/completions",
            "api_key_env": "NGC_API_KEY",
            "tasks": ["simple_evals.mmlu"],
            "interceptors": ["caching", "response_stats"],
            "output_dir": "./results",
        }

        yaml_config = cmd._build_yaml_config(config)

        # Check defaults section
        assert "defaults" in yaml_config
        assert {"execution": "local"} in yaml_config["defaults"]
        assert {"deployment": "none"} in yaml_config["defaults"]

        # Check execution section
        assert yaml_config["execution"]["output_dir"] == "./results"

        # Check target section
        assert "target" in yaml_config
        assert "api_endpoint" in yaml_config["target"]
        assert yaml_config["target"]["api_endpoint"]["model_id"] == "meta/llama-3.2-3b-instruct"

        # Check evaluation tasks
        assert "evaluation" in yaml_config
        assert len(yaml_config["evaluation"]["tasks"]) == 1
        assert yaml_config["evaluation"]["tasks"][0]["name"] == "simple_evals.mmlu"

    def test_slurm_executor_config(self):
        """Test building SLURM executor config."""
        cmd = Cmd()
        config = {
            "executor": "slurm",
            "deployment": "none",
            "model": "meta/llama-3.2-3b-instruct",
            "slurm_hostname": "cluster.example.com",
            "slurm_account": "my-account",
            "slurm_partition": "gpu",
            "slurm_walltime": "02:00:00",
            "tasks": ["simple_evals.mmlu"],
            "interceptors": [],
            "output_dir": "./results",
        }

        yaml_config = cmd._build_yaml_config(config)

        # Check SLURM execution config
        assert {"execution": "slurm/default"} in yaml_config["defaults"]
        assert yaml_config["execution"]["hostname"] == "cluster.example.com"
        assert yaml_config["execution"]["account"] == "my-account"
        assert yaml_config["execution"]["partition"] == "gpu"
        assert yaml_config["execution"]["walltime"] == "02:00:00"

    def test_vllm_deployment_config(self):
        """Test building vLLM deployment config."""
        cmd = Cmd()
        config = {
            "executor": "local",
            "deployment": "vllm",
            "hf_model": "meta-llama/Llama-3.2-3B-Instruct",
            "model_name": "llama-3.2-3b",
            "tensor_parallel": 2,
            "tasks": ["simple_evals.mmlu"],
            "interceptors": [],
            "output_dir": "./results",
        }

        yaml_config = cmd._build_yaml_config(config)

        # Check deployment config
        assert {"deployment": "vllm"} in yaml_config["defaults"]
        assert "deployment" in yaml_config
        assert yaml_config["deployment"]["hf_model_handle"] == "meta-llama/Llama-3.2-3B-Instruct"
        assert yaml_config["deployment"]["model_name"] == "llama-3.2-3b"
        assert yaml_config["deployment"]["tensor_parallel"] == 2

    def test_reasoning_config(self):
        """Test building config with reasoning enabled."""
        cmd = Cmd()
        config = {
            "executor": "local",
            "deployment": "none",
            "model": "meta/llama-3.2-3b-instruct",
            "tasks": ["simple_evals.mmlu"],
            "reasoning": {
                "process_reasoning_traces": True,
                "use_system_prompt": True,
                "custom_system_prompt": "/think",
            },
            "interceptors": ["caching"],
            "output_dir": "./results",
        }

        yaml_config = cmd._build_yaml_config(config)

        # Check adapter config contains reasoning
        adapter_config = yaml_config["target"]["api_endpoint"]["adapter_config"]
        assert adapter_config["process_reasoning_traces"] is True
        assert adapter_config["use_system_prompt"] is True
        assert adapter_config["custom_system_prompt"] == "/think"

    def test_interceptors_in_config(self):
        """Test interceptors are correctly added to config."""
        cmd = Cmd()
        config = {
            "executor": "local",
            "deployment": "none",
            "model": "meta/llama-3.2-3b-instruct",
            "tasks": ["simple_evals.mmlu"],
            "interceptors": ["caching", "response_stats", "request_logging"],
            "output_dir": "./results",
        }

        yaml_config = cmd._build_yaml_config(config)

        interceptors = yaml_config["target"]["api_endpoint"]["adapter_config"]["interceptors"]

        # Check interceptors are present (plus endpoint at the end)
        interceptor_names = [i["name"] for i in interceptors]
        assert "caching" in interceptor_names
        assert "response_stats" in interceptor_names
        assert "request_logging" in interceptor_names
        assert "endpoint" in interceptor_names

        # Endpoint should be last
        assert interceptors[-1]["name"] == "endpoint"

        # Caching should have config
        caching_interceptor = next(i for i in interceptors if i["name"] == "caching")
        assert caching_interceptor["config"]["cache_dir"] == "/results/cache"
        assert caching_interceptor["config"]["reuse_cached_responses"] is True

    def test_limit_samples_config(self):
        """Test limit_samples is correctly added to config."""
        cmd = Cmd()
        config = {
            "executor": "local",
            "deployment": "none",
            "model": "meta/llama-3.2-3b-instruct",
            "tasks": ["simple_evals.mmlu"],
            "interceptors": [],
            "output_dir": "./results",
            "limit_samples": 100,
        }

        yaml_config = cmd._build_yaml_config(config)

        assert "nemo_evaluator_config" in yaml_config["evaluation"]
        assert yaml_config["evaluation"]["nemo_evaluator_config"]["config"]["params"]["limit_samples"] == 100

    def test_multiple_tasks(self):
        """Test multiple tasks are correctly added."""
        cmd = Cmd()
        config = {
            "executor": "local",
            "deployment": "none",
            "model": "meta/llama-3.2-3b-instruct",
            "tasks": ["simple_evals.mmlu", "simple_evals.gpqa_diamond", "ifeval"],
            "interceptors": [],
            "output_dir": "./results",
        }

        yaml_config = cmd._build_yaml_config(config)

        tasks = yaml_config["evaluation"]["tasks"]
        assert len(tasks) == 3
        assert {"name": "simple_evals.mmlu"} in tasks
        assert {"name": "simple_evals.gpqa_diamond"} in tasks
        assert {"name": "ifeval"} in tasks

    def test_nim_deployment_config(self):
        """Test building NIM deployment config."""
        cmd = Cmd()
        config = {
            "executor": "local",
            "deployment": "nim",
            "nim_model": "llama3-8b-instruct",
            "tasks": ["simple_evals.mmlu"],
            "interceptors": [],
            "output_dir": "./results",
        }

        yaml_config = cmd._build_yaml_config(config)

        assert {"deployment": "nim"} in yaml_config["defaults"]
        assert yaml_config["deployment"]["nim_model"] == "llama3-8b-instruct"

    def test_payload_modifier_config(self):
        """Test payload modifier is correctly added."""
        cmd = Cmd()
        config = {
            "executor": "local",
            "deployment": "none",
            "model": "meta/llama-3.2-3b-instruct",
            "tasks": ["simple_evals.mmlu"],
            "payload_modifier": {
                "params_to_add": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                }
            },
            "interceptors": ["caching"],
            "output_dir": "./results",
        }

        yaml_config = cmd._build_yaml_config(config)

        interceptors = yaml_config["target"]["api_endpoint"]["adapter_config"]["interceptors"]
        payload_modifier = next((i for i in interceptors if i["name"] == "payload_modifier"), None)

        assert payload_modifier is not None
        assert payload_modifier["config"]["params_to_add"]["temperature"] == 0.7
        assert payload_modifier["config"]["params_to_add"]["top_p"] == 0.9


class TestSaveConfig:
    """Tests for _save_config method."""

    def test_save_config_creates_file(self, tmp_path):
        """Test that _save_config creates a valid YAML file."""
        import yaml

        cmd = Cmd()
        config = {
            "executor": "local",
            "deployment": "none",
            "model": "meta/llama-3.2-3b-instruct",
            "tasks": ["simple_evals.mmlu"],
            "interceptors": ["caching", "response_stats"],
            "output_dir": "./results",
            "exporters": [],
        }

        config_path = tmp_path / "test-config.yaml"
        cmd._save_config(config, str(config_path))

        # Verify file exists
        assert config_path.exists()

        # Verify it's valid YAML
        with open(config_path) as f:
            loaded = yaml.safe_load(f)

        assert "defaults" in loaded
        assert "execution" in loaded
        assert "evaluation" in loaded

    def test_save_config_with_exporters(self, tmp_path, capsys):
        """Test that _save_config shows export commands when exporters configured."""
        cmd = Cmd()
        config = {
            "executor": "local",
            "deployment": "none",
            "model": "meta/llama-3.2-3b-instruct",
            "tasks": ["simple_evals.mmlu"],
            "interceptors": [],
            "output_dir": "./results",
            "exporters": [
                {"dest": "local", "format": "json"},
                {"dest": "mlflow", "tracking_uri": "http://localhost:5000"},
            ],
        }

        config_path = tmp_path / "test-config.yaml"
        cmd._save_config(config, str(config_path))

        captured = capsys.readouterr()
        assert "Export after run" in captured.out
        assert "nel export <id> --dest local" in captured.out
        assert "nel export <id> --dest mlflow" in captured.out


class TestShowSummary:
    """Tests for _show_summary method."""

    def test_show_summary_basic(self, capsys):
        """Test summary displays key config values."""
        cmd = Cmd()
        config = {
            "executor": "local",
            "deployment": "none",
            "model": "meta/llama-3.2-3b-instruct",
            "tasks": ["simple_evals.mmlu", "ifeval"],
            "interceptors": ["caching"],
            "output_dir": "./results",
        }

        cmd._show_summary(config)

        captured = capsys.readouterr()
        assert "local" in captured.out
        assert "none" in captured.out
        assert "simple_evals.mmlu" in captured.out

    def test_show_summary_truncates_many_tasks(self, capsys):
        """Test summary truncates task list when more than 3 tasks."""
        cmd = Cmd()
        config = {
            "executor": "local",
            "deployment": "none",
            "model": "meta/llama-3.2-3b-instruct",
            "tasks": ["task1", "task2", "task3", "task4", "task5"],
            "interceptors": [],
            "output_dir": "./results",
        }

        cmd._show_summary(config)

        captured = capsys.readouterr()
        assert "+2 more" in captured.out


class TestCLIIntegration:
    """Integration tests for wizard CLI."""

    def test_wizard_help(self):
        """Test wizard --help works."""
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "nemo_evaluator_launcher.cli.main", "wizard", "--help"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "wizard" in result.stdout.lower() or "configuration" in result.stdout.lower()
        assert "--save-only" in result.stdout or "save_only" in result.stdout
        assert "--output" in result.stdout or "-o" in result.stdout
