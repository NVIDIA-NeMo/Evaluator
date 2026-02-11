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
"""Tests for the validation module."""

from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

import pytest
from omegaconf import MISSING, OmegaConf

from nemo_evaluator_launcher.common.validation import (
    ValidationError,
    ValidationResult,
    validate_config,
    validate_deployment,
    validate_executor,
    validate_missing_values,
    validate_target_api_endpoint,
    validate_tasks,
)


class TestValidationResult:
    """Tests for ValidationResult class."""

    def test_empty_result_is_valid(self):
        """Test that empty result is valid."""
        result = ValidationResult()
        assert result.valid is True
        assert len(result.errors) == 0
        assert len(result.warnings) == 0

    def test_add_error(self):
        """Test adding an error."""
        result = ValidationResult()
        result.add_error("path.to.field", "Missing value", "Add --flag")
        assert result.valid is False
        assert len(result.errors) == 1
        assert result.errors[0].path == "path.to.field"
        assert result.errors[0].message == "Missing value"
        assert result.errors[0].suggestion == "Add --flag"

    def test_add_warning(self):
        """Test adding a warning doesn't make result invalid."""
        result = ValidationResult()
        result.add_warning("some.path", "Warning message")
        assert result.valid is True  # Warnings don't invalidate
        assert len(result.warnings) == 1

    def test_print_errors(self):
        """Test error printing."""
        result = ValidationResult()
        result.add_error("execution.hostname", "Required field", "--slurm-hostname")
        result.add_warning("env.NGC_API_KEY", "Not set", "export NGC_API_KEY=...")

        output = StringIO()
        with redirect_stdout(output):
            result.print_errors()

        printed = output.getvalue()
        assert "execution.hostname" in printed
        assert "Required field" in printed
        assert "env.NGC_API_KEY" in printed


class TestValidateExecutor:
    """Tests for validate_executor function."""

    def test_local_executor_requires_output_dir(self):
        """Test local executor requires output_dir."""
        cfg = OmegaConf.create({
            "execution": {"type": "local"},
        })
        result = ValidationResult()
        validate_executor(cfg, result)
        assert result.valid is False
        assert any("output_dir" in e.path for e in result.errors)

    def test_local_executor_valid(self):
        """Test valid local executor config."""
        cfg = OmegaConf.create({
            "execution": {"type": "local", "output_dir": "/tmp/results"},
        })
        result = ValidationResult()
        validate_executor(cfg, result)
        assert result.valid is True

    def test_slurm_executor_requires_all_fields(self):
        """Test slurm executor requires hostname, account, output_dir."""
        cfg = OmegaConf.create({
            "execution": {"type": "slurm"},
        })
        result = ValidationResult()
        validate_executor(cfg, result)
        assert result.valid is False
        assert len(result.errors) == 3  # hostname, account, output_dir

    def test_slurm_executor_valid(self):
        """Test valid slurm executor config."""
        cfg = OmegaConf.create({
            "execution": {
                "type": "slurm",
                "hostname": "cluster.example.com",
                "account": "my-account",
                "output_dir": "/shared/results",
            },
        })
        result = ValidationResult()
        validate_executor(cfg, result)
        assert result.valid is True


class TestValidateDeployment:
    """Tests for validate_deployment function."""

    def test_none_deployment_no_requirements(self):
        """Test deployment=none has no deployment requirements."""
        cfg = OmegaConf.create({
            "deployment": {"type": "none"},
        })
        result = ValidationResult()
        validate_deployment(cfg, result)
        assert result.valid is True

    def test_vllm_deployment_requires_fields(self):
        """Test vllm deployment requires checkpoint and model name."""
        cfg = OmegaConf.create({
            "deployment": {"type": "vllm"},
        })
        result = ValidationResult()
        validate_deployment(cfg, result)
        assert result.valid is False
        assert any("checkpoint_path" in e.path for e in result.errors)
        assert any("served_model_name" in e.path for e in result.errors)

    def test_vllm_deployment_valid(self):
        """Test valid vllm deployment config."""
        cfg = OmegaConf.create({
            "deployment": {
                "type": "vllm",
                "checkpoint_path": "/path/to/model",
                "served_model_name": "my-model",
            },
        })
        result = ValidationResult()
        validate_deployment(cfg, result)
        assert result.valid is True

    def test_nim_deployment_requires_model_name(self):
        """Test nim deployment requires model_name."""
        cfg = OmegaConf.create({
            "deployment": {"type": "nim"},
        })
        result = ValidationResult()
        validate_deployment(cfg, result)
        assert result.valid is False
        assert any("model_name" in e.path for e in result.errors)

    def test_warning_when_url_set_with_deployment(self):
        """Test warning when target.api_endpoint.url is set with deployment."""
        cfg = OmegaConf.create({
            "deployment": {
                "type": "vllm",
                "checkpoint_path": "/path",
                "served_model_name": "model",
            },
            "target": {
                "api_endpoint": {
                    "url": "http://example.com",
                }
            },
        })
        result = ValidationResult()
        validate_deployment(cfg, result)
        # Should have warnings
        assert len(result.warnings) >= 1


class TestValidateTargetApiEndpoint:
    """Tests for validate_target_api_endpoint function."""

    def test_requires_model_id_for_none_deployment(self):
        """Test model_id is required when deployment=none."""
        cfg = OmegaConf.create({
            "deployment": {"type": "none"},
        })
        result = ValidationResult()
        validate_target_api_endpoint(cfg, result)
        assert result.valid is False
        assert any("model_id" in e.path for e in result.errors)

    def test_valid_with_model_id(self):
        """Test valid when model_id is provided."""
        cfg = OmegaConf.create({
            "deployment": {"type": "none"},
            "target": {
                "api_endpoint": {
                    "model_id": "meta/llama-3.2-3b-instruct",
                }
            },
        })
        result = ValidationResult()
        validate_target_api_endpoint(cfg, result)
        assert result.valid is True

    def test_skipped_for_non_none_deployment(self):
        """Test validation is skipped for non-none deployments."""
        cfg = OmegaConf.create({
            "deployment": {"type": "vllm"},
        })
        result = ValidationResult()
        validate_target_api_endpoint(cfg, result)
        assert result.valid is True  # No errors because deployment handles this


class TestValidateTasks:
    """Tests for validate_tasks function."""

    def test_no_tasks_error(self):
        """Test error when no tasks specified."""
        cfg = OmegaConf.create({
            "evaluation": {"tasks": []},
        })
        result = ValidationResult()
        validate_tasks(cfg, result)
        assert result.valid is False
        assert any("tasks" in e.path for e in result.errors)

    def test_valid_with_tasks(self):
        """Test valid when tasks are specified."""
        cfg = OmegaConf.create({
            "evaluation": {"tasks": [{"name": "ifeval"}]},
        })
        result = ValidationResult()
        # Mock the mapping lookup to avoid loading actual mapping
        with patch("nemo_evaluator_launcher.common.mapping.load_tasks_mapping") as mock:
            mock.return_value = {("helm", "ifeval"): {"task": "ifeval"}}
            validate_tasks(cfg, result)
        assert result.valid is True


class TestValidateMissingValues:
    """Tests for validate_missing_values function."""

    def test_detects_missing_values(self):
        """Test that ??? values are detected."""
        # Create a config with MISSING value using OmegaConf.structured approach
        cfg = OmegaConf.create({
            "execution": {
                "hostname": "???",  # Use string "???" which OmegaConf interprets as MISSING
                "output_dir": "/tmp/results",
            },
        })
        result = ValidationResult()
        validate_missing_values(cfg, result)
        assert result.valid is False
        assert any("hostname" in e.path for e in result.errors)

    def test_no_missing_values(self):
        """Test no errors when all values are present."""
        cfg = OmegaConf.create({
            "execution": {
                "hostname": "cluster.example.com",
                "output_dir": "/tmp/results",
            },
        })
        result = ValidationResult()
        validate_missing_values(cfg, result)
        assert result.valid is True


class TestValidateConfig:
    """Tests for the main validate_config function."""

    def test_comprehensive_validation(self):
        """Test that all validations run together."""
        cfg = OmegaConf.create({
            "execution": {"type": "slurm"},  # Missing required fields
            "deployment": {"type": "none"},  # Missing model_id
            "evaluation": {"tasks": []},  # No tasks
        })

        result = validate_config(cfg)
        assert result.valid is False
        # Should have errors from multiple validators
        assert len(result.errors) >= 3

    def test_valid_config(self):
        """Test valid configuration passes all validations."""
        cfg = OmegaConf.create({
            "execution": {
                "type": "local",
                "output_dir": "/tmp/results",
            },
            "deployment": {"type": "none"},
            "target": {
                "api_endpoint": {
                    "model_id": "meta/llama-3.2-3b-instruct",
                }
            },
            "evaluation": {"tasks": [{"name": "ifeval"}]},
        })

        with patch("nemo_evaluator_launcher.common.mapping.load_tasks_mapping") as mock:
            mock.return_value = {("helm", "ifeval"): {"task": "ifeval"}}
            result = validate_config(cfg)

        assert result.valid is True

    def test_skip_task_validation(self):
        """Test skipping task validation."""
        cfg = OmegaConf.create({
            "execution": {
                "type": "local",
                "output_dir": "/tmp/results",
            },
            "deployment": {"type": "none"},
            "target": {
                "api_endpoint": {
                    "model_id": "meta/llama-3.2-3b-instruct",
                }
            },
            "evaluation": {"tasks": [{"name": "unknown-task"}]},
        })

        result = validate_config(cfg, skip_task_validation=True)
        # Should be valid because task validation is skipped
        assert result.valid is True
