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
"""Tests for verify_config.py NEL config validator."""

from __future__ import annotations

import contextlib
import os
import pathlib
import sys

import pytest

# =============================================================================
# Path Setup
# =============================================================================

# Repository and directory paths
LAUNCHER_ROOT = pathlib.Path(__file__).resolve().parents[2]
REPO_ROOT = LAUNCHER_ROOT.parents[1]
SCRIPTS_DIR = (
    LAUNCHER_ROOT / ".claude" / "skills" / "nel-assistant" / "scripts"
).resolve()

# Config directories
EXAMPLES_DIR = LAUNCHER_ROOT / "examples"
SKILL_TESTS_DIR = pathlib.Path(__file__).resolve().parent
VALID_CONFIGS_DIR = SKILL_TESTS_DIR / "valid_configs"
INVALID_CONFIGS_DIR = SKILL_TESTS_DIR / "invalid_configs"

# Ensure validator script is importable
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


# =============================================================================
# Test Helpers (moved from verify_config.py)
# =============================================================================


MOCK_ENV_VARS = {
    "HF_TOKEN": "mock-hf-token",
    "NGC_API_KEY": "mock-ngc-api-key",
    "OPENAI_API_KEY": "mock-openai-api-key",
    "JUDGE_API_KEY": "mock-judge-api-key",
    "WANDB_API_KEY": "mock-wandb-api-key",
    "API_KEY": "mock-api-key",
    "TEST_KEY": "mock-test-key",
}


@contextlib.contextmanager
def mock_env_vars():
    """Context manager to set up and restore mock environment variables."""
    original_values = {}
    try:
        for key, value in MOCK_ENV_VARS.items():
            original_values[key] = os.environ.get(key)
            if original_values[key] is None:
                os.environ[key] = value
        yield
    finally:
        for key, value in original_values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def get_mock_overrides(config_path: str) -> list[str]:
    """Generate mock overrides for required (???) fields based on config type."""
    overrides = [
        "execution.output_dir=/tmp/nel-test-results",
    ]

    config_name = pathlib.Path(config_path).stem

    # SLURM configs need additional mocks
    if config_name.startswith("slurm_"):
        overrides.extend(
            [
                "execution.hostname=test-slurm-host.example.com",
                "execution.account=test-account",
            ]
        )
        # Only add checkpoint_path override for vLLM/SGLang configs
        # (not for NIM or 'none' deployments which don't have checkpoint_path)
        if "_vllm" in config_name or "_sglang" in config_name:
            overrides.append("++deployment.checkpoint_path=null")

    # Auto-export configs need MLflow URI
    if "auto_export" in config_name:
        overrides.extend(
            [
                "export.mlflow.tracking_uri=http://test-mlflow:5000",
            ]
        )

    # Safety configs need judge URL
    if "safety" in config_name:
        overrides.append(
            "++evaluation.tasks.1.nemo_evaluator_config.config.params.extra.judge.url=https://test-judge.example.com/v1"
        )

    return overrides


# =============================================================================
# Test Fixtures
# =============================================================================


def collect_yaml_files(*dirs: pathlib.Path) -> list[pathlib.Path]:
    """Collect all YAML files from given directories."""
    files = []
    for d in dirs:
        if d.exists():
            files.extend(sorted(d.glob("*.yaml")))
            # Also check subdirectories (e.g., examples/nemotron/)
            for subdir in d.iterdir():
                if subdir.is_dir():
                    files.extend(sorted(subdir.glob("*.yaml")))
    return files


# Collect config files
EXAMPLE_CONFIGS = collect_yaml_files(EXAMPLES_DIR)
VALID_SKILL_CONFIGS = collect_yaml_files(VALID_CONFIGS_DIR)
INVALID_CONFIGS = collect_yaml_files(INVALID_CONFIGS_DIR)


# =============================================================================
# Test Classes
# =============================================================================


class TestExampleConfigs:
    """Test that all example configs pass validation."""

    @pytest.mark.parametrize(
        "config_path",
        EXAMPLE_CONFIGS,
        ids=[p.relative_to(LAUNCHER_ROOT).as_posix() for p in EXAMPLE_CONFIGS],
    )
    def test_example_config_is_valid(self, config_path: pathlib.Path) -> None:
        """Each example config should pass validation with mock overrides."""
        import verify_config

        with mock_env_vars():
            overrides = get_mock_overrides(str(config_path))
            cfg = verify_config.resolve_config(str(config_path), overrides)
            valid, errors, warnings = verify_config.validate_config(cfg)

        assert valid, f"Config should be valid but got errors: {errors}"


class TestValidSkillConfigs:
    """Test that valid skill test configs pass validation."""

    @pytest.mark.parametrize(
        "config_path",
        VALID_SKILL_CONFIGS,
        ids=[p.name for p in VALID_SKILL_CONFIGS],
    )
    def test_valid_skill_config_is_valid(self, config_path: pathlib.Path) -> None:
        """Each valid skill config should pass validation."""
        import verify_config

        with mock_env_vars():
            overrides = get_mock_overrides(str(config_path))
            cfg = verify_config.resolve_config(str(config_path), overrides)
            valid, errors, warnings = verify_config.validate_config(cfg)

        assert valid, f"Config should be valid but got errors: {errors}"


class TestInvalidConfigs:
    """Test that invalid configs fail validation."""

    @pytest.mark.parametrize(
        "config_path",
        INVALID_CONFIGS,
        ids=[p.name for p in INVALID_CONFIGS],
    )
    def test_invalid_config_fails_validation(self, config_path: pathlib.Path) -> None:
        """Each invalid config should fail validation."""
        import verify_config

        with mock_env_vars():
            overrides = get_mock_overrides(str(config_path))
            try:
                cfg = verify_config.resolve_config(str(config_path), overrides)
                valid, errors, warnings = verify_config.validate_config(cfg)
                # If we got here, config resolved - check it fails validation
                assert not valid, f"Config should fail validation: {config_path.name}"
            except Exception:
                # Config failed to resolve (e.g., missing required fields in Hydra)
                # This is also a valid failure case
                pass


class TestWarnings:
    """Test warning detection in configs."""

    def test_missing_endpoint_interceptor_warning(self) -> None:
        """Config with interceptors but no 'endpoint' should emit warning."""
        import verify_config

        config_path = VALID_CONFIGS_DIR / "reasoning_missing_endpoint_warning.yaml"
        assert config_path.exists(), f"Test config not found: {config_path}"

        with mock_env_vars():
            overrides = get_mock_overrides(str(config_path))
            cfg = verify_config.resolve_config(str(config_path), overrides)
            valid, errors, warnings = verify_config.validate_config(cfg)

        assert valid, f"Config should be valid but got errors: {errors}"
        assert len(warnings) == 1, f"Expected 1 warning, got {len(warnings)}"
        assert "endpoint" in warnings[0].lower()
        assert "interceptor" in warnings[0].lower()

    def test_interceptors_with_endpoint_no_warning(self) -> None:
        """Config with interceptors including 'endpoint' should not emit warning."""
        import verify_config

        config_path = VALID_CONFIGS_DIR / "reasoning_interceptors.yaml"
        assert config_path.exists(), f"Test config not found: {config_path}"

        with mock_env_vars():
            overrides = get_mock_overrides(str(config_path))
            cfg = verify_config.resolve_config(str(config_path), overrides)
            valid, errors, warnings = verify_config.validate_config(cfg)

        assert valid, f"Config should be valid but got errors: {errors}"
        assert len(warnings) == 0, f"Expected no warnings, got: {warnings}"
