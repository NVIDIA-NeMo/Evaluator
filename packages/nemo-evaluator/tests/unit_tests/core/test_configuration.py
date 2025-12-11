# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for configuration validation with YAML and CLI overrides."""

import importlib
import importlib.machinery
import importlib.util
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from pydantic import ValidationError

from nemo_evaluator.core.entrypoint import run_eval
from nemo_evaluator.core.utils import MisconfigurationError


@pytest.fixture
def n():
    # Reuse dummy framework package from installed_modules/1
    return 1


# Common target configuration used across test cases
DEFAULT_TARGET = {
    "api_endpoint": {
        "url": "http://localhost",
        "model_id": "test",
        "type": "chat",
    }
}


def make_config(config_dict, target_dict=None):
    """Helper to create test configuration with default target."""
    if target_dict is None:
        target_dict = DEFAULT_TARGET
    return {
        "config": config_dict,
        "target": target_dict,
    }


@pytest.fixture(autouse=True)
def installed_modules(n: int, monkeypatch):
    if not n:
        monkeypatch.setitem(sys.modules, "core_evals", None)
        yield
        return

    pkg = "core_evals"
    # Prefer a sibling installed_modules/<n> directory; fallback to functional_tests location.
    search_paths = [
        os.path.join(Path(__file__).parent.absolute(), f"installed_modules/{n}"),
        os.path.join(
            Path(__file__).parents[2].absolute(),
            "functional_tests",
            "installed_modules",
            str(n),
        ),
    ]
    spec = None
    for sp in search_paths:
        spec = importlib.machinery.PathFinder().find_spec(pkg, [sp])
        if spec is not None:
            break
    if spec is None:
        raise RuntimeError(
            f"Could not locate dummy package for {pkg} in {search_paths}"
        )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[pkg] = mod
    spec.loader.exec_module(mod)
    sys.modules[pkg] = importlib.import_module(pkg)
    yield
    sys.modules.pop(pkg)


@pytest.mark.parametrize(
    "yaml_config,cli_overrides,should_fail,error_keyword",
    [
        # Valid configs - should succeed
        (
            make_config(
                {
                    "type": "dummy_task",
                    "params": {"max_new_tokens": 100, "temperature": 0.7},
                }
            ),
            [],
            False,
            None,
        ),
        # Typo in YAML config params: max_tokens instead of max_new_tokens
        (
            make_config(
                {
                    "type": "dummy_task",
                    "params": {"max_tokens": 100},  # Typo!
                }
            ),
            [],
            True,
            "max_tokens",
        ),
        # Typo in YAML target: model instead of model_id
        (
            {
                "config": {"type": "dummy_task"},
                "target": {
                    "api_endpoint": {"url": "http://localhost", "model": "test"}
                },  # Typo!
            },
            [],
            True,
            "model",
        ),
        # Valid YAML config with CLI override using --overrides
        (
            make_config({"type": "dummy_task"}),
            ["--overrides", "config.params.max_new_tokens=200"],
            False,
            None,
        ),
        # Valid YAML with multiple CLI overrides using --overrides
        (
            make_config({"type": "dummy_task"}),
            [
                "--overrides",
                "config.params.max_new_tokens=200,config.params.temperature=0.5",
            ],
            False,
            None,
        ),
        # Valid config with interceptors and valid model fields
        (
            {
                "config": {"type": "dummy_task"},
                "target": {
                    "api_endpoint": {
                        "url": "http://localhost",
                        "model_id": "test",
                        "type": "completions",
                        "adapter_config": {
                            "interceptors": [{"name": "endpoint"}],
                            "log_failed_requests": True,
                            "endpoint_type": "completions",
                        },
                    }
                },
            },
            [],
            False,
            None,
        ),
        # Typo in adapter_config: use_cacing instead of use_caching
        (
            {
                "config": {"type": "dummy_task"},
                "target": {
                    "api_endpoint": {
                        "url": "http://localhost",
                        "model_id": "test",
                        "adapter_config": {"use_cacing": True},  # Typo!
                    }
                },
            },
            [],
            True,
            "use_cacing",
        ),
        # Typo in adapter_config with CLI overrides: log_faild_requests
        (
            {
                "config": {"type": "dummy_task"},
                "target": {
                    "api_endpoint": {
                        "url": "http://localhost",
                        "model_id": "test",
                        "adapter_config": {"log_faild_requests": True},  # Typo!
                    }
                },
            },
            ["--overrides", "config.params.temperature=0.7"],  # Valid CLI override
            True,
            "log_faild_requests",
        ),
        # Typo in CLI override: max_tokens instead of max_new_tokens
        (
            make_config({"type": "dummy_task"}),
            ["--overrides", "config.params.max_tokens=100"],  # Typo in override!
            True,
            "max_tokens",
        ),
        # Typo in second CLI override: temprature instead of temperature
        (
            make_config({"type": "dummy_task"}),
            [
                "--overrides",
                "config.params.max_new_tokens=100,config.params.temprature=0.5",
            ],  # Typo in 2nd param!
            True,
            "temprature",
        ),
        # Typo in CLI override for adapter_config: use_cacing instead of use_caching
        (
            make_config({"type": "dummy_task"}),
            [
                "--overrides",
                "target.api_endpoint.adapter_config.use_cacing=true",
            ],  # Typo in override!
            True,
            "use_cacing",
        ),
        # Extra field in target: unknown_field
        (
            {
                "config": {"type": "dummy_task"},
                "target": {
                    "api_endpoint": {"url": "http://localhost", "model_id": "test"},
                    "extra_target_field": "value",  # Extra field!
                },
            },
            [],
            True,
            "extra",
        ),
        # Extra field in config: unknown_config_param
        (
            make_config({"type": "dummy_task", "unknown_config_param": "value"}),
            [],
            True,
            "extra",
        ),
        # Extra field in CLI override: config.unknown_field
        (
            make_config({"type": "dummy_task"}),
            ["--overrides", "config.unknown_field=value"],
            True,
            "extra",
        ),
        # Invalid type: temperature passed as string instead of float
        (
            make_config(
                {"type": "dummy_task", "params": {"temperature": "not_a_number"}}
            ),
            [],
            True,
            "temperature",
        ),
        # Invalid: params.extra key not used in command
        (
            make_config(
                {"type": "dummy_task", "params": {"extra": {"non_existing": 789}}}
            ),
            [],
            True,
            "non_existing",
        ),
        # Invalid: CLI override with params.extra key not in command
        (
            make_config({"type": "dummy_task"}),
            ["--overrides", "config.params.extra.invalid_param=value"],
            True,
            "invalid_param",
        ),
        # Valid: Override params.extra.dummy_score that exists in command
        (
            make_config(
                {"type": "dummy_task", "params": {"extra": {"dummy_score": 999}}}
            ),
            [],
            False,
            None,
        ),
        # Valid: Standard param temperature IS in command
        (
            make_config({"type": "dummy_task", "params": {"temperature": 0.9}}),
            [],
            False,
            None,
        ),
        # Invalid: CLI override with standard param NOT in command
        (
            make_config({"type": "dummy_task"}),
            ["--overrides", "config.params.limit_samples=100"],
            True,
            "limit_samples",
        ),
        # Valid: Existing standard param in YAML - max_new_tokens IS in command
        (
            make_config({"type": "dummy_task", "params": {"max_new_tokens": 500}}),
            [],
            False,
            None,
        ),
        # Invalid: Non-existing standard param in YAML - unknown_param NOT in command
        (
            make_config({"type": "dummy_task", "params": {"unknown_param": 123}}),
            [],
            True,
            "unknown_param",
        ),
        # Valid: Existing standard param in CLI override - top_p IS in command
        (
            make_config({"type": "dummy_task"}),
            ["--overrides", "config.params.top_p=0.95"],
            False,
            None,
        ),
        # Invalid: Non-existing standard param in CLI override - bad_param NOT in command
        (
            make_config({"type": "dummy_task"}),
            ["--overrides", "config.params.bad_param=999"],
            True,
            "bad_param",
        ),
        # Valid: Existing extra param in YAML - dummy_score IS in command
        (
            make_config(
                {"type": "dummy_task", "params": {"extra": {"dummy_score": 888}}}
            ),
            [],
            False,
            None,
        ),
        # Invalid: Non-existing extra param in YAML - not_in_command NOT in command
        (
            make_config(
                {"type": "dummy_task", "params": {"extra": {"not_in_command": 777}}}
            ),
            [],
            True,
            "not_in_command",
        ),
        # Valid: Existing extra param in CLI override - dummy_score IS in command
        (
            make_config({"type": "dummy_task"}),
            ["--overrides", "config.params.extra.dummy_score=555"],
            False,
            None,
        ),
        # Invalid: Non-existing extra param in CLI override - missing_extra NOT in command
        (
            make_config({"type": "dummy_task"}),
            ["--overrides", "config.params.extra.missing_extra=444"],
            True,
            "missing_extra",
        ),
        # Valid: Multiple existing params in YAML (standard + extra)
        (
            make_config(
                {
                    "type": "dummy_task",
                    "params": {
                        "max_new_tokens": 200,
                        "temperature": 0.8,
                        "top_p": 0.9,
                        "extra": {"dummy_score": 333},
                    },
                }
            ),
            [],
            False,
            None,
        ),
        # Invalid: Mix of existing and non-existing params in YAML
        (
            make_config(
                {
                    "type": "dummy_task",
                    "params": {"max_new_tokens": 200, "non_existent": 100},
                }
            ),
            [],
            True,
            "non_existent",
        ),
        # Valid: Multiple existing params in CLI override
        (
            make_config({"type": "dummy_task"}),
            [
                "--overrides",
                "config.params.max_new_tokens=300,config.params.temperature=0.6,config.params.extra.dummy_score=111",
            ],
            False,
            None,
        ),
        # Invalid: Mix of existing and non-existing in CLI override
        (
            make_config({"type": "dummy_task"}),
            [
                "--overrides",
                "config.params.max_new_tokens=300,config.params.invalid_cli=50",
            ],
            True,
            "invalid_cli",
        ),
    ],
)
@patch("nemo_evaluator.core.entrypoint.evaluate")
def test_run_configuration_validation_yaml_and_cli(
    mock_evaluate,
    yaml_config,
    cli_overrides,
    should_fail,
    error_keyword,
    monkeypatch,
    tmp_path,
):
    """Test run configuration validation with both YAML config and CLI overrides."""
    # GIVEN a YAML config file and optional CLI overrides
    if "output_dir" not in yaml_config.get("config", {}):
        yaml_config.setdefault("config", {})["output_dir"] = str(tmp_path)

    cfg_path = tmp_path / "run.yml"
    cfg_path.write_text(yaml.safe_dump(yaml_config))

    cli_args = ["prog", "run_eval", "--run_config", str(cfg_path)] + cli_overrides
    monkeypatch.setattr(sys, "argv", cli_args)

    if should_fail:
        # WHEN running evaluation with invalid config (typos/extra fields)
        with pytest.raises((ValidationError, MisconfigurationError)) as exc_info:
            run_eval()

        # THEN validation error should mention the problematic field
        error_str = str(exc_info.value).lower()
        assert error_keyword.lower() in error_str, (
            f"Expected '{error_keyword}' in error message"
        )

        # AND evaluate function should not be called
        mock_evaluate.assert_not_called()
    else:
        # WHEN running evaluation with valid config
        run_eval()

        # THEN evaluate function should be called successfully
        mock_evaluate.assert_called_once()
