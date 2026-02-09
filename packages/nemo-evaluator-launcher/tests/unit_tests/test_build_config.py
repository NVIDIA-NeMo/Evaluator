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
"""Tests for build_config NEL config builder."""

from __future__ import annotations

import contextlib
import os
import pathlib
from typing import List

import pytest
import yaml

from nemo_evaluator_launcher.common.build_config import (
    _load_template,
    build_config,
    deep_merge,
    resolve_output_path,
)

# =============================================================================
# Test-only utilities (moved from build_config.py)
# =============================================================================

MOCK_ENV_VARS = {
    "HF_TOKEN": "mock-hf-token",
    "NGC_API_KEY": "mock-ngc-api-key",
    "OPENAI_API_KEY": "mock-openai-api-key",
    "JUDGE_API_KEY": "mock-judge-api-key",
    "WANDB_API_KEY": "mock-wandb-api-key",
    "API_KEY": "mock-api-key",
    "TEST_KEY": "mock-test-key",
    "HF_TOKEN_FOR_GPQA_DIAMOND": "mock-hf-token-gpqa",
    "HF_TOKEN_FOR_AEGIS_V2": "mock-hf-token-aegis",
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


def get_mock_overrides(
    execution: str,
    deployment: str,
    export: str,
    model_type: str,
    benchmarks: List[str],
) -> List[str]:
    """Generate mock overrides for required (???) fields based on config choices."""
    overrides = [
        "execution.output_dir=/tmp/nel-test-results",
    ]

    # SLURM configs need additional mocks
    if execution == "slurm":
        overrides.extend(
            [
                "execution.hostname=test-slurm-host.example.com",
                "execution.account=test-account",
            ]
        )

    # Deployment-specific mocks
    if deployment == "none":
        overrides.extend(
            [
                "target.api_endpoint.model_id=test/model",
                "target.api_endpoint.url=https://test-api.example.com/v1/chat/completions",
                "target.api_endpoint.api_key_name=TEST_KEY",
            ]
        )
    elif deployment == "vllm":
        overrides.extend(
            [
                "deployment.hf_model_handle=test/model",
                "deployment.served_model_name=test/model",
            ]
        )
    elif deployment == "sglang":
        overrides.extend(
            [
                "deployment.hf_model_handle=test/model",
                "deployment.served_model_name=test/model",
            ]
        )
    elif deployment == "nim":
        overrides.extend(
            [
                "deployment.image=nvcr.io/nim/test/model:latest",
                "deployment.served_model_name=test/model",
            ]
        )
    elif deployment == "trtllm":
        overrides.extend(
            [
                "deployment.checkpoint_path=/path/to/checkpoint",
                "deployment.served_model_name=test/model",
            ]
        )

    # Export-specific mocks
    if export == "mlflow":
        overrides.append("export.mlflow.tracking_uri=http://test-mlflow:5000")
    elif export == "wandb":
        overrides.append("export.wandb.project=test-project")

    # Model type specific mocks
    if model_type == "base":
        overrides.append(
            "evaluation.nemo_evaluator_config.config.params.extra.tokenizer=test/tokenizer"
        )
    elif model_type == "reasoning":
        overrides.append(
            "+evaluation.nemo_evaluator_config.target.api_endpoint.adapter_config.params_to_add.chat_template_kwargs.enable_thinking=true"
        )

    # Safety benchmark mocks (need judge URL)
    if "safety" in benchmarks:
        try:
            _load_template(f"evaluation/{model_type}/safety.yaml")
        except FileNotFoundError:
            pass
        else:
            overrides.append(
                "++evaluation.tasks.1.nemo_evaluator_config.config.params.extra.judge.url=https://test-judge.example.com/v1"
            )

    return overrides


# =============================================================================
# Unit Tests: deep_merge
# =============================================================================


class TestDeepMerge:
    """Unit tests for the deep_merge helper."""

    def test_flat_dicts(self) -> None:
        assert deep_merge({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}

    def test_override_value(self) -> None:
        assert deep_merge({"a": 1}, {"a": 2}) == {"a": 2}

    def test_nested_merge(self) -> None:
        base = {"x": {"a": 1, "b": 2}}
        override = {"x": {"b": 3, "c": 4}}
        assert deep_merge(base, override) == {"x": {"a": 1, "b": 3, "c": 4}}

    def test_list_extension(self) -> None:
        base = {"tasks": [{"name": "a"}]}
        override = {"tasks": [{"name": "b"}]}
        result = deep_merge(base, override)
        assert result["tasks"] == [{"name": "a"}, {"name": "b"}]

    def test_empty_base(self) -> None:
        assert deep_merge({}, {"a": 1}) == {"a": 1}

    def test_empty_override(self) -> None:
        assert deep_merge({"a": 1}, {}) == {"a": 1}

    def test_does_not_mutate_base(self) -> None:
        base = {"a": {"b": 1}}
        deep_merge(base, {"a": {"b": 2}})
        assert base == {"a": {"b": 1}}


# =============================================================================
# Unit Tests: resolve_output_path
# =============================================================================


class TestResolveOutputPath:
    def test_explicit_yaml_file(self, tmp_path: pathlib.Path) -> None:
        out = tmp_path / "my_config.yaml"
        result = resolve_output_path(out, "local", "vllm", "chat", ["standard"])
        assert result == out

    def test_directory_auto_generates_name(self, tmp_path: pathlib.Path) -> None:
        result = resolve_output_path(tmp_path, "slurm", "nim", "base", ["code"])
        assert result.parent == tmp_path
        assert result.name == "slurm_nim_base_code.yaml"

    def test_benchmarks_sorted_in_filename(self, tmp_path: pathlib.Path) -> None:
        result = resolve_output_path(
            tmp_path, "local", "vllm", "chat", ["code", "standard"]
        )
        assert result.name == "local_vllm_chat_code_standard.yaml"

    def test_none_uses_cwd(self, tmp_path: pathlib.Path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        result = resolve_output_path(None, "local", "vllm", "chat", ["standard"])
        assert result.parent == tmp_path
        assert result.name == "local_vllm_chat_standard.yaml"

    def test_creates_parent_dir(self, tmp_path: pathlib.Path) -> None:
        new_dir = tmp_path / "sub" / "dir"
        result = resolve_output_path(new_dir, "local", "vllm", "chat", ["standard"])
        assert new_dir.is_dir()
        assert result.parent == new_dir

    def test_existing_file_gets_suffix(self, tmp_path: pathlib.Path) -> None:
        (tmp_path / "local_vllm_chat_standard.yaml").touch()
        result = resolve_output_path(tmp_path, "local", "vllm", "chat", ["standard"])
        assert result.name == "local_vllm_chat_standard_1.yaml"

    def test_multiple_existing_increments(self, tmp_path: pathlib.Path) -> None:
        for name in (
            "local_vllm_chat_standard.yaml",
            "local_vllm_chat_standard_1.yaml",
            "local_vllm_chat_standard_3.yaml",
        ):
            (tmp_path / name).touch()
        result = resolve_output_path(tmp_path, "local", "vllm", "chat", ["standard"])
        assert result.name == "local_vllm_chat_standard_2.yaml"


# =============================================================================
# Unit Tests: get_mock_overrides
# =============================================================================


class TestGetMockOverrides:
    def test_local_vllm_none_chat(self) -> None:
        overrides = get_mock_overrides("local", "vllm", "none", "chat", ["standard"])
        override_str = " ".join(overrides)
        assert "output_dir" in override_str
        assert "hf_model_handle" in override_str
        assert "served_model_name" in override_str
        # Should NOT contain slurm-specific overrides
        assert "hostname" not in override_str
        assert "account" not in override_str

    def test_slurm_adds_hostname_and_account(self) -> None:
        overrides = get_mock_overrides("slurm", "vllm", "none", "chat", ["standard"])
        override_str = " ".join(overrides)
        assert "hostname" in override_str
        assert "account" in override_str

    def test_deployment_none_adds_target(self) -> None:
        overrides = get_mock_overrides("local", "none", "none", "chat", ["standard"])
        override_str = " ".join(overrides)
        assert "target.api_endpoint.model_id" in override_str
        assert "target.api_endpoint.url" in override_str
        assert "target.api_endpoint.api_key_name" in override_str

    def test_deployment_nim_adds_image(self) -> None:
        overrides = get_mock_overrides("local", "nim", "none", "chat", ["standard"])
        override_str = " ".join(overrides)
        assert "deployment.image" in override_str

    def test_deployment_trtllm_adds_checkpoint(self) -> None:
        overrides = get_mock_overrides("local", "trtllm", "none", "chat", ["standard"])
        override_str = " ".join(overrides)
        assert "deployment.checkpoint_path" in override_str

    def test_deployment_sglang_adds_model_handle(self) -> None:
        overrides = get_mock_overrides("local", "sglang", "none", "chat", ["standard"])
        override_str = " ".join(overrides)
        assert "deployment.hf_model_handle" in override_str

    def test_export_mlflow(self) -> None:
        overrides = get_mock_overrides("local", "vllm", "mlflow", "chat", ["standard"])
        override_str = " ".join(overrides)
        assert "mlflow.tracking_uri" in override_str

    def test_export_wandb(self) -> None:
        overrides = get_mock_overrides("local", "vllm", "wandb", "chat", ["standard"])
        override_str = " ".join(overrides)
        assert "wandb.project" in override_str

    def test_model_type_base_adds_tokenizer(self) -> None:
        overrides = get_mock_overrides("local", "vllm", "none", "base", ["standard"])
        override_str = " ".join(overrides)
        assert "tokenizer" in override_str

    def test_model_type_reasoning_adds_params_to_add(self) -> None:
        overrides = get_mock_overrides(
            "local", "vllm", "none", "reasoning", ["standard"]
        )
        override_str = " ".join(overrides)
        assert "params_to_add" in override_str

    def test_safety_benchmark_adds_judge_url(self) -> None:
        # chat has safety benchmark
        overrides = get_mock_overrides("local", "vllm", "none", "chat", ["safety"])
        override_str = " ".join(overrides)
        assert "judge.url" in override_str

    def test_safety_benchmark_ignored_for_base(self) -> None:
        # base does NOT have safety benchmark file
        overrides = get_mock_overrides("local", "vllm", "none", "base", ["safety"])
        override_str = " ".join(overrides)
        assert "judge.url" not in override_str


# =============================================================================
# Integration Tests: build_config structure
# =============================================================================

# Each tuple: (execution, deployment, export, model_type, benchmarks, test_id)
# Designed so every possible value appears at least once.
BUILD_CONFIG_CASES = [
    # 1. local, vllm, none, chat, [standard]
    ("local", "vllm", "none", "chat", ["standard"], "local_vllm_none_chat_standard"),
    # 2. slurm, nim, mlflow, reasoning, [code]
    (
        "slurm",
        "nim",
        "mlflow",
        "reasoning",
        ["code"],
        "slurm_nim_mlflow_reasoning_code",
    ),
    # 3. local, sglang, wandb, base, [multilingual]
    (
        "local",
        "sglang",
        "wandb",
        "base",
        ["multilingual"],
        "local_sglang_wandb_base_multilingual",
    ),
    # 4. local, none, none, chat, [safety]
    ("local", "none", "none", "chat", ["safety"], "local_none_none_chat_safety"),
    # 5. slurm, trtllm, none, reasoning, [math_reasoning]
    (
        "slurm",
        "trtllm",
        "none",
        "reasoning",
        ["math_reasoning"],
        "slurm_trtllm_none_reasoning_math_reasoning",
    ),
    # 6. local, vllm, mlflow, chat, [standard, code, math_reasoning]
    (
        "local",
        "vllm",
        "mlflow",
        "chat",
        ["standard", "code", "math_reasoning"],
        "local_vllm_mlflow_chat_multi",
    ),
    # 7. local, vllm, none, reasoning, [standard, safety]
    (
        "local",
        "vllm",
        "none",
        "reasoning",
        ["standard", "safety"],
        "local_vllm_none_reasoning_std_safety",
    ),
    # 8. slurm, sglang, wandb, chat, [standard, multilingual, safety]
    (
        "slurm",
        "sglang",
        "wandb",
        "chat",
        ["standard", "multilingual", "safety"],
        "slurm_sglang_wandb_chat_multi",
    ),
    # 9. local, nim, none, base, [standard, code]
    (
        "local",
        "nim",
        "none",
        "base",
        ["standard", "code"],
        "local_nim_none_base_std_code",
    ),
    # 10. local, trtllm, mlflow, chat, [standard, code, math_reasoning, safety, multilingual]
    (
        "local",
        "trtllm",
        "mlflow",
        "chat",
        ["standard", "code", "math_reasoning", "safety", "multilingual"],
        "local_trtllm_mlflow_chat_all",
    ),
]


class TestBuildConfigStructure:
    """Test that build_config produces valid config dicts with expected structure."""

    @pytest.mark.parametrize(
        "execution,deployment,export,model_type,benchmarks",
        [c[:5] for c in BUILD_CONFIG_CASES],
        ids=[c[5] for c in BUILD_CONFIG_CASES],
    )
    def test_returns_dict_with_required_keys(
        self,
        execution: str,
        deployment: str,
        export: str,
        model_type: str,
        benchmarks: list[str],
    ) -> None:
        """build_config must return a dict with defaults, execution, and deployment."""
        config = build_config(
            execution=execution,
            deployment=deployment,
            export=export,
            model_type=model_type,
            benchmarks=benchmarks,
        )
        assert isinstance(config, dict)
        assert "defaults" in config
        assert "_self_" in config["defaults"], "defaults must end with _self_"
        assert config["defaults"][-1] == "_self_", "_self_ must be last"

    @pytest.mark.parametrize(
        "execution,deployment,export,model_type,benchmarks",
        [c[:5] for c in BUILD_CONFIG_CASES],
        ids=[c[5] for c in BUILD_CONFIG_CASES],
    )
    def test_has_evaluation_with_tasks(
        self,
        execution: str,
        deployment: str,
        export: str,
        model_type: str,
        benchmarks: list[str],
    ) -> None:
        """Config must have evaluation section with at least one task."""
        config = build_config(
            execution=execution,
            deployment=deployment,
            export=export,
            model_type=model_type,
            benchmarks=benchmarks,
        )
        assert "evaluation" in config
        assert "tasks" in config["evaluation"]
        assert len(config["evaluation"]["tasks"]) >= 1

    @pytest.mark.parametrize(
        "execution,deployment,export,model_type,benchmarks",
        [c[:5] for c in BUILD_CONFIG_CASES],
        ids=[c[5] for c in BUILD_CONFIG_CASES],
    )
    def test_writes_valid_yaml(
        self,
        execution: str,
        deployment: str,
        export: str,
        model_type: str,
        benchmarks: list[str],
        tmp_path: pathlib.Path,
    ) -> None:
        """Config written to disk must be parseable YAML matching the returned dict."""
        output = tmp_path / "config.yaml"
        config = build_config(
            execution=execution,
            deployment=deployment,
            export=export,
            model_type=model_type,
            benchmarks=benchmarks,
            output=output,
        )
        assert output.exists()
        with open(output) as f:
            loaded = yaml.safe_load(f)
        assert loaded == config


# =============================================================================
# Edge-case Tests: specific option behaviour
# =============================================================================


class TestBuildConfigEdgeCases:
    def test_export_none_has_no_export_section(self) -> None:
        """export='none' asset is a comment-only file, so no export key expected."""
        config = build_config("local", "vllm", "none", "chat", ["standard"])
        # With none export, there should be no mlflow/wandb config
        export = config.get("export")
        if export is not None:
            assert "mlflow" not in export
            assert "wandb" not in export

    def test_export_mlflow_has_tracking_uri(self) -> None:
        config = build_config("local", "vllm", "mlflow", "chat", ["standard"])
        assert "export" in config
        assert "mlflow" in config["export"]
        assert "tracking_uri" in config["export"]["mlflow"]

    def test_export_wandb_has_project(self) -> None:
        config = build_config("local", "vllm", "wandb", "chat", ["standard"])
        assert "export" in config
        assert "wandb" in config["export"]
        assert "project" in config["export"]["wandb"]

    def test_slurm_has_hostname_placeholder(self) -> None:
        config = build_config("slurm", "vllm", "none", "chat", ["standard"])
        assert config["execution"]["hostname"] == "???"

    def test_local_has_output_dir(self) -> None:
        config = build_config("local", "vllm", "none", "chat", ["standard"])
        assert "output_dir" in config["execution"]

    def test_deployment_none_has_target(self) -> None:
        config = build_config("local", "none", "none", "chat", ["standard"])
        assert "target" in config
        assert "api_endpoint" in config["target"]

    def test_deployment_vllm_has_deployment(self) -> None:
        config = build_config("local", "vllm", "none", "chat", ["standard"])
        assert "deployment" in config
        assert "hf_model_handle" in config["deployment"]

    def test_deployment_nim_has_image(self) -> None:
        config = build_config("local", "nim", "none", "chat", ["standard"])
        assert "deployment" in config
        assert "image" in config["deployment"]

    def test_deployment_trtllm_has_checkpoint_path(self) -> None:
        config = build_config("local", "trtllm", "none", "chat", ["standard"])
        assert "deployment" in config
        assert "checkpoint_path" in config["deployment"]

    def test_deployment_sglang_has_hf_model_handle(self) -> None:
        config = build_config("local", "sglang", "none", "chat", ["standard"])
        assert "deployment" in config
        assert "hf_model_handle" in config["deployment"]

    def test_base_model_has_tokenizer(self) -> None:
        config = build_config("local", "vllm", "none", "base", ["standard"])
        params = config["evaluation"]["nemo_evaluator_config"]["config"]["params"]
        assert "extra" in params
        assert "tokenizer" in params["extra"]

    def test_reasoning_model_has_adapter_config(self) -> None:
        config = build_config("local", "vllm", "none", "reasoning", ["standard"])
        target = config["evaluation"]["nemo_evaluator_config"]["target"]
        assert "api_endpoint" in target
        assert "adapter_config" in target["api_endpoint"]
        assert (
            target["api_endpoint"]["adapter_config"]["process_reasoning_traces"] is True
        )

    def test_missing_benchmark_file_does_not_raise(self) -> None:
        """base model has no safety benchmark; build_config should warn, not crash."""
        config = build_config("local", "vllm", "none", "base", ["safety"])
        # Should still return a valid config (just without safety tasks)
        assert isinstance(config, dict)

    def test_multiple_benchmarks_accumulate_tasks(self) -> None:
        single = build_config("local", "vllm", "none", "chat", ["standard"])
        double = build_config("local", "vllm", "none", "chat", ["standard", "code"])
        assert len(double["evaluation"]["tasks"]) > len(single["evaluation"]["tasks"])

    def test_no_duplicate_self_in_defaults(self) -> None:
        """_self_ should appear exactly once and be last."""
        config = build_config(
            "local",
            "vllm",
            "mlflow",
            "chat",
            ["standard", "code", "math_reasoning", "safety", "multilingual"],
        )
        self_count = config["defaults"].count("_self_")
        assert self_count == 1
        assert config["defaults"][-1] == "_self_"
