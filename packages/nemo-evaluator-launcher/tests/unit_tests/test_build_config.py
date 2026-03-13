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
    "COMPLIANCE_JUDGE_SERVICE_API_KEY": "mock-compliance-judge-api-key",
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
    elif model_type == "chat_reasoning":
        overrides.append(
            "+evaluation.nemo_evaluator_config.target.api_endpoint.adapter_config.params_to_add.chat_template_kwargs.enable_thinking=true"
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
        result = resolve_output_path(
            out, "local", "vllm", "chat_reasoning", ["core_reasoning"]
        )
        assert result == out

    def test_directory_auto_generates_name(self, tmp_path: pathlib.Path) -> None:
        result = resolve_output_path(tmp_path, "slurm", "nim", "base", ["coding"])
        assert result.parent == tmp_path
        assert result.name == "slurm_nim_base_coding.yaml"

    def test_benchmarks_sorted_in_filename(self, tmp_path: pathlib.Path) -> None:
        result = resolve_output_path(
            tmp_path,
            "local",
            "vllm",
            "chat_reasoning",
            ["multilingual", "core_reasoning"],
        )
        assert (
            result.name == "local_vllm_chat_reasoning_core_reasoning_multilingual.yaml"
        )

    def test_none_uses_cwd(self, tmp_path: pathlib.Path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        result = resolve_output_path(
            None, "local", "vllm", "chat_reasoning", ["core_reasoning"]
        )
        assert result.parent == tmp_path
        assert result.name == "local_vllm_chat_reasoning_core_reasoning.yaml"

    def test_creates_parent_dir(self, tmp_path: pathlib.Path) -> None:
        new_dir = tmp_path / "sub" / "dir"
        result = resolve_output_path(
            new_dir, "local", "vllm", "chat_reasoning", ["core_reasoning"]
        )
        assert new_dir.is_dir()
        assert result.parent == new_dir

    def test_existing_file_gets_suffix(self, tmp_path: pathlib.Path) -> None:
        (tmp_path / "local_vllm_chat_reasoning_core_reasoning.yaml").touch()
        result = resolve_output_path(
            tmp_path, "local", "vllm", "chat_reasoning", ["core_reasoning"]
        )
        assert result.name == "local_vllm_chat_reasoning_core_reasoning_1.yaml"

    def test_multiple_existing_increments(self, tmp_path: pathlib.Path) -> None:
        for name in (
            "local_vllm_chat_reasoning_core_reasoning.yaml",
            "local_vllm_chat_reasoning_core_reasoning_1.yaml",
            "local_vllm_chat_reasoning_core_reasoning_3.yaml",
        ):
            (tmp_path / name).touch()
        result = resolve_output_path(
            tmp_path, "local", "vllm", "chat_reasoning", ["core_reasoning"]
        )
        assert result.name == "local_vllm_chat_reasoning_core_reasoning_2.yaml"


# =============================================================================
# Unit Tests: get_mock_overrides
# =============================================================================


class TestGetMockOverrides:
    def test_local_vllm_none_chat_reasoning(self) -> None:
        overrides = get_mock_overrides(
            "local", "vllm", "none", "chat_reasoning", ["core_reasoning"]
        )
        override_str = " ".join(overrides)
        assert "output_dir" in override_str
        assert "hf_model_handle" in override_str
        assert "served_model_name" in override_str
        # Should NOT contain slurm-specific overrides
        assert "hostname" not in override_str
        assert "account" not in override_str

    def test_slurm_adds_hostname_and_account(self) -> None:
        overrides = get_mock_overrides(
            "slurm", "vllm", "none", "chat_reasoning", ["core_reasoning"]
        )
        override_str = " ".join(overrides)
        assert "hostname" in override_str
        assert "account" in override_str

    def test_deployment_none_adds_target(self) -> None:
        overrides = get_mock_overrides(
            "local", "none", "none", "chat_reasoning", ["core_reasoning"]
        )
        override_str = " ".join(overrides)
        assert "target.api_endpoint.model_id" in override_str
        assert "target.api_endpoint.url" in override_str
        assert "target.api_endpoint.api_key_name" in override_str

    def test_deployment_nim_adds_image(self) -> None:
        overrides = get_mock_overrides(
            "local", "nim", "none", "chat_reasoning", ["core_reasoning"]
        )
        override_str = " ".join(overrides)
        assert "deployment.image" in override_str

    def test_deployment_trtllm_adds_checkpoint(self) -> None:
        overrides = get_mock_overrides(
            "local", "trtllm", "none", "chat_reasoning", ["core_reasoning"]
        )
        override_str = " ".join(overrides)
        assert "deployment.checkpoint_path" in override_str

    def test_deployment_sglang_adds_model_handle(self) -> None:
        overrides = get_mock_overrides(
            "local", "sglang", "none", "chat_reasoning", ["core_reasoning"]
        )
        override_str = " ".join(overrides)
        assert "deployment.hf_model_handle" in override_str

    def test_export_mlflow(self) -> None:
        overrides = get_mock_overrides(
            "local", "vllm", "mlflow", "chat_reasoning", ["core_reasoning"]
        )
        override_str = " ".join(overrides)
        assert "mlflow.tracking_uri" in override_str

    def test_export_wandb(self) -> None:
        overrides = get_mock_overrides(
            "local", "vllm", "wandb", "chat_reasoning", ["core_reasoning"]
        )
        override_str = " ".join(overrides)
        assert "wandb.project" in override_str

    def test_model_type_base_adds_tokenizer(self) -> None:
        overrides = get_mock_overrides(
            "local", "vllm", "none", "base", ["general_knowledge"]
        )
        override_str = " ".join(overrides)
        assert "tokenizer" in override_str

    def test_model_type_chat_reasoning_adds_params_to_add(self) -> None:
        overrides = get_mock_overrides(
            "local", "vllm", "none", "chat_reasoning", ["core_reasoning"]
        )
        override_str = " ".join(overrides)
        assert "params_to_add" in override_str


# =============================================================================
# Integration Tests: build_config structure
# =============================================================================

# Each tuple: (execution, deployment, export, model_type, benchmarks, test_id)
# Designed so every possible value appears at least once.
BUILD_CONFIG_CASES = [
    # 1. local, vllm, none, chat_reasoning, [core_reasoning]
    (
        "local",
        "vllm",
        "none",
        "chat_reasoning",
        ["core_reasoning"],
        "local_vllm_none_chat_reasoning_core_reasoning",
    ),
    # 2. slurm, nim, mlflow, chat_reasoning, [agentic]
    (
        "slurm",
        "nim",
        "mlflow",
        "chat_reasoning",
        ["agentic"],
        "slurm_nim_mlflow_chat_reasoning_agentic",
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
    # 4. local, none, none, chat_reasoning, [agentic]
    (
        "local",
        "none",
        "none",
        "chat_reasoning",
        ["agentic"],
        "local_none_none_chat_reasoning_agentic",
    ),
    # 5. slurm, trtllm, none, chat_reasoning, [long_context]
    (
        "slurm",
        "trtllm",
        "none",
        "chat_reasoning",
        ["long_context"],
        "slurm_trtllm_none_chat_reasoning_long_context",
    ),
    # 6. local, vllm, mlflow, chat_reasoning, [core_reasoning, multilingual]
    (
        "local",
        "vllm",
        "mlflow",
        "chat_reasoning",
        ["core_reasoning", "multilingual"],
        "local_vllm_mlflow_chat_reasoning_multi",
    ),
    # 7. local, vllm, none, chat_reasoning, [core_reasoning, long_context]
    (
        "local",
        "vllm",
        "none",
        "chat_reasoning",
        ["core_reasoning", "long_context"],
        "local_vllm_none_chat_reasoning_core_reasoning_long_context",
    ),
    # 8. slurm, sglang, wandb, chat_reasoning, [agentic, multilingual]
    (
        "slurm",
        "sglang",
        "wandb",
        "chat_reasoning",
        ["agentic", "multilingual"],
        "slurm_sglang_wandb_chat_reasoning_multi",
    ),
    # 9. local, nim, none, base, [general_knowledge, coding]
    (
        "local",
        "nim",
        "none",
        "base",
        ["general_knowledge", "coding"],
        "local_nim_none_base_general_knowledge_coding",
    ),
    # 10. local, trtllm, mlflow, chat_reasoning, all merged Nemotron groups
    (
        "local",
        "trtllm",
        "mlflow",
        "chat_reasoning",
        [
            "core_reasoning",
            "agentic",
            "long_context",
            "multilingual",
        ],
        "local_trtllm_mlflow_chat_reasoning_all",
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
        config = build_config(
            "local", "vllm", "none", "chat_reasoning", ["core_reasoning"]
        )
        # With none export, there should be no mlflow/wandb config
        export = config.get("export")
        if export is not None:
            assert "mlflow" not in export
            assert "wandb" not in export

    def test_export_mlflow_has_tracking_uri(self) -> None:
        config = build_config(
            "local", "vllm", "mlflow", "chat_reasoning", ["core_reasoning"]
        )
        assert "export" in config
        assert "mlflow" in config["export"]
        assert "tracking_uri" in config["export"]["mlflow"]

    def test_export_wandb_has_project(self) -> None:
        config = build_config(
            "local", "vllm", "wandb", "chat_reasoning", ["core_reasoning"]
        )
        assert "export" in config
        assert "wandb" in config["export"]
        assert "project" in config["export"]["wandb"]

    def test_slurm_has_hostname_placeholder(self) -> None:
        config = build_config(
            "slurm", "vllm", "none", "chat_reasoning", ["core_reasoning"]
        )
        assert config["execution"]["hostname"] == "???"

    def test_local_has_output_dir(self) -> None:
        config = build_config(
            "local", "vllm", "none", "chat_reasoning", ["core_reasoning"]
        )
        assert "output_dir" in config["execution"]

    def test_deployment_none_has_target(self) -> None:
        config = build_config(
            "local", "none", "none", "chat_reasoning", ["core_reasoning"]
        )
        assert "target" in config
        assert "api_endpoint" in config["target"]

    @pytest.mark.parametrize(
        "model_type,benchmarks",
        [
            ("base", ["general_knowledge"]),
            ("chat_reasoning", ["core_reasoning"]),
        ],
    )
    def test_deployment_none_overrides_parallelism_to_one(
        self, model_type: str, benchmarks: list[str]
    ) -> None:
        config = build_config("local", "none", "none", model_type, benchmarks)
        params = config["evaluation"]["nemo_evaluator_config"]["config"]["params"]
        assert params["parallelism"] == 1

    def test_deployment_vllm_has_deployment(self) -> None:
        config = build_config(
            "local", "vllm", "none", "chat_reasoning", ["core_reasoning"]
        )
        assert "deployment" in config
        assert "hf_model_handle" in config["deployment"]

    def test_deployment_nim_has_image(self) -> None:
        config = build_config(
            "local", "nim", "none", "chat_reasoning", ["core_reasoning"]
        )
        assert "deployment" in config
        assert "image" in config["deployment"]

    def test_deployment_trtllm_has_checkpoint_path(self) -> None:
        config = build_config(
            "local", "trtllm", "none", "chat_reasoning", ["core_reasoning"]
        )
        assert "deployment" in config
        assert "checkpoint_path" in config["deployment"]

    def test_deployment_sglang_has_hf_model_handle(self) -> None:
        config = build_config(
            "local", "sglang", "none", "chat_reasoning", ["core_reasoning"]
        )
        assert "deployment" in config
        assert "hf_model_handle" in config["deployment"]

    def test_chat_reasoning_templates_include_hf_env_vars(self) -> None:
        config = build_config(
            "local", "vllm", "none", "chat_reasoning", ["core_reasoning"]
        )
        env_vars = config["evaluation"]["env_vars"]
        assert env_vars["HF_TOKEN"] == "host:HF_TOKEN"

    def test_base_model_uses_nemotron_super_base_defaults(self) -> None:
        config = build_config("local", "vllm", "none", "base", ["general_knowledge"])
        params = config["evaluation"]["nemo_evaluator_config"]["config"]["params"]
        assert params["max_retries"] == 5
        assert params["request_timeout"] == 3600
        assert params["parallelism"] == 16
        assert "extra" in params
        assert "tokenizer" in params["extra"]
        assert params["extra"]["tokenizer_backend"] == "huggingface"

    def test_base_general_knowledge_keeps_curated_subset(self) -> None:
        config = build_config("local", "vllm", "none", "base", ["general_knowledge"])
        task_names = [task["name"] for task in config["evaluation"]["tasks"]]
        assert len(task_names) == 5
        assert "adlr_gpqa_diamond_cot_5_shot" in task_names

    def test_base_long_context_uses_completions_tasks(self) -> None:
        config = build_config("local", "vllm", "none", "base", ["long_context"])
        task_names = [task["name"] for task in config["evaluation"]["tasks"]]
        assert task_names == [
            "ruler-128k-completions",
            "ruler-256k-completions",
            "ruler-512k-completions",
        ]

    def test_chat_reasoning_long_context_uses_chat_tasks(self) -> None:
        config = build_config(
            "local", "vllm", "none", "chat_reasoning", ["long_context"]
        )
        task_names = [task["name"] for task in config["evaluation"]["tasks"]]
        assert task_names == [
            "ruler-128k-chat",
            "ruler-256k-chat",
            "ruler-512k-chat",
        ]

    def test_chat_reasoning_model_uses_nemotron_super_defaults(self) -> None:
        config = build_config(
            "local", "vllm", "none", "chat_reasoning", ["core_reasoning"]
        )
        params = config["evaluation"]["nemo_evaluator_config"]["config"]["params"]
        assert params["parallelism"] == 16
        assert params["temperature"] == 1.0
        assert params["top_p"] == 0.95
        assert params["max_new_tokens"] == 131072
        assert params["max_retries"] == 10
        assert params["extra"]["tokenizer_backend"] == "huggingface"
        assert "tokenizer" in params["extra"]
        target = config["evaluation"]["nemo_evaluator_config"]["target"]
        assert "api_endpoint" in target
        adapter_config = target["api_endpoint"]["adapter_config"]
        assert adapter_config["process_reasoning_traces"] is True
        assert adapter_config["params_to_add"] == "???"
        env_vars = config["evaluation"]["env_vars"]
        assert env_vars["HF_TOKEN"] == "host:HF_TOKEN"

    def test_agentic_tau2_uses_qwen_user_key(self) -> None:
        config = build_config("local", "vllm", "none", "chat_reasoning", ["agentic"])
        tau2_task = config["evaluation"]["tasks"][1]
        assert tau2_task["name"] == "tau2_bench_telecom"
        assert tau2_task["env_vars"]["USER_API_KEY"] == "host:NVIDIA_API_KEY"
        params = tau2_task["nemo_evaluator_config"]["config"]["params"]
        assert params["parallelism"] == 1
        assert params["extra"]["skip_failed_samples"] is True
        user_config = tau2_task["nemo_evaluator_config"]["config"]["params"]["extra"][
            "user"
        ]
        assert user_config["model_id"] == "qwen/qwen3.5-397b-a17b"
        assert (
            user_config["url"] == "https://integrate.api.nvidia.com/v1/chat/completions"
        )
        assert user_config["api_key"] == "USER_API_KEY"

    @pytest.mark.parametrize(
        "model_type,benchmarks",
        [
            ("base", ["core_reasoning"]),
            ("base", ["agentic"]),
            ("chat_reasoning", ["general_knowledge"]),
            ("chat_reasoning", ["coding"]),
        ],
    )
    def test_invalid_model_type_benchmark_combinations_raise(
        self, model_type: str, benchmarks: list[str]
    ) -> None:
        with pytest.raises(ValueError, match="not supported for model_type"):
            build_config("local", "vllm", "none", model_type, benchmarks)

    def test_multiple_benchmarks_accumulate_tasks(self) -> None:
        single = build_config(
            "local", "vllm", "none", "chat_reasoning", ["core_reasoning"]
        )
        double = build_config(
            "local",
            "vllm",
            "none",
            "chat_reasoning",
            ["core_reasoning", "multilingual"],
        )
        assert len(double["evaluation"]["tasks"]) > len(single["evaluation"]["tasks"])

    def test_no_duplicate_self_in_defaults(self) -> None:
        """_self_ should appear exactly once and be last."""
        config = build_config(
            "local",
            "vllm",
            "mlflow",
            "chat_reasoning",
            [
                "core_reasoning",
                "agentic",
                "long_context",
                "multilingual",
            ],
        )
        self_count = config["defaults"].count("_self_")
        assert self_count == 1
        assert config["defaults"][-1] == "_self_"
