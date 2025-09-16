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
"""Tests for Lepton deployment helper functions."""

from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

from nemo_evaluator_launcher.executors.lepton.deployment_helpers import (
    deep_merge,
    generate_lepton_spec,
    replace_placeholders,
)


class TestDeepMerge:
    """Test the deep_merge utility function."""

    def test_deep_merge_empty_dicts(self):
        """Test merging empty dictionaries."""
        base = {}
        override = {}
        result = deep_merge(base, override)
        assert result == {}

    def test_deep_merge_simple_override(self):
        """Test simple key override."""
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = deep_merge(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_deep_merge_nested_dicts(self):
        """Test merging nested dictionaries."""
        base = {"level1": {"a": 1, "b": 2}, "other": "value"}
        override = {"level1": {"b": 3, "c": 4}, "new": "added"}
        result = deep_merge(base, override)

        expected = {
            "level1": {"a": 1, "b": 3, "c": 4},
            "other": "value",
            "new": "added",
        }
        assert result == expected

    def test_deep_merge_deeply_nested(self):
        """Test merging deeply nested structures."""
        base = {
            "level1": {
                "level2": {"a": 1, "b": 2},
                "other": "keep",
            }
        }
        override = {
            "level1": {
                "level2": {"b": 3, "c": 4},
                "new": "added",
            }
        }
        result = deep_merge(base, override)

        expected = {
            "level1": {
                "level2": {"a": 1, "b": 3, "c": 4},
                "other": "keep",
                "new": "added",
            }
        }
        assert result == expected

    def test_deep_merge_non_dict_override(self):
        """Test that non-dict values override completely."""
        base = {"config": {"nested": {"value": 1}}}
        override = {"config": "simple_string"}
        result = deep_merge(base, override)

        expected = {"config": "simple_string"}
        assert result == expected

    def test_deep_merge_preserves_original(self):
        """Test that original dictionaries are not modified."""
        base = {"a": {"b": 1}}
        override = {"a": {"c": 2}}
        original_base = {"a": {"b": 1}}
        original_override = {"a": {"c": 2}}

        result = deep_merge(base, override)

        # Original dicts should be unchanged
        assert base == original_base
        assert override == original_override
        # Result should have merged content
        assert result == {"a": {"b": 1, "c": 2}}

    def test_deep_merge_with_lists(self):
        """Test merging when values contain lists."""
        base = {"items": [1, 2], "config": {"nested": True}}
        override = {"items": [3, 4], "config": {"other": False}}
        result = deep_merge(base, override)

        expected = {
            "items": [3, 4],  # Lists are replaced, not merged
            "config": {"nested": True, "other": False},
        }
        assert result == expected


class TestReplacePlaceholders:
    """Test the replace_placeholders utility function."""

    def test_replace_placeholders_empty_data(self):
        """Test replacing placeholders in empty data."""
        data = {}
        replacements = {"KEY": "value"}
        result = replace_placeholders(data, replacements)
        assert result == {}

    def test_replace_placeholders_no_placeholders(self):
        """Test data with no placeholders."""
        data = {"config": {"path": "/some/path", "count": 42}}
        replacements = {"KEY": "value"}
        result = replace_placeholders(data, replacements)
        assert result == data

    def test_replace_placeholders_simple_string(self):
        """Test replacing placeholders in simple strings."""
        data = {"path": "/cache/{{MODEL_NAME}}/data"}
        replacements = {"MODEL_NAME": "llama-3-8b"}
        result = replace_placeholders(data, replacements)

        expected = {"path": "/cache/llama-3-8b/data"}
        assert result == expected

    def test_replace_placeholders_multiple_in_string(self):
        """Test multiple placeholders in single string."""
        data = {"command": "{{BINARY}} --model {{MODEL}} --port {{PORT}}"}
        replacements = {
            "BINARY": "vllm",
            "MODEL": "llama-3-8b",
            "PORT": "8000",
        }
        result = replace_placeholders(data, replacements)

        expected = {"command": "vllm --model llama-3-8b --port 8000"}
        assert result == expected

    def test_replace_placeholders_nested_dict(self):
        """Test replacing placeholders in nested dictionaries."""
        data = {
            "deployment": {
                "image": "{{IMAGE_NAME}}",
                "config": {
                    "model_path": "/models/{{MODEL_CACHE_NAME}}",
                    "port": 8000,
                },
            }
        }
        replacements = {
            "IMAGE_NAME": "nvcr.io/llama:latest",
            "MODEL_CACHE_NAME": "llama-3-8b-instruct",
        }
        result = replace_placeholders(data, replacements)

        expected = {
            "deployment": {
                "image": "nvcr.io/llama:latest",
                "config": {
                    "model_path": "/models/llama-3-8b-instruct",
                    "port": 8000,
                },
            }
        }
        assert result == expected

    def test_replace_placeholders_in_lists(self):
        """Test replacing placeholders in lists."""
        data = {
            "commands": [
                "{{BINARY}} serve {{MODEL}}",
                "echo {{MESSAGE}}",
                {"nested": "{{VALUE}}"},
            ]
        }
        replacements = {
            "BINARY": "vllm",
            "MODEL": "llama-3-8b",
            "MESSAGE": "starting",
            "VALUE": "nested_val",
        }
        result = replace_placeholders(data, replacements)

        expected = {
            "commands": [
                "vllm serve llama-3-8b",
                "echo starting",
                {"nested": "nested_val"},
            ]
        }
        assert result == expected

    def test_replace_placeholders_non_string_values(self):
        """Test that non-string values are preserved."""
        data = {
            "string": "{{PLACEHOLDER}}",
            "number": 42,
            "boolean": True,
            "null": None,
            "list": [1, 2, 3],
        }
        replacements = {"PLACEHOLDER": "replaced"}
        result = replace_placeholders(data, replacements)

        expected = {
            "string": "replaced",
            "number": 42,
            "boolean": True,
            "null": None,
            "list": [1, 2, 3],
        }
        assert result == expected

    def test_replace_placeholders_missing_replacement(self):
        """Test behavior when placeholder has no replacement."""
        data = {"path": "/cache/{{MISSING_KEY}}/data"}
        replacements = {"OTHER_KEY": "value"}
        result = replace_placeholders(data, replacements)

        # Missing placeholders should remain unchanged
        expected = {"path": "/cache/{{MISSING_KEY}}/data"}
        assert result == expected

    def test_replace_placeholders_preserves_original(self):
        """Test that original data is not modified."""
        original_data = {"path": "/cache/{{MODEL}}/data"}
        data = {"path": "/cache/{{MODEL}}/data"}
        replacements = {"MODEL": "llama"}

        result = replace_placeholders(data, replacements)

        # Original should be unchanged
        assert data == original_data
        # Result should be modified
        assert result == {"path": "/cache/llama/data"}


class TestGenerateLeptonSpec:
    """Test the generate_lepton_spec function."""

    def test_generate_lepton_spec_missing_lepton_config(self):
        """Test that missing lepton_config raises ValueError."""
        cfg = OmegaConf.create(
            {
                "deployment": {
                    "type": "vllm",
                    "image": "test:latest",
                }
            }
        )

        with pytest.raises(ValueError, match="deployment.lepton_config is required"):
            generate_lepton_spec(cfg)

    def test_generate_lepton_spec_minimal_nim_config(self):
        """Test generating spec for minimal NIM configuration."""
        cfg = OmegaConf.create(
            {
                "deployment": {
                    "type": "nim",
                    "image": "nvcr.io/nim/meta/llama-3.1-8b-instruct:1.8.6",
                    "port": 8000,
                    "served_model_name": "meta/llama-3.1-8b-instruct",
                    "lepton_config": {
                        "resource_shape": "gpu.1xh200",
                        "min_replicas": 1,
                        "max_replicas": 2,
                        "auto_scaler": {"type": "cpu"},
                    },
                }
            }
        )

        result = generate_lepton_spec(cfg)

        # Check basic structure
        assert "resource_requirement" in result
        assert "auto_scaler" in result
        assert "container" in result
        assert "envs" in result

        # Check resource requirements
        assert result["resource_requirement"]["resource_shape"] == "gpu.1xh200"
        assert result["resource_requirement"]["min_replicas"] == 1
        assert result["resource_requirement"]["max_replicas"] == 2

        # Check container spec
        assert (
            result["container"]["image"]
            == "nvcr.io/nim/meta/llama-3.1-8b-instruct:1.8.6"
        )
        assert result["container"]["ports"] == [{"container_port": 8000}]
        # NIM should not have custom command
        assert "command" not in result["container"]

        # Check auto-populated environment variables
        env_names = [env["name"] for env in result["envs"]]
        assert "SERVED_MODEL_NAME" in env_names
        assert "MODEL_PORT" in env_names
        assert "NIM_MODEL_NAME" in env_names

    def test_generate_lepton_spec_vllm_with_platform_defaults(self):
        """Test generating spec for vLLM with platform defaults."""
        cfg = OmegaConf.create(
            {
                "execution": {
                    "lepton_platform": {
                        "deployment": {
                            "platform_defaults": {
                                "queue_config": {"priority": "high"},
                                "image_pull_secrets": ["secret1", "secret2"],
                            },
                            "node_group": "gpu-nodes",
                        }
                    }
                },
                "deployment": {
                    "type": "vllm",
                    "image": "vllm/vllm-openai:latest",
                    "port": 8000,
                    "checkpoint_path": "meta-llama/Llama-3.1-8B-Instruct",
                    "served_model_name": "llama-3.1-8b-instruct",
                    "tensor_parallel_size": 2,
                    "pipeline_parallel_size": 1,
                    "data_parallel_size": 1,
                    "extra_args": "--max-model-len 8192",
                    "lepton_config": {
                        "resource_shape": "gpu.2xh200",
                        "min_replicas": 1,
                        "max_replicas": 4,
                        "auto_scaler": {"type": "gpu", "target_utilization": 0.8},
                    },
                },
            }
        )

        result = generate_lepton_spec(cfg)

        # Check platform defaults are applied
        assert result["queue_config"]["priority"] == "high"
        assert result["image_pull_secrets"] == ["secret1", "secret2"]

        # Check node group affinity
        assert "affinity" in result["resource_requirement"]
        assert result["resource_requirement"]["affinity"][
            "allowed_dedicated_node_groups"
        ] == ["gpu-nodes"]

        # Check vLLM command generation
        command = result["container"]["command"]
        assert command[0] == "vllm"
        assert command[1] == "serve"
        assert "meta-llama/Llama-3.1-8B-Instruct" in command
        assert "--tensor-parallel-size=2" in command
        assert "--max-model-len" in command
        assert "8192" in command

        # Check vLLM-specific environment variables
        env_dict = {env["name"]: env["value"] for env in result["envs"]}
        assert env_dict["MODEL_PATH"] == "meta-llama/Llama-3.1-8B-Instruct"
        assert env_dict["TENSOR_PARALLEL_SIZE"] == "2"

    def test_generate_lepton_spec_sglang_with_envs_and_tokens(self):
        """Test generating spec for SGLang with custom environment variables and API tokens."""
        cfg = OmegaConf.create(
            {
                "deployment": {
                    "type": "sglang",
                    "image": "lmsysorg/sglang:latest",
                    "port": 30000,
                    "checkpoint_path": "microsoft/DialoGPT-medium",
                    "served_model_name": "dialogpt-medium",
                    "tensor_parallel_size": 1,
                    "data_parallel_size": 1,
                    "lepton_config": {
                        "resource_shape": "gpu.1xa100",
                        "min_replicas": 1,
                        "max_replicas": 1,
                        "auto_scaler": {"type": "disabled"},
                        "envs": {
                            "CUSTOM_VAR": "custom_value",
                            "SECRET_VAR": {
                                "value_from": {"secret_name_ref": "my-secret"}
                            },
                        },
                        "api_tokens": [
                            "token123",
                            {"value": "token456"},
                            {"value_from": {"secret_name_ref": "token-secret"}},
                        ],
                    },
                }
            }
        )

        result = generate_lepton_spec(cfg)

        # Check SGLang command generation
        command = result["container"]["command"]
        assert command[0] == "python3"
        assert command[1] == "-m"
        assert command[2] == "sglang.launch_server"
        assert "--model-path=microsoft/DialoGPT-medium" in command
        assert "--port=30000" in command
        assert "--tp=1" in command

        # Check custom environment variables
        env_vars = result["envs"]
        custom_env = next(env for env in env_vars if env["name"] == "CUSTOM_VAR")
        assert custom_env["value"] == "custom_value"

        secret_env = next(env for env in env_vars if env["name"] == "SECRET_VAR")
        assert secret_env["value_from"]["secret_name_ref"] == "my-secret"

        # Check API tokens
        api_tokens = result["api_tokens"]
        assert len(api_tokens) == 3
        assert api_tokens[0]["value"] == "token123"
        assert api_tokens[1]["value"] == "token456"
        assert api_tokens[2]["value_from"]["secret_name_ref"] == "token-secret"

    def test_generate_lepton_spec_with_mounts(self):
        """Test generating spec with mount configurations."""
        cfg = OmegaConf.create(
            {
                "execution": {
                    "lepton_platform": {
                        "tasks": {"mounts": [{"from": "custom-nfs:shared-storage"}]}
                    }
                },
                "deployment": {
                    "type": "nim",
                    "image": "nim:latest",
                    "port": 8000,
                    "lepton_config": {
                        "resource_shape": "cpu.small",
                        "min_replicas": 1,
                        "max_replicas": 1,
                        "auto_scaler": {"type": "disabled"},
                        "mounts": {
                            "enabled": True,
                            "cache_path": "/cache/models",
                            "mount_path": "/mnt/shared",
                        },
                    },
                },
            }
        )

        result = generate_lepton_spec(cfg)

        # Check mounts configuration
        assert "mounts" in result
        mounts = result["mounts"]
        assert len(mounts) == 1
        mount = mounts[0]
        assert mount["path"] == "/cache/models"
        assert mount["from"] == "custom-nfs:shared-storage"
        assert mount["mount_path"] == "/mnt/shared"
        assert mount["mount_options"] == {}

    def test_generate_lepton_spec_with_health_check(self):
        """Test generating spec with health check configuration."""
        cfg = OmegaConf.create(
            {
                "deployment": {
                    "type": "nim",
                    "image": "nim:latest",
                    "port": 8000,
                    "lepton_config": {
                        "resource_shape": "cpu.small",
                        "min_replicas": 1,
                        "max_replicas": 1,
                        "auto_scaler": {"type": "disabled"},
                        "health": {
                            "readiness_probe": {
                                "http_get": {"path": "/health", "port": 8000},
                                "initial_delay_seconds": 30,
                            }
                        },
                    },
                }
            }
        )

        result = generate_lepton_spec(cfg)

        # Check health configuration
        assert "health" in result
        health = result["health"]
        assert health["readiness_probe"]["http_get"]["path"] == "/health"
        assert health["readiness_probe"]["initial_delay_seconds"] == 30

    def test_generate_lepton_spec_legacy_api_token(self):
        """Test backward compatibility with legacy single api_token."""
        cfg = OmegaConf.create(
            {
                "deployment": {
                    "type": "nim",
                    "image": "nim:latest",
                    "port": 8000,
                    "lepton_config": {
                        "resource_shape": "cpu.small",
                        "min_replicas": 1,
                        "max_replicas": 1,
                        "auto_scaler": {"type": "disabled"},
                        "api_token": "legacy_token_123",
                    },
                }
            }
        )

        result = generate_lepton_spec(cfg)

        # Check legacy api_token is converted to api_tokens list
        assert "api_tokens" in result
        api_tokens = result["api_tokens"]
        assert len(api_tokens) == 1
        assert api_tokens[0]["value"] == "legacy_token_123"

    @patch(
        "nemo_evaluator_launcher.executors.lepton.deployment_helpers._generate_model_cache_name"
    )
    def test_generate_lepton_spec_placeholder_replacement(
        self, mock_generate_cache_name
    ):
        """Test that placeholders are replaced correctly."""
        mock_generate_cache_name.return_value = "llama-3-1-8b-instruct"

        cfg = OmegaConf.create(
            {
                "deployment": {
                    "type": "nim",
                    "image": "nvcr.io/nim/meta/llama-3.1-8b-instruct:1.8.6",
                    "port": 8000,
                    "lepton_config": {
                        "resource_shape": "gpu.1xh200",
                        "min_replicas": 1,
                        "max_replicas": 1,
                        "auto_scaler": {"type": "disabled"},
                        "envs": {
                            "CACHE_DIR": "/cache/{{MODEL_CACHE_NAME}}",
                        },
                    },
                }
            }
        )

        result = generate_lepton_spec(cfg)

        # Check that placeholder was replaced
        cache_env = next(env for env in result["envs"] if env["name"] == "CACHE_DIR")
        assert cache_env["value"] == "/cache/llama-3-1-8b-instruct"

        # Verify the mock was called with correct image
        mock_generate_cache_name.assert_called_once_with(
            "nvcr.io/nim/meta/llama-3.1-8b-instruct:1.8.6"
        )

    def test_generate_lepton_spec_invalid_image_pull_secrets(self):
        """Test handling of invalid image_pull_secrets."""
        cfg = OmegaConf.create(
            {
                "execution": {
                    "lepton_platform": {
                        "deployment": {
                            "platform_defaults": {
                                "image_pull_secrets": "invalid_string_instead_of_list"
                            }
                        }
                    }
                },
                "deployment": {
                    "type": "nim",
                    "image": "nim:latest",
                    "port": 8000,
                    "lepton_config": {
                        "resource_shape": "cpu.small",
                        "min_replicas": 1,
                        "max_replicas": 1,
                        "auto_scaler": {"type": "disabled"},
                    },
                },
            }
        )

        result = generate_lepton_spec(cfg)

        # Invalid image_pull_secrets should be removed
        assert "image_pull_secrets" not in result
