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

"""Tests for AdapterConfig class."""

from nvidia_eval_commons.adapters.adapter_config import AdapterConfig, InterceptorConfig


def test_get_validated_config_with_valid_config():
    """Test get_validated_config with valid configuration."""
    run_config = {
        "target": {
            "api_endpoint": {
                "adapter_config": {
                    "interceptors": [
                        {
                            "name": "request_logging",
                            "enabled": True,
                            "config": {"output_dir": "/tmp/logs"},
                        }
                    ],
                    "endpoint_type": "chat",
                    "caching_dir": "/tmp/cache",
                    "generate_html_report": True,
                }
            }
        }
    }

    adapter_config = AdapterConfig.get_validated_config(run_config)

    assert adapter_config is not None
    assert len(adapter_config.interceptors) == 1  # request_logging
    assert adapter_config.interceptors[0].name == "request_logging"
    assert adapter_config.interceptors[0].enabled is True
    assert adapter_config.interceptors[0].config == {"output_dir": "/tmp/logs"}
    assert adapter_config.endpoint_type == "chat"
    assert adapter_config.caching_dir == "/tmp/cache"
    assert adapter_config.generate_html_report is True


def test_get_validated_config_with_discovery_config():
    """Test get_validated_config with discovery configuration."""
    run_config = {
        "target": {
            "api_endpoint": {
                "adapter_config": {
                    "discovery": {
                        "modules": ["my_module.interceptor", "another_module.plugin"],
                        "dirs": ["/path/to/plugins", "/another/path"],
                    },
                    "interceptors": [
                        {
                            "name": "request_logging",
                            "enabled": True,
                            "config": {"output_dir": "/tmp/logs"},
                        }
                    ],
                }
            }
        }
    }

    adapter_config = AdapterConfig.get_validated_config(run_config)

    assert adapter_config is not None
    assert adapter_config.discovery.modules == [
        "my_module.interceptor",
        "another_module.plugin",
    ]
    assert adapter_config.discovery.dirs == ["/path/to/plugins", "/another/path"]
    assert len(adapter_config.interceptors) == 1  # request_logging


def test_get_validated_config_with_global_discovery_config():
    """Test get_validated_config with global discovery configuration."""
    run_config = {
        "global_adapter_config": {
            "discovery": {
                "modules": ["global_module.interceptor"],
                "dirs": ["/global/path"],
            }
        },
        "target": {
            "api_endpoint": {
                "adapter_config": {
                    "interceptors": [
                        {
                            "name": "request_logging",
                            "enabled": True,
                            "config": {"output_dir": "/tmp/logs"},
                        }
                    ],
                }
            }
        },
    }

    adapter_config = AdapterConfig.get_validated_config(run_config)

    assert adapter_config is not None
    assert adapter_config.discovery.modules == ["global_module.interceptor"]
    assert adapter_config.discovery.dirs == ["/global/path"]
    assert len(adapter_config.interceptors) == 1  # request_logging


def test_get_validated_config_with_merged_discovery_configs():
    """Test get_validated_config with both global and local discovery configs."""
    run_config = {
        "global_adapter_config": {
            "discovery": {
                "modules": ["global_module.interceptor"],
                "dirs": ["/global/path"],
            }
        },
        "target": {
            "api_endpoint": {
                "adapter_config": {
                    "discovery": {
                        "modules": ["local_module.interceptor"],
                        "dirs": ["/local/path"],
                    },
                    "interceptors": [
                        {
                            "name": "request_logging",
                            "enabled": True,
                            "config": {"output_dir": "/tmp/logs"},
                        }
                    ],
                }
            }
        },
    }

    adapter_config = AdapterConfig.get_validated_config(run_config)

    assert adapter_config is not None
    # Global and local modules should be merged
    assert adapter_config.discovery.modules == [
        "global_module.interceptor",
        "local_module.interceptor",
    ]
    # Global and local dirs should be merged
    assert adapter_config.discovery.dirs == ["/global/path", "/local/path"]
    assert len(adapter_config.interceptors) == 1  # request_logging


def test_get_validated_config_with_global_only():
    """Test get_validated_config with only global adapter config."""
    run_config = {
        "global_adapter_config": {
            "discovery": {
                "modules": ["global_module.interceptor"],
                "dirs": ["/global/path"],
            },
            "interceptors": [
                {
                    "name": "global_interceptor",
                    "enabled": True,
                    "config": {"global": True},
                }
            ],
        }
    }

    adapter_config = AdapterConfig.get_validated_config(run_config)

    assert adapter_config is not None
    assert adapter_config.discovery.modules == ["global_module.interceptor"]
    assert adapter_config.discovery.dirs == ["/global/path"]
    assert len(adapter_config.interceptors) == 1
    assert adapter_config.interceptors[0].name == "global_interceptor"


def test_get_validated_config_without_adapter_config():
    """Test get_validated_config when adapter_config is not present."""
    run_config = {"target": {"api_endpoint": {"url": "https://api.example.com"}}}

    adapter_config = AdapterConfig.get_validated_config(run_config)

    assert adapter_config is None


def test_get_validated_config_without_api_endpoint():
    """Test get_validated_config when api_endpoint is not present."""
    run_config = {"target": {}}

    adapter_config = AdapterConfig.get_validated_config(run_config)

    assert adapter_config is None


def test_get_validated_config_without_target():
    """Test get_validated_config when target is not present."""
    run_config = {}

    adapter_config = AdapterConfig.get_validated_config(run_config)

    assert adapter_config is None


def test_discovery_config_defaults():
    """Test DiscoveryConfig default values."""
    adapter_config = AdapterConfig()

    assert adapter_config.discovery.modules == []
    assert adapter_config.discovery.dirs == []


def test_discovery_config_with_values():
    """Test DiscoveryConfig with specific values."""
    adapter_config = AdapterConfig(
        discovery={
            "modules": ["test.module"],
            "dirs": ["/test/path"],
        }
    )

    assert adapter_config.discovery.modules == ["test.module"]
    assert adapter_config.discovery.dirs == ["/test/path"]


def test_from_legacy_config():
    """Test from_legacy_config method."""
    legacy_config = {
        "use_request_logging": True,
        "use_response_logging": True,
        "use_caching": True,
        "caching_dir": "/tmp/cache",
        "reuse_cached_responses": True,
        "save_responses": True,
        "use_reasoning": True,
        "end_reasoning_token": "</think>",
        "output_dir": "/tmp/output",
        "generate_html_report": True,
        "include_json": True,
        "endpoint_type": "chat",
        "start_reasoning_token": "<think>",
        "include_if_reasoning_not_finished": True,
        "track_reasoning": True,
    }

    config = AdapterConfig.from_legacy_config(legacy_config)

    assert (
        len(config.interceptors) == 6
    )  # request_logging, caching, endpoint, response_logging, reasoning, progress_tracking
    assert len(config.post_eval_hooks) == 2  # post_eval_report, progress_tracking

    # Check specific interceptors
    interceptor_names = [ic.name for ic in config.interceptors]
    assert "request_logging" in interceptor_names
    assert "caching" in interceptor_names
    assert "endpoint" in interceptor_names
    assert "response_logging" in interceptor_names
    assert "reasoning" in interceptor_names
    assert "progress_tracking" in interceptor_names
    assert config.caching_dir == "/tmp/cache"
    assert config.generate_html_report is True

    # Check post-eval hooks
    hook_names = [hook.name for hook in config.post_eval_hooks]
    assert "post_eval_report" in hook_names
    assert "progress_tracking" in hook_names


def test_from_legacy_config_with_nvcf():
    """Test conversion from legacy configuration format with use_nvcf."""
    legacy_config = {
        "use_nvcf": True,
        "use_request_logging": True,
        "use_response_logging": True,
    }

    config = AdapterConfig.from_legacy_config(legacy_config)

    assert (
        len(config.interceptors) == 3
    )  # request_logging, nvcf, response_logging (no endpoint)

    # Check specific interceptors
    interceptor_names = [ic.name for ic in config.interceptors]
    assert "request_logging" in interceptor_names
    assert "nvcf" in interceptor_names
    assert "endpoint" not in interceptor_names  # endpoint should not be present
    assert "response_logging" in interceptor_names

    # Check that nvcf interceptor is enabled
    nvcf_interceptor = next(ic for ic in config.interceptors if ic.name == "nvcf")
    assert nvcf_interceptor.enabled is True


def test_get_validated_config_with_legacy_nvcf():
    """Test that get_validated_config properly handles legacy use_nvcf configuration."""
    run_config = {
        "target": {
            "api_endpoint": {
                "adapter_config": {
                    "use_nvcf": True,
                    "use_request_logging": True,
                }
            }
        }
    }

    config = AdapterConfig.get_validated_config(run_config)
    assert config is not None

    # Should have interceptors due to legacy conversion
    assert len(config.interceptors) == 2  # request_logging, nvcf (no endpoint)

    # Check specific interceptors
    interceptor_names = [ic.name for ic in config.interceptors]
    assert "request_logging" in interceptor_names
    assert "nvcf" in interceptor_names
    assert "endpoint" not in interceptor_names  # endpoint should not be present


def test_get_interceptor_configs():
    """Test get_interceptor_configs method."""
    adapter_config = AdapterConfig(
        interceptors=[
            InterceptorConfig(
                name="test_interceptor",
                enabled=True,
                config={"key": "value"},
            ),
            InterceptorConfig(
                name="disabled_interceptor",
                enabled=False,
                config={"disabled": True},
            ),
        ]
    )

    configs = adapter_config.get_interceptor_configs()

    assert "test_interceptor" in configs
    assert configs["test_interceptor"] == {"key": "value"}
    assert "disabled_interceptor" not in configs


def test_legacy_params_to_add_creates_payload_modifier():
    """Test that legacy params_to_add configuration creates payload_modifier interceptor."""
    legacy_config = {
        "params_to_add": {"chat_template_kwargs": {"enable_thinking": False}}
    }

    adapter_config = AdapterConfig.from_legacy_config(legacy_config)

    # Verify payload_modifier interceptor was created
    payload_modifier_interceptors = [
        interceptor
        for interceptor in adapter_config.interceptors
        if interceptor.name == "payload_modifier"
    ]

    assert len(payload_modifier_interceptors) == 1
    payload_modifier = payload_modifier_interceptors[0]

    # Verify configuration
    assert payload_modifier.enabled is True
    assert payload_modifier.config["params_to_add"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }


def test_legacy_params_to_add_with_other_interceptors():
    """Test that params_to_add works with other legacy interceptors."""
    legacy_config = {
        "use_request_logging": True,
        "use_caching": True,
        "params_to_add": {"chat_template_kwargs": {"enable_thinking": False}},
    }

    adapter_config = AdapterConfig.from_legacy_config(legacy_config)

    # Verify all interceptors are present
    interceptor_names = [
        interceptor.name for interceptor in adapter_config.interceptors
    ]

    assert "request_logging" in interceptor_names
    assert "caching" in interceptor_names
    assert "payload_modifier" in interceptor_names
    assert "endpoint" in interceptor_names

    # Verify payload_modifier configuration
    payload_modifier = next(
        interceptor
        for interceptor in adapter_config.interceptors
        if interceptor.name == "payload_modifier"
    )

    assert payload_modifier.config["params_to_add"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }


def test_legacy_params_to_add_complex_config():
    """Test params_to_add with complex nested configuration."""
    legacy_config = {
        "params_to_add": {
            "chat_template_kwargs": {"enable_thinking": False},
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            "custom_param": "custom_value",
        }
    }

    adapter_config = AdapterConfig.from_legacy_config(legacy_config)

    payload_modifier = next(
        interceptor
        for interceptor in adapter_config.interceptors
        if interceptor.name == "payload_modifier"
    )

    expected_config = {
        "chat_template_kwargs": {"enable_thinking": False},
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        "custom_param": "custom_value",
    }

    assert payload_modifier.config["params_to_add"] == expected_config


def test_legacy_config_without_params_to_add():
    """Test that payload_modifier is not added when params_to_add is not specified."""
    legacy_config = {"use_request_logging": True, "use_caching": True}

    adapter_config = AdapterConfig.from_legacy_config(legacy_config)

    # Verify payload_modifier is not present
    payload_modifier_interceptors = [
        interceptor
        for interceptor in adapter_config.interceptors
        if interceptor.name == "payload_modifier"
    ]

    assert len(payload_modifier_interceptors) == 0


def test_legacy_reasoning_interceptor_basic():
    """Test basic legacy reasoning interceptor configuration."""
    legacy_config = {
        "use_reasoning": True,
        "end_reasoning_token": "</think>",
    }

    adapter_config = AdapterConfig.from_legacy_config(legacy_config)

    # Verify reasoning interceptor was created
    reasoning_interceptors = [
        interceptor
        for interceptor in adapter_config.interceptors
        if interceptor.name == "reasoning"
    ]

    assert len(reasoning_interceptors) == 1
    reasoning = reasoning_interceptors[0]

    # Verify basic configuration
    assert reasoning.enabled is True
    assert reasoning.config["end_reasoning_token"] == "</think>"


def test_legacy_reasoning_interceptor_with_start_token():
    """Test legacy reasoning interceptor with start token configuration."""
    legacy_config = {
        "use_reasoning": True,
        "start_reasoning_token": "<think>",
        "end_reasoning_token": "</think>",
    }

    adapter_config = AdapterConfig.from_legacy_config(legacy_config)

    reasoning = next(
        interceptor
        for interceptor in adapter_config.interceptors
        if interceptor.name == "reasoning"
    )

    assert reasoning.config["start_reasoning_token"] == "<think>"
    assert reasoning.config["end_reasoning_token"] == "</think>"


def test_legacy_reasoning_interceptor_with_include_if_not_finished():
    """Test legacy reasoning interceptor with include_if_not_finished parameter."""
    legacy_config = {
        "use_reasoning": True,
        "include_if_reasoning_not_finished": False,
        "end_reasoning_token": "</think>",
    }

    adapter_config = AdapterConfig.from_legacy_config(legacy_config)

    reasoning = next(
        interceptor
        for interceptor in adapter_config.interceptors
        if interceptor.name == "reasoning"
    )

    assert reasoning.config["include_if_not_finished"] is False
    assert reasoning.config["end_reasoning_token"] == "</think>"


def test_legacy_reasoning_interceptor_with_track_reasoning():
    """Test legacy reasoning interceptor with track_reasoning parameter."""
    legacy_config = {
        "use_reasoning": True,
        "track_reasoning": False,
        "end_reasoning_token": "</think>",
    }

    adapter_config = AdapterConfig.from_legacy_config(legacy_config)

    reasoning = next(
        interceptor
        for interceptor in adapter_config.interceptors
        if interceptor.name == "reasoning"
    )

    assert reasoning.config["enable_reasoning_tracking"] is False
    assert reasoning.config["end_reasoning_token"] == "</think>"


def test_legacy_reasoning_interceptor_all_parameters():
    """Test legacy reasoning interceptor with all parameters configured."""
    legacy_config = {
        "use_reasoning": True,
        "start_reasoning_token": "<think>",
        "end_reasoning_token": "</think>",
        "include_if_reasoning_not_finished": True,
        "track_reasoning": True,
    }

    adapter_config = AdapterConfig.from_legacy_config(legacy_config)

    reasoning = next(
        interceptor
        for interceptor in adapter_config.interceptors
        if interceptor.name == "reasoning"
    )

    assert reasoning.config["start_reasoning_token"] == "<think>"
    assert reasoning.config["end_reasoning_token"] == "</think>"
    assert reasoning.config["include_if_not_finished"] is True
    assert reasoning.config["enable_reasoning_tracking"] is True


def test_legacy_reasoning_interceptor_defaults():
    """Test legacy reasoning interceptor with minimal configuration (defaults)."""
    legacy_config = {
        "use_reasoning": True,
    }

    adapter_config = AdapterConfig.from_legacy_config(legacy_config)

    reasoning = next(
        interceptor
        for interceptor in adapter_config.interceptors
        if interceptor.name == "reasoning"
    )

    # Should only have end_reasoning_token with default value
    assert reasoning.config["end_reasoning_token"] == "</think>"
    # Other parameters should not be present when not specified
    assert "start_reasoning_token" not in reasoning.config
    assert "include_if_not_finished" not in reasoning.config
    assert "enable_reasoning_tracking" not in reasoning.config
