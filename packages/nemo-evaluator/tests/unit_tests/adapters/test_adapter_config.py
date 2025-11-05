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

import pytest

from nemo_evaluator.adapters.adapter_config import AdapterConfig, InterceptorConfig


def assert_default_config_with_interceptors(adapter_config):
    """Helper function to assert default config with caching, endpoint, and response_stats interceptors.

    Args:
        adapter_config: The adapter config to check
    """
    assert adapter_config is not None
    assert len(adapter_config.interceptors) == 3  # caching, endpoint, response_stats
    interceptor_names = [ic.name for ic in adapter_config.interceptors]
    assert "caching" in interceptor_names
    assert "endpoint" in interceptor_names
    assert "response_stats" in interceptor_names
    assert adapter_config.generate_html_report is True
    assert adapter_config.html_report_size == 5


def assert_interceptor_order(adapter_config, expected_order):
    """Helper function to assert interceptor order.

    Args:
        adapter_config: The adapter config to check
        expected_order: List of expected interceptor names in order
    """
    interceptor_names = [ic.name for ic in adapter_config.interceptors]
    assert interceptor_names == expected_order, (
        f"Expected order {expected_order}, got {interceptor_names}"
    )


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

    # Should return default config with caching enabled
    assert_default_config_with_interceptors(adapter_config)
    # Check interceptor order: caching, endpoint, response_stats
    assert_interceptor_order(adapter_config, ["caching", "endpoint", "response_stats"])


def test_get_validated_config_without_api_endpoint():
    """Test get_validated_config when api_endpoint is not present."""
    run_config = {"target": {}}

    adapter_config = AdapterConfig.get_validated_config(run_config)

    # Should return default config with caching enabled
    assert_default_config_with_interceptors(adapter_config)
    # Check interceptor order: caching, endpoint, response_stats
    assert_interceptor_order(adapter_config, ["caching", "endpoint", "response_stats"])


def test_get_validated_config_without_target():
    """Test get_validated_config when target is not present."""
    run_config = {}

    adapter_config = AdapterConfig.get_validated_config(run_config)

    # Should return default config with caching enabled
    assert_default_config_with_interceptors(adapter_config)
    # Check interceptor order: caching, endpoint, response_stats
    assert_interceptor_order(adapter_config, ["caching", "endpoint", "response_stats"])


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
        "save_responses": True,
        "process_reasoning_traces": True,
        "end_reasoning_token": "</think>",
        "output_dir": "/tmp/output",
        "generate_html_report": True,
        "include_json": True,
        "endpoint_type": "chat",
        "start_reasoning_token": "<think>",
        "include_if_reasoning_not_finished": True,
        "track_reasoning": True,
        "tracking_requests_stats": True,
        "use_progress_tracking": True,
        "progress_tracking_url": "https://progress.example.com",
        "progress_tracking_interval": 5,
    }

    config = AdapterConfig.from_legacy_config(legacy_config)

    assert (
        len(config.interceptors) == 7
    )  # request_logging, caching, response_stats, response_logging, reasoning, progress_tracking, endpoint
    assert (
        len(config.post_eval_hooks) == 1
    )  # post_eval_report (progress_tracking is added automatically by server)

    # Check specific interceptors
    interceptor_names = [ic.name for ic in config.interceptors]
    assert "request_logging" in interceptor_names
    assert "caching" in interceptor_names
    assert "endpoint" in interceptor_names
    assert (
        "response_stats" in interceptor_names
    )  # Added by default when tracking_requests_stats=True
    assert "response_logging" in interceptor_names
    assert "reasoning" in interceptor_names
    assert "progress_tracking" in interceptor_names
    assert config.caching_dir == "/tmp/cache"
    assert config.generate_html_report is True

    # Verify response_stats interceptor cache_dir is set correctly using _get_default_cache_dir
    response_stats_interceptor = next(
        ic for ic in config.interceptors if ic.name == "response_stats"
    )
    assert "cache_dir" in response_stats_interceptor.config
    # Since caching_dir is provided, should use caching_dir/response_stats_cache
    assert (
        response_stats_interceptor.config["cache_dir"]
        == "/tmp/cache/response_stats_cache"
    )

    # Check post-eval hooks
    hook_names = [hook.name for hook in config.post_eval_hooks]
    assert "post_eval_report" in hook_names
    # Note: progress_tracking is not explicitly added as a post-eval hook here
    # because it's an interceptor that implements PostEvalHook, so the server
    # automatically adds it to post-eval hooks at runtime


def test_from_legacy_config_with_html_report_size():
    """Test from_legacy_config method with html_report_size parameter - max_saved behavior."""
    legacy_config = {
        "caching_dir": "/tmp/cache",
        "save_requests": True,
        "save_responses": True,
        "max_saved_requests": 10,
        "max_saved_responses": 5,
        "html_report_size": 15,
        "generate_html_report": True,  # Enable HTML report generation
        "use_caching": False,  # Explicitly set to False to test html_report_size override
    }

    config = AdapterConfig.from_legacy_config(legacy_config)

    # Find the caching interceptor
    caching_interceptor = next(ic for ic in config.interceptors if ic.name == "caching")

    # html_report_size should override the smaller max_saved limits
    assert (
        caching_interceptor.config["max_saved_requests"] == 15
    )  # Overridden by html_report_size
    assert (
        caching_interceptor.config["max_saved_responses"] == 15
    )  # Overridden by html_report_size
    assert config.html_report_size == 15

    # Check that html_report_size is passed to the post_eval_report hook
    hook_configs = config.get_post_eval_hook_configs()
    assert "post_eval_report" in hook_configs
    assert hook_configs["post_eval_report"]["html_report_size"] == 15


def test_from_legacy_config_with_html_report_size_smaller():
    """Test from_legacy_config method with html_report_size smaller than existing max_saved limits."""
    legacy_config = {
        "caching_dir": "/tmp/cache",
        "save_requests": True,
        "save_responses": True,
        "max_saved_requests": 20,
        "max_saved_responses": 25,
        "html_report_size": 15,
        "use_caching": False,  # Explicitly set to False to test html_report_size behavior
    }

    config = AdapterConfig.from_legacy_config(legacy_config)

    # Find the caching interceptor
    caching_interceptor = next(ic for ic in config.interceptors if ic.name == "caching")

    # html_report_size should not override larger max_saved limits
    assert caching_interceptor.config["max_saved_requests"] == 20  # Not overridden
    assert caching_interceptor.config["max_saved_responses"] == 25  # Not overridden
    assert config.html_report_size == 15


@pytest.mark.parametrize(
    "test_name, legacy_config, expected_config",
    [
        (
            "html_report_size_auto_enables_save_requests_responses",
            {
                "use_caching": True,
                "caching_dir": "/tmp/cache",
                "html_report_size": 25,
                "generate_html_report": True,
                # Note: save_requests and save_responses are NOT set
            },
            {
                "max_saved_requests": 25,
                "max_saved_responses": 25,
                "save_requests": True,
                "save_responses": True,
                "html_report_size": 25,
            },
        ),
        (
            "html_report_size_overrides_explicit_save_settings",
            {
                "use_caching": True,
                "caching_dir": "/tmp/cache",
                "save_requests": False,  # Explicitly set to False
                "save_responses": False,  # Explicitly set to False
                "html_report_size": 30,
                "generate_html_report": True,
            },
            {
                "max_saved_requests": 30,
                "max_saved_responses": 30,
                "save_requests": True,  # Always overridden by html_report_size
                "save_responses": True,  # Always overridden by html_report_size
                "html_report_size": 30,
            },
        ),
        (
            "html_report_size_smaller_than_existing_limits",
            {
                "use_caching": True,
                "caching_dir": "/tmp/cache",
                "save_requests": True,
                "save_responses": True,
                "max_saved_requests": 20,
                "max_saved_responses": 25,
                "html_report_size": 15,
                "generate_html_report": True,
            },
            {
                "max_saved_requests": 20,  # Not overridden (html_report_size is smaller)
                "max_saved_responses": 25,  # Not overridden (html_report_size is smaller)
                "save_requests": True,  # Preserved
                "save_responses": True,  # Preserved
                "html_report_size": 15,
            },
        ),
        (
            "html_report_size_larger_than_existing_limits",
            {
                "use_caching": True,
                "caching_dir": "/tmp/cache",
                "save_requests": True,
                "save_responses": True,
                "max_saved_requests": 10,
                "max_saved_responses": 5,
                "html_report_size": 30,
                "generate_html_report": True,
            },
            {
                "max_saved_requests": 30,  # Overridden by html_report_size
                "max_saved_responses": 30,  # Overridden by html_report_size
                "save_requests": True,  # Preserved
                "save_responses": True,  # Preserved
                "html_report_size": 30,
            },
        ),
    ],
    ids=[
        "html_report_size_auto_enables_save_requests_responses",
        "html_report_size_overrides_explicit_save_settings",
        "html_report_size_smaller_than_existing_limits",
        "html_report_size_larger_than_existing_limits",
    ],
)
def test_from_legacy_config_html_report_size_behavior(
    test_name, legacy_config, expected_config
):
    """Test various html_report_size behavior scenarios in adapter config."""
    config = AdapterConfig.from_legacy_config(legacy_config)

    # Find the caching interceptor
    caching_interceptor = next(ic for ic in config.interceptors if ic.name == "caching")

    # Verify all expected configuration values
    for key, expected_value in expected_config.items():
        if key == "html_report_size":
            # html_report_size is a top-level config attribute, not in interceptor config
            assert config.html_report_size == expected_value
        else:
            # All other values are in the interceptor config
            assert caching_interceptor.config[key] == expected_value, (
                f"Failed for {key}"
            )

    # Additional checks for specific scenarios
    if (
        "generate_html_report" in legacy_config
        and legacy_config["generate_html_report"]
    ):
        # Check that html_report_size is passed to the post_eval_report hook
        hook_configs = config.get_post_eval_hook_configs()
        assert "post_eval_report" in hook_configs
        if "html_report_size" in expected_config:
            assert (
                hook_configs["post_eval_report"]["html_report_size"]
                == expected_config["html_report_size"]
            )


@pytest.mark.parametrize(
    "include_json, expected_types",
    [
        (True, ["html", "json"]),
        (False, ["html"]),
        (None, ["html", "json"]),
    ],
    ids=["include_json_true", "include_json_false", "include_json_default"],
)
def test_legacy_html_report_types_parametrized(include_json, expected_types):
    legacy_config = {"generate_html_report": True}
    if include_json is not None:
        legacy_config["include_json"] = include_json

    config = AdapterConfig.from_legacy_config(legacy_config)

    hook_configs = config.get_post_eval_hook_configs()
    assert "post_eval_report" in hook_configs

    report_types = hook_configs["post_eval_report"]["report_types"]
    assert isinstance(report_types, list)
    assert len(report_types) == len(expected_types)
    assert set(report_types) == set(expected_types)
    assert all(isinstance(rt, str) for rt in report_types)


def test_from_legacy_config_with_nvcf():
    """Test conversion from legacy configuration format with use_nvcf."""
    legacy_config = {
        "use_nvcf": True,
        "use_request_logging": True,
        "use_response_logging": True,
    }

    config = AdapterConfig.from_legacy_config(legacy_config)

    assert (
        len(config.interceptors) == 5
    )  # request_logging, nvcf, response_stats, response_logging, caching (no endpoint)

    # Check specific interceptors
    interceptor_names = [ic.name for ic in config.interceptors]
    assert "request_logging" in interceptor_names
    assert "nvcf" in interceptor_names
    assert (
        "caching" in interceptor_names
    )  # Now included by default due to generate_html_report=True
    assert (
        "response_stats" in interceptor_names
    )  # Added by default when tracking_requests_stats=True
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
    assert (
        len(config.interceptors) == 4
    )  # request_logging, nvcf, response_stats, caching (no endpoint)

    # Check specific interceptors
    interceptor_names = [ic.name for ic in config.interceptors]
    assert "request_logging" in interceptor_names
    assert "nvcf" in interceptor_names
    assert (
        "caching" in interceptor_names
    )  # Now included by default due to generate_html_report=True
    assert (
        "response_stats" in interceptor_names
    )  # Added by default when tracking_requests_stats=True
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
    assert "endpoint" in interceptor_names  # Now included by default

    request_logging_id = interceptor_names.index("request_logging")
    payload_modifier_id = interceptor_names.index("payload_modifier")
    assert request_logging_id > payload_modifier_id

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
    legacy_config = {
        "use_request_logging": True,
        "use_caching": True,
    }

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
        "process_reasoning_traces": True,
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

    # Verify cache_dir is set correctly using _get_default_cache_dir
    assert "cache_dir" in reasoning.config
    # Since caching_dir and output_dir are None, should fallback to /tmp/reasoning_stats_cache
    assert reasoning.config["cache_dir"] == "/tmp/reasoning_stats_cache"


def test_legacy_reasoning_interceptor_with_start_token():
    """Test legacy reasoning interceptor with start token configuration."""
    legacy_config = {
        "process_reasoning_traces": True,
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

    # Verify cache_dir is set correctly using _get_default_cache_dir
    assert "cache_dir" in reasoning.config
    # Since caching_dir and output_dir are None, should fallback to /tmp/reasoning_stats_cache
    assert reasoning.config["cache_dir"] == "/tmp/reasoning_stats_cache"


def test_legacy_reasoning_interceptor_with_include_if_not_finished():
    """Test legacy reasoning interceptor with include_if_not_finished parameter."""
    legacy_config = {
        "process_reasoning_traces": True,
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

    # Verify cache_dir is set correctly using _get_default_cache_dir
    assert "cache_dir" in reasoning.config
    # Since caching_dir and output_dir are None, should fallback to /tmp/reasoning_stats_cache
    assert reasoning.config["cache_dir"] == "/tmp/reasoning_stats_cache"


def test_legacy_reasoning_interceptor_with_track_reasoning():
    """Test legacy reasoning interceptor with track_reasoning parameter."""
    legacy_config = {
        "process_reasoning_traces": True,
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

    # Verify cache_dir is set correctly using _get_default_cache_dir
    assert "cache_dir" in reasoning.config
    # Since caching_dir and output_dir are None, should fallback to /tmp/reasoning_stats_cache
    assert reasoning.config["cache_dir"] == "/tmp/reasoning_stats_cache"


def test_legacy_reasoning_interceptor_all_parameters():
    """Test legacy reasoning interceptor with all parameters configured."""
    legacy_config = {
        "process_reasoning_traces": True,
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

    # Verify cache_dir is set correctly using _get_default_cache_dir
    assert "cache_dir" in reasoning.config
    # Since caching_dir and output_dir are None, should fallback to /tmp/reasoning_stats_cache
    assert reasoning.config["cache_dir"] == "/tmp/reasoning_stats_cache"


def test_legacy_reasoning_interceptor_defaults():
    """Test legacy reasoning interceptor with minimal configuration (defaults)."""
    legacy_config = {
        "process_reasoning_traces": True,
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

    # Verify cache_dir is set correctly using _get_default_cache_dir
    assert "cache_dir" in reasoning.config
    # Since caching_dir and output_dir are None, should fallback to /tmp/reasoning_stats_cache
    assert reasoning.config["cache_dir"] == "/tmp/reasoning_stats_cache"


@pytest.mark.parametrize(
    "test_name, legacy_config, expected_cache_dirs",
    [
        (
            "no_output_dir_no_caching_dir",
            {
                "process_reasoning_traces": True,
                "tracking_requests_stats": True,
                "use_caching": True,
                "generate_html_report": True,
            },
            {
                "reasoning": "/tmp/reasoning_stats_cache",
                "response_stats": "/tmp/response_stats_cache",
                "caching": "/tmp/cache",
            },
        ),
        (
            "with_output_dir_no_caching_dir",
            {
                "process_reasoning_traces": True,
                "tracking_requests_stats": True,
                "use_caching": True,
                "generate_html_report": True,
                "output_dir": "/tmp/output",
            },
            {
                "reasoning": "/tmp/output/reasoning_stats_cache",
                "response_stats": "/tmp/output/response_stats_cache",
                "caching": "/tmp/output/cache",
            },
        ),
        (
            "with_caching_dir",
            {
                "process_reasoning_traces": True,
                "tracking_requests_stats": True,
                "use_caching": True,
                "generate_html_report": True,
                "caching_dir": "/tmp/cache",
            },
            {
                "reasoning": "/tmp/cache/reasoning_stats_cache",
                "response_stats": "/tmp/cache/response_stats_cache",
                "caching": "/tmp/cache",
            },
        ),
    ],
)
def test_interceptor_cache_dir_verification(
    test_name, legacy_config, expected_cache_dirs
):
    """Test that all interceptors use _get_default_cache_dir correctly."""
    adapter_config = AdapterConfig.from_legacy_config(legacy_config)

    # Check reasoning interceptor
    if "reasoning" in expected_cache_dirs:
        reasoning = next(
            (ic for ic in adapter_config.interceptors if ic.name == "reasoning"), None
        )
        assert reasoning is not None, "Reasoning interceptor should be present"
        assert "cache_dir" in reasoning.config
        assert reasoning.config["cache_dir"] == expected_cache_dirs["reasoning"]

    # Check response_stats interceptor
    if "response_stats" in expected_cache_dirs:
        response_stats = next(
            (ic for ic in adapter_config.interceptors if ic.name == "response_stats"),
            None,
        )
        assert response_stats is not None, (
            "Response stats interceptor should be present"
        )
        assert "cache_dir" in response_stats.config
        assert (
            response_stats.config["cache_dir"] == expected_cache_dirs["response_stats"]
        )

    # Check caching interceptor
    if "caching" in expected_cache_dirs:
        caching = next(
            (ic for ic in adapter_config.interceptors if ic.name == "caching"), None
        )
        assert caching is not None, "Caching interceptor should be present"
        assert "cache_dir" in caching.config
        assert caching.config["cache_dir"] == expected_cache_dirs["caching"]


@pytest.mark.parametrize(
    "test_name, legacy_config, expected_config",
    [
        (
            "html_report_generation_enabled_with_size",
            {
                "use_caching": True,
                "caching_dir": "/tmp/cache",
                "html_report_size": 15,
                "generate_html_report": True,
                "save_requests": False,
                "save_responses": False,
                "max_saved_requests": 5,
                "max_saved_responses": 10,
            },
            {
                "save_requests": True,
                "save_responses": True,
                "max_saved_requests": 15,  # Overridden by html_report_size
                "max_saved_responses": 15,  # Overridden by html_report_size
            },
        ),
        (
            "html_report_generation_enabled_without_size",
            {
                "use_caching": True,
                "caching_dir": "/tmp/cache",
                "html_report_size": None,
                "generate_html_report": True,
                "save_requests": False,
                "save_responses": False,
                "max_saved_requests": 5,
                "max_saved_responses": 10,
            },
            {
                "save_requests": True,
                "save_responses": True,
                "max_saved_requests": 5,  # Uses existing value when html_report_size is None
                "max_saved_responses": 10,  # Uses existing value when html_report_size is None
            },
        ),
        (
            "html_report_generation_disabled_with_size",
            {
                "use_caching": True,
                "caching_dir": "/tmp/cache",
                "html_report_size": 15,
                "generate_html_report": False,
                "save_requests": False,
                "save_responses": False,
                "max_saved_requests": 5,
                "max_saved_responses": 10,
            },
            {
                "save_requests": False,  # Respects explicit setting
                "save_responses": False,  # Respects explicit setting
                "max_saved_requests": 5,  # Respects explicit setting
                "max_saved_responses": 10,  # Respects explicit setting
            },
        ),
        (
            "html_report_generation_disabled_without_size",
            {
                "use_caching": True,
                "caching_dir": "/tmp/cache",
                "html_report_size": None,
                "generate_html_report": False,
                "save_requests": False,
                "save_responses": False,
                "max_saved_requests": 5,
                "max_saved_responses": 10,
            },
            {
                "save_requests": False,  # Respects explicit setting
                "save_responses": False,  # Respects explicit setting
                "max_saved_requests": 5,  # Respects explicit setting
                "max_saved_responses": 10,  # Respects explicit setting
            },
        ),
        (
            "html_report_size_smaller_than_existing_limits",
            {
                "use_caching": True,
                "caching_dir": "/tmp/cache",
                "html_report_size": 3,
                "generate_html_report": True,
                "save_requests": False,
                "save_responses": False,
                "max_saved_requests": 5,
                "max_saved_responses": 10,
            },
            {
                "save_requests": True,
                "save_responses": True,
                "max_saved_requests": 5,  # Not overridden (html_report_size is smaller)
                "max_saved_responses": 10,  # Not overridden (html_report_size is smaller)
            },
        ),
        (
            "html_report_size_larger_than_existing_limits",
            {
                "use_caching": True,
                "caching_dir": "/tmp/cache",
                "html_report_size": 20,
                "generate_html_report": True,
                "save_requests": False,
                "save_responses": False,
                "max_saved_requests": 5,
                "max_saved_responses": 10,
            },
            {
                "save_requests": True,
                "save_responses": True,
                "max_saved_requests": 20,  # Overridden by html_report_size
                "max_saved_responses": 20,  # Overridden by html_report_size
            },
        ),
    ],
    ids=[
        "html_report_enabled_with_size",
        "html_report_enabled_without_size",
        "html_report_disabled_with_size",
        "html_report_disabled_without_size",
        "html_report_size_smaller_than_limits",
        "html_report_size_larger_than_limits",
    ],
)
def test_html_report_generation_logic(test_name, legacy_config, expected_config):
    """Test HTML report generation logic in adapter config."""
    config = AdapterConfig.from_legacy_config(legacy_config)

    # Find the caching interceptor
    caching_interceptors = [ic for ic in config.interceptors if ic.name == "caching"]

    # If generate_html_report is False and no other caching settings, no interceptor should be created
    if not legacy_config.get("generate_html_report", True) and not any(
        [
            legacy_config.get("use_caching", False),
            legacy_config.get("save_responses", False),
            legacy_config.get("save_requests", False),
        ]
    ):
        assert len(caching_interceptors) == 0, (
            "No caching interceptor should be created when generate_html_report is False and no other caching settings"
        )
        return

    assert len(caching_interceptors) == 1, "Expected exactly one caching interceptor"
    caching_interceptor = caching_interceptors[0]

    # Verify all expected configuration values
    for key, expected_value in expected_config.items():
        actual_value = caching_interceptor.config.get(key)
        assert actual_value == expected_value, (
            f"Failed for {key}: expected {expected_value}, got {actual_value}"
        )


@pytest.mark.parametrize(
    "legacy_config,expected_caching_enabled,expected_config",
    [
        # Test cases for caching interceptor activation
        (
            {"use_caching": True},
            True,
            {
                "reuse_cached_responses": True,  # Maps to use_caching=True
                "save_requests": True,  # Default from generate_html_report=True
                "save_responses": True,  # Default from generate_html_report=True
                "max_saved_requests": 5,  # Default from html_report_size=5
                "max_saved_responses": 5,  # Default from html_report_size=5
                "cache_dir": "/tmp/cache",  # Default fallback when no caching_dir or output_dir
            },
        ),
        (
            {"save_responses": True},
            True,
            {
                "reuse_cached_responses": True,  # Default when caching is enabled
                "save_requests": True,  # Default from generate_html_report=True
                "save_responses": True,
                "max_saved_requests": 5,  # Default from html_report_size=5
                "max_saved_responses": 5,  # Default from html_report_size=5
                "cache_dir": "/tmp/cache",  # Default fallback when no caching_dir or output_dir
            },
        ),
        (
            {"save_requests": True},
            True,
            {
                "reuse_cached_responses": True,  # Default when caching is enabled
                "save_requests": True,
                "save_responses": True,  # Default from generate_html_report=True
                "max_saved_requests": 5,  # Default from html_report_size=5
                "max_saved_responses": 5,  # Default from html_report_size=5
            },
        ),
        (
            {"generate_html_report": True},
            True,
            {
                "reuse_cached_responses": True,  # Default when caching is enabled
                "save_requests": True,
                "save_responses": True,
                "max_saved_requests": 5,  # Default from html_report_size=5
                "max_saved_responses": 5,  # Default from html_report_size=5
            },
        ),
        (
            {"generate_html_report": False},
            False,  # Not enabled when generate_html_report=False and no other caching settings
            {},
        ),
        # Test case where caching should be activated due to defaults
        (
            {},  # No caching-related settings
            True,  # Still enabled because generate_html_report defaults to True
            {
                "reuse_cached_responses": True,  # Default when caching is enabled
                "save_requests": True,  # Default from generate_html_report=True
                "save_responses": True,  # Default from generate_html_report=True
                "max_saved_requests": 5,  # Default from html_report_size=5
                "max_saved_responses": 5,  # Default from html_report_size=5
            },
        ),
        # Test with custom html_report_size
        (
            {"html_report_size": 10},
            True,
            {
                "reuse_cached_responses": True,  # Default when caching is enabled
                "save_requests": True,  # Default from generate_html_report=True
                "save_responses": True,  # Default from generate_html_report=True
                "max_saved_requests": 10,  # From html_report_size=10
                "max_saved_responses": 10,  # From html_report_size=10
                "cache_dir": "/tmp/cache",  # Default fallback when no caching_dir or output_dir
            },
        ),
        # Test with html_report_size and existing max values
        (
            {
                "html_report_size": 15,
                "max_saved_requests": 10,
                "max_saved_responses": 20,
            },
            True,
            {
                "reuse_cached_responses": True,  # Default when caching is enabled
                "save_requests": True,  # Default from generate_html_report=True
                "save_responses": True,  # Default from generate_html_report=True
                "max_saved_requests": 15,  # max(15, 10) = 15
                "max_saved_responses": 20,  # max(15, 20) = 20
                "cache_dir": "/tmp/cache",  # Default fallback when no caching_dir or output_dir
            },
        ),
        # Test with None html_report_size
        (
            {"html_report_size": None},
            True,
            {
                "reuse_cached_responses": True,  # Default when caching is enabled
                "save_requests": True,  # Default from generate_html_report=True
                "save_responses": True,  # Default from generate_html_report=True
                "max_saved_requests": None,  # No html_report_size override
                "max_saved_responses": None,  # No html_report_size override
                "cache_dir": "/tmp/cache",  # Default fallback when no caching_dir or output_dir
            },
        ),
        # Test with output_dir but no caching_dir
        (
            {"html_report_size": 5, "output_dir": "/tmp/output"},
            True,
            {
                "reuse_cached_responses": True,  # Default when caching is enabled
                "save_requests": True,  # Default from generate_html_report=True
                "save_responses": True,  # Default from generate_html_report=True
                "max_saved_requests": 5,  # Default from html_report_size=5
                "max_saved_responses": 5,  # Default from html_report_size=5
                "cache_dir": "/tmp/output/cache",  # Should use output_dir/cache
            },
        ),
        # Test with explicit caching_dir
        (
            {"html_report_size": 5, "caching_dir": "/custom/cache"},
            True,
            {
                "reuse_cached_responses": True,  # Default when caching is enabled
                "save_requests": True,  # Default from generate_html_report=True
                "save_responses": True,  # Default from generate_html_report=True
                "max_saved_requests": 5,  # Default from html_report_size=5
                "max_saved_responses": 5,  # Default from html_report_size=5
                "cache_dir": "/custom/cache",  # Should use explicit caching_dir
            },
        ),
    ],
    ids=[
        "use_caching_enabled",
        "save_responses_enabled",
        "save_requests_enabled",
        "generate_html_report_enabled",
        "generate_html_report_disabled",
        "no_caching_settings_defaults",
        "custom_html_report_size",
        "html_report_size_with_existing_limits",
        "html_report_size_none",
        "with_output_dir_no_caching_dir",
        "with_explicit_caching_dir",
    ],
)
def test_caching_interceptor_activation_and_config(
    legacy_config, expected_caching_enabled, expected_config
):
    """Test caching interceptor activation and configuration with various combinations."""
    config = AdapterConfig.from_legacy_config(legacy_config)

    # Check if caching interceptor is present
    caching_interceptors = [ic for ic in config.interceptors if ic.name == "caching"]

    if expected_caching_enabled:
        assert len(caching_interceptors) == 1, "Caching interceptor should be present"
        caching_interceptor = caching_interceptors[0]
        assert caching_interceptor.enabled is True, (
            "Caching interceptor should be enabled"
        )

        # Verify all expected configuration values
        for key, expected_value in expected_config.items():
            actual_value = caching_interceptor.config.get(key)
            assert actual_value == expected_value, (
                f"Failed for {key}: expected {expected_value}, got {actual_value}"
            )

        # Verify cache_dir is set correctly using _get_default_cache_dir
        assert "cache_dir" in caching_interceptor.config
        assert caching_interceptor.config["cache_dir"].endswith("cache")
    else:
        assert len(caching_interceptors) == 1, "Caching interceptor should be present"


@pytest.mark.parametrize(
    "legacy_config,expected_adapter_config_values",
    [
        # Test default values in AdapterConfig
        (
            {},
            {
                "generate_html_report": True,
                "html_report_size": 5,
            },
        ),
        # Test with explicit values
        (
            {
                "generate_html_report": False,
                "html_report_size": 10,
            },
            {
                "generate_html_report": False,
                "html_report_size": 10,
            },
        ),
        # Test with explicit True values (None values cause Pydantic validation errors)
        (
            {
                "generate_html_report": True,
                "html_report_size": 5,
            },
            {
                "generate_html_report": True,  # Explicit
                "html_report_size": 5,  # Explicit
            },
        ),
    ],
    ids=[
        "default_values",
        "explicit_values",
        "explicit_true_values",
    ],
)
def test_adapter_config_defaults(legacy_config, expected_adapter_config_values):
    """Test that AdapterConfig uses correct default values."""
    config = AdapterConfig.from_legacy_config(legacy_config)

    for key, expected_value in expected_adapter_config_values.items():
        actual_value = getattr(config, key)
        assert actual_value == expected_value, (
            f"Failed for {key}: expected {expected_value}, got {actual_value}"
        )


@pytest.mark.parametrize(
    "legacy_config,expected_config",
    [
        # Test default behavior (HTML report enabled, caching enabled)
        (
            {},
            {
                "has_caching_interceptor": True,
                "has_post_eval_report_hook": True,
                "caching_config": {
                    "reuse_cached_responses": True,
                    "save_requests": True,
                    "save_responses": True,
                    "max_saved_requests": 5,
                    "max_saved_responses": 5,
                },
                "post_eval_config": {
                    "html_report_size": 5,
                },
            },
        ),
        # Test with explicit HTML report disabled
        (
            {"generate_html_report": False},
            {
                "has_caching_interceptor": True,
                "has_post_eval_report_hook": False,
                "caching_config": {},
                "post_eval_config": {},
            },
        ),
        # Test with caching enabled but no HTML report
        (
            {"use_caching": True, "generate_html_report": False},
            {
                "has_caching_interceptor": True,
                "has_post_eval_report_hook": False,
                "caching_config": {
                    "reuse_cached_responses": True,
                    "save_requests": False,
                    "save_responses": False,
                    "max_saved_requests": None,
                    "max_saved_responses": None,
                },
                "post_eval_config": {},
            },
        ),
        # Test with custom HTML report size
        (
            {"html_report_size": 10},
            {
                "has_caching_interceptor": True,
                "has_post_eval_report_hook": True,
                "caching_config": {
                    "reuse_cached_responses": True,
                    "save_requests": True,
                    "save_responses": True,
                    "max_saved_requests": 10,
                    "max_saved_responses": 10,
                },
                "post_eval_config": {
                    "html_report_size": 10,
                },
            },
        ),
        # Test with both caching and HTML report enabled
        (
            {"use_caching": True, "html_report_size": 15},
            {
                "has_caching_interceptor": True,
                "has_post_eval_report_hook": True,
                "caching_config": {
                    "reuse_cached_responses": True,
                    "save_requests": True,
                    "save_responses": True,
                    "max_saved_requests": 15,
                    "max_saved_responses": 15,
                },
                "post_eval_config": {
                    "html_report_size": 15,
                },
            },
        ),
    ],
    ids=[
        "default_behavior",
        "html_report_disabled",
        "caching_only",
        "custom_html_size",
        "caching_and_html_enabled",
    ],
)
def test_html_report_and_caching_integration(legacy_config, expected_config):
    """Test integration between HTML report generation and caching parameters."""
    config = AdapterConfig.from_legacy_config(legacy_config)

    # Check caching interceptor presence and configuration
    caching_interceptors = [ic for ic in config.interceptors if ic.name == "caching"]
    has_caching_interceptor = len(caching_interceptors) > 0

    assert has_caching_interceptor == expected_config["has_caching_interceptor"], (
        f"Expected caching interceptor presence: {expected_config['has_caching_interceptor']}, "
        f"got: {has_caching_interceptor}"
    )

    if has_caching_interceptor:
        caching_interceptor = caching_interceptors[0]
        assert caching_interceptor.enabled is True, (
            "Caching interceptor should be enabled"
        )

        # Check caching configuration
        for key, expected_value in expected_config["caching_config"].items():
            if expected_value is not None:
                assert caching_interceptor.config[key] == expected_value, (
                    f"Expected {key}={expected_value}, got {caching_interceptor.config[key]}"
                )
            else:
                assert caching_interceptor.config.get(key) is None, (
                    f"Expected {key} to be None, got {caching_interceptor.config.get(key)}"
                )

    # Check post-eval hook presence and configuration
    hook_configs = config.get_post_eval_hook_configs()
    has_post_eval_report_hook = "post_eval_report" in hook_configs

    assert has_post_eval_report_hook == expected_config["has_post_eval_report_hook"], (
        f"Expected post-eval report hook presence: {expected_config['has_post_eval_report_hook']}, "
        f"got: {has_post_eval_report_hook}"
    )

    if has_post_eval_report_hook:
        hook_config = hook_configs["post_eval_report"]

        # Check post-eval configuration
        for key, expected_value in expected_config["post_eval_config"].items():
            if expected_value is not None:
                assert hook_config[key] == expected_value, (
                    f"Expected {key}={expected_value}, got {hook_config[key]}"
                )
            else:
                assert hook_config.get(key) is None, (
                    f"Expected {key} to be None, got {hook_config.get(key)}"
                )


@pytest.mark.parametrize(
    "legacy_config,expected_interceptor_attributes",
    [
        # Test when use_caching is True
        (
            {"use_caching": True, "html_report_size": 10},
            {
                "should_save_requests": True,
                "should_save_responses": True,
                "max_requests_limit": 10,
                "max_responses_limit": None,
            },
        ),
        # Test when generate_html_report is False
        (
            {"generate_html_report": False, "use_caching": True},
            {
                "should_save_requests": False,
                "should_save_responses": True,
                "max_requests_limit": None,
                "max_responses_limit": None,
            },
        ),
    ],
    ids=[
        "reuse_cached_with_limits",
        "generate_html_false",
    ],
)
def test_caching_interceptor_internal_attributes(
    legacy_config, expected_interceptor_attributes
):
    """Test the internal attributes and behavior of the CachingInterceptor class."""
    from nemo_evaluator.adapters.adapter_config import AdapterConfig
    from nemo_evaluator.adapters.interceptors.caching_interceptor import (
        CachingInterceptor,
    )

    # Create adapter config and get the caching interceptor config
    adapter_config = AdapterConfig.from_legacy_config(legacy_config)
    caching_interceptors = [
        ic for ic in adapter_config.interceptors if ic.name == "caching"
    ]

    assert len(caching_interceptors) == 1, "Caching interceptor should be present"
    caching_config = caching_interceptors[0].config

    # Create the actual CachingInterceptor instance using the Params class
    interceptor = CachingInterceptor(CachingInterceptor.Params(**caching_config))

    # Test basic attributes
    assert (
        interceptor.save_requests
        == expected_interceptor_attributes["should_save_requests"]
    )
    assert (
        interceptor.save_responses
        == expected_interceptor_attributes["should_save_responses"]
    )

    # Test limits
    if expected_interceptor_attributes["max_requests_limit"] is not None:
        assert (
            interceptor.max_saved_requests
            == expected_interceptor_attributes["max_requests_limit"]
        )
    if expected_interceptor_attributes["max_responses_limit"] is not None:
        assert (
            interceptor.max_saved_responses
            == expected_interceptor_attributes["max_responses_limit"]
        )

    # Test post-eval hook for HTML report generation
    hook_configs = adapter_config.get_post_eval_hook_configs()
    has_post_eval_report_hook = "post_eval_report" in hook_configs

    defaults = AdapterConfig.get_legacy_defaults()
    should_enable_html = legacy_config.get(
        "generate_html_report", defaults["generate_html_report"]
    )

    if should_enable_html:
        assert has_post_eval_report_hook, (
            "Should have post_eval_report hook when generate_html_report is True"
        )
        if has_post_eval_report_hook:
            hook_config = hook_configs["post_eval_report"]
            expected_html_size = legacy_config.get(
                "html_report_size", defaults["html_report_size"]
            )
            assert hook_config["html_report_size"] == expected_html_size
    else:
        assert not has_post_eval_report_hook, (
            "Should not have post_eval_report hook when generate_html_report is False"
        )


@pytest.mark.parametrize(
    "legacy_config,expected_behavior_flags",
    [
        # Test caching behavior
        (
            {"use_caching": True, "html_report_size": 5},
            {
                "should_cache_responses": True,
                "should_save_requests": True,
                "should_save_responses": True,
            },
        ),
        # Test when generate_html_report is False
        (
            {"generate_html_report": False, "use_caching": True},
            {
                "should_cache_responses": True,
                "should_save_requests": False,
                "should_save_responses": True,
            },
        ),
    ],
    ids=[
        "full_caching_behavior",
        "cache_only_behavior",
    ],
)
def test_caching_interceptor_behavior_flags(legacy_config, expected_behavior_flags):
    """Test the behavior flags and capabilities of the CachingInterceptor."""
    from nemo_evaluator.adapters.adapter_config import AdapterConfig
    from nemo_evaluator.adapters.interceptors.caching_interceptor import (
        CachingInterceptor,
    )

    # Create adapter config and get the caching interceptor config
    adapter_config = AdapterConfig.from_legacy_config(legacy_config)
    caching_interceptors = [
        ic for ic in adapter_config.interceptors if ic.name == "caching"
    ]

    assert len(caching_interceptors) == 1, "Caching interceptor should be present"
    caching_config = caching_interceptors[0].config

    # Create the actual CachingInterceptor instance using the Params class
    interceptor = CachingInterceptor(CachingInterceptor.Params(**caching_config))

    # Test basic behavior flags
    assert (
        interceptor.reuse_cached_responses
        == expected_behavior_flags["should_cache_responses"]
    )
    assert interceptor.save_requests == expected_behavior_flags["should_save_requests"]
    assert (
        interceptor.save_responses == expected_behavior_flags["should_save_responses"]
    )

    # Test post-eval hook for HTML report generation
    hook_configs = adapter_config.get_post_eval_hook_configs()
    has_post_eval_report_hook = "post_eval_report" in hook_configs

    defaults = AdapterConfig.get_legacy_defaults()
    should_enable_html = legacy_config.get(
        "generate_html_report", defaults["generate_html_report"]
    )

    if should_enable_html:
        assert has_post_eval_report_hook, (
            "Should have post_eval_report hook when generate_html_report is True"
        )
    else:
        assert not has_post_eval_report_hook, (
            "Should not have post_eval_report hook when generate_html_report is False"
        )
