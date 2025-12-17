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

"""Tests for adapter_config merging flow: framework defaults + user legacy params + user overrides.

This module tests the complete configuration merge flow to ensure:
1. Framework defaults (e.g., mode: client) are preserved
2. User legacy parameters (e.g., use_caching=True) are converted to interceptors
3. User explicit overrides take precedence
4. The final adapter_config is always an AdapterConfig object with correct mode and interceptors
"""

import tempfile

from nemo_evaluator.adapters.adapter_config import AdapterConfig
from nemo_evaluator.core.input import validate_configuration
from nemo_evaluator.core.utils import deep_update, dotlist_to_dict


class TestAdapterConfigMergeFlow:
    """Test complete adapter_config merge flow from CLI args through to final Evaluation."""

    def test_framework_defaults_mode_client_preserved_with_legacy_params(
        self, monkeypatch
    ):
        """Test that framework's mode: client is preserved when user adds legacy params via CLI overrides."""
        # Mock framework that returns simple-evals with mode: client
        from nemo_evaluator.api.api_dataclasses import Evaluation

        mock_framework_defaults = {
            "simple_evals": {
                "framework_name": "simple_evals",
                "pkg_name": "simple_evals",
                "command": "echo test",
                "config": {
                    "type": "mmlu_pro",
                    "params": {"task": "mmlu_pro"},
                },
                "target": {
                    "api_endpoint": {
                        "type": "chat",
                        "adapter_config": {
                            "mode": "client"  # Framework default from simple-evals
                        },
                    }
                },
            }
        }

        # Create an Evaluation object from framework defaults
        framework_eval = Evaluation(**mock_framework_defaults["simple_evals"])
        mock_framework_eval_mappings = {"simple_evals": {"mmlu_pro": framework_eval}}
        mock_eval_name_mapping = {"mmlu_pro": framework_eval}

        def mock_get_available_evaluations():
            return (
                mock_framework_eval_mappings,
                mock_framework_defaults,
                mock_eval_name_mapping,
            )

        monkeypatch.setattr(
            "nemo_evaluator.core.input.get_available_evaluations",
            mock_get_available_evaluations,
        )

        # Simulate CLI args with legacy parameters (like e2e test does)
        with tempfile.TemporaryDirectory() as temp_dir:
            run_config = {
                "config": {
                    "type": "mmlu_pro",
                    "output_dir": temp_dir,
                    "params": {"limit_samples": 2},
                },
                "target": {
                    "api_endpoint": {
                        "model_id": "test-model",
                        "url": "http://localhost:8000/v1/chat/completions",
                        "type": "chat",
                        "adapter_config": {
                            # User adds legacy params via CLI overrides
                            "use_caching": True,
                            "use_request_logging": True,
                            "use_response_logging": True,
                            "use_system_prompt": True,
                            "custom_system_prompt": "You are a helpful AI assistant.",
                            # NOTE: user does NOT specify mode
                        },
                    }
                },
            }

            # Validate configuration (this is what nemo-evaluator run_eval does)
            evaluation = validate_configuration(run_config)

            # Verify adapter_config is an AdapterConfig object
            assert evaluation.target.api_endpoint.adapter_config is not None
            adapter_config = evaluation.target.api_endpoint.adapter_config
            assert isinstance(adapter_config, AdapterConfig), (
                "adapter_config should be AdapterConfig object"
            )

            # Verify mode: client from framework defaults is preserved
            assert adapter_config.mode == "client", (
                "Framework default mode: client should be preserved"
            )

            # Verify legacy params were converted to interceptors
            interceptor_names = [ic.name for ic in adapter_config.interceptors]
            assert "caching" in interceptor_names, (
                "use_caching should create caching interceptor"
            )
            assert "request_logging" in interceptor_names, (
                "use_request_logging should create request_logging interceptor"
            )
            assert "response_logging" in interceptor_names, (
                "use_response_logging should create response_logging interceptor"
            )
            assert "system_message" in interceptor_names, (
                "use_system_prompt should create system_message interceptor"
            )

            # Verify system message interceptor has correct config
            system_msg_interceptor = next(
                ic for ic in adapter_config.interceptors if ic.name == "system_message"
            )
            assert (
                system_msg_interceptor.config.get("system_message")
                == "You are a helpful AI assistant."
            )

    def test_user_explicit_mode_overrides_framework_default(self, monkeypatch):
        """Test that user's explicit mode: server overrides framework's mode: client."""
        from nemo_evaluator.api.api_dataclasses import Evaluation

        mock_framework_defaults = {
            "simple_evals": {
                "framework_name": "simple_evals",
                "pkg_name": "simple_evals",
                "command": "echo test",
                "config": {"type": "mmlu_pro", "params": {"task": "mmlu_pro"}},
                "target": {
                    "api_endpoint": {
                        "type": "chat",
                        "adapter_config": {"mode": "client"},  # Framework default
                    }
                },
            }
        }

        framework_eval = Evaluation(**mock_framework_defaults["simple_evals"])
        mock_framework_eval_mappings = {"simple_evals": {"mmlu_pro": framework_eval}}
        mock_eval_name_mapping = {"mmlu_pro": framework_eval}

        def mock_get_available_evaluations():
            return (
                mock_framework_eval_mappings,
                mock_framework_defaults,
                mock_eval_name_mapping,
            )

        monkeypatch.setattr(
            "nemo_evaluator.core.input.get_available_evaluations",
            mock_get_available_evaluations,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            run_config = {
                "config": {
                    "type": "mmlu_pro",
                    "output_dir": temp_dir,
                },
                "target": {
                    "api_endpoint": {
                        "model_id": "test-model",
                        "url": "http://localhost:8000",
                        "type": "chat",
                        "adapter_config": {
                            "mode": "server",  # User explicitly sets mode
                            "use_caching": True,
                        },
                    }
                },
            }

            evaluation = validate_configuration(run_config)

            adapter_config = evaluation.target.api_endpoint.adapter_config
            assert isinstance(adapter_config, AdapterConfig)
            assert adapter_config.mode == "server", (
                "User's explicit mode: server should override framework default"
            )

    def test_dotlist_override_merging_preserves_framework_mode(self, monkeypatch):
        """Test that dotlist-style CLI overrides preserve framework mode: client."""
        from nemo_evaluator.api.api_dataclasses import Evaluation

        mock_framework_defaults = {
            "simple_evals": {
                "framework_name": "simple_evals",
                "pkg_name": "simple_evals",
                "command": "echo test",
                "config": {"type": "mmlu_pro", "params": {"task": "mmlu_pro"}},
                "target": {
                    "api_endpoint": {
                        "type": "chat",
                        "adapter_config": {"mode": "client"},
                    }
                },
            }
        }

        framework_eval = Evaluation(**mock_framework_defaults["simple_evals"])
        mock_framework_eval_mappings = {"simple_evals": {"mmlu_pro": framework_eval}}
        mock_eval_name_mapping = {"mmlu_pro": framework_eval}

        def mock_get_available_evaluations():
            return (
                mock_framework_eval_mappings,
                mock_framework_defaults,
                mock_eval_name_mapping,
            )

        monkeypatch.setattr(
            "nemo_evaluator.core.input.get_available_evaluations",
            mock_get_available_evaluations,
        )

        # Simulate dotlist-style CLI overrides (like e2e test)
        override_str = (
            "target.api_endpoint.adapter_config.use_caching=True,"
            "target.api_endpoint.adapter_config.use_request_logging=True,"
            "target.api_endpoint.adapter_config.use_system_prompt=True,"
            "target.api_endpoint.adapter_config.custom_system_prompt=Test prompt"
        )
        overrides = dotlist_to_dict(override_str.split(","))

        with tempfile.TemporaryDirectory() as temp_dir:
            base_config = {
                "config": {"type": "mmlu_pro", "output_dir": temp_dir},
                "target": {
                    "api_endpoint": {
                        "model_id": "test-model",
                        "url": "http://localhost:8000",
                        "type": "chat",
                    }
                },
            }

            # Merge with overrides (simulating what _parse_cli_args does)
            run_config = deep_update(base_config, overrides, skip_nones=True)

            evaluation = validate_configuration(run_config)

            adapter_config = evaluation.target.api_endpoint.adapter_config
            assert isinstance(adapter_config, AdapterConfig)

            # Verify mode: client from framework is preserved
            assert adapter_config.mode == "client", (
                "Framework mode: client should be preserved with dotlist overrides"
            )

            # Verify legacy params were converted
            interceptor_names = [ic.name for ic in adapter_config.interceptors]
            assert "caching" in interceptor_names
            assert "request_logging" in interceptor_names
            assert "system_message" in interceptor_names

    def test_complex_merge_framework_plus_legacy_plus_overrides(self, monkeypatch):
        """Test complex merge: framework defaults + legacy params + user overrides."""
        from nemo_evaluator.api.api_dataclasses import Evaluation

        mock_framework_defaults = {
            "simple_evals": {
                "framework_name": "simple_evals",
                "pkg_name": "simple_evals",
                "command": "echo test",
                "config": {"type": "mmlu_pro", "params": {"task": "mmlu_pro"}},
                "target": {
                    "api_endpoint": {
                        "type": "chat",
                        "adapter_config": {
                            "mode": "client",
                            "endpoint_type": "chat",
                            "log_failed_requests": False,
                        },
                    }
                },
            }
        }

        framework_eval = Evaluation(**mock_framework_defaults["simple_evals"])
        mock_framework_eval_mappings = {"simple_evals": {"mmlu_pro": framework_eval}}
        mock_eval_name_mapping = {"mmlu_pro": framework_eval}

        def mock_get_available_evaluations():
            return (
                mock_framework_eval_mappings,
                mock_framework_defaults,
                mock_eval_name_mapping,
            )

        monkeypatch.setattr(
            "nemo_evaluator.core.input.get_available_evaluations",
            mock_get_available_evaluations,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            run_config = {
                "config": {"type": "mmlu_pro", "output_dir": temp_dir},
                "target": {
                    "api_endpoint": {
                        "model_id": "test-model",
                        "url": "http://localhost:8000",
                        "type": "chat",
                        "adapter_config": {
                            # User adds legacy params
                            "use_caching": True,
                            "use_request_logging": True,
                            # User overrides framework default
                            "log_failed_requests": True,
                            # User does NOT specify mode
                        },
                    }
                },
            }

            evaluation = validate_configuration(run_config)

            adapter_config = evaluation.target.api_endpoint.adapter_config
            assert isinstance(adapter_config, AdapterConfig)

            # Verify framework defaults preserved
            assert adapter_config.mode == "client", "Framework mode preserved"
            assert adapter_config.endpoint_type == "chat", (
                "Framework endpoint_type preserved"
            )

            # Verify user override applied
            assert adapter_config.log_failed_requests is True, (
                "User override should take precedence"
            )

            # Verify legacy params converted to interceptors
            interceptor_names = [ic.name for ic in adapter_config.interceptors]
            assert "caching" in interceptor_names
            assert "request_logging" in interceptor_names
            assert "endpoint" in interceptor_names

    def test_adapter_config_conversion_happens_before_pydantic(self, monkeypatch):
        """Test that adapter_config dict is converted to AdapterConfig BEFORE Pydantic validation."""
        from nemo_evaluator.api.api_dataclasses import Evaluation

        mock_framework_defaults = {
            "simple_evals": {
                "framework_name": "simple_evals",
                "pkg_name": "simple_evals",
                "command": "echo test",
                "config": {"type": "test_task", "params": {"task": "test_task"}},
                "target": {
                    "api_endpoint": {
                        "type": "chat",
                        "adapter_config": {"mode": "client"},
                    }
                },
            }
        }

        framework_eval = Evaluation(**mock_framework_defaults["simple_evals"])
        mock_framework_eval_mappings = {"simple_evals": {"test_task": framework_eval}}
        mock_eval_name_mapping = {"test_task": framework_eval}

        def mock_get_available_evaluations():
            return (
                mock_framework_eval_mappings,
                mock_framework_defaults,
                mock_eval_name_mapping,
            )

        monkeypatch.setattr(
            "nemo_evaluator.core.input.get_available_evaluations",
            mock_get_available_evaluations,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            # This simulates what happens when user runs nemo-evaluator with --overrides
            run_config = {
                "config": {"type": "test_task", "output_dir": temp_dir},
                "target": {
                    "api_endpoint": {
                        "model_id": "test-model",
                        "url": "http://localhost:8000",
                        "type": "chat",
                        "adapter_config": {
                            # Dict with legacy params (this is what dotlist_to_dict produces)
                            "use_caching": True,
                            "caching_dir": f"{temp_dir}/cache",
                            "use_request_logging": True,
                        },
                    }
                },
            }

            # Before validate_configuration, adapter_config is a dict
            assert isinstance(
                run_config["target"]["api_endpoint"]["adapter_config"], dict
            )

            evaluation = validate_configuration(run_config)

            # After validate_configuration, adapter_config should be AdapterConfig object
            adapter_config = evaluation.target.api_endpoint.adapter_config
            assert isinstance(adapter_config, AdapterConfig), (
                "adapter_config should be converted to AdapterConfig object"
            )

            # Verify it has correct mode from framework
            assert adapter_config.mode == "client"

            # Verify it has interceptors from legacy params
            assert len(adapter_config.interceptors) > 0
            interceptor_names = [ic.name for ic in adapter_config.interceptors]
            assert "caching" in interceptor_names
            assert "request_logging" in interceptor_names

    def test_e2e_simulation_comprehensive_overrides(self, monkeypatch):
        """Simulate exactly what e2e test does with comprehensive interceptor overrides."""
        from nemo_evaluator.api.api_dataclasses import Evaluation

        mock_framework_defaults = {
            "simple_evals": {
                "framework_name": "simple_evals",
                "pkg_name": "simple_evals",
                "command": "echo test",
                "config": {"type": "mmlu_pro", "params": {"task": "mmlu_pro"}},
                "target": {
                    "api_endpoint": {
                        "type": "chat",
                        "adapter_config": {"mode": "client"},
                    }
                },
            }
        }

        framework_eval = Evaluation(**mock_framework_defaults["simple_evals"])
        mock_framework_eval_mappings = {"simple_evals": {"mmlu_pro": framework_eval}}
        mock_eval_name_mapping = {"mmlu_pro": framework_eval}

        def mock_get_available_evaluations():
            return (
                mock_framework_eval_mappings,
                mock_framework_defaults,
                mock_eval_name_mapping,
            )

        monkeypatch.setattr(
            "nemo_evaluator.core.input.get_available_evaluations",
            mock_get_available_evaluations,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            # Simulate exactly what e2e test does
            override_str = (
                f"config.params.limit_samples=2,"
                f"target.api_endpoint.adapter_config.use_system_prompt=True,"
                f"target.api_endpoint.adapter_config.custom_system_prompt=You are a helpful AI assistant.,"
                f"target.api_endpoint.adapter_config.use_request_logging=True,"
                f"target.api_endpoint.adapter_config.use_response_logging=True,"
                f"target.api_endpoint.adapter_config.use_caching=True,"
                f"target.api_endpoint.adapter_config.reuse_cached_responses=True,"
                f"target.api_endpoint.adapter_config.save_requests=True,"
                f"target.api_endpoint.adapter_config.max_saved_requests=1,"
                f"target.api_endpoint.adapter_config.html_report_size=5,"
                f"target.api_endpoint.adapter_config.caching_dir={temp_dir}/cache,"
                f"target.api_endpoint.adapter_config.process_reasoning_traces=True,"
                f"target.api_endpoint.adapter_config.use_progress_tracking=True,"
                f"target.api_endpoint.adapter_config.progress_tracking_interval=1,"
                f'target.api_endpoint.adapter_config.params_to_add={{"comprehensive_test": true}},'
                f"target.api_endpoint.adapter_config.tracking_requests_stats=True,"
                f"target.api_endpoint.adapter_config.response_stats_cache_dir={temp_dir}/cache/response_stats_cache,"
                f"target.api_endpoint.adapter_config.generate_html_report=True"
            )
            overrides = dotlist_to_dict(override_str.split(","))

            base_config = {
                "config": {"type": "mmlu_pro", "output_dir": temp_dir},
                "target": {
                    "api_endpoint": {
                        "model_id": "Qwen/Qwen3-8B",
                        "url": "http://localhost:8000/v1/chat/completions",
                        "type": "chat",
                    }
                },
            }

            run_config = deep_update(base_config, overrides, skip_nones=True)

            evaluation = validate_configuration(run_config)

            adapter_config = evaluation.target.api_endpoint.adapter_config
            assert isinstance(adapter_config, AdapterConfig)

            # Critical: mode: client from framework should be preserved
            assert adapter_config.mode == "client", (
                "Framework mode: client should be preserved with comprehensive overrides"
            )

            # Verify all legacy params were converted to interceptors
            interceptor_names = [ic.name for ic in adapter_config.interceptors]
            assert "system_message" in interceptor_names
            assert "request_logging" in interceptor_names
            assert "response_logging" in interceptor_names
            assert "caching" in interceptor_names
            assert "reasoning" in interceptor_names
            assert "response_stats" in interceptor_names
            assert "progress_tracking" in interceptor_names
            assert "payload_modifier" in interceptor_names

            # Verify post-eval hooks were created
            hook_names = [hook.name for hook in adapter_config.post_eval_hooks]
            assert "post_eval_report" in hook_names

            # Verify specific interceptor configs
            caching_interceptor = next(
                ic for ic in adapter_config.interceptors if ic.name == "caching"
            )
            assert caching_interceptor.config.get("reuse_cached_responses") is True
            assert caching_interceptor.config.get("max_saved_requests") == 5


class TestAdapterConfigLegacyConversion:
    """Test from_legacy_config preserves mode correctly."""

    def test_from_legacy_config_preserves_explicit_mode(self):
        """Test that mode is preserved when converting legacy config."""
        legacy_config = {
            "mode": "client",
            "use_caching": True,
            "use_request_logging": True,
        }

        adapter_config = AdapterConfig.from_legacy_config(legacy_config)

        assert adapter_config.mode == "client", "Explicit mode should be preserved"
        assert len(adapter_config.interceptors) > 0
        interceptor_names = [ic.name for ic in adapter_config.interceptors]
        assert "caching" in interceptor_names
        assert "request_logging" in interceptor_names

    def test_from_legacy_config_fills_missing_with_defaults(self):
        """Test that missing fields are filled with defaults."""
        legacy_config = {
            "mode": "client",
            "use_caching": True,
        }

        adapter_config = AdapterConfig.from_legacy_config(legacy_config)

        assert adapter_config.mode == "client"
        assert adapter_config.endpoint_type == "chat"  # From defaults
        assert adapter_config.log_failed_requests is False  # From defaults

    def test_from_legacy_config_does_not_overwrite_existing(self):
        """Test that existing values are not overwritten by defaults."""
        legacy_config = {
            "mode": "client",
            "endpoint_type": "completions",
            "log_failed_requests": True,
            "use_caching": True,
        }

        adapter_config = AdapterConfig.from_legacy_config(legacy_config)

        # Existing values should be preserved
        assert adapter_config.mode == "client"
        assert adapter_config.endpoint_type == "completions"
        assert adapter_config.log_failed_requests is True
