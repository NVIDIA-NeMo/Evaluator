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

"""Tests for configuration merging logic, specifically adapter_config merging."""

import pytest

from nemo_evaluator.adapters.adapter_config import AdapterConfig
from nemo_evaluator.api.api_dataclasses import (
    ApiEndpoint,
    EvaluationConfig,
    EvaluationTarget,
)
from nemo_evaluator.core.input import get_evaluation


class TestAdapterConfigMerge:
    """Test that adapter_config merges correctly preserve framework defaults."""

    def test_framework_mode_preserved_with_user_legacy_params(self, monkeypatch):
        """Test that framework's mode: client is preserved when user adds legacy params."""
        # Mock framework defaults with mode: client
        mock_framework_defaults = {
            "simple_evals": {
                "framework_name": "simple_evals",
                "pkg_name": "simple_evals",
                "command": "echo test",  # Required field
                "config": {"type": "test_task", "params": {"task": "test_task"}},
                "target": {
                    "api_endpoint": {
                        "type": "chat",
                        "adapter_config": {"mode": "client"},  # Framework default
                    }
                },
            }
        }

        mock_framework_eval_mappings = {"simple_evals": {}}
        mock_eval_name_mapping = {}

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

        # User config with legacy parameters but no mode specified
        eval_config = EvaluationConfig(
            type="simple_evals.test_task", params={"task": "test_task"}
        )
        target_config = EvaluationTarget(
            api_endpoint=ApiEndpoint(
                model_id="test-model",
                url="http://localhost:8000",
                type="chat",
                adapter_config=AdapterConfig(
                    # User only sets legacy params via overrides
                    # These would come from CLI: target.api_endpoint.adapter_config.use_caching=True
                    interceptors=[],  # Empty means it will be converted from legacy
                ),
            )
        )

        # Get the merged evaluation
        evaluation = get_evaluation(eval_config, target_config)

        # Check that mode: client from framework defaults is preserved
        assert evaluation.target.api_endpoint.adapter_config is not None
        adapter_config_dict = evaluation.target.api_endpoint.adapter_config
        if isinstance(adapter_config_dict, AdapterConfig):
            adapter_config_dict = adapter_config_dict.model_dump()

        assert (
            adapter_config_dict.get("mode") == "client"
        ), "Framework default mode: client should be preserved"

    def test_user_mode_overrides_framework_default(self, monkeypatch):
        """Test that user's explicit mode setting overrides framework default."""
        # Mock framework defaults with mode: client
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

        mock_framework_eval_mappings = {"simple_evals": {}}
        mock_eval_name_mapping = {}

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

        # User explicitly sets mode: server
        eval_config = EvaluationConfig(
            type="simple_evals.test_task", params={"task": "test_task"}
        )
        target_config = EvaluationTarget(
            api_endpoint=ApiEndpoint(
                model_id="test-model",
                url="http://localhost:8000",
                type="chat",
                adapter_config=AdapterConfig(
                    mode="server",  # User explicitly sets mode
                    interceptors=[],
                ),
            )
        )

        # Get the merged evaluation
        evaluation = get_evaluation(eval_config, target_config)

        # Check that user's explicit mode: server overrides framework default
        adapter_config_dict = evaluation.target.api_endpoint.adapter_config
        if isinstance(adapter_config_dict, AdapterConfig):
            adapter_config_dict = adapter_config_dict.model_dump()

        assert (
            adapter_config_dict.get("mode") == "server"
        ), "User's explicit mode: server should override framework default"

    def test_no_framework_defaults_uses_user_config(self, monkeypatch):
        """Test that when no framework defaults exist, user config is used."""
        mock_framework_defaults = {
            "simple_evals": {
                "framework_name": "simple_evals",
                "pkg_name": "simple_evals",
                "command": "echo test",
                "config": {"type": "test_task", "params": {"task": "test_task"}},
                "target": {"api_endpoint": {"type": "chat"}},  # No adapter_config
            }
        }

        mock_framework_eval_mappings = {"simple_evals": {}}
        mock_eval_name_mapping = {}

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

        eval_config = EvaluationConfig(
            type="simple_evals.test_task", params={"task": "test_task"}
        )
        target_config = EvaluationTarget(
            api_endpoint=ApiEndpoint(
                model_id="test-model",
                url="http://localhost:8000",
                type="chat",
                adapter_config=AdapterConfig(mode="client", interceptors=[]),
            )
        )

        evaluation = get_evaluation(eval_config, target_config)

        adapter_config_dict = evaluation.target.api_endpoint.adapter_config
        if isinstance(adapter_config_dict, AdapterConfig):
            adapter_config_dict = adapter_config_dict.model_dump()

        assert (
            adapter_config_dict.get("mode") == "client"
        ), "User config should be used when no framework defaults"

    def test_framework_defaults_with_multiple_adapter_params(self, monkeypatch):
        """Test merging when framework has multiple adapter config parameters."""
        mock_framework_defaults = {
            "simple_evals": {
                "framework_name": "simple_evals",
                "pkg_name": "simple_evals",
                "command": "echo test",
                "config": {"type": "test_task", "params": {"task": "test_task"}},
                "target": {
                    "api_endpoint": {
                        "type": "chat",
                        "adapter_config": {
                            "mode": "client",
                            "endpoint_type": "chat",
                            "log_failed_requests": True,
                        },
                    }
                },
            }
        }

        mock_framework_eval_mappings = {"simple_evals": {}}
        mock_eval_name_mapping = {}

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

        eval_config = EvaluationConfig(
            type="simple_evals.test_task", params={"task": "test_task"}
        )
        target_config = EvaluationTarget(
            api_endpoint=ApiEndpoint(
                model_id="test-model",
                url="http://localhost:8000",
                type="chat",
                adapter_config=AdapterConfig(
                    log_failed_requests=False,  # User overrides this
                    interceptors=[],
                ),
            )
        )

        evaluation = get_evaluation(eval_config, target_config)

        adapter_config_dict = evaluation.target.api_endpoint.adapter_config
        if isinstance(adapter_config_dict, AdapterConfig):
            adapter_config_dict = adapter_config_dict.model_dump()

        # Framework defaults should be preserved except for user overrides
        assert adapter_config_dict.get("mode") == "client", "Framework mode preserved"
        assert (
            adapter_config_dict.get("endpoint_type") == "chat"
        ), "Framework endpoint_type preserved"
        assert (
            adapter_config_dict.get("log_failed_requests") is False
        ), "User override applied"

    def test_exclude_unset_preserves_framework_defaults(self):
        """Test that model_dump(exclude_unset=True) correctly excludes Pydantic defaults."""
        # Create an AdapterConfig with only some fields set
        config = AdapterConfig(log_failed_requests=True)

        # exclude_unset=True should only include explicitly set fields
        explicit_fields = config.model_dump(exclude_unset=True)

        assert "log_failed_requests" in explicit_fields
        assert explicit_fields["log_failed_requests"] is True

        # mode was not explicitly set, so it should not be in explicit_fields
        assert "mode" not in explicit_fields, "Pydantic default 'mode' should be excluded"

        # Full dump includes all fields including defaults
        all_fields = config.model_dump()
        assert (
            all_fields["mode"] == "server"
        ), "Full dump includes Pydantic default mode"


class TestAdapterConfigLegacyMerge:
    """Test that from_legacy_config correctly preserves mode during conversion."""

    def test_from_legacy_config_preserves_mode(self):
        """Test that mode: client is preserved when converting from legacy config."""
        legacy_config = {
            "mode": "client",
            "use_caching": True,
            "use_request_logging": True,
        }

        adapter_config = AdapterConfig.from_legacy_config(legacy_config)

        assert (
            adapter_config.mode == "client"
        ), "mode: client should be preserved from legacy config"
        assert len(adapter_config.interceptors) > 0, "Legacy params should be converted"

    def test_from_legacy_config_fills_missing_defaults(self):
        """Test that from_legacy_config fills missing keys with defaults."""
        legacy_config = {
            "mode": "client",
            "use_caching": True,
        }

        adapter_config = AdapterConfig.from_legacy_config(legacy_config)

        assert adapter_config.mode == "client", "Explicit mode preserved"
        assert (
            adapter_config.endpoint_type == "chat"
        ), "Missing endpoint_type filled with default"
        assert (
            adapter_config.log_failed_requests is False
        ), "Missing log_failed_requests filled with default"

    def test_from_legacy_config_no_mode_uses_default(self):
        """Test that when no mode is specified, default is used."""
        legacy_config = {
            "use_caching": True,
        }

        adapter_config = AdapterConfig.from_legacy_config(legacy_config)

        # When mode is not specified, the default from get_legacy_defaults() is used
        defaults = AdapterConfig.get_legacy_defaults()
        assert (
            adapter_config.mode == defaults["mode"]
        ), "Should use default mode when not specified"

