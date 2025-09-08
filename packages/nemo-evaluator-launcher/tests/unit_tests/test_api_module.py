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
"""Tests for the API functional module."""

import pytest
from omegaconf import OmegaConf

from nemo_evaluator_launcher.api.functional import (
    _check_api_endpoint_when_deployment_is_configured,
)


class TestApiEndpointConfigurationCheck:
    """Test the _check_api_endpoint_when_deployment_is_configured function."""

    def test_deployment_none_passes(self):
        """Test that when deployment type is 'none', no validation error occurs."""
        cfg = OmegaConf.create(
            {
                "deployment": {"type": "none"},
                "target": {
                    "api_endpoint": {
                        "url": "https://api.example.com",
                        "model_id": "some-model",
                    }
                },
            }
        )

        # Should not raise any exception
        _check_api_endpoint_when_deployment_is_configured(cfg)

    def test_no_target_passes(self):
        """Test that when target is missing, no validation error occurs."""
        cfg = OmegaConf.create({"deployment": {"type": "some_deployment"}})

        # Should not raise any exception
        _check_api_endpoint_when_deployment_is_configured(cfg)

    def test_target_not_dict_passes(self):
        """Test that when target is not a DictConfig, no validation error occurs."""
        cfg = OmegaConf.create(
            {"deployment": {"type": "some_deployment"}, "target": "string_value"}
        )

        # Should not raise any exception
        _check_api_endpoint_when_deployment_is_configured(cfg)

    def test_no_api_endpoint_passes(self):
        """Test that when api_endpoint is missing, no validation error occurs."""
        cfg = OmegaConf.create(
            {
                "deployment": {"type": "some_deployment"},
                "target": {"other_config": "value"},
            }
        )

        # Should not raise any exception
        _check_api_endpoint_when_deployment_is_configured(cfg)

    def test_api_endpoint_not_dict_passes(self):
        """Test that when api_endpoint is not a DictConfig, no validation error occurs."""
        cfg = OmegaConf.create(
            {
                "deployment": {"type": "some_deployment"},
                "target": {"api_endpoint": "string_value"},
            }
        )

        # Should not raise any exception
        _check_api_endpoint_when_deployment_is_configured(cfg)

    def test_url_field_with_deployment_raises_error(self):
        """Test that url field in api_endpoint raises ValueError when deployment is configured."""
        cfg = OmegaConf.create(
            {
                "deployment": {"type": "kubernetes"},
                "target": {
                    "api_endpoint": {
                        "url": "https://api.example.com",
                        "api_key": "secret",
                    }
                },
            }
        )

        with pytest.raises(
            ValueError,
            match="when deployment is configured, url field should not exist in target.api_endpoint",
        ):
            _check_api_endpoint_when_deployment_is_configured(cfg)

    def test_model_id_field_with_deployment_raises_error(self):
        """Test that model_id field in api_endpoint raises ValueError when deployment is configured."""
        cfg = OmegaConf.create(
            {
                "deployment": {"type": "slurm"},
                "target": {"api_endpoint": {"model_id": "gpt-4", "api_key": "secret"}},
            }
        )

        with pytest.raises(
            ValueError,
            match="when deployment is configured, model_id field should not exist in target.api_endpoint",
        ):
            _check_api_endpoint_when_deployment_is_configured(cfg)

    def test_both_url_and_model_id_with_deployment_raises_error(self):
        """Test that having both url and model_id fields raises ValueError (url checked first)."""
        cfg = OmegaConf.create(
            {
                "deployment": {"type": "lepton"},
                "target": {
                    "api_endpoint": {
                        "url": "https://api.example.com",
                        "model_id": "gpt-4",
                        "api_key": "secret",
                    }
                },
            }
        )

        # Should raise error for url field (checked first)
        with pytest.raises(
            ValueError,
            match="when deployment is configured, url field should not exist in target.api_endpoint",
        ):
            _check_api_endpoint_when_deployment_is_configured(cfg)

    def test_valid_api_endpoint_with_deployment_passes(self):
        """Test that valid api_endpoint configuration with deployment passes."""
        cfg = OmegaConf.create(
            {
                "deployment": {"type": "kubernetes"},
                "target": {
                    "api_endpoint": {
                        "api_key": "secret",
                        "timeout": 30,
                        "max_retries": 3,
                    }
                },
            }
        )

        # Should not raise any exception
        _check_api_endpoint_when_deployment_is_configured(cfg)

    def test_empty_api_endpoint_with_deployment_passes(self):
        """Test that empty api_endpoint configuration with deployment passes."""
        cfg = OmegaConf.create(
            {"deployment": {"type": "local"}, "target": {"api_endpoint": {}}}
        )

        # Should not raise any exception
        _check_api_endpoint_when_deployment_is_configured(cfg)

    def test_deployment_none_with_url_and_model_id_passes(self):
        """Test that url and model_id are allowed when deployment type is 'none'."""
        cfg = OmegaConf.create(
            {
                "deployment": {"type": "none"},
                "target": {
                    "api_endpoint": {
                        "url": "https://api.openai.com",
                        "model_id": "gpt-4-turbo",
                        "api_key": "secret",
                    }
                },
            }
        )

        # Should not raise any exception
        _check_api_endpoint_when_deployment_is_configured(cfg)
