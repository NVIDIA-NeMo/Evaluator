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

"""Tests for the Lepton executor."""

import pytest
from omegaconf import OmegaConf


class TestLeptonExecutor:
    """Test Lepton executor functionality."""

    def test_lepton_executor_import(self):
        """Test that Lepton executor can be imported conditionally."""
        try:
            from nemo_evaluator_launcher.executors.lepton.executor import LeptonExecutor

            assert LeptonExecutor is not None
        except ImportError:
            pytest.skip("Lepton executor not available (missing leptonai dependency)")

    def test_lepton_deployment_helpers_import(self):
        """Test that Lepton deployment helpers can be imported."""
        try:
            from nemo_evaluator_launcher.executors.lepton.deployment_helpers import (
                create_lepton_endpoint,
                delete_lepton_endpoint,
                get_lepton_endpoint_status,
                wait_for_lepton_endpoint_ready,
            )

            assert create_lepton_endpoint is not None
            assert delete_lepton_endpoint is not None
            assert get_lepton_endpoint_status is not None
            assert wait_for_lepton_endpoint_ready is not None
        except ImportError:
            pytest.skip("Lepton deployment helpers not available")

    def test_lepton_job_helpers_import(self):
        """Test that Lepton job helpers can be imported."""
        try:
            from nemo_evaluator_launcher.executors.lepton.job_helpers import (
                create_lepton_job,
                delete_lepton_job,
                get_lepton_job_status,
            )

            assert create_lepton_job is not None
            assert get_lepton_job_status is not None
            assert delete_lepton_job is not None
        except ImportError:
            pytest.skip("Lepton job helpers not available")

    def test_lepton_config_validation(self):
        """Test Lepton configuration validation."""
        # Test NIM config
        nim_config = {
            "deployment": {
                "type": "nim",
                "image": "nvcr.io/nim/meta/llama-3.1-8b-instruct:1.8.6",
                "served_model_name": "meta/llama-3.1-8b-instruct",
                "lepton_config": {
                    "endpoint_name": "test-endpoint",
                    "resource_shape": "cpu.small",
                },
            }
        }
        cfg = OmegaConf.create(nim_config)
        assert cfg.deployment.type == "nim"
        assert cfg.deployment.lepton_config.endpoint_name == "test-endpoint"

        # Test vLLM config
        vllm_config = {
            "deployment": {
                "type": "vllm",
                "checkpoint_path": "meta-llama/Llama-3.1-8B-Instruct",
                "served_model_name": "llama-3.1-8b-instruct",
                "lepton_config": {
                    "endpoint_name": "test-vllm-endpoint",
                    "resource_shape": "gpu.1xh200",
                },
            }
        }
        cfg = OmegaConf.create(vllm_config)
        assert cfg.deployment.type == "vllm"
        assert cfg.deployment.checkpoint_path == "meta-llama/Llama-3.1-8B-Instruct"

        # Test none deployment config
        none_config = {
            "deployment": {"type": "none"},
            "target": {
                "api_endpoint": {
                    "url": "https://existing-endpoint.lepton.run/v1/chat/completions"
                }
            },
        }
        cfg = OmegaConf.create(none_config)
        assert cfg.deployment.type == "none"
        assert "existing-endpoint.lepton.run" in cfg.target.api_endpoint.url
