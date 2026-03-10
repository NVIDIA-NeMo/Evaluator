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
"""Tests for the select_not_null resolver and vllm deployment command construction."""

import pytest
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from nemo_evaluator_launcher.api.types import RunConfig

# NOTE(martas): import to register the resolver
from nemo_evaluator_launcher.configs import select_not_null  # noqa: F401


class TestSelectNotNullResolver:
    """Tests for the select_not_null custom OmegaConf resolver."""

    def test_string_value_returned_as_is(self):
        cfg = OmegaConf.create(
            {
                "some_config": {"some_field": "some_value"},
                "result": "${select_not_null:some_config.some_field,default_value}",
            }
        )
        assert cfg.result == "some_value"

    def test_null_value_returns_default(self):
        cfg = OmegaConf.create(
            {
                "some_config": {"some_field": None},
                "result": "${select_not_null:some_config.some_field,default_value}",
            }
        )
        assert cfg.result == "default_value"

    def test_missing_key_returns_default(self):
        cfg = OmegaConf.create(
            {
                "some_config": {"some_other_field": "some_value"},
                "result": "${select_not_null:some_config.some_field,default_value}",
            }
        )
        assert cfg.result == "default_value"


class TestVllmDeploymentCommand:
    """Tests for the vllm deployment command string construction."""

    @pytest.fixture(autouse=True)
    def clear_hydra(self):
        GlobalHydra.instance().clear()
        yield
        GlobalHydra.instance().clear()

    def _load_vllm_cfg(self, checkpoint_path, **kwargs):
        """Load config via Hydra with deployment=vllm, the same way nel run does."""
        overrides = [
            "deployment=vllm",
            "deployment.served_model_name=test-model",
            f"deployment.checkpoint_path={checkpoint_path}",
        ]
        for key, value in kwargs.items():
            if value is None:
                value = "null"
            overrides.append(f"++deployment.{key}={value}")
        return RunConfig.from_hydra(hydra_overrides=overrides)

    def test_checkpoint_path_set_hf_handle_missing_uses_checkpoint(self):
        """hf_model_handle absent from config → command falls back to /checkpoint."""
        cfg = self._load_vllm_cfg(checkpoint_path="/my/checkpoint")
        assert cfg.deployment.command.startswith("vllm serve /checkpoint")

    def test_checkpoint_path_set_hf_handle_null_uses_checkpoint(self):
        """hf_model_handle: null → command falls back to /checkpoint."""
        cfg = self._load_vllm_cfg(
            checkpoint_path="/my/checkpoint", hf_model_handle=None
        )
        assert cfg.deployment.command.startswith("vllm serve /checkpoint")

    def test_both_set_hf_handle_takes_precedence(self):
        """Both checkpoint_path and hf_model_handle set → hf_model_handle used."""
        cfg = self._load_vllm_cfg(
            checkpoint_path="/my/checkpoint",
            hf_model_handle="meta-llama/Llama-3.1-8B",
        )
        command = cfg.deployment.command
        assert "meta-llama/Llama-3.1-8B" in command
        assert command.startswith("vllm serve meta-llama/Llama-3.1-8B")
