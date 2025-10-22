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
"""Type definitions for the nemo-evaluator-launcher public API.

This module defines data structures and helpers for configuration and type safety in the API layer.
"""

import os
import warnings
from dataclasses import dataclass
from typing import cast

# ruff: noqa: E402
# Later when adding optional module to hydra, since the internal package is optional,
# will generate a hydra warning. We suppress it as distraction and bad UX, before hydra gets invoked.
warnings.filterwarnings(
    "ignore",
    message="provider=hydra.searchpath.*path=nemo_evaluator_launcher_internal.*is not available\\.",
)

import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from nemo_evaluator_launcher.common.logging_utils import logger


@dataclass
class RunConfig(DictConfig):
    @staticmethod
    def from_hydra(
        config_name: str = "default",
        config_dir: str | None = None,
        hydra_overrides: list[str] = [],
        dict_overrides: dict = {},
    ) -> "RunConfig":
        """Load configuration from Hydra and merge with dictionary overrides.

        Args:
            config_name: Name of the Hydra configuration to load.
            hydra_overrides: List of Hydra command-line style overrides.
            dict_overrides: Dictionary of configuration overrides to merge.
            config_dir: Optional path to user config directory. If provided, Hydra will
                       search in this directory first, then fall back to internal configs.

        Returns:
            RunConfig: Merged configuration object.
        """
        overrides = hydra_overrides.copy()
        # Check if a GlobalHydra instance is already initialized and clear it
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()

        if config_dir:
            # Convert relative path to absolute path if needed
            if not os.path.isabs(config_dir):
                config_dir = os.path.abspath(config_dir)

            hydra.initialize_config_dir(
                config_dir=config_dir,
                version_base=None,
            )
        else:
            hydra.initialize_config_module(
                config_module="nemo_evaluator_launcher.configs",
                version_base=None,
            )
        overrides = overrides + [
            "hydra.searchpath=[pkg://nemo_evaluator_launcher.configs,pkg://nemo_evaluator_launcher_internal.configs]"
        ]
        cfg = hydra.compose(config_name=config_name, overrides=overrides)

        # Merge dict_overrides if provided
        if dict_overrides:
            cfg = OmegaConf.merge(cfg, dict_overrides)

        logger.debug(
            "Loaded run config from hydra",
            config_name=config_name,
            config_dir=config_dir,
            overrides=hydra_overrides,
            dict_overrides=dict_overrides,
            result=cfg,
        )
        return cast("RunConfig", cfg)
