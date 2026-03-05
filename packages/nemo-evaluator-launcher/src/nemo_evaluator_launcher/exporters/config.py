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
"""Config loading and override-parsing utilities for exporters."""

import os
import re
from typing import List

import yaml
from omegaconf import DictConfig, OmegaConf

from nemo_evaluator_launcher.common.logging_utils import logger


def load_export_config_from_file(config_path: str, dest: str) -> DictConfig:
    """Load export configuration from a YAML file.

    Args:
        config_path: Path to the config file.
        dest: Export destination name used to extract destination-specific section.

    Returns:
        OmegaConf object containing the export configuration.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file {config_path} not found")
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    if not isinstance(config_dict, dict):
        raise ValueError(
            f"Config file {config_path} must contain a dictionary at the root level"
        )

    export_config = config_dict.get("export", {})
    if export_config:
        if dest in export_config:
            return OmegaConf.create(export_config[dest])
        else:
            raise ValueError(
                f"Export destination {dest} not found in config file {config_path}"
            )
    else:
        logger.warning(
            f"Config file {config_path} does not have an 'export' section. Using top-level config."
        )
        return OmegaConf.create(config_dict)


def apply_export_overrides(config: DictConfig, dest: str, overrides: List[str]) -> None:
    """Apply Hydra-style overrides to an export config in-place.

    Only overrides matching ``export.<dest>.*`` are applied. The ``~`` prefix
    deletes a key; ``+`` / ``++`` prefixes are accepted but treated as plain
    set operations (matching Hydra semantics for new / forced-override keys).

    Args:
        config: OmegaConf DictConfig to mutate.
        dest: Export destination name (e.g. ``"wandb"``).
        overrides: List of override strings (e.g. ``["export.wandb.entity=org"]``).
    """
    if not overrides:
        return

    hydra_pattern = re.compile(rf"^(?:~|\+\+|\+)?export\.{re.escape(dest)}\.(.+)$")

    to_del: List[str] = []
    filtered_overrides: List[str] = []

    for override in overrides:
        match = hydra_pattern.match(override)
        if not match:
            logger.debug("Ignoring non-applicable override", override=override)
            continue
        rest = match.group(1)
        if override.startswith("~"):
            to_del.append(rest)
            logger.debug("Adding to delete list", config_key=rest, override=override)
            continue
        logger.debug("Adding to update list", config_key=rest, override=override)
        filtered_overrides.append(rest)

    config_updates = OmegaConf.from_dotlist(filtered_overrides)
    if config_updates:
        config.update(config_updates)

    for key_to_remove in to_del:
        *parent_path, leaf_key = key_to_remove.rsplit(".", 1)

        if len(parent_path) == 0:
            config.pop(leaf_key, None)
        elif len(parent_path) == 1:
            parent = OmegaConf.select(config, parent_path[0])
            if parent is not None:
                del parent[leaf_key]
            else:
                logger.warning(
                    f"Could not remove {key_to_remove} from the config (key not found)"
                )
        else:
            raise RuntimeError(f"Invalid key to remove: {key_to_remove}")

    logger.debug("Final config after applying overrides", config=config)
