# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""
Global configuration for NeMo Evaluator.

Manages persistent settings stored in $XDG_CONFIG_HOME/nemo-evaluator/config.yaml.
"""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path

import yaml
from pydantic import BaseModel

from nemo_evaluator.logging import get_logger

logger = get_logger(__name__)

_xdg_config_home = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
CONFIG_FILE = _xdg_config_home / "nemo-evaluator" / "config.yaml"


class TelemetryLevel(int, Enum):
    """Telemetry reporting levels."""

    OFF = 0  # No reporting
    MINIMAL = 1  # Usage metrics
    DEFAULT = 2  # Usage metrics + model_id


class TelemetryConfig(BaseModel):
    """Telemetry section of the config file."""

    level: TelemetryLevel = TelemetryLevel.DEFAULT


class NemoEvaluatorConfig(BaseModel):
    """Root schema."""

    telemetry: TelemetryConfig = TelemetryConfig()


def load_config(path: Path | None = None) -> NemoEvaluatorConfig:
    """Read and validate the config file.

    Args:
        path: Config file path. Defaults to CONFIG_FILE.

    Returns defaults if the file doesn't exist.

    Raises:
        ValueError: If the config file exists but contains invalid data.
    """
    config_path = path or CONFIG_FILE
    if not config_path.exists():
        return NemoEvaluatorConfig()
    raw = yaml.safe_load(config_path.read_text()) or {}
    return NemoEvaluatorConfig.model_validate(raw)


def save_config(config: NemoEvaluatorConfig, path: Path | None = None) -> None:
    """Write config to the config file.

    Args:
        config: Config object to persist.
        path: Config file path. Defaults to CONFIG_FILE.
    """
    config_path = path or CONFIG_FILE
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        yaml.safe_dump(
            config.model_dump(mode="json"), default_flow_style=False, sort_keys=False
        )
    )
