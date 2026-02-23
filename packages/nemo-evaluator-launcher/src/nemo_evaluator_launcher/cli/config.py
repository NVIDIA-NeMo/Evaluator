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
"""CLI config subcommands for managing persistent settings."""

from __future__ import annotations

from dataclasses import dataclass

from simple_parsing import field

from nemo_evaluator.config import (
    CONFIG_FILE,
    TelemetryLevel,
    load_config,
    save_config,
)
from nemo_evaluator.telemetry import get_telemetry_level
from nemo_evaluator_launcher.common.logging_utils import logger


@dataclass
class SetCmd:
    """Set a config value."""

    key: str = field(positional=True, metavar="KEY")
    value: str = field(positional=True, metavar="VALUE")

    def execute(self) -> None:
        if self.key == "telemetry.level":
            try:
                level = int(self.value)
                TelemetryLevel(level)  # validate
            except (ValueError, KeyError):
                logger.error(
                    f"Invalid telemetry level '{self.value}'. Must be 0, 1, or 2."
                )
                return
            cfg = load_config()
            cfg.telemetry.level = TelemetryLevel(level)
            save_config(cfg)
            label = TelemetryLevel(level).name
            logger.info(
                f"Telemetry level set to {level} ({label})",
                path=str(CONFIG_FILE),
            )
        else:
            logger.error(f"Unknown config key '{self.key}'.")


@dataclass
class GetCmd:
    """Get the effective value of a config key."""

    key: str = field(positional=True, metavar="KEY")

    def execute(self) -> None:
        if self.key == "telemetry.level":
            level = get_telemetry_level()
            logger.info(f"{level.value} ({level.name})")
        else:
            logger.error(f"Unknown config key '{self.key}'.")


@dataclass
class ShowCmd:
    """Show the full config file."""

    def execute(self) -> None:
        if not CONFIG_FILE.exists():
            logger.info(f"No config file found at {CONFIG_FILE}")
            return
        logger.info(CONFIG_FILE.read_text().rstrip())
