# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

from pathlib import Path
from typing import Any, Literal, Optional

import hydra
from hydra.core.global_hydra import GlobalHydra
from jinja2 import Environment
from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from nemo_evaluator_launcher.api.functional import RunConfig
from nemo_evaluator_launcher.common.logging_utils import logger

DEFAULT_READY_MARKERS = ["metadata.json", "config.yaml", "config.json"]
DEFAULT_CHECKPOINT_PATTERNS = ["iter_*", "step_*"]
DEFAULT_INTERVAL = 300

CHECKPOINT_FIELD = "deployment.checkpoint_path"
WATCH_STATE_DIR = Path.home() / ".nemo-evaluator" / "watch-state"


class ClusterConfig(BaseModel):
    """Configuration for a cluster."""

    model_config = ConfigDict(extra="forbid")
    username: str = Field(
        description="Username to use for SSH connections and sbatch submission."
    )
    hostname: str = Field(
        default="localhost",
        description="Hostname of the cluster. Defaults to localhost "
        "(script launched from the cluster login node).",
    )
    account: str = Field(
        description="SLURM account allocation to charge for conversion, deployment and evaluation jobs"
    )
    partition: str = Field(
        default="batch", description="Partition of the cluster. Defaults to batch."
    )
    output_dir: str = Field(
        description="Path to the output directory for storing results.",
    )
    sbatch_extra_flags: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional flags to pass to the sbatch command as '#SBATCH --<key> [\"<value>\"]'.",
    )


class MonitoringConfig(BaseModel):
    """Configuration for monitoring directories for new checkpoints."""

    model_config = ConfigDict(extra="forbid")

    directories: list[str] = Field(
        description="Checkpoint directories to watch for new subdirectories. "
        "Use absolute paths accessible from the cluster login node.",
    )
    interval: Optional[int] = Field(
        default=DEFAULT_INTERVAL,
        description=(
            "Polling interval in seconds between directory scans. "
            "Set to None to scan once and exit."
        ),
    )
    ready_markers: list[str] = Field(
        default_factory=lambda: list(DEFAULT_READY_MARKERS),
        description=(
            "A checkpoint directory is considered ready when ANY of these files exist inside it."
        ),
    )
    checkpoint_patterns: list[str] = Field(
        default_factory=lambda: list(DEFAULT_CHECKPOINT_PATTERNS),
        description=(
            "Glob patterns for checkpoint subdirectory names. "
            "Only directories matching ANY pattern are considered."
        ),
    )
    order: Literal["last", "first"] = Field(
        default="last",
        description=(
            "Processing order for newly discovered checkpoints based on directory name: "
            "'last' processes the highest name (e.g. step_10000) first, "
            "'first' processes the lowest name (e.g. step_1000) first."
        ),
    )

    @field_validator("interval")
    @classmethod
    def interval_must_be_positive(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v <= 0:
            raise ValueError(f"interval must be a positive integer or None, got {v}")
        return v

    @field_validator("ready_markers", "checkpoint_patterns", "directories")
    @classmethod
    def list_must_be_non_empty(cls, v: list) -> list:
        if not v:
            raise ValueError("must contain at least one entry")
        return v


class MountConfig(BaseModel):
    """Configuration for a mount directory."""

    model_config = ConfigDict(extra="forbid")
    target: str = Field(description="Target directory to mount (inside the container).")
    source: str = Field(description="Source directory to mount (on the login node).")


class ConversionConfig(BaseModel):
    """Configuration for a checkpoint conversion job run before evaluation."""

    model_config = ConfigDict(extra="forbid")

    container: str = Field(description="Docker image to use for the conversion job.")
    mounts: list[MountConfig] = Field(
        default_factory=list,
        description="Mount directories to pass to the conversion job.",
    )
    command_pattern: str = Field(
        description=(
            "Jinja2 command template to run inside the container. "
            "Must contain '{{ input_path }}' and '{{ output_path }}' placeholders. "
        )
    )
    command_params: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Extra parameters passed as Jinja2 variables when rendering command_pattern. "
            "Available alongside the required '{{ input_path }}' and '{{ output_path }}' variables."
        ),
    )

    def render_command(self, input_path: str, output_path: str) -> str:
        """Render the command_pattern template with input/output paths and command_params."""
        return (
            Environment()
            .from_string(self.command_pattern)
            .render(
                input_path=input_path,
                output_path=output_path,
                **self.command_params,
            )
        )

    @field_validator("command_pattern")
    @classmethod
    def command_pattern_must_have_placeholders(cls, v: str) -> str:
        # FIXME verify if all the command_params are used in the command_pattern
        # and if there are no {{ xxx }} that are not in command_params
        missing = [p for p in ("{{ input_path }}", "{{ output_path }}") if p not in v]
        if missing:
            raise ValueError(
                f"command_pattern is missing required Jinja2 placeholder(s): {', '.join(missing)}"
            )
        return v


class WatchConfig(BaseModel):
    """Top-level configuration for nel-watch, loadable from a YAML file."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    cluster_config: ClusterConfig = Field(
        description="Configuration for the cluster (used for monitoring, conversion and evaluation jobs).",
    )
    monitoring_config: MonitoringConfig = Field(
        description="Configuration for monitoring directories for new checkpoints.",
    )
    conversion_config: Optional[ConversionConfig] = Field(
        default=None,
        description="Conversion config to run for each discovered checkpoint before evaluation. "
        "If not provided the original checkpoint is used for evaluation.",
    )
    evaluation_configs: list[RunConfig] = Field(
        description="NeMo Evaluator Launcher configs to run for each discovered checkpoint.",
    )

    @model_validator(mode="after")
    def validate_eval_config_structure(self) -> "WatchConfig":
        for i, cfg in enumerate(self.evaluation_configs):
            if "deployment" not in cfg or cfg.get("deployment") is None:
                raise ValueError(
                    f"evaluation_configs[{i}] must have a 'deployment' section"
                )
            if OmegaConf.select(cfg, CHECKPOINT_FIELD, default=None) is not None:
                logger.warning(
                    f"evaluation_configs[{i}] pre-defines '{CHECKPOINT_FIELD}' "
                    f"as watch mode will override it for each discovered checkpoint"
                )
            if "execution" not in cfg or cfg.get("execution") is None:
                raise ValueError(
                    f"evaluation_configs[{i}] must have an 'execution' section"
                )
            if cfg.get("execution", {}).get("output_dir") is not None:
                logger.warning(
                    f"Ignoring evaluation_configs[{i}] pre-define 'execution.output_dir' "
                    f"as watch mode sets it per checkpoint"
                )
            if cfg.get("execution", {}).get("type") != "slurm":
                raise ValueError(
                    f"Only SLURM execution is supported, but found {cfg.get('execution', {}).get('type')} for evaluation_configs[{i}]"
                )
        return self

    @classmethod
    def from_hydra(
        cls, path: Path, overrides: list[str] | None = None
    ) -> "WatchConfig":
        """Load a WatchConfig from a YAML file with optional Hydra overrides.

        The ``evaluation_configs`` field should contain paths to evaluation config YAML files.
        Each path is loaded via :func:`RunConfig.from_hydra` before validation.
        """
        overrides = list(overrides or [])

        eval_overrides = []
        for override in overrides:
            key = override.lstrip("+~").split("=", 1)[0]
            if key.startswith("evaluation_configs") and key != "evaluation_configs":
                eval_overrides.append(override)
        if eval_overrides:
            raise ValueError(
                "Overrides to individual evaluation configs parameters are not supported but found: "
                f"{', '.join(eval_overrides)}. "
                "Edit the evaluation configs directly."
            )

        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()

        config_path = path.expanduser()
        if not config_path.is_absolute():
            config_path = (Path.cwd() / config_path).resolve()

        hydra.initialize_config_dir(
            config_dir=str(config_path.parent), version_base=None
        )
        overrides = overrides + [
            "hydra.searchpath=[pkg://nemo_evaluator_launcher.configs/watcher]"
        ]
        cfg = hydra.compose(config_name=config_path.stem, overrides=overrides)

        cfg = OmegaConf.to_container(cfg, resolve=True)
        raw_eval_configs = cfg.pop("evaluation_configs", [])
        loaded_eval_configs = [
            RunConfig.from_hydra(config=str(p)) for p in raw_eval_configs
        ]
        cfg = cls.model_validate({**cfg, "evaluation_configs": loaded_eval_configs})
        logger.debug(
            "Loaded watch config",
            cluster_config=cfg.cluster_config,
            monitoring_config=cfg.monitoring_config,
            conversion_config=cfg.conversion_config,
            evaluation_configs=cfg.evaluation_configs,
        )

        return cfg
