from pathlib import Path
from typing import Literal, Optional

import yaml
from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from nemo_evaluator_launcher.api.functional import RunConfig
from nemo_evaluator_launcher.common.logging_utils import logger

DEFAULT_READY_MARKERS = ["metadata.json", "config.yaml", "config.json"]
DEFAULT_CHECKPOINT_PATTERNS = ["iter_*", "step_*"]
DEFAULT_INTERVAL = 300

CHECKPOINT_FIELD = "deployment.checkpoint_path"
WATCH_STATE_DIR = Path.home() / ".nemo-evaluator" / "watch-state"


class MonitoringConfig(BaseModel):
    """Configuration for monitoring directories for new checkpoints."""

    model_config = ConfigDict(extra="forbid")

    directories: list[str] = Field(
        description="Checkpoint directories to watch for new subdirectories. "
        "Use absolute local paths or '[user@]hostname:/path' for remote directories.",
    )
    interval: Optional[int] = Field(
        default=DEFAULT_INTERVAL,
        description=(
            "Polling interval in seconds between directory scans. "
            "Set to null to scan once and exit."
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
            raise ValueError(f"interval must be a positive integer or null, got {v}")
        return v

    @field_validator("ready_markers", "checkpoint_patterns", "directories")
    @classmethod
    def list_must_be_non_empty(cls, v: list) -> list:
        if not v:
            raise ValueError("must contain at least one entry")
        return v


class ConversionConfig(BaseModel):
    """Configuration for a checkpoint conversion job run before evaluation."""

    model_config = ConfigDict(extra="forbid")

    container: str = Field(description="Docker image to use for the conversion job.")
    mounts: Optional[dict[str, str]] = Field(
        default=None,
        description="Mount directories (source:target format) to pass to the conversion job.",
    )
    command_pattern: str = Field(
        description=(
            "Command template to run inside the container. "
            "Must contain '{input_path}' and '{output_path}' placeholders. "
            "It can also contain other placeholders that will be populated during runtime."
        )
    )

    @field_validator("command_pattern")
    @classmethod
    def command_pattern_must_have_placeholders(cls, v: str) -> str:
        missing = [p for p in ("{input_path}", "{output_path}") if p not in v]
        if missing:
            raise ValueError(
                f"command_pattern is missing required placeholder(s): {', '.join(missing)}"
            )
        return v


class WatchConfig(BaseModel):
    """Top-level configuration for nel-watch, loadable from a YAML file."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    monitoring_config: MonitoringConfig = Field(
        description="Configuration for monitoring directories for new checkpoints.",
    )
    conversion_config: Optional[ConversionConfig] = Field(
        default=None,
        description="Conversion config to run for each discovered checkpoint before evaluation. "
        "If not provided the original checkpoint is used for evaluation.",
    )
    eval_configs: list[RunConfig] = Field(
        description="Evaluation configs to run for each discovered checkpoint.",
    )

    @model_validator(mode="after")
    def validate_eval_config_structure(self) -> "WatchConfig":
        for i, cfg in enumerate(self.eval_configs):
            if "deployment" not in cfg or cfg.get("deployment") is None:
                raise ValueError(f"eval_configs[{i}] must have a 'deployment' section")
            if OmegaConf.select(cfg, CHECKPOINT_FIELD, default=None) is not None:
                logger.warning(
                    f"eval_configs[{i}] pre-defines '{CHECKPOINT_FIELD}' "
                    f"— watch mode will override it for each discovered checkpoint"
                )
            if "execution" not in cfg or cfg.get("execution") is None:
                raise ValueError(f"eval_configs[{i}] must have an 'execution' section")
            if cfg.get("execution", {}).get("output_dir") is not None:
                raise ValueError(
                    f"eval_configs[{i}] must not pre-define 'execution.output_dir' "
                    f"— watch mode sets it per checkpoint"
                )
        return self

    @classmethod
    def from_yaml(cls, path: Path) -> "WatchConfig":
        """Load a WatchConfig from a YAML file.

        The ``eval_configs`` field should contain paths to evaluation config YAML files.
        Each path is loaded via :func:`RunConfig.from_hydra` before validation.
        """
        data = yaml.safe_load(path.read_text()) or {}
        raw_eval_configs = data.pop("eval_configs", [])
        loaded = [RunConfig.from_hydra(config=str(p)) for p in raw_eval_configs]
        return cls.model_validate({**data, "eval_configs": loaded})
