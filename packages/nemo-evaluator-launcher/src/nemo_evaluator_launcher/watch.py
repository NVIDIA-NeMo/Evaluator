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
"""Watch mode: poll checkpoint directories and trigger evaluations for new checkpoints."""

import fnmatch
import json
import re
import signal
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

import yaml
from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from nemo_evaluator_launcher.api.functional import RunConfig, run_eval
from nemo_evaluator_launcher.common.execdb import generate_invocation_id
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
            if cfg.get("deployment", {}).get("checkpoint_path") is not None:
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


@dataclass
class SubmittedCheckpoint:
    """Record of a submitted checkpoint evaluation."""

    checkpoint: str
    invocation_id: str
    timestamp: str
    watch_dir: str


class WatchState:
    """Append-only JSONL log of submitted checkpoint evaluations.

    Each call to :meth:`append` writes one JSON line to ``WATCH_STATE_FILE`` and
    updates the in-memory index, mirroring the pattern used by
    :class:`~nemo_evaluator_launcher.common.execdb.ExecutionDB`.
    """

    def __init__(self, path: Optional[Path] = None) -> None:
        if path is None:
            session_id = generate_invocation_id()
            path = WATCH_STATE_DIR / f"watch-state.{session_id}.v1.jsonl"
        self._path = path
        self._submitted: dict[str, SubmittedCheckpoint] = {}
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            with open(self._path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        self._submitted[record["checkpoint"]] = SubmittedCheckpoint(
                            **record
                        )
                    except json.JSONDecodeError as e:
                        logger.warning("Failed to parse watch state line", error=str(e))
        except OSError as e:
            logger.warning(
                f"Failed to load watch state from {self._path}, starting fresh",
                error=str(e),
            )

    def submitted_paths(self) -> set[str]:
        return set(self._submitted.keys())

    def append(self, record: SubmittedCheckpoint) -> None:
        """Append a submitted checkpoint to the log and update in-memory state."""
        self._submitted[record.checkpoint] = record
        try:
            with open(self._path, "a") as f:
                f.write(json.dumps(record.model_dump()) + "\n")
        except OSError as e:
            logger.error(
                "Failed to write to watch state", path=str(self._path), error=str(e)
            )
            raise


def _natural_sort_key(name: str) -> list:
    """Split a string into text and numeric parts for natural sorting."""
    parts = re.split(r"(\d+)", name)
    result = []
    for part in parts:
        if part.isdigit():
            result.append(int(part))
        else:
            result.append(part.lower())
    return result


def discover_checkpoints(
    watch_dir: Path,
    ready_markers: list[str],
    checkpoint_patterns: Optional[list[str]] = None,
) -> list[Path]:
    """Find checkpoint subdirectories that contain any ready marker file.

    Args:
        watch_dir: Directory to scan for checkpoint subdirectories.
        ready_markers: A checkpoint is ready if ANY of these files exist in it.
        checkpoint_patterns: Glob patterns for checkpoint directory names.
            Only subdirectories matching ANY pattern are considered.
            If None, all subdirectories are candidates.

    Returns:
        Paths sorted in natural order by name (lowest first).
    """
    if not watch_dir.is_dir():
        return []
    checkpoints = []
    for child in watch_dir.iterdir():
        if not child.is_dir():
            continue
        # Check if directory name matches any checkpoint pattern
        if checkpoint_patterns is not None:
            if not any(fnmatch.fnmatch(child.name, pat) for pat in checkpoint_patterns):
                continue
        # Check if any ready marker exists
        if not any((child / marker).exists() for marker in ready_markers):
            continue
        checkpoints.append(child)
    checkpoints.sort(key=lambda p: _natural_sort_key(p.name))
    return checkpoints


def watch_checkpoints(
    watch_config: WatchConfig,
    dry_run: bool = False,
    state_file: Optional[Path] = None,
) -> list[SubmittedCheckpoint]:
    """Watch directories for new checkpoints and submit evaluations.

    Args:
        watch_config: Full watch configuration.
        dry_run: If True, show what would be submitted without actually submitting.
        state_file: Path to the state file. If None, a unique per-session file is created under WATCH_STATE_DIR.

    Returns:
        List of all submitted checkpoints during this session.
    """
    state = WatchState(state_file)
    session_submissions: list[SubmittedCheckpoint] = []

    stop_requested = False

    def _handle_sigint(signum, frame):
        nonlocal stop_requested
        stop_requested = True
        logger.info("Received SIGINT, finishing current cycle...")

    original_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, _handle_sigint)

    try:
        while True:
            for watch_dir_str in watch_config.monitoring_config.directories:
                if stop_requested:
                    break

                wd = Path(watch_dir_str)
                checkpoints = discover_checkpoints(
                    wd,
                    watch_config.monitoring_config.ready_markers,
                    watch_config.monitoring_config.checkpoint_patterns,
                )

                if watch_config.monitoring_config.order == "last":
                    checkpoints = list(reversed(checkpoints))

                already_submitted = state.submitted_paths()
                new_checkpoints = [
                    cp for cp in checkpoints if str(cp) not in already_submitted
                ]

                if new_checkpoints:
                    logger.info(
                        f"Found {len(new_checkpoints)} new checkpoint(s)",
                        watch_dir=str(wd),
                    )
                else:
                    logger.debug("No new checkpoints found", watch_dir=str(wd))

                for cp in new_checkpoints:
                    if stop_requested:
                        break

                    logger.info(f"Processing checkpoint: {cp.name}", path=str(cp))

                    for eval_config in watch_config.eval_configs:
                        cfg_copy = OmegaConf.create(
                            OmegaConf.to_container(eval_config, resolve=True)
                        )
                        OmegaConf.update(cfg_copy, CHECKPOINT_FIELD, str(cp))

                        if dry_run:
                            logger.info(
                                f"[dry-run] Would submit eval for checkpoint: {cp.name}",
                                checkpoint_path=str(cp),
                            )
                        else:
                            try:
                                invocation_id = run_eval(cfg_copy)
                                logger.info(
                                    f"Submitted eval for checkpoint: {cp.name}",
                                    invocation_id=invocation_id,
                                )
                            except Exception as e:
                                logger.error(
                                    f"Failed to submit eval for checkpoint: {cp.name}",
                                    error=str(e),
                                )
                                continue

                            record = SubmittedCheckpoint(
                                checkpoint=str(cp),
                                invocation_id=invocation_id,
                                timestamp=datetime.now(timezone.utc).isoformat(),
                                watch_dir=str(wd),
                            )
                            state.append(record)
                            session_submissions.append(record)

            if watch_config.monitoring_config.interval is None or stop_requested:
                break

            logger.debug(
                f"Sleeping {watch_config.monitoring_config.interval}s before next poll"
            )
            elapsed = 0
            while (
                elapsed < watch_config.monitoring_config.interval and not stop_requested
            ):
                time.sleep(min(1, watch_config.monitoring_config.interval - elapsed))
                elapsed += 1

    finally:
        signal.signal(signal.SIGINT, original_handler)

    return session_submissions
