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
import re
import signal
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml
from omegaconf import OmegaConf

from nemo_evaluator_launcher.api.functional import RunConfig, run_eval
from nemo_evaluator_launcher.common.logging_utils import logger

DEFAULT_READY_MARKERS = ["metadata.json", "config.yaml"]
DEFAULT_CHECKPOINT_PATTERNS = ["iter_*", "step_*"]
DEFAULT_INTERVAL = 300


@dataclass
class WatchDirConfig:
    """Configuration for a single watched directory."""

    checkpoint_dir: Path
    output_dir: Optional[Path] = None
    checkpoint_field: Optional[str] = None


@dataclass
class SubmittedCheckpoint:
    """Record of a submitted checkpoint evaluation."""

    checkpoint: str
    invocation_id: Optional[str]
    timestamp: str
    watch_dir: Optional[str] = None


@dataclass
class WatchState:
    """Persistent state for the watch loop."""

    submitted: list[SubmittedCheckpoint] = field(default_factory=list)

    def submitted_paths(self) -> set[str]:
        return {s.checkpoint for s in self.submitted}

    def save(self, path: Path) -> None:
        data = {
            "submitted": [
                {
                    k: v
                    for k, v in {
                        "checkpoint": s.checkpoint,
                        "invocation_id": s.invocation_id,
                        "timestamp": s.timestamp,
                        "watch_dir": s.watch_dir,
                    }.items()
                    if v is not None
                }
                for s in self.submitted
            ]
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))

    @classmethod
    def load(cls, path: Path) -> "WatchState":
        if not path.exists():
            return cls()
        try:
            data = yaml.safe_load(path.read_text()) or {}
        except Exception:
            logger.warning(f"Failed to load watch state from {path}, starting fresh")
            return cls()
        submitted = []
        for entry in data.get("submitted", []):
            submitted.append(
                SubmittedCheckpoint(
                    checkpoint=entry["checkpoint"],
                    invocation_id=entry.get("invocation_id"),
                    timestamp=entry.get("timestamp", ""),
                    watch_dir=entry.get("watch_dir"),
                )
            )
        return cls(submitted=submitted)


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
        Paths sorted in natural order (oldest first).
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


def _set_nested_value(cfg: "OmegaConf", dotted_key: str, value: str) -> None:
    """Set a value in an OmegaConf config using a dot-separated key path."""
    keys = dotted_key.split(".")
    obj = cfg
    for key in keys[:-1]:
        obj = getattr(obj, key)
    OmegaConf.update(obj, keys[-1], value)


def _load_watch_config(path: Path) -> tuple[list[WatchDirConfig], str]:
    """Load a watch config YAML file.

    Returns:
        Tuple of (watch_dirs, global_checkpoint_field).
    """
    data = yaml.safe_load(path.read_text())
    global_field = data.get("checkpoint_field", "deployment.hf_model_handle")
    watch_dirs = []
    for entry in data.get("watch_dirs", []):
        watch_dirs.append(
            WatchDirConfig(
                checkpoint_dir=Path(entry["checkpoint_dir"]),
                output_dir=Path(entry["output_dir"]) if entry.get("output_dir") else None,
                checkpoint_field=entry.get("checkpoint_field"),
            )
        )
    return watch_dirs, global_field


def watch_checkpoints(
    config: RunConfig,
    watch_dir: Optional[Path] = None,
    watch_dirs: Optional[list[WatchDirConfig]] = None,
    interval: int = DEFAULT_INTERVAL,
    ready_markers: Optional[list[str]] = None,
    checkpoint_patterns: Optional[list[str]] = None,
    checkpoint_field: str = "deployment.hf_model_handle",
    order: str = "newest",
    dry_run: bool = False,
    once: bool = False,
    state_file: Optional[Path] = None,
) -> list[SubmittedCheckpoint]:
    """Watch directories for new checkpoints and submit evaluations.

    Args:
        config: Base evaluation config.
        watch_dir: Single directory to watch (simple mode).
        watch_dirs: List of directory configs to watch (multi-dir mode).
        interval: Polling interval in seconds.
        ready_markers: Files to check for readiness (ANY must exist). Defaults to DEFAULT_READY_MARKERS.
        checkpoint_patterns: Glob patterns for checkpoint dir names. Defaults to DEFAULT_CHECKPOINT_PATTERNS.
        checkpoint_field: Dot-separated config field to override with checkpoint path.
        order: Processing order — "newest" (reverse natural sort) or "oldest" (natural sort).
        dry_run: If True, show what would be submitted without actually submitting.
        once: If True, scan once and exit without looping.
        state_file: Path to the state file. If None, auto-generated.

    Returns:
        List of all submitted checkpoints during this session.
    """
    if ready_markers is None:
        ready_markers = list(DEFAULT_READY_MARKERS)
    if checkpoint_patterns is None:
        checkpoint_patterns = list(DEFAULT_CHECKPOINT_PATTERNS)

    # Build the list of directories to watch
    if watch_dirs is not None:
        dirs_to_watch = watch_dirs
    elif watch_dir is not None:
        dirs_to_watch = [WatchDirConfig(checkpoint_dir=watch_dir)]
    else:
        raise ValueError("Either watch_dir or watch_dirs must be provided.")

    # Determine state file location
    if state_file is None:
        # Try first output_dir from watch_dirs, then config output_dir
        first_output = None
        for d in dirs_to_watch:
            if d.output_dir is not None:
                first_output = d.output_dir
                break
        if first_output is None:
            if hasattr(config, "execution") and hasattr(config.execution, "output_dir"):
                first_output = Path(str(config.execution.output_dir))
            else:
                first_output = Path(".")
        state_file = first_output / ".watch_state.yaml"

    state = WatchState.load(state_file)
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
            for dir_cfg in dirs_to_watch:
                if stop_requested:
                    break

                wd = dir_cfg.checkpoint_dir
                field_override = dir_cfg.checkpoint_field or checkpoint_field

                checkpoints = discover_checkpoints(wd, ready_markers, checkpoint_patterns)

                # Apply ordering
                if order == "newest":
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

                    # Create a copy of the config with the checkpoint path overridden
                    cfg_copy = OmegaConf.create(
                        OmegaConf.to_container(config, resolve=True)
                    )
                    _set_nested_value(cfg_copy, field_override, str(cp))

                    # Override output_dir if specified in watch dir config
                    if dir_cfg.output_dir is not None:
                        _set_nested_value(cfg_copy, "execution.output_dir", str(dir_cfg.output_dir))

                    invocation_id = None
                    if dry_run:
                        logger.info(
                            f"[dry-run] Would submit eval for checkpoint: {cp.name}",
                            checkpoint_field=field_override,
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
                    state.submitted.append(record)
                    session_submissions.append(record)

                    if not dry_run:
                        state.save(state_file)

            if once or stop_requested:
                break

            logger.debug(f"Sleeping {interval}s before next poll")
            elapsed = 0
            while elapsed < interval and not stop_requested:
                time.sleep(min(1, interval - elapsed))
                elapsed += 1

    finally:
        signal.signal(signal.SIGINT, original_handler)

    return session_submissions
