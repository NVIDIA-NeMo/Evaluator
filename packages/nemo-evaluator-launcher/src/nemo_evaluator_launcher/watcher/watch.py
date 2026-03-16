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

import copy
import fnmatch
import re
import signal
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from omegaconf import OmegaConf

from nemo_evaluator_launcher.api.functional import run_eval
from nemo_evaluator_launcher.common.logging_utils import logger
from nemo_evaluator_launcher.watcher.configs import CHECKPOINT_FIELD, WatchConfig
from nemo_evaluator_launcher.watcher.watchdb import SubmittedCheckpoint, WatchStateDB


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


def watch_and_evaluate(
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
    state = WatchStateDB(state_file)
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

                if not new_checkpoints:
                    logger.debug("No new checkpoints found", watch_dir=str(wd))
                    continue
                logger.info(
                    f"Found {len(new_checkpoints)} new checkpoint(s)", watch_dir=str(wd)
                )

                for cp in new_checkpoints:
                    if stop_requested:
                        break

                    logger.info(f"Processing checkpoint: {cp.name}", path=str(cp))
                    # FIXME: trigger conversion job if defined
                    # this should be done in a non-blocking manner
                    # conversion is sent to the background, and we should have
                    # a concurrent process to wait for the conversion to complete
                    # and then trigger the evaluation
                    # this main blocking loop should only register that the job was submitted
                    # so the checkpoint is not picked up again

                    for eval_config in watch_config.evaluation_configs:
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
                                watch_config=copy.deepcopy(watch_config),
                                eval_config=cfg_copy,
                            )
                            state.append(record)
                            session_submissions.append(record)

            if (
                watch_config.monitoring_config.interval is None
                or stop_requested
                or dry_run
            ):
                break

            logger.debug(
                f"Sleeping {watch_config.monitoring_config.interval}s before next poll"
            )
            # Sleep 1 second at a time so SIGINT is handled promptly.
            for _ in range(watch_config.monitoring_config.interval):
                if stop_requested:
                    break
                time.sleep(1)

    finally:
        signal.signal(signal.SIGINT, original_handler)

    return session_submissions
