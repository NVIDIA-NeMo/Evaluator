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
import shlex
import signal
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from jinja2 import Environment, FileSystemLoader
from omegaconf import OmegaConf

from nemo_evaluator_launcher.api.functional import run_eval
from nemo_evaluator_launcher.api.types import RunConfig
from nemo_evaluator_launcher.common.execdb import generate_invocation_id
from nemo_evaluator_launcher.common.logging_utils import logger
from nemo_evaluator_launcher.common.ssh_utils import (
    close_master_connection,
    make_remote_dir,
    open_master_connection,
    rsync_upload,
)
from nemo_evaluator_launcher.watcher.configs import (
    CHECKPOINT_FIELD,
    ConversionConfig,
    WatchConfig,
)
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


def _submit_conversion_job(
    input_path: Path,
    conversion_config: ConversionConfig,
    dry_run: bool = False,
) -> tuple[Optional[str], str]:
    """Submit a conversion job for a checkpoint.

    Returns:
        Tuple of (slurm_job_id, output_path). slurm_job_id is None in dry-run mode.
    """
    # 1. Resolve output path
    output_path = Path(conversion_config.output_dir) / input_path.name

    # 2. Render the command
    command = conversion_config.command_pattern.format(
        input_path=str(input_path),
        output_path=str(output_path),
    )

    # 3. Build mounts: start from configured mounts, auto-add input and output paths
    mounts_str = ""
    for src, dst in conversion_config.mounts.items():
        mounts_str += f"{src}:{dst},"
    mounts_str += (
        f"{str(input_path)}:{str(input_path)},{str(output_path)}:{str(output_path)}"
    )

    # 4. Extract SSH connection params; remaining keys become #SBATCH directives
    execution_params = dict(conversion_config.execution_params)
    username = execution_params.pop("username")
    hostname = execution_params.pop("hostname")

    # 5. Render sbatch script from Jinja template
    template_dir = Path(__file__).parent
    env = Environment(loader=FileSystemLoader(str(template_dir)))
    template = env.get_template("conversion_sbatch.template")
    sbatch_script = template.render(
        execution_params=execution_params,
        container=conversion_config.container,
        mounts=mounts_str,
        command=command,
        input_path=str(input_path),
        output_path=str(output_path),
    )

    if dry_run:
        logger.info(
            "[dry-run] Would submit conversion job", sbatch_script=sbatch_script
        )
        return None, str(output_path)

    # 6. Submit via SSH
    invocation_id = generate_invocation_id()
    remote_dir = f"/tmp/nel-conversion-{invocation_id}"

    with tempfile.TemporaryDirectory() as tmpdir:
        local_script_dir = Path(tmpdir) / "conversion"
        local_script_dir.mkdir()
        (local_script_dir / "conversion.sh").write_text(sbatch_script)

        socket = str(Path(tmpdir) / "socket")
        socket_or_none = open_master_connection(
            username=username, hostname=hostname, socket=socket
        )
        if socket_or_none is None:
            raise RuntimeError(
                f"Failed to connect to {hostname} as {username}. "
                "Please check your SSH configuration."
            )
        try:
            make_remote_dir(
                dirpath=remote_dir,
                username=username,
                hostname=hostname,
                socket=socket_or_none,
            )
            rsync_upload(
                local_sources=[local_script_dir],
                remote_target=remote_dir,
                username=username,
                hostname=hostname,
            )
            slurm_job_ids = _sbatch_remote(
                remote_script=f"{remote_dir}/conversion/conversion.sh",
                username=username,
                hostname=hostname,
                socket=socket_or_none,
            )
        finally:
            close_master_connection(
                username=username, hostname=hostname, socket=socket_or_none
            )

    job_id = slurm_job_ids[0]
    logger.info(
        "Submitted conversion job",
        job_id=job_id,
        input_path=str(input_path),
        output_path=str(output_path),
    )
    return job_id, str(output_path)


def _sbatch_remote(
    remote_script: str,
    username: str,
    hostname: str,
    socket: str | None,
) -> list[str]:
    """Submit a single sbatch script on a remote host and return the SLURM job ID."""
    ssh_command = ["ssh"]
    if socket is not None:
        ssh_command.append(f"-S {socket}")
    ssh_command.append(f"{username}@{hostname}")
    ssh_command.append(f"sbatch {remote_script}")
    ssh_command_str = " ".join(ssh_command)
    logger.info("Running sbatch", cmd=ssh_command_str)
    completed = subprocess.run(
        args=shlex.split(ssh_command_str),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"failed to submit sbatch script\n{completed.stderr.decode()}"
        )
    stdout = completed.stdout.decode()
    job_ids = re.findall(r"(?<=Submitted batch job )\d+", stdout)
    if not job_ids:
        raise RuntimeError(f"Could not parse SLURM job ID from sbatch output: {stdout}")
    logger.info("Started sbatch successfully", slurm_job_ids=job_ids)
    return job_ids


def _convert_and_evaluate(
    checkpoint: Path,
    conversion_config: ConversionConfig | None,
    evaluation_config: RunConfig,
    dry_run: bool = False,
) -> tuple[str, RunConfig]:
    """Convert a checkpoint and evaluate it."""
    # TODO:
    # 1. trigger the conversion job using _submit_conversion_job function
    # 2. add dependency on the conversion job to the config
    # 3. return invocation_id and the evaluation config with the dependency added

    cfg_copy = OmegaConf.create(OmegaConf.to_container(evaluation_config, resolve=True))

    if conversion_config is not None:
        conversion_slurm_id, converted_checkpoint_path = _submit_conversion_job(
            checkpoint, conversion_config, dry_run=dry_run
        )
        evaluation_config.execution.sbatch_dependency = f"afterok:{conversion_slurm_id}"
        OmegaConf.update(cfg_copy, CHECKPOINT_FIELD, str(converted_checkpoint_path))
    else:
        OmegaConf.update(cfg_copy, CHECKPOINT_FIELD, str(checkpoint))

    if dry_run:
        logger.info(
            f"[dry-run] Would submit eval for checkpoint: {checkpoint.name}",
            checkpoint_path=str(checkpoint),
        )
        return None, cfg_copy
    try:
        invocation_id = run_eval(cfg_copy)
        return invocation_id, cfg_copy
    except Exception as e:
        logger.error(
            f"Failed to submit eval for checkpoint: {checkpoint.name}",
            error=str(e),
        )
        return None, cfg_copy


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

                    for eval_config in watch_config.evaluation_configs:
                        invocation_id, resolved_eval_config = _convert_and_evaluate(
                            cp,
                            watch_config.conversion_config,
                            eval_config,
                            dry_run=dry_run,
                        )
                        if invocation_id is not None:
                            logger.info(
                                f"Submitted eval for checkpoint: {cp.name}",
                                invocation_id=invocation_id,
                            )

                            record = SubmittedCheckpoint(
                                checkpoint=str(cp),
                                invocation_id=invocation_id,
                                timestamp=datetime.now(timezone.utc).isoformat(),
                                watch_config=copy.deepcopy(watch_config),
                                eval_config=resolved_eval_config,
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
