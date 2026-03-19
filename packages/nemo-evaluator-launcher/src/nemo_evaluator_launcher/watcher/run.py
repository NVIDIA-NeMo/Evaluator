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

import re
import shlex
import signal
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
    master_connection,
    rsync_upload,
    run_remote_command,
)
from nemo_evaluator_launcher.watcher.configs import (
    CHECKPOINT_FIELD,
    ClusterConfig,
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
    cluster_config: ClusterConfig,
    ready_markers: list[str],
    checkpoint_patterns: list[str],
    socket: Optional[str] = None,
) -> list[Path]:
    """Find checkpoint subdirectories that contain any ready marker file.

    Args:
        watch_dir: Directory to scan for checkpoint subdirectories.
        cluster_config: Provides SSH connection details for remote discovery.
        ready_markers: A checkpoint is ready if ANY of these files exist in it.
        checkpoint_patterns: Glob patterns for checkpoint directory names.
            Only subdirectories matching ANY pattern are considered.
        socket: Optional path to a multiplexing socket for connection reuse.

    Returns:
        Paths sorted in natural order by name (lowest first).
    """
    # Build the find command, optionally filtering by checkpoint name patterns.
    name_exprs = " -o ".join(f"-name {shlex.quote(p)}" for p in checkpoint_patterns)
    find_cmd = (
        f"find {shlex.quote(str(watch_dir))} -maxdepth 1 -mindepth 1 -type d"
        f" \\( {name_exprs} \\)"
    )

    # Pipe into a loop that checks for any ready marker.
    marker_tests = " || ".join(f'[ -f "$d/{m}" ]' for m in ready_markers)
    remote_cmd = (
        f"{find_cmd} 2>/dev/null"
        f" | while IFS= read -r d; do"
        f' if {marker_tests}; then echo "$d"; fi;'
        f" done"
    )

    try:
        output = run_remote_command(
            command=remote_cmd,
            username=cluster_config.username,
            hostname=cluster_config.hostname,
            socket=socket,
        )
    except RuntimeError as e:
        logger.warning(f"Failed to discover checkpoints in {watch_dir}", error=str(e))
        return []

    paths = [Path(line.strip()) for line in output.splitlines() if line.strip()]
    paths.sort(key=lambda p: _natural_sort_key(p.name))
    return paths


def _submit_conversion_job(
    *,
    input_path: Path,
    output_dir: Path,
    conversion_config: ConversionConfig,
    cluster_config: ClusterConfig,
    dry_run: bool = False,
) -> tuple[Optional[str], str]:
    """Submit a conversion job for a checkpoint.

    Returns:
        Tuple of (slurm_job_id, output_path). slurm_job_id is None in dry-run mode.
    """
    # 1. Resolve output path

    output_path = output_dir / "conversion" / "converted_checkpoint"
    logs_path = output_dir / "conversion" / "logs"

    # 2. Render the command
    command = conversion_config.render_command(
        input_path=str(input_path),
        output_path=str(output_path),
    )

    # 3. Build mounts: start from configured mounts, auto-add input and output paths
    mounts_str = ""
    for mount in conversion_config.mounts:
        src = mount.source
        dst = mount.target
        mounts_str += f"{src}:{dst},"
    mounts_str += f"{input_path}:{input_path}:ro,{output_path}:{output_path}"

    # 4. Render sbatch script from Jinja template
    template_dir = Path(__file__).parent
    env = Environment(loader=FileSystemLoader(str(template_dir)))
    template = env.get_template("conversion_sbatch.template")
    execution_params = {
        "account": cluster_config.account,
        "partition": cluster_config.partition,
        "output": str(logs_path / "slurm-%A.log"),
    }
    for flag, value in cluster_config.sbatch_extra_flags.items():
        # skip flags that are False - we just should not set them
        if isinstance(value, bool) and value is False:
            continue
        execution_params[flag] = value
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

    # 5. Submit via SSH

    with tempfile.TemporaryDirectory() as tmpdir:
        local_script_dir = Path(tmpdir) / "conversion"
        local_script_dir.mkdir()
        (local_script_dir / "run.sub").write_text(sbatch_script)

        with master_connection(
            username=cluster_config.username,
            hostname=cluster_config.hostname,
        ) as socket:
            run_remote_command(
                command=f"mkdir -p {output_path}",
                username=cluster_config.username,
                hostname=cluster_config.hostname,
                socket=socket,
            )
            run_remote_command(
                command=f"mkdir -p {logs_path}",
                username=cluster_config.username,
                hostname=cluster_config.hostname,
                socket=socket,
            )
            rsync_upload(
                local_sources=[local_script_dir],
                remote_target=str(output_dir),
                username=cluster_config.username,
                hostname=cluster_config.hostname,
            )
            stdout = run_remote_command(
                command=f"sbatch --parsable {str(output_dir / 'conversion' / 'run.sub')}",
                username=cluster_config.username,
                hostname=cluster_config.hostname,
                socket=socket,
            )
            job_id = stdout.strip().split(";")[0]
            if not job_id:
                raise RuntimeError(
                    f"Could not parse SLURM job ID from sbatch output: {stdout!r}"
                )

    logger.info(
        "Submitted conversion job",
        job_id=job_id,
        input_path=str(input_path),
        output_path=str(output_path),
    )
    return job_id, str(output_path)


def _convert_and_evaluate(
    *,
    output_dir: Path,
    checkpoint: Path,
    conversion_config: ConversionConfig | None,
    evaluation_config: RunConfig,
    cluster_config: ClusterConfig,
    dry_run: bool = False,
) -> tuple[str, RunConfig]:
    """Convert a checkpoint and evaluate it."""

    cfg_copy = OmegaConf.create(OmegaConf.to_container(evaluation_config, resolve=True))

    job_dir = output_dir / checkpoint.name
    if conversion_config is not None:
        conversion_slurm_id, converted_checkpoint_path = _submit_conversion_job(
            output_dir=job_dir,
            input_path=checkpoint,
            conversion_config=conversion_config,
            cluster_config=cluster_config,
            dry_run=dry_run,
        )
        OmegaConf.update(
            cfg_copy,
            "execution.sbatch_dependency",
            f"afterok:{conversion_slurm_id}",
        )
        OmegaConf.update(cfg_copy, CHECKPOINT_FIELD, str(converted_checkpoint_path))
    else:
        OmegaConf.update(cfg_copy, CHECKPOINT_FIELD, str(checkpoint))

    # TODO(martas): do we want to update served_model_name too?

    cfg_copy = OmegaConf.merge(
        cfg_copy,
        OmegaConf.create(
            # TODO(martas): we need to decide if the cluster config should be shared or not
            # if yes, pass other flags to execution section; check for supported ones and pass
            # the rest as sbatch_extra_flags.
            # if no, move all but basic params to conversion_config.
            {
                "execution": {
                    "hostname": cluster_config.hostname,
                    "account": cluster_config.account,
                    "partition": cluster_config.partition,
                    "username": cluster_config.username,
                    "output_dir": str(job_dir / "evaluation"),
                }
            }
        ),
    )

    if dry_run:
        logger.info(
            f"[dry-run] Would submit eval for checkpoint: {checkpoint.name}",
            checkpoint_path=str(checkpoint),
        )

    try:
        invocation_id = run_eval(cfg_copy, dry_run=dry_run)
        if dry_run:
            return None, cfg_copy
        return invocation_id, cfg_copy
    except Exception as e:
        logger.error(
            f"Failed to submit eval for checkpoint: {checkpoint.name}",
            error=str(e),
        )
        return None, cfg_copy


def watch_and_evaluate(
    watch_config: WatchConfig,
    resubmit_previous_sessions: bool = False,
    dry_run: bool = False,
) -> list[SubmittedCheckpoint]:
    """Watch directories for new checkpoints and submit evaluations.

    Args:
        watch_config: Full watch configuration.
        resubmit_previous_sessions: If True, resubmit checkpoints evaluated during previous sessions.
        dry_run: If True, show what would be submitted without actually submitting.

    Returns:
        List of all submitted checkpoints during this session.
    """

    state = WatchStateDB()
    session_id = generate_invocation_id()
    session_submissions: list[SubmittedCheckpoint] = []
    logger.debug(
        "Starting watch and evaluate",
        resubmit_previous_sessions=resubmit_previous_sessions,
        dry_run=dry_run,
        session_id=session_id,
    )

    stop_requested = False

    def _handle_sigint(signum, frame):
        nonlocal stop_requested
        stop_requested = True
        logger.info("Received SIGINT, finishing current cycle...")

    original_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, _handle_sigint)

    # FIXME improve code structure
    try:
        with master_connection(
            username=watch_config.cluster_config.username,
            hostname=watch_config.cluster_config.hostname,
        ) as socket:
            while True:
                for watch_dir_str in watch_config.monitoring_config.directories:
                    if stop_requested:
                        break

                    wd = Path(watch_dir_str)
                    checkpoints = discover_checkpoints(
                        wd,
                        watch_config.cluster_config,
                        watch_config.monitoring_config.ready_markers,
                        watch_config.monitoring_config.checkpoint_patterns,
                        socket=socket,
                    )

                    if watch_config.monitoring_config.order == "last":
                        checkpoints = list(reversed(checkpoints))

                    logger.debug(
                        "Found checkpoints",
                        num_checkpoints=len(checkpoints),
                        watch_dir=str(wd),
                    )

                    session_ids = [session_id] if resubmit_previous_sessions else None

                    already_submitted = state.submitted_paths(session_ids=session_ids)
                    logger.debug(
                        "Already submitted checkpoints",
                        num_already_submitted=len(already_submitted),
                        this_session_only=not resubmit_previous_sessions,
                    )
                    new_checkpoints = [
                        cp for cp in checkpoints if str(cp) not in already_submitted
                    ]

                    if not new_checkpoints:
                        logger.debug("No new checkpoints found", watch_dir=str(wd))
                        continue
                    logger.info(
                        f"Found {len(new_checkpoints)} new checkpoint(s)",
                        watch_dir=str(wd),
                    )

                    for cp in new_checkpoints:
                        if stop_requested:
                            break

                        logger.info(f"Processing checkpoint: {cp.name}", path=str(cp))

                        for eval_config in watch_config.evaluation_configs:
                            invocation_id, resolved_eval_config = _convert_and_evaluate(
                                output_dir=Path(watch_config.cluster_config.output_dir)
                                / session_id,
                                checkpoint=cp,
                                conversion_config=watch_config.conversion_config,
                                evaluation_config=eval_config,
                                cluster_config=watch_config.cluster_config,
                                dry_run=dry_run,
                            )
                            if invocation_id is not None:
                                logger.info(
                                    f"Submitted eval for checkpoint: {cp.name}",
                                    invocation_id=invocation_id,
                                )

                                record = SubmittedCheckpoint(
                                    checkpoint=str(cp),
                                    session_id=session_id,
                                    invocation_id=invocation_id,
                                    timestamp=datetime.now(timezone.utc).isoformat(),
                                    watch_config={
                                        **watch_config.model_dump(
                                            exclude={"evaluation_configs"}
                                        ),
                                        "evaluation_configs": [
                                            OmegaConf.to_container(cfg, resolve=True)
                                            for cfg in watch_config.evaluation_configs
                                        ],
                                    },
                                    eval_config=OmegaConf.to_container(
                                        resolved_eval_config, resolve=True
                                    ),
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
