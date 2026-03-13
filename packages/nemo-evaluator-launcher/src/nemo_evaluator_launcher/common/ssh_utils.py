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
"""SSH/rsync utilities for interacting with remote SLURM clusters."""

import shlex
import subprocess
from pathlib import Path
from typing import List, Optional

from nemo_evaluator_launcher.common.logging_utils import logger


def open_master_connection(
    username: str,
    hostname: str,
    socket: str,
) -> str | None:
    """Open a persistent multiplexed SSH master connection.

    Args:
        username: SSH username.
        hostname: SSH hostname.
        socket: Path to the Unix socket file for connection multiplexing.

    Returns:
        The socket path on success, None on failure.
    """
    ssh_command = f"ssh -MNf -S {socket} {username}@{hostname}"
    logger.info("Opening master connection", cmd=ssh_command)
    completed_process = subprocess.run(args=shlex.split(ssh_command))
    if completed_process.returncode == 0:
        logger.info("Opened master connection successfully", cmd=ssh_command)
        return socket
    logger.error("Failed to open master connection", code=completed_process.returncode)
    return None


def close_master_connection(
    username: str,
    hostname: str,
    socket: str | None,
) -> None:
    """Close a previously opened SSH master connection.

    Args:
        username: SSH username.
        hostname: SSH hostname.
        socket: Path to the Unix socket file. No-op if None.

    Raises:
        RuntimeError: If the connection could not be closed.
    """
    if socket is None:
        return
    ssh_command = f"ssh -O exit -S {socket} {username}@{hostname}"
    completed_process = subprocess.run(args=shlex.split(ssh_command))
    if completed_process.returncode != 0:
        raise RuntimeError(
            "failed to close the master connection\n{}".format(
                completed_process.stderr.decode("utf-8")
            )
        )


def make_remote_dir(
    dirpath: str,
    username: str,
    hostname: str,
    socket: Optional[str] = None,
) -> None:
    """Create a directory on a remote host via SSH.

    Args:
        dirpath: Absolute path to create on the remote host.
        username: SSH username.
        hostname: SSH hostname.
        socket: Optional path to a multiplexing socket for connection reuse.

    Raises:
        RuntimeError: If the remote mkdir fails.
    """
    mkdir_command = f"mkdir -p {dirpath}"
    ssh_command = ["ssh"]
    if socket is not None:
        ssh_command.append(f"-S {socket}")
    ssh_command.append(f"{username}@{hostname}")
    ssh_command.append(mkdir_command)
    ssh_command = " ".join(ssh_command)
    logger.info("Creating remote dir", cmd=ssh_command)
    completed_process = subprocess.run(
        args=shlex.split(ssh_command), stderr=subprocess.PIPE
    )
    if completed_process.returncode != 0:
        error_msg = (
            completed_process.stderr.decode("utf-8")
            if completed_process.stderr
            else "Unknown error"
        )
        logger.error(
            "Error creating remote dir",
            code=completed_process.returncode,
            msg=error_msg,
        )
        raise RuntimeError(
            "failed to make a remote execution output dir\n{}".format(error_msg)
        )


def rsync_upload(
    local_sources: List[Path],
    remote_target: str,
    username: str,
    hostname: str,
) -> None:
    """Upload local paths to a remote host using rsync over SSH.

    Args:
        local_sources: List of local Path objects to upload (must be directories).
        remote_target: Remote destination directory path.
        username: SSH username.
        hostname: SSH hostname.

    Raises:
        RuntimeError: If rsync fails.
    """
    for local_source in local_sources:
        assert local_source.is_dir()
    remote_destination_str = f"{username}@{hostname}:{remote_target}"
    local_sources_str = " ".join(map(str, local_sources))
    rsync_upload_command = f"rsync -qcaz {local_sources_str} {remote_destination_str}"
    logger.info("Rsyncing to remote dir", cmd=rsync_upload_command)
    completed_process = subprocess.run(
        args=shlex.split(rsync_upload_command),
        stderr=subprocess.PIPE,
    )
    if completed_process.returncode != 0:
        error_msg = (
            completed_process.stderr.decode("utf-8")
            if completed_process.stderr
            else "Unknown error"
        )
        logger.error(
            "Error rsyncing to remote dir",
            code=completed_process.returncode,
            msg=error_msg,
        )
        raise RuntimeError("failed to upload local sources\n{}".format(error_msg))
