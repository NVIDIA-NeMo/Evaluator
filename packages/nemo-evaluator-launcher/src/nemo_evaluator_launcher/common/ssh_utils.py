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
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, List, Optional

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
    ssh_command = [
        "ssh",
        "-Nf",
        "-o",
        "ControlMaster=auto",
        "-S",
        socket,
        f"{username}@{hostname}",
    ]
    logger.info("Opening master connection", cmd=" ".join(ssh_command))
    completed_process = subprocess.run(args=ssh_command)
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


def run_remote_command(
    command: str,
    username: str,
    hostname: str,
    socket: Optional[str] = None,
) -> str:
    """Run a command on a remote host via SSH and return stdout.

    Args:
        command: Shell command to execute on the remote host.
        username: SSH username.
        hostname: SSH hostname.
        socket: Optional path to a multiplexing socket for connection reuse.

    Returns:
        stdout of the remote command as a string.

    Raises:
        RuntimeError: If the remote command exits with a non-zero return code.
    """
    ssh_cmd = ["ssh"]
    if socket is not None:
        ssh_cmd += ["-S", socket]
    ssh_cmd.append(f"{username}@{hostname}")
    ssh_cmd.append(command)
    completed = subprocess.run(
        args=ssh_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Remote command failed on {hostname}:\n{completed.stderr.decode()}"
        )
    return completed.stdout.decode()


@contextmanager
def master_connection(username: str, hostname: str) -> Generator[str, None, None]:
    """Context manager that opens a persistent SSH master connection and closes it on exit.

    Args:
        username: SSH username.
        hostname: SSH hostname.

    Yields:
        Path to the Unix socket file for connection multiplexing.

    Raises:
        RuntimeError: If the connection cannot be opened.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
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
            yield socket_or_none
        finally:
            close_master_connection(
                username=username, hostname=hostname, socket=socket_or_none
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
