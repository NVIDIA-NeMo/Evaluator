# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""SSH submission: copy scripts to a SLURM login node and submit via sbatch."""

from __future__ import annotations

import logging
import shlex
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


class SSHError(RuntimeError):
    pass


def _ssh_opts(target: str) -> list[str]:
    """Return SSH options that enable connection multiplexing.

    First call for a given target opens a ControlMaster; subsequent calls
    reuse it — so the user authenticates only once.  ControlPersist handles
    automatic teardown after the last client disconnects; we intentionally
    do NOT register an atexit hook because that would kill the shared socket
    while other nel processes (e.g. ``nel eval run`` + ``nel eval status``)
    may still be using it.
    """
    socket_dir = Path.home() / ".ssh" / "nel"
    socket_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    if socket_dir.stat().st_mode & 0o077:
        socket_dir.chmod(0o700)
    control_path = str(socket_dir / "%C")

    return [
        "-o",
        f"ControlPath={control_path}",
        "-o",
        "ControlMaster=auto",
        "-o",
        "ControlPersist=120",
    ]


def _run(cmd: list[str], timeout: float = 30.0) -> str:
    logger.debug("Running: %s", " ".join(cmd))
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        raise SSHError(f"Timed out after {timeout}s: {' '.join(cmd)}") from e
    if result.returncode != 0:
        raise SSHError(f"Command failed ({result.returncode}): {' '.join(cmd)}\n{result.stderr}")
    return result.stdout.strip()


def _ensure_master(target: str) -> None:
    """Open a ControlMaster connection if one isn't already active.

    Runs with inherited stdin/stdout so the user can type a password
    or respond to host-key prompts interactively.
    """
    opts = _ssh_opts(target)
    check = subprocess.run(
        ["ssh", *opts, "-O", "check", target],
        capture_output=True,
        timeout=5,
    )
    if check.returncode == 0:
        return

    logger.info("Opening SSH connection to %s ...", target)
    result = subprocess.run(
        ["ssh", *opts, target, "true"],
        timeout=60,
    )
    if result.returncode != 0:
        raise SSHError(f"Failed to open SSH connection to {target}")


def _ssh(target: str, remote_cmd: str, timeout: float = 30.0) -> str:
    _ensure_master(target)
    return _run(["ssh", *_ssh_opts(target), target, remote_cmd], timeout=timeout)


def ssh_run(hostname: str, remote_cmd: str, username: str | None = None, timeout: float = 30.0) -> str:
    """Public API: run a command on a remote host via SSH."""
    target = f"{username}@{hostname}" if username else hostname
    return _ssh(target, remote_cmd, timeout=timeout)


def _scp(local_path: str, remote_dest: str, target: str, timeout: float = 60.0) -> str:
    _ensure_master(target)
    return _run(["scp", "-p", *_ssh_opts(target), local_path, remote_dest], timeout=timeout)


def _ssh_target(hostname: str, username: str | None = None) -> str:
    if username:
        return f"{username}@{hostname}"
    return hostname


def copy_from_remote(
    hostname: str,
    remote_pattern: str,
    local_dir: Path,
    username: str | None = None,
    timeout: float = 120.0,
) -> list[Path]:
    """Download files matching a remote glob pattern to a local directory via scp.

    Returns the list of local paths that were downloaded.
    """
    target = _ssh_target(hostname, username)
    local_dir.mkdir(parents=True, exist_ok=True)

    _ensure_master(target)
    try:
        _run(
            ["scp", "-p", *_ssh_opts(target), f"{target}:{remote_pattern}", str(local_dir) + "/"],
            timeout=timeout,
        )
    except SSHError as e:
        if "No such file" in str(e) or "not a regular file" in str(e):
            return []
        raise

    return sorted(local_dir.iterdir())


def copy_tree_from_remote(
    hostname: str,
    remote_dir: str,
    local_dir: Path,
    username: str | None = None,
    timeout: float = 300.0,
) -> None:
    """Recursively download a remote directory to a local path via scp -r."""
    target = _ssh_target(hostname, username)
    local_dir.mkdir(parents=True, exist_ok=True)
    _ensure_master(target)
    _run(
        ["scp", "-rp", *_ssh_opts(target), f"{target}:{remote_dir}/.", str(local_dir) + "/"],
        timeout=timeout,
    )


def copy_to_remote(
    hostname: str,
    local_paths: list[Path],
    remote_dir: str,
    username: str | None = None,
) -> None:
    """Copy files to the remote host via scp, preserving permissions."""
    target = _ssh_target(hostname, username)

    has_secrets = any(p.name == ".secrets.env" for p in local_paths)
    if has_secrets:
        _ssh(
            target,
            f"mkdir -p {shlex.quote(remote_dir)}/logs && "
            f"install -m 600 /dev/null {shlex.quote(remote_dir + '/.secrets.env')}",
            timeout=15.0,
        )
    else:
        _ssh(target, f"mkdir -p {shlex.quote(remote_dir)}/logs", timeout=15.0)

    for p in local_paths:
        _scp(str(p), f"{target}:{remote_dir}/", target)
        logger.info("Copied %s -> %s:%s/", p.name, target, remote_dir)


def submit_sbatch(
    hostname: str,
    remote_script: str,
    username: str | None = None,
) -> str:
    """Submit an sbatch script on the remote host and return the SLURM job ID."""
    target = _ssh_target(hostname, username)
    output = _ssh(target, f"sbatch {shlex.quote(remote_script)}")

    # sbatch output: "Submitted batch job 12345"
    parts = output.split()
    if len(parts) >= 4 and parts[-1].isdigit():
        return parts[-1]
    raise SSHError(f"Could not parse sbatch output: {output}")


def check_job_status(
    hostname: str,
    job_id: str,
    username: str | None = None,
) -> dict[str, str]:
    """Check SLURM job status via squeue, falling back to sacct."""
    target = _ssh_target(hostname, username)
    try:
        output = _ssh(
            target,
            f"squeue --job {shlex.quote(job_id)} --noheader --format=%i|%j|%T|%M|%N",
            timeout=15.0,
        )
    except SSHError:
        return {"job_id": job_id, "state": "UNKNOWN"}

    if not output.strip():
        return _sacct_status(target, job_id)

    fields = output.strip().split("|")
    if len(fields) >= 5:
        return {
            "job_id": fields[0],
            "name": fields[1],
            "state": fields[2],
            "time": fields[3],
            "node": fields[4],
        }
    return {"job_id": job_id, "state": output.strip()}


def _sacct_status(target: str, job_id: str) -> dict[str, str]:
    """Fall back to sacct for completed/failed jobs no longer in squeue."""
    try:
        output = _ssh(
            target,
            f"sacct -j {shlex.quote(job_id)} --noheader --parsable2 --format=JobID,State,ExitCode -n",
            timeout=15.0,
        )
    except SSHError:
        return {"job_id": job_id, "state": "UNKNOWN"}

    for line in output.strip().splitlines():
        fields = line.split("|")
        if len(fields) >= 3 and fields[0] == job_id:
            return {
                "job_id": job_id,
                "state": fields[1],
                "exit_code": fields[2],
            }
    return {"job_id": job_id, "state": "UNKNOWN"}


def batch_check_job_status(
    hostname: str,
    job_ids: list[str],
    username: str | None = None,
) -> dict[str, dict[str, str]]:
    """Check multiple SLURM jobs in one squeue + sacct round-trip."""
    if not job_ids:
        return {}
    target = _ssh_target(hostname, username)
    ids_str = ",".join(shlex.quote(j) for j in job_ids)

    results: dict[str, dict[str, str]] = {}
    try:
        output = _ssh(
            target,
            f"squeue --jobs {ids_str} --noheader --format=%i|%j|%T|%M|%N",
            timeout=15.0,
        )
        for line in output.strip().splitlines():
            if not line.strip():
                continue
            fields = line.strip().split("|")
            if len(fields) >= 5:
                results[fields[0]] = {
                    "job_id": fields[0],
                    "name": fields[1],
                    "state": fields[2],
                    "time": fields[3],
                    "node": fields[4],
                }
    except SSHError:
        pass

    missing = [jid for jid in job_ids if jid not in results]
    if missing:
        sacct_ids = ",".join(shlex.quote(j) for j in missing)
        try:
            output = _ssh(
                target,
                f"sacct -j {sacct_ids} --noheader --parsable2 --format=JobID,State,ExitCode -n",
                timeout=15.0,
            )
            for line in output.strip().splitlines():
                fields = line.split("|")
                if len(fields) >= 3 and fields[0] in missing:
                    results[fields[0]] = {
                        "job_id": fields[0],
                        "state": fields[1],
                        "exit_code": fields[2],
                    }
        except SSHError:
            pass

    for jid in job_ids:
        if jid not in results:
            results[jid] = {"job_id": jid, "state": "UNKNOWN"}

    return results


def cancel_job(
    hostname: str,
    job_id: str,
    username: str | None = None,
) -> None:
    """Cancel a SLURM job via scancel."""
    target = _ssh_target(hostname, username)
    _ssh(target, f"scancel {shlex.quote(job_id)}", timeout=15.0)
    logger.info("Cancelled SLURM job %s", job_id)


def submit_eval(
    script_path: Path,
    hostname: str,
    remote_dir: str,
    username: str | None = None,
    extra_files: list[Path] | None = None,
) -> dict[str, str]:
    """Full submission flow: copy files, sbatch, return job metadata."""
    files = [script_path] + (extra_files or [])
    copy_to_remote(hostname, files, remote_dir, username)

    remote_script = f"{remote_dir}/{script_path.name}"
    job_id = submit_sbatch(hostname, remote_script, username)

    return {
        "job_id": job_id,
        "hostname": hostname,
        "username": username or "",
        "remote_dir": remote_dir,
        "script": remote_script,
    }
