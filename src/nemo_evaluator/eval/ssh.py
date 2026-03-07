"""SSH submission: copy scripts to a SLURM login node and submit via sbatch."""
from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


class SSHError(RuntimeError):
    pass


def _run(cmd: list[str], timeout: float = 30.0) -> str:
    logger.debug("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise SSHError(f"Command failed ({result.returncode}): {' '.join(cmd)}\n{result.stderr}")
    return result.stdout.strip()


def _ssh_target(hostname: str, username: str | None = None) -> str:
    if username:
        return f"{username}@{hostname}"
    return hostname


def copy_to_remote(
    hostname: str,
    local_paths: list[Path],
    remote_dir: str,
    username: str | None = None,
) -> None:
    """Copy files to the remote host via scp."""
    target = _ssh_target(hostname, username)

    _run(["ssh", target, "mkdir", "-p", remote_dir], timeout=15.0)

    for p in local_paths:
        _run(["scp", str(p), f"{target}:{remote_dir}/"], timeout=60.0)
        logger.info("Copied %s -> %s:%s/", p.name, target, remote_dir)


def submit_sbatch(
    hostname: str,
    remote_script: str,
    username: str | None = None,
) -> str:
    """Submit an sbatch script on the remote host and return the SLURM job ID."""
    target = _ssh_target(hostname, username)
    output = _run(["ssh", target, "sbatch", remote_script], timeout=30.0)

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
    """Check SLURM job status via squeue."""
    target = _ssh_target(hostname, username)
    try:
        output = _run([
            "ssh", target,
            "squeue", "--job", job_id, "--noheader",
            "--format=%i|%j|%T|%M|%N",
        ], timeout=15.0)
    except SSHError:
        return {"job_id": job_id, "state": "UNKNOWN"}

    if not output.strip():
        return {"job_id": job_id, "state": "COMPLETED"}

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


def cancel_job(
    hostname: str,
    job_id: str,
    username: str | None = None,
) -> None:
    """Cancel a SLURM job via scancel."""
    target = _ssh_target(hostname, username)
    _run(["ssh", target, "scancel", job_id], timeout=15.0)
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

    metadata = {
        "job_id": job_id,
        "hostname": hostname,
        "username": username or "",
        "remote_dir": remote_dir,
        "script": remote_script,
    }

    meta_path = script_path.parent / "slurm_job.json"
    meta_path.write_text(json.dumps(metadata, indent=2))
    logger.info("SLURM job %s submitted. Metadata: %s", job_id, meta_path)

    return metadata
