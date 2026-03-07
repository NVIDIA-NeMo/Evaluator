"""Executor-aware lifecycle management for nel eval and nel serve.

Each executor persists state to a metadata file:
  - local:  nel.pid       (PID of the background process)
  - slurm:  slurm_job.json (job ID, hostname, username)
  - docker: docker.json    (container ID)

The lifecycle module reads these files and dispatches status/stop
operations to the appropriate backend.
"""
from __future__ import annotations

import json
import logging
import os
import signal
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ProcessState:
    executor: str
    running: bool
    details: dict


# ---------------------------------------------------------------------------
# Local lifecycle
# ---------------------------------------------------------------------------

def write_local_pid(output_dir: str | Path, pid: int | None = None) -> Path:
    """Write the current (or given) process PID to nel.pid."""
    p = Path(output_dir) / "nel.pid"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(str(pid or os.getpid()))
    return p


def read_local_pid(output_dir: str | Path) -> int | None:
    p = Path(output_dir) / "nel.pid"
    if not p.exists():
        return None
    try:
        return int(p.read_text().strip())
    except ValueError:
        return None


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def local_status(output_dir: str | Path) -> ProcessState:
    pid = read_local_pid(output_dir)
    if pid is None:
        return ProcessState("local", False, {"error": "No PID file found"})
    alive = _pid_alive(pid)
    return ProcessState("local", alive, {"pid": pid})


def local_stop(output_dir: str | Path) -> bool:
    pid = read_local_pid(output_dir)
    if pid is None:
        logger.warning("No PID file found in %s", output_dir)
        return False
    if not _pid_alive(pid):
        logger.info("Process %d already stopped", pid)
        _cleanup_pid(output_dir)
        return True
    try:
        os.kill(pid, signal.SIGTERM)
        logger.info("Sent SIGTERM to %d", pid)
        _cleanup_pid(output_dir)
        return True
    except OSError as e:
        logger.error("Failed to stop process %d: %s", pid, e)
        return False


def _cleanup_pid(output_dir: str | Path) -> None:
    p = Path(output_dir) / "nel.pid"
    if p.exists():
        p.unlink()


# ---------------------------------------------------------------------------
# SLURM lifecycle
# ---------------------------------------------------------------------------

def read_slurm_meta(output_dir: str | Path) -> dict | None:
    p = Path(output_dir) / "slurm_job.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def slurm_status(output_dir: str | Path) -> ProcessState:
    meta = read_slurm_meta(output_dir)
    if meta is None:
        return ProcessState("slurm", False, {"error": "No slurm_job.json found"})

    from nemo_evaluator.eval.ssh import check_job_status
    info = check_job_status(
        hostname=meta["hostname"],
        job_id=meta["job_id"],
        username=meta.get("username") or None,
    )
    running = info.get("state", "") in ("PENDING", "RUNNING", "CONFIGURING", "COMPLETING")
    return ProcessState("slurm", running, info)


def slurm_stop(output_dir: str | Path) -> bool:
    meta = read_slurm_meta(output_dir)
    if meta is None:
        logger.warning("No slurm_job.json found in %s", output_dir)
        return False

    from nemo_evaluator.eval.ssh import cancel_job
    try:
        cancel_job(
            hostname=meta["hostname"],
            job_id=meta["job_id"],
            username=meta.get("username") or None,
        )
        return True
    except Exception as e:
        logger.error("Failed to cancel SLURM job %s: %s", meta["job_id"], e)
        return False


# ---------------------------------------------------------------------------
# Docker lifecycle
# ---------------------------------------------------------------------------

def read_docker_meta(output_dir: str | Path) -> dict | None:
    p = Path(output_dir) / "docker.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def docker_status(output_dir: str | Path) -> ProcessState:
    import subprocess

    meta = read_docker_meta(output_dir)
    if meta is None:
        return ProcessState("docker", False, {"error": "No docker.json found"})

    container_id = meta.get("container_id", "")
    try:
        result = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Status}}", container_id],
            capture_output=True, text=True, timeout=10,
        )
        status = result.stdout.strip()
        running = status == "running"
        return ProcessState("docker", running, {"container_id": container_id, "status": status})
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        return ProcessState("docker", False, {"container_id": container_id, "error": str(e)})


def docker_stop(output_dir: str | Path) -> bool:
    import subprocess

    meta = read_docker_meta(output_dir)
    if meta is None:
        logger.warning("No docker.json found in %s", output_dir)
        return False

    container_id = meta.get("container_id", "")
    try:
        subprocess.run(
            ["docker", "stop", container_id],
            capture_output=True, timeout=30,
        )
        logger.info("Stopped Docker container %s", container_id)
        return True
    except Exception as e:
        logger.error("Failed to stop container %s: %s", container_id, e)
        return False


# ---------------------------------------------------------------------------
# Unified dispatch
# ---------------------------------------------------------------------------

def detect_executor(output_dir: str | Path) -> str:
    """Detect which executor was used based on metadata files."""
    d = Path(output_dir)
    if (d / "slurm_job.json").exists():
        return "slurm"
    if (d / "docker.json").exists():
        return "docker"
    if (d / "nel.pid").exists():
        return "local"
    return "unknown"


def status(output_dir: str | Path) -> ProcessState:
    executor = detect_executor(output_dir)
    dispatch = {
        "local": local_status,
        "slurm": slurm_status,
        "docker": docker_status,
    }
    fn = dispatch.get(executor)
    if fn is None:
        return ProcessState("unknown", False, {"error": f"No lifecycle metadata in {output_dir}"})
    return fn(output_dir)


def stop(output_dir: str | Path) -> bool:
    executor = detect_executor(output_dir)
    dispatch = {
        "local": local_stop,
        "slurm": slurm_stop,
        "docker": docker_stop,
    }
    fn = dispatch.get(executor)
    if fn is None:
        logger.warning("Cannot determine executor for %s", output_dir)
        return False
    return fn(output_dir)
