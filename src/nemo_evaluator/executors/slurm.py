"""SLURM executor: generate sbatch scripts, submit, and manage jobs."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from nemo_evaluator.executors import ProcessState

logger = logging.getLogger(__name__)

_META_FILE = "slurm_job.json"


class SlurmExecutor:
    name = "slurm"

    def run(self, config, *, dry_run=False, resume=False,
            background=False, submit=False) -> None:
        import click

        from nemo_evaluator.eval.slurm_gen import write_sbatch

        script_path = write_sbatch(config)
        click.echo(f"Generated: {script_path}")

        if dry_run:
            click.echo(f"\nTo submit locally:  sbatch {script_path}")
            if config.cluster.hostname:
                click.echo(f"To submit via SSH:  nel eval run {script_path} --submit")
            return

        if submit and config.cluster.hostname:
            from nemo_evaluator.eval.ssh import submit_eval
            meta = submit_eval(
                script_path=script_path,
                hostname=config.cluster.hostname,
                remote_dir=config.output.dir,
                username=config.cluster.username,
            )
            click.echo(f"SLURM job submitted: {meta['job_id']}")
            click.echo(f"Check status: nel eval status -o {config.output.dir}")
            return

        import subprocess
        result = subprocess.run(
            ["sbatch", str(script_path)], capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise click.ClickException(f"sbatch failed: {result.stderr}")
        click.echo(result.stdout.strip())

    def status(self, output_dir: str | Path) -> ProcessState:
        meta = _read_meta(output_dir)
        if meta is None:
            return ProcessState("slurm", False, {"error": f"No {_META_FILE} found"})

        from nemo_evaluator.eval.ssh import check_job_status
        info = check_job_status(
            hostname=meta["hostname"],
            job_id=meta["job_id"],
            username=meta.get("username") or None,
        )
        running = info.get("state", "") in (
            "PENDING", "RUNNING", "CONFIGURING", "COMPLETING",
        )
        return ProcessState("slurm", running, info)

    def stop(self, output_dir: str | Path) -> bool:
        meta = _read_meta(output_dir)
        if meta is None:
            logger.warning("No %s found in %s", _META_FILE, output_dir)
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

    @staticmethod
    def detect(output_dir: str | Path) -> bool:
        return (Path(output_dir) / _META_FILE).exists()


def _read_meta(output_dir: str | Path) -> dict | None:
    p = Path(output_dir) / _META_FILE
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return None
