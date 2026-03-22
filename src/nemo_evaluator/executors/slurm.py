"""SLURM executor: generate sbatch scripts, submit, and manage jobs."""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
from pathlib import Path

from nemo_evaluator.executors import ProcessState

logger = logging.getLogger(__name__)

_META_FILE = "slurm_job.json"


class SlurmExecutor:
    name = "slurm"

    def run(self, config, *, dry_run=False, resume=False, background=False, **_kwargs) -> None:
        import click

        from nemo_evaluator.eval.slurm_gen import write_sbatch

        is_remote = bool(config.cluster.hostname)
        local_staging = None
        if is_remote:
            local_staging = Path(tempfile.mkdtemp(prefix="nel-slurm-"))
            script_path, extra_paths = write_sbatch(config, output_dir=local_staging)
        else:
            script_path, extra_paths = write_sbatch(config)

        click.echo(f"Generated: {script_path}")
        for sp in extra_paths:
            if sp.name == ".secrets.env":
                click.echo(f"  secrets: {sp}  (chmod 600)")
            else:
                click.echo(f"  sidecar: {sp}")

        if dry_run:
            if is_remote:
                target = config.cluster.hostname
                click.echo(f"\nFiles staged in: {local_staging}")
                click.echo(f"Remote target:   {target}:{config.output.dir}")
                click.echo("\nTo submit manually:")
                click.echo(f"  scp {script_path} {' '.join(str(p) for p in extra_paths)} {target}:{config.output.dir}/")
                click.echo(f"  ssh {target} sbatch {config.output.dir}/{script_path.name}")
            else:
                click.echo(f"\nTo submit:  sbatch {script_path}")
            return

        if is_remote:
            from nemo_evaluator.eval.ssh import submit_eval

            try:
                meta = submit_eval(
                    script_path=script_path,
                    hostname=config.cluster.hostname,
                    remote_dir=config.output.dir,
                    username=config.cluster.username,
                    extra_files=extra_paths,
                )
            finally:
                if local_staging:
                    shutil.rmtree(local_staging, ignore_errors=True)

            meta_dir = Path.home() / ".local" / "share" / "nel" / "jobs" / meta["job_id"]
            meta_dir.mkdir(parents=True, exist_ok=True)
            meta_path = meta_dir / _META_FILE
            meta_path.write_text(json.dumps(meta, indent=2))

            jid = meta["job_id"]
            rdir = config.output.dir
            host = config.cluster.hostname
            click.echo(f"\nSLURM job submitted: {jid}")
            click.echo(f"Remote dir: {host}:{rdir}")
            click.echo(f"Log:        {host}:{rdir}/slurm-{jid}.log")
            click.echo(f"Metadata:   {meta_path}")
            click.echo(f"\nTail logs:  ssh {host} tail -f {rdir}/slurm-{jid}.log")
            return

        import subprocess

        result = subprocess.run(
            ["sbatch", str(script_path)],
            capture_output=True,
            text=True,
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
            "PENDING",
            "RUNNING",
            "CONFIGURING",
            "COMPLETING",
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
        return _read_meta(output_dir) is not None


def _read_meta(output_dir: str | Path) -> dict | None:
    # 1. Direct path (local cluster or explicit meta dir)
    p = Path(output_dir) / _META_FILE
    if p.exists():
        try:
            return json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            return None

    # 2. Search local jobs store by remote_dir match
    jobs_dir = Path.home() / ".local" / "share" / "nel" / "jobs"
    if jobs_dir.is_dir():
        out_str = str(output_dir)
        for meta_path in jobs_dir.glob(f"*/{_META_FILE}"):
            try:
                meta = json.loads(meta_path.read_text())
                if meta.get("remote_dir") == out_str:
                    return meta
            except (json.JSONDecodeError, OSError):
                continue

    return None
