"""Local executor: in-process or background fork."""
from __future__ import annotations

import logging
import os
import signal
from pathlib import Path

from nemo_evaluator.executors import ProcessState

logger = logging.getLogger(__name__)


def _write_pid(output_dir: str | Path, pid: int | None = None) -> Path:
    p = Path(output_dir) / "nel.pid"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(str(pid or os.getpid()))
    return p


def _read_pid(output_dir: str | Path) -> int | None:
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


class LocalExecutor:
    name = "local"

    def run(self, config, *, dry_run=False, resume=False,
            background=False, submit=False) -> None:
        if dry_run:
            import click
            click.echo(f"Dry-run: would run {len(config.benchmarks)} benchmark(s) locally")
            for b in config.benchmarks:
                click.echo(f"  - {b.name} (repeats={b.repeats})")
            click.echo(f"Output: {config.output.dir}")
            return
        if background:
            self._run_background(config, resume=resume)
        else:
            self._run_foreground(config, resume=resume)

    def _run_foreground(self, config, *, resume: bool = False) -> None:
        import click

        from nemo_evaluator.eval.local_runner import run_local

        _write_pid(config.output.dir)
        try:
            bundles = run_local(config, resume=resume)
            completed = sum(1 for b in bundles if not b.get("_failed"))
            failed = sum(1 for b in bundles if b.get("_failed"))
            msg = f"\nCompleted {completed} benchmark(s)."
            if failed:
                msg += f" {failed} failed."
            msg += f" Results: {config.output.dir}"
            click.echo(msg)
        finally:
            pid_file = Path(config.output.dir) / "nel.pid"
            if pid_file.exists():
                pid_file.unlink()

    def _run_background(self, config, *, resume: bool = False) -> None:
        import sys

        import click

        pid = os.fork()
        if pid > 0:
            _write_pid(config.output.dir, pid)
            click.echo(f"Background evaluation started (PID {pid})")
            click.echo(f"Check status: nel eval status -o {config.output.dir}")
            click.echo(f"Stop:         nel eval stop -o {config.output.dir}")
            return

        os.setsid()
        sys.stdin.close()

        log_path = Path(config.output.dir) / "nel_eval.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fd = open(log_path, "w")
        os.dup2(log_fd.fileno(), 1)
        os.dup2(log_fd.fileno(), 2)

        try:
            from nemo_evaluator.eval.local_runner import run_local
            run_local(config, resume=resume)
        finally:
            pid_file = Path(config.output.dir) / "nel.pid"
            if pid_file.exists():
                pid_file.unlink()
            log_fd.close()
            os._exit(0)

    def status(self, output_dir: str | Path) -> ProcessState:
        pid = _read_pid(output_dir)
        if pid is None:
            return ProcessState("local", False, {"error": "No PID file found"})
        return ProcessState("local", _pid_alive(pid), {"pid": pid})

    def stop(self, output_dir: str | Path) -> bool:
        pid = _read_pid(output_dir)
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

    @staticmethod
    def detect(output_dir: str | Path) -> bool:
        return (Path(output_dir) / "nel.pid").exists()


def _cleanup_pid(output_dir: str | Path) -> None:
    p = Path(output_dir) / "nel.pid"
    if p.exists():
        p.unlink()
