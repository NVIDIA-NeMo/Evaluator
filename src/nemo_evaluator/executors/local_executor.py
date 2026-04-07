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
"""Local executor: in-process or background fork."""

from __future__ import annotations

import logging
import os
import signal
from pathlib import Path

from nemo_evaluator.executors import Executor, ProcessState

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


class LocalExecutor(Executor):
    name = "local"

    def run(self, config, *, dry_run=False, resume=False, background=False, submit=False) -> None:
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
        import sys

        import click

        from nemo_evaluator.orchestration.orchestrator import run_local

        _write_pid(config.output.dir)
        self._save_run_meta(config)
        try:
            bundles = run_local(config, resume=resume)
            completed = sum(1 for b in bundles if not b.get("_failed"))
            failed = sum(1 for b in bundles if b.get("_failed"))
            msg = f"\nCompleted {completed} benchmark(s)."
            if failed:
                msg += f" {failed} failed."
            msg += f" Results: {config.output.dir}"
            click.echo(msg)
            if failed:
                sys.exit(1)
        finally:
            pid_file = Path(config.output.dir) / "nel.pid"
            if pid_file.exists():
                pid_file.unlink()

    @staticmethod
    def _save_run_meta(config) -> str:
        """Write unified RunMeta, return the run_id."""
        from datetime import datetime, timezone

        from nemo_evaluator.run_store import (
            RunMeta,
            config_summary,
            generate_run_id,
        )

        run_id = generate_run_id(config)
        run_meta = RunMeta(
            run_id=run_id,
            executor="local",
            output_dir=str(Path(config.output.dir).resolve()),
            started_at=datetime.now(timezone.utc).isoformat(),
            config_summary=config_summary(config),
            details={"pid": os.getpid()},
        )
        run_meta.save()
        return run_id

    def _run_background(self, config, *, resume: bool = False) -> None:
        import sys

        import click

        pid = os.fork()
        if pid > 0:
            _write_pid(config.output.dir, pid)
            run_id = self._save_run_meta(config)
            click.echo(f"Background evaluation started (PID {pid}, run_id: {run_id})")
            click.echo(f"Check status: nel eval status -r {run_id}")
            click.echo(f"Stop:         nel eval stop -r {run_id}")
            return

        os.setsid()
        sys.stdin.close()

        log_path = Path(config.output.dir) / "nel_eval.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fd = open(log_path, "w")
        os.dup2(log_fd.fileno(), 1)
        os.dup2(log_fd.fileno(), 2)

        bundles: list = []
        try:
            from nemo_evaluator.orchestration.orchestrator import run_local

            bundles = run_local(config, resume=resume)
        finally:
            pid_file = Path(config.output.dir) / "nel.pid"
            if pid_file.exists():
                pid_file.unlink()
            log_fd.close()
            os._exit(1 if any(b.get("_failed") for b in bundles) else 0)

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

    def resume_run(self, run_meta, **kwargs) -> None:
        import click

        output_dir = run_meta.output_dir
        config_path = Path(output_dir) / "_docker_config.json"
        if not config_path.exists():
            raise click.ClickException(f"Cannot resume local run: no saved config in {output_dir}")

        import json

        from nemo_evaluator.config import parse_eval_config

        raw = json.loads(config_path.read_text(encoding="utf-8"))
        config = parse_eval_config(raw)
        config.output.dir = output_dir
        self._run_foreground(config, resume=True)

    @staticmethod
    def detect(output_dir: str | Path) -> bool:
        return (Path(output_dir) / "nel.pid").exists()


def _cleanup_pid(output_dir: str | Path) -> None:
    p = Path(output_dir) / "nel.pid"
    if p.exists():
        p.unlink()
