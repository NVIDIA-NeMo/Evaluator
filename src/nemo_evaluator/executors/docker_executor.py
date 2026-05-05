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
"""Docker executor: run evaluation inside a container."""

from __future__ import annotations

import json
import logging
import os
import shlex
import subprocess
import warnings
from pathlib import Path

from nemo_evaluator.executors import Executor, ProcessState

logger = logging.getLogger(__name__)

_META_FILE = "docker.json"
_LOCAL_TAG_PREFIX = "nemo-evaluator"


def _is_repo_root(p: Path) -> bool:
    return (p / "pyproject.toml").exists() and (p / "docker").is_dir() and (p / "src" / "nemo_evaluator").is_dir()


def _find_repo_root() -> Path | None:
    """Locate the NEL repo root by walking up from CWD (then __file__ as fallback)."""
    for start in (Path.cwd().resolve(), Path(__file__).resolve().parent):
        p = start
        for _ in range(10):
            if _is_repo_root(p):
                return p
            parent = p.parent
            if parent == p:
                break
            p = parent
    return None


def _build_local_image(variant: str, *, quiet: bool = False) -> str:
    """Build a local Docker image from ``docker/Dockerfile.<variant>``.

    Returns the local image tag, e.g. ``nemo-evaluator:local-gym``.
    """
    repo = _find_repo_root()
    if repo is None:
        raise FileNotFoundError(
            "Cannot find repo root (need pyproject.toml + docker/ directory). "
            "Local builds only work from a source checkout."
        )

    suffix = variant if variant and variant != "base" else "base"
    dockerfile = repo / "docker" / f"Dockerfile.{suffix}"
    if not dockerfile.exists():
        raise FileNotFoundError(f"Dockerfile not found: {dockerfile}")

    tag = f"{_LOCAL_TAG_PREFIX}:local" if suffix == "base" else f"{_LOCAL_TAG_PREFIX}:local-{suffix}"

    cmd = ["docker", "build", "-f", str(dockerfile), "-t", tag, str(repo)]
    logger.info("Building local image: %s", shlex.join(cmd))
    if not quiet:
        import click

        click.echo(f"Building {tag} from {dockerfile.relative_to(repo)} …")

    result = subprocess.run(cmd, capture_output=quiet, text=True)
    if result.returncode != 0:
        msg = result.stderr.strip() if quiet else "(see output above)"
        raise RuntimeError(f"docker build failed: {msg}")
    return tag


class DockerExecutor(Executor):
    name = "docker"

    def run(self, config, *, dry_run=False, resume=False, background=False, submit=False) -> None:
        import click

        from nemo_evaluator.orchestration.image_resolver import resolve_eval_image, scheme_to_variant

        base = getattr(config.cluster, "image", None) or getattr(config.cluster, "container_image", None)
        local_build = not base or base == "local"

        if local_build:
            variant = "base"
            if config.benchmarks:
                variant = scheme_to_variant(config.benchmarks[0].name) or "base"
            if dry_run:
                suffix = variant if variant and variant != "base" else "base"
                image = f"{_LOCAL_TAG_PREFIX}:local" if suffix == "base" else f"{_LOCAL_TAG_PREFIX}:local-{suffix}"
                click.echo(f"Would build: docker/Dockerfile.{suffix} → {image}")
            else:
                image = _build_local_image(variant)
        else:
            image = resolve_eval_image(
                config.benchmarks[0].name if config.benchmarks else "",
                base_override=base,
            )

        output_dir = str(Path(config.output.dir).resolve())

        mount_args = ["-v", f"{output_dir}:{output_dir}"]
        for m in getattr(config.cluster, "container_mounts", []):
            mount_args.extend(["-v", m])
        if getattr(config.cluster, "mount_home", True):
            home = os.environ.get("HOME", "")
            if home:
                mount_args.extend(["-v", f"{home}:{home}"])

        container_env = dict(getattr(config.cluster, "container_env", {}))

        _FORWARD_ENVS = (
            "NEMO_API_KEY",
            "NEMO_MODEL_URL",
            "NEMO_MODEL_ID",
            "HF_TOKEN",
            "MLFLOW_TRACKING_URI",
            "MLFLOW_TRACKING_TOKEN",
        )
        forwarded: list[str] = []
        for key in _FORWARD_ENVS:
            val = os.environ.get(key)
            if val and key not in container_env:
                container_env[key] = val
                forwarded.append(key)
        if forwarded:
            warnings.warn(
                f"Auto-forwarded host env vars {forwarded} into container. "
                "Use cluster.container_env instead for explicit control. "
                "_FORWARD_ENVS auto-forwarding is deprecated.",
                DeprecationWarning,
                stacklevel=2,
            )

        container_env["NEL_INNER_EXECUTION"] = "1"
        container_env["PYTHONUNBUFFERED"] = "1"

        env_file_path = Path(output_dir) / ".docker.env"
        env_file_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [f"{k}={v}" for k, v in container_env.items()]
        env_file_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        env_file_path.chmod(0o600)
        env_args: list[str] = ["--env-file", str(env_file_path)]

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        cfg_path = Path(output_dir) / "_docker_config.json"
        cfg_path.write_text(
            json.dumps(config.model_dump(), default=str),
            encoding="utf-8",
        )

        shm = getattr(config.cluster, "shm_size", None)
        if not shm:
            if any(s.type == "gym" for s in config.services.values()):
                shm = "2g"
        extra_docker_args: list[str] = []
        if shm:
            extra_docker_args.extend([f"--shm-size={shm}"])

        cmd = [
            "docker",
            "run",
            "-d",
            "--name",
            f"nel-eval-{os.getpid()}",
            *extra_docker_args,
            *mount_args,
            *env_args,
            image,
            "nel",
            "eval",
            "run",
            str(cfg_path),
        ]
        if resume:
            cmd.append("--resume")

        if dry_run:
            click.echo(f"Docker command:\n  {shlex.join(cmd)}")
            click.echo(f"\nEnv file ({env_file_path}):")
            for k, v in container_env.items():
                if k in ("NEL_INNER_EXECUTION", "PYTHONUNBUFFERED"):
                    click.echo(f"  {k}={v}")
                elif len(v) > 4:
                    click.echo(f"  {k}=***{v[-4:]}")
                else:
                    click.echo(f"  {k}=***")
            return

        click.echo(f"Running in Docker: {image}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise click.ClickException(f"docker run failed: {result.stderr.strip()}")

        container_id = result.stdout.strip()
        meta_path = Path(output_dir) / _META_FILE
        meta_path.write_text(
            json.dumps({"container_id": container_id, "image": image}),
            encoding="utf-8",
        )

        from datetime import datetime, timezone

        from nemo_evaluator.run_store import (
            RunMeta,
            generate_run_id,
        )
        from nemo_evaluator.run_store import (
            config_summary as _config_summary,
        )

        run_id = generate_run_id(config)
        run_meta = RunMeta(
            run_id=run_id,
            executor="docker",
            output_dir=output_dir,
            started_at=datetime.now(timezone.utc).isoformat(),
            config_summary=_config_summary(config),
            details={
                "container_id": container_id,
                "image": image,
            },
        )
        run_meta.save()

        click.echo(f"Container started: {container_id[:12]}  (run_id: {run_id})")
        click.echo(f"Check status: nel eval status -r {run_id}")
        click.echo(f"Stop:         nel eval stop -r {run_id}")
        click.echo(f"Logs:         nel eval logs -r {run_id}")

    def status(self, output_dir: str | Path) -> ProcessState:
        meta = _read_meta(output_dir)
        if meta is None:
            return ProcessState("docker", False, {"error": f"No {_META_FILE} found"})

        container_id = meta.get("container_id", "")
        try:
            result = subprocess.run(
                ["docker", "inspect", "-f", "{{.State.Status}}", container_id],
                capture_output=True,
                text=True,
                timeout=10,
            )
            st = result.stdout.strip()
            return ProcessState("docker", st == "running", {"container_id": container_id, "status": st})
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            return ProcessState("docker", False, {"container_id": container_id, "error": str(e)})

    def stop(self, output_dir: str | Path) -> bool:
        meta = _read_meta(output_dir)
        if meta is None:
            logger.warning("No %s found in %s", _META_FILE, output_dir)
            return False

        container_id = meta.get("container_id", "")
        try:
            subprocess.run(
                ["docker", "stop", container_id],
                capture_output=True,
                timeout=30,
            )
            logger.info("Stopped Docker container %s", container_id)
            return True
        except Exception as e:
            logger.error("Failed to stop container %s: %s", container_id, e)
            return False

    def logs(self, output_dir: str | Path, *, follow: bool = False, tail: int | None = None) -> str | None:
        meta = _read_meta(output_dir)
        if meta is None:
            return super().logs(output_dir, follow=follow, tail=tail)
        container_id = meta.get("container_id", "")
        cmd = ["docker", "logs"]
        if tail:
            cmd.extend(["--tail", str(tail)])
        cmd.append(container_id)
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return result.stdout + result.stderr
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return super().logs(output_dir, follow=follow, tail=tail)

    def resume_run(self, run_meta, **kwargs) -> None:
        details = run_meta.details
        container_id = details.get("container_id", "")
        if not container_id:
            raise RuntimeError("No container_id in run metadata")
        try:
            result = subprocess.run(
                ["docker", "start", container_id],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                raise RuntimeError(f"docker start failed: {result.stderr.strip()}")
            logger.info("Resumed Docker container %s", container_id)
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Timed out trying to start container {container_id}")

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
