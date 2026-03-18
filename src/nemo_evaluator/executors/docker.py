"""Docker executor: run evaluation inside a container."""
from __future__ import annotations

import json
import logging
import os
import shlex
import subprocess
from pathlib import Path

from nemo_evaluator.executors import ProcessState

logger = logging.getLogger(__name__)

_META_FILE = "docker.json"
_LOCAL_TAG_PREFIX = "nemo-evaluator"


def _is_repo_root(p: Path) -> bool:
    return (
        (p / "pyproject.toml").exists()
        and (p / "docker").is_dir()
        and (p / "src" / "nemo_evaluator").is_dir()
    )


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


class DockerExecutor:
    name = "docker"

    def run(self, config, *, dry_run=False, resume=False,
            background=False, submit=False) -> None:
        import click

        from nemo_evaluator.eval.containers import resolve_eval_image, scheme_to_variant

        base = config.cluster.container_image
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
        for m in config.cluster.container_mounts:
            mount_args.extend(["-v", m])
        if config.cluster.mount_home:
            home = os.environ.get("HOME", "")
            if home:
                mount_args.extend(["-v", f"{home}:{home}"])

        env_args: list[str] = []
        for k, v in config.cluster.container_env.items():
            env_args.extend(["-e", f"{k}={v}"])
        _FORWARD_ENVS = (
            "NEMO_API_KEY", "NEMO_MODEL_URL", "NEMO_MODEL_ID",
            "HF_TOKEN", "MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_TOKEN",
        )
        for key in _FORWARD_ENVS:
            val = os.environ.get(key)
            if val:
                env_args.extend(["-e", f"{key}={val}"])

        env_args.extend(["-e", "NEL_INNER_EXECUTION=1"])

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        cfg_path = Path(output_dir) / "_docker_config.json"
        cfg_path.write_text(
            json.dumps(config.model_dump(), default=str), encoding="utf-8",
        )

        shm = config.cluster.shm_size
        if not shm:
            services = config.resolved_services() if hasattr(config, "resolved_services") else {}
            if any(s.type == "gym" for s in services.values()):
                shm = "2g"
        extra_docker_args: list[str] = []
        if shm:
            extra_docker_args.extend([f"--shm-size={shm}"])

        cmd = [
            "docker", "run", "-d",
            "--name", f"nel-eval-{os.getpid()}",
            *extra_docker_args,
            *mount_args,
            *env_args,
            image,
            "nel", "eval", "run", str(cfg_path),
        ]
        if resume:
            cmd.append("--resume")

        if dry_run:
            click.echo(f"Docker command:\n  {shlex.join(cmd)}")
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
        click.echo(f"Container started: {container_id[:12]}")
        click.echo(f"Check status: nel eval status -o {output_dir}")
        click.echo(f"Stop:         nel eval stop -o {output_dir}")
        click.echo(f"Logs:         docker logs -f {container_id[:12]}")

    def status(self, output_dir: str | Path) -> ProcessState:
        meta = _read_meta(output_dir)
        if meta is None:
            return ProcessState("docker", False, {"error": f"No {_META_FILE} found"})

        container_id = meta.get("container_id", "")
        try:
            result = subprocess.run(
                ["docker", "inspect", "-f", "{{.State.Status}}", container_id],
                capture_output=True, text=True, timeout=10,
            )
            st = result.stdout.strip()
            return ProcessState("docker", st == "running",
                                {"container_id": container_id, "status": st})
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            return ProcessState("docker", False,
                                {"container_id": container_id, "error": str(e)})

    def stop(self, output_dir: str | Path) -> bool:
        meta = _read_meta(output_dir)
        if meta is None:
            logger.warning("No %s found in %s", _META_FILE, output_dir)
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
