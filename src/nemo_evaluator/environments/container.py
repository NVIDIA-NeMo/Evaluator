"""ContainerEnvironment: run legacy eval-factory containers as opaque environments.

Wraps any legacy evaluation container image, runs it via `docker run`,
and parses the output into the NEL artifact bundle format. The container
owns the full eval loop (dataset, model call, scoring).

Usage in config:
    benchmarks:
      - name: container://nvcr.io/eval-factory/mmlu:latest#mmlu_pro
"""
from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

import yaml

from nemo_evaluator.environments.base import EvalEnvironment, SeedResult, VerifyResult

logger = logging.getLogger(__name__)

# Maps well-known harness names to NVCR container images.
# Users can override via the full container:// URI.
DEFAULT_IMAGE_MAP: dict[str, str] = {}


class ContainerEnvironment(EvalEnvironment):
    """Runs a legacy eval-factory container and parses its results.

    The container is expected to produce results at /results/results.yml
    inside the container. The host mounts a temp directory for output.
    """

    def __init__(
        self,
        image: str,
        task: str = "",
        model_url: str | None = None,
        model_id: str | None = None,
        api_key: str | None = None,
        extra_env: dict[str, str] | None = None,
        extra_mounts: list[str] | None = None,
        pre_cmd: str | None = None,
        timeout: float = 3600.0,
    ) -> None:
        super().__init__()
        self.name = f"container/{task or image.split('/')[-1].split(':')[0]}"
        self._image = image
        self._task = task
        self._model_url = model_url
        self._model_id = model_id
        self._api_key = api_key
        self._extra_env = extra_env or {}
        self._extra_mounts = extra_mounts or []
        self._pre_cmd = pre_cmd
        self._timeout = timeout

    async def dataset_size(self) -> int:
        return 0

    async def seed(self, idx: int) -> SeedResult:
        raise NotImplementedError("ContainerEnvironment uses run_batch()")

    async def verify(self, response: str, expected: str, **metadata: Any) -> VerifyResult:
        raise NotImplementedError("ContainerEnvironment uses run_batch()")

    async def run_batch(self, solver: Any = None, config: dict[str, Any] | None = None) -> dict[str, Any]:
        """Run the container and parse results."""
        config = config or {}
        model_url = self._model_url or config.get("base_url", "")
        model_id = self._model_id or config.get("model", "")
        api_key = self._api_key or config.get("api_key", "")

        with tempfile.TemporaryDirectory(prefix="nel_container_") as tmpdir:
            results_dir = Path(tmpdir) / "results"
            results_dir.mkdir()

            container_config = self._build_container_config(model_url, model_id)
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text(yaml.dump(container_config), encoding="utf-8")

            cmd = self._build_docker_cmd(tmpdir, results_dir, config_path)

            env_vars = {
                "NEMO_MODEL_URL": model_url,
                "NEMO_MODEL_ID": model_id,
            }
            if api_key:
                env_vars["NEMO_API_KEY"] = api_key
            env_vars.update(self._extra_env)

            for k, v in env_vars.items():
                cmd.extend(["-e", f"{k}={v}"])

            cmd.append(self._image)

            if self._pre_cmd:
                cmd.extend(["bash", "-c", self._pre_cmd])

            logger.info("Running container: %s", " ".join(cmd[:10]) + "...")

            import asyncio as _aio
            proc = await _aio.create_subprocess_exec(
                *cmd,
                stdout=_aio.subprocess.PIPE,
                stderr=_aio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await _aio.wait_for(proc.communicate(), timeout=self._timeout)
            except _aio.TimeoutError:
                proc.kill()
                await proc.wait()
                stdout, stderr = b"", b"timeout"

            if proc.returncode != 0:
                logger.error("Container failed (exit %d): %s",
                             proc.returncode, (stderr or b"").decode()[:1000])

            return self._parse_results(results_dir, proc.returncode or 0)

    def _build_container_config(self, model_url: str, model_id: str) -> dict:
        return {
            "task": self._task,
            "model": {"url": model_url, "id": model_id},
        }

    def _build_docker_cmd(self, tmpdir: str, results_dir: Path, config_path: Path) -> list[str]:
        cmd = [
            "docker", "run", "--rm",
            "-v", f"{results_dir}:/results",
            "-v", f"{config_path}:/config/config.yaml",
        ]
        for mount in self._extra_mounts:
            cmd.extend(["-v", mount])
        return cmd

    def _parse_results(self, results_dir: Path, exit_code: int) -> dict[str, Any]:
        """Parse container output into NEL bundle format."""
        results_file = results_dir / "results.yml"
        results_json = results_dir / "results.json"

        raw: dict[str, Any] = {}
        if results_file.exists():
            raw = yaml.safe_load(results_file.read_text()) or {}
        elif results_json.exists():
            raw = json.loads(results_json.read_text())

        scores: dict[str, Any] = {}
        source = raw.get("metrics") or raw.get("scores") or {}
        for key, value in source.items():
            if isinstance(value, (int, float)):
                scores[key] = {"value": round(float(value), 4)}
            elif isinstance(value, dict) and "value" in value:
                scores[key] = value

        return {
            "benchmark": {
                "name": self.name,
                "samples": raw.get("samples", raw.get("n_samples", 0)),
                "scores": scores,
            },
            "config": {
                "benchmark": self.name,
                "image": self._image,
                "task": self._task,
                "framework": "container",
            },
            "_container_exit_code": exit_code,
        }
