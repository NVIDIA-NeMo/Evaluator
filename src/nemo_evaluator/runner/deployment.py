"""Model server deployment: start, health-check, stop."""
from __future__ import annotations

import logging
import os
import signal
import socket
import subprocess
import time
from typing import Protocol

import httpx

from nemo_evaluator.executors.base import DeployConfig

logger = logging.getLogger(__name__)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class ModelDeployment(Protocol):
    def start(self) -> str: ...
    def health_wait(self, timeout: float = 600.0) -> None: ...
    def stop(self) -> None: ...


class APIDeployment:
    """External API -- no server to manage."""

    def __init__(self, base_url: str) -> None:
        self._url = base_url

    def start(self) -> str:
        return self._url

    def health_wait(self, timeout: float = 600.0) -> None:
        pass

    def stop(self) -> None:
        pass


class DockerModelDeployment:
    """Starts a model server in a Docker container."""

    def __init__(self, config: DeployConfig) -> None:
        self._config = config
        self._port = config.port or _find_free_port()
        self._container_name = f"nel-model-{self._port}"
        self._process_started = False

    def _build_docker_cmd(self) -> list[str]:
        c = self._config
        cmd = [
            "docker", "run", "--rm", "-d",
            "--name", self._container_name,
            "-p", f"{self._port}:{self._port}",
        ]

        if c.gpus > 0:
            cmd += ["--gpus", f'"device={",".join(str(i) for i in range(c.gpus))}"']

        for k, v in c.extra_env.items():
            cmd += ["-e", f"{k}={v}"]

        cmd.append(c.image)
        cmd.extend(c.extra_args)
        return cmd

    def start(self) -> str:
        cmd = self._build_docker_cmd()
        logger.info("Starting model server: %s", " ".join(cmd))

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to start model container: {result.stderr}")

        self._process_started = True
        self.health_wait(self._config.startup_timeout)
        return f"http://localhost:{self._port}/v1"

    def health_wait(self, timeout: float = 600.0) -> None:
        url = f"http://localhost:{self._port}{self._config.health_path}"
        deadline = time.monotonic() + timeout
        logger.info("Waiting for model server health at %s (timeout=%.0fs)", url, timeout)

        while time.monotonic() < deadline:
            try:
                r = httpx.get(url, timeout=5.0)
                if r.status_code == 200:
                    logger.info("Model server healthy")
                    return
            except (httpx.ConnectError, httpx.TimeoutException):
                pass
            time.sleep(5.0)

        self.stop()
        raise TimeoutError(f"Model server not healthy within {timeout}s")

    def stop(self) -> None:
        if not self._process_started:
            return
        logger.info("Stopping model container: %s", self._container_name)
        subprocess.run(["docker", "stop", self._container_name],
                       capture_output=True, timeout=30)
        self._process_started = False


class ProcessModelDeployment:
    """Starts a model server as a local process (for vLLM, sglang without Docker)."""

    def __init__(self, config: DeployConfig) -> None:
        self._config = config
        self._port = config.port or _find_free_port()
        self._process: subprocess.Popen | None = None

    def _build_cmd(self) -> list[str]:
        c = self._config
        if c.type == "vllm":
            return [
                "python", "-m", "vllm.entrypoints.openai.api_server",
                "--model", c.model or "",
                "--port", str(self._port),
                *c.extra_args,
            ]
        if c.type == "sglang":
            return [
                "python", "-m", "sglang.launch_server",
                "--model-path", c.model or "",
                "--port", str(self._port),
                *c.extra_args,
            ]
        raise ValueError(f"ProcessModelDeployment does not support type={c.type!r}")

    def start(self) -> str:
        cmd = self._build_cmd()
        logger.info("Starting model process: %s", " ".join(cmd))

        env = {**os.environ, **self._config.extra_env}
        self._process = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        self.health_wait(self._config.startup_timeout)
        return f"http://localhost:{self._port}/v1"

    def health_wait(self, timeout: float = 600.0) -> None:
        url = f"http://localhost:{self._port}{self._config.health_path}"
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            if self._process and self._process.poll() is not None:
                raise RuntimeError(f"Model process exited with code {self._process.returncode}")
            try:
                r = httpx.get(url, timeout=5.0)
                if r.status_code == 200:
                    logger.info("Model server healthy at %s", url)
                    return
            except (httpx.ConnectError, httpx.TimeoutException):
                pass
            time.sleep(5.0)

        self.stop()
        raise TimeoutError(f"Model not healthy within {timeout}s")

    def stop(self) -> None:
        if self._process is None:
            return
        logger.info("Stopping model process (pid=%d)", self._process.pid)
        try:
            self._process.send_signal(signal.SIGTERM)
            self._process.wait(timeout=15)
        except (subprocess.TimeoutExpired, OSError):
            self._process.kill()
            self._process.wait(timeout=5)
        self._process = None


# NIM-specific: uses Docker with NIM-specific env vars
class NIMDeployment(DockerModelDeployment):
    """NIM container deployment with standard NIM env vars."""

    def _build_docker_cmd(self) -> list[str]:
        c = self._config
        c.extra_env.setdefault("NIM_SERVED_MODEL_NAME", c.model or "model")
        c.health_path = c.health_path or "/v1/health/ready"
        return super()._build_docker_cmd()


_DEPLOYMENT_MAP = {
    "api": lambda c: APIDeployment(c.extra_env.get("base_url", "")),
    "nim": NIMDeployment,
    "docker": DockerModelDeployment,
    "vllm": ProcessModelDeployment,
    "sglang": ProcessModelDeployment,
}


def get_deployment(config: DeployConfig) -> ModelDeployment:
    factory = _DEPLOYMENT_MAP.get(config.type)
    if factory is None:
        raise ValueError(f"Unknown deployment type: {config.type!r}. "
                         f"Available: {', '.join(sorted(_DEPLOYMENT_MAP))}")
    return factory(config)
