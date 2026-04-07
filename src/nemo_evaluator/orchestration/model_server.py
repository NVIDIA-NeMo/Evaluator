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
"""Model server deployment: start, health-check, stop."""

from __future__ import annotations

import logging
import os
import signal
import socket
import subprocess
import time
from dataclasses import dataclass, field
from typing import Protocol

import urllib.error
import urllib.request

logger = logging.getLogger(__name__)


@dataclass
class DeployConfig:
    type: str = "api"
    image: str | None = None
    model: str | None = None
    gpus: int = 1
    port: int = 8000
    health_path: str = "/health"
    startup_timeout: float = 600.0
    extra_env: dict[str, str] = field(default_factory=dict)
    cluster_env: dict[str, str] = field(default_factory=dict)
    extra_args: list[str] = field(default_factory=list)
    nodes: int = 1
    pipeline_parallel_size: int | None = None


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
            "docker",
            "run",
            "--rm",
            "-d",
            "--name",
            self._container_name,
            "-p",
            f"{self._port}:{self._port}",
        ]

        if c.gpus > 0:
            cmd += ["--gpus", f'"device={",".join(str(i) for i in range(c.gpus))}"']

        merged_env = {**c.cluster_env, **c.extra_env}
        for k, v in merged_env.items():
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
                with urllib.request.urlopen(url, timeout=5.0) as r:
                    if r.status == 200:
                        logger.info("Model server healthy")
                        return
            except (urllib.error.URLError, OSError):
                pass
            time.sleep(5.0)

        self.stop()
        raise TimeoutError(f"Model server not healthy within {timeout}s")

    def stop(self) -> None:
        if not self._process_started:
            return
        logger.info("Stopping model container: %s", self._container_name)
        subprocess.run(["docker", "stop", self._container_name], capture_output=True, timeout=30)
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
                "python",
                "-m",
                "vllm.entrypoints.openai.api_server",
                "--model",
                c.model or "",
                "--port",
                str(self._port),
                *c.extra_args,
            ]
        if c.type == "sglang":
            return [
                "python",
                "-m",
                "sglang.launch_server",
                "--model-path",
                c.model or "",
                "--port",
                str(self._port),
                *c.extra_args,
            ]
        raise ValueError(f"ProcessModelDeployment does not support type={c.type!r}")

    def start(self) -> str:
        cmd = self._build_cmd()
        logger.info("Starting model process: %s", " ".join(cmd))

        env = {**os.environ, **self._config.cluster_env, **self._config.extra_env}
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
                with urllib.request.urlopen(url, timeout=5.0) as r:
                    if r.status == 200:
                        logger.info("Model server healthy at %s", url)
                        return
            except (urllib.error.URLError, OSError):
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
        c.health_path = c.health_path or "/health"
        return super()._build_docker_cmd()


class RayMultiNodeDeployment:
    """Multi-node vLLM deployment via Ray for tensor + pipeline parallelism."""

    def __init__(self, config: DeployConfig) -> None:
        self._config = config
        self._port = config.port or _find_free_port()
        self._head_process: subprocess.Popen | None = None
        self._vllm_process: subprocess.Popen | None = None
        self._fallback: ProcessModelDeployment | None = None

    def start(self) -> str:
        c = self._config
        nodes = c.nodes
        pp = c.pipeline_parallel_size or 1

        if nodes <= 1 and pp <= 1:
            logger.info("Single-node deployment, delegating to ProcessModelDeployment")
            self._fallback = ProcessModelDeployment(c)
            return self._fallback.start()

        logger.info("Starting Ray head node for %d-node deployment", nodes)
        self._head_process = subprocess.Popen(
            ["ray", "start", "--head", "--port=6379", "--dashboard-port=8265"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        time.sleep(5)

        cmd = [
            "python",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            c.model or "",
            "--port",
            str(self._port),
        ]

        # vLLM uses Ray automatically for multi-GPU/multi-node when these are set
        tp = next(
            (a for i, a in enumerate(c.extra_args) if a == "--tensor-parallel-size" and i + 1 < len(c.extra_args)), None
        )
        if tp is None and hasattr(c, "gpus") and c.gpus > 1:
            cmd.extend(["--tensor-parallel-size", str(c.gpus)])

        if pp > 1:
            cmd.extend(["--pipeline-parallel-size", str(pp)])

        cmd.extend(c.extra_args)

        env = {**os.environ, **c.cluster_env, **c.extra_env}
        logger.info("Starting vLLM with Ray: %s", " ".join(cmd))
        self._vllm_process = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        self.health_wait(c.startup_timeout)
        return f"http://localhost:{self._port}/v1"

    def health_wait(self, timeout: float = 600.0) -> None:
        url = f"http://localhost:{self._port}{self._config.health_path}"
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            if self._vllm_process and self._vllm_process.poll() is not None:
                raise RuntimeError(f"vLLM process exited with code {self._vllm_process.returncode}")
            try:
                with urllib.request.urlopen(url, timeout=5.0) as r:
                    if r.status == 200:
                        logger.info("Multi-node vLLM healthy at %s", url)
                        return
            except (urllib.error.URLError, OSError):
                pass
            time.sleep(5.0)

        self.stop()
        raise TimeoutError(f"Multi-node vLLM not healthy within {timeout}s")

    def stop(self) -> None:
        if self._fallback is not None:
            self._fallback.stop()
            return

        for proc, name in [(self._vllm_process, "vLLM"), (self._head_process, "Ray head")]:
            if proc is None:
                continue
            logger.info("Stopping %s (pid=%d)", name, proc.pid)
            try:
                proc.send_signal(signal.SIGTERM)
                proc.wait(timeout=15)
            except (subprocess.TimeoutExpired, OSError):
                proc.kill()
                proc.wait(timeout=5)

        self._vllm_process = None
        self._head_process = None

        try:
            subprocess.run(["ray", "stop"], capture_output=True, timeout=10)
        except Exception:
            pass


_DEPLOYMENT_MAP = {
    "api": lambda c: APIDeployment(c.extra_env.get("base_url", "")),
    "nim": NIMDeployment,
    "docker": DockerModelDeployment,
    "vllm": lambda c: (
        RayMultiNodeDeployment(c) if (c.nodes > 1 or (c.pipeline_parallel_size or 0) > 1) else ProcessModelDeployment(c)
    ),
    "sglang": ProcessModelDeployment,
}


def get_deployment(config: DeployConfig) -> ModelDeployment:
    factory = _DEPLOYMENT_MAP.get(config.type)
    if factory is None:
        raise ValueError(f"Unknown deployment type: {config.type!r}. Available: {', '.join(sorted(_DEPLOYMENT_MAP))}")
    return factory(config)
