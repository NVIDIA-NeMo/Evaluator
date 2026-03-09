"""Gym environments: remote client + managed server orchestration."""
from __future__ import annotations

import logging
import os
import signal
import socket
import subprocess
import sys
import time
from typing import TYPE_CHECKING, Any

import httpx

from nemo_evaluator.environments.base import EvalEnvironment, SeedResult, VerifyResult

if TYPE_CHECKING:
    from nemo_evaluator.sandbox.base import Sandbox

logger = logging.getLogger(__name__)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class GymEnvironment(EvalEnvironment):
    """Remote environment client using seed_session/verify REST calls."""

    def __init__(self, endpoint: str, timeout: float = 60.0) -> None:
        super().__init__()
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout
        self.name = f"gym@{self.endpoint}"
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def seed(self, idx: int) -> SeedResult:
        c = await self._get_client()
        r = await c.post(f"{self.endpoint}/seed_session", json={"idx": idx})
        r.raise_for_status()
        d = r.json()
        sandbox_spec = None
        if d.get("sandbox_spec"):
            from nemo_evaluator.sandbox.base import SandboxSpec
            sandbox_spec = SandboxSpec(**d["sandbox_spec"])
        return SeedResult(
            prompt=d.get("prompt", ""), expected_answer=d.get("expected_answer", ""),
            metadata=d.get("metadata", {}),
            messages=d.get("messages"), system=d.get("system"),
            sandbox_spec=sandbox_spec,
        )

    async def verify(self, response: str, expected: str,
                     sandbox: Sandbox | None = None, **meta: Any) -> VerifyResult:
        c = await self._get_client()
        r = await c.post(
            f"{self.endpoint}/verify",
            json={"response": response, "expected": expected, "metadata": meta},
        )
        r.raise_for_status()
        d = r.json()
        return VerifyResult(
            reward=float(d.get("reward", 0.0)),
            extracted_answer=d.get("extracted_answer"),
            scoring_details=d.get("scoring_details", {}),
            metadata=d.get("metadata", {}),
        )

    async def dataset_size(self) -> int:
        try:
            c = await self._get_client()
            r = await c.get(f"{self.endpoint}/dataset_size")
            r.raise_for_status()
            return r.json().get("size", -1)
        except (httpx.HTTPError, KeyError, ValueError):
            return -1

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None


class ManagedGymEnvironment(EvalEnvironment):
    """Starts a Gym-compatible server, runs eval through it, tears it down."""

    def __init__(
        self,
        server_cmd: str | None = None,
        server_module: str | None = None,
        nel_benchmark: str | None = None,
        host: str = "127.0.0.1",
        port: int | None = None,
        startup_timeout: float = 60.0,
        request_timeout: float = 120.0,
    ) -> None:
        super().__init__()
        self._server_cmd = server_cmd
        self._server_module = server_module
        self._nel_benchmark = nel_benchmark
        self._host = host
        self._port = port or _find_free_port()
        self._startup_timeout = startup_timeout
        self._process: subprocess.Popen | None = None
        self._inner: GymEnvironment | None = None
        self._request_timeout = request_timeout
        self.name = f"managed-gym@{host}:{self._port}"

    @property
    def endpoint(self) -> str:
        return f"http://{self._host}:{self._port}"

    def start(self) -> None:
        if self._process is not None:
            return
        cmd = self._build_cmd()
        logger.info("Starting managed Gym server: %s", " ".join(cmd) if isinstance(cmd, list) else cmd)
        env = {**os.environ, "PORT": str(self._port)}
        self._process = subprocess.Popen(
            cmd, shell=isinstance(cmd, str), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
        )
        self._wait_for_health()
        self._inner = GymEnvironment(self.endpoint, timeout=self._request_timeout)
        logger.info("Managed Gym server ready at %s (pid=%d)", self.endpoint, self._process.pid)

    def _build_cmd(self) -> list[str] | str:
        if self._server_cmd:
            return f"{self._server_cmd} --port {self._port}"
        if self._server_module:
            return [sys.executable, "-m", "uvicorn", f"{self._server_module}:app",
                    "--host", self._host, "--port", str(self._port)]
        if self._nel_benchmark:
            return [sys.executable, "-m", "nemo_evaluator.cli.main", "serve",
                    "--benchmark", self._nel_benchmark, "--port", str(self._port), "--host", self._host]
        raise ValueError("ManagedGymEnvironment requires one of: server_cmd, server_module, or nel_benchmark")

    def _wait_for_health(self) -> None:
        deadline = time.monotonic() + self._startup_timeout
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                raise RuntimeError(f"Server exited with code {self._process.returncode} during startup")
            try:
                r = httpx.get(f"{self.endpoint}/health", timeout=2.0)
                if r.status_code == 200:
                    return
            except (httpx.ConnectError, httpx.TimeoutException):
                pass
            time.sleep(0.5)
        self.stop()
        raise TimeoutError(f"Server at {self.endpoint} not healthy within {self._startup_timeout}s")

    def stop(self) -> None:
        if self._process is None:
            return
        logger.info("Stopping managed Gym server (pid=%d)", self._process.pid)
        try:
            self._process.send_signal(signal.SIGTERM)
            self._process.wait(timeout=10)
        except (subprocess.TimeoutExpired, OSError):
            self._process.kill()
            self._process.wait(timeout=5)
        self._process = None

    async def seed(self, idx: int) -> SeedResult:
        if self._inner is None:
            self.start()
        return await self._inner.seed(idx)

    async def verify(self, response: str, expected: str,
                     sandbox: Sandbox | None = None, **meta: Any) -> VerifyResult:
        return await self._inner.verify(response, expected, sandbox=sandbox, **meta)

    async def dataset_size(self) -> int:
        if self._inner is None:
            self.start()
        return await self._inner.dataset_size()

    async def close(self) -> None:
        if self._inner:
            await self._inner.close()
        self.stop()

    def __del__(self):
        self.stop()
