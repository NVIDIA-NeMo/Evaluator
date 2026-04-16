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
"""Gym environments: unified client + managed server lifecycle.

Two classes:

- ``GymEnvironment``: HTTP client that talks to any Gym-compatible server.
  Supports two protocols (``evaluator`` and ``native``) and an optional
  ``GymDataset`` for environments where the server doesn't serve task data.

- ``ManagedGymEnvironment``: Starts a subprocess server, delegates to
  ``GymEnvironment``, and tears down the process on close.

One helper:

- ``GymDataset``: Loads Gym JSONL task files.  Provides ``__len__`` and
  ``__getitem__`` for task data without mixing file I/O into the HTTP client.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import aiohttp

from nemo_evaluator.environments.base import EvalEnvironment, SeedResult, VerifyResult
from nemo_evaluator.environments.gym_protocol import (
    extract_prompt_from_rcp,
    messages_from_rcp,
    wrap_text_as_gym_response,
    wrap_text_as_responses_create_params,
)

if TYPE_CHECKING:
    from nemo_evaluator.sandbox.base import Sandbox

logger = logging.getLogger(__name__)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


# ---------------------------------------------------------------------------
# GymDataset -- JSONL loader, separate from the HTTP client
# ---------------------------------------------------------------------------


class GymDataset:
    """Loads a Gym-format JSONL file (responses_create_params + metadata per row).

    Each row is expected to have at least ``responses_create_params``
    with an ``input`` field containing the prompt messages.  Additional
    fields (``db_id``, ``gold_sql``, ``verifier_metadata``, etc.) are
    benchmark-specific and are forwarded to the server at verify time.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._rows: list[dict[str, Any]] = []
        if self._path.exists():
            with open(self._path) as f:
                self._rows = [json.loads(line) for line in f if line.strip()]

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self._rows[idx]

    @property
    def name(self) -> str:
        return self._path.stem


# ---------------------------------------------------------------------------
# GymEnvironment -- unified HTTP client
# ---------------------------------------------------------------------------


class GymEnvironment(EvalEnvironment):
    """HTTP client for Gym-compatible servers.

    Parameters
    ----------
    endpoint:
        Base URL of the server (e.g. ``http://localhost:8000``).
    protocol:
        ``"evaluator"`` -- sends plain text to ``/verify`` (for ``nel serve``).
        ``"native"`` -- wraps text in NeMoGymResponse envelope and forwards
        all per-task metadata (for native Gym resource servers).
    dataset:
        Optional ``GymDataset``.  When provided, ``seed()`` reads from the
        dataset instead of calling ``/seed_session``, and ``dataset_size()``
        returns the dataset length.  Required for ``protocol="native"``
        since native Gym servers don't serve task data via HTTP.
    timeout:
        HTTP request timeout in seconds.
    """

    def __init__(
        self,
        endpoint: str,
        *,
        protocol: Literal["evaluator", "native"] = "evaluator",
        dataset: GymDataset | None = None,
        timeout: float = 60.0,
    ) -> None:
        super().__init__()
        self.endpoint = endpoint.rstrip("/")
        self.protocol = protocol
        self._dataset = dataset
        self.timeout = timeout
        self._client: aiohttp.ClientSession | None = None

        ds_label = dataset.name if dataset else self.endpoint
        self.name = f"gym@{ds_label}" if protocol == "evaluator" else f"gym-native@{ds_label}"

    async def _get_client(self) -> aiohttp.ClientSession:
        if self._client is None or self._client.closed:
            self._client = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            )
        return self._client

    # -- seed ---------------------------------------------------------------

    async def seed(self, idx: int) -> SeedResult:
        if self._dataset is not None:
            return self._seed_from_dataset(idx)
        return await self._seed_from_server(idx)

    def _seed_from_dataset(self, idx: int) -> SeedResult:
        row = self._dataset[idx]
        rcp = row.get("responses_create_params", {})
        prompt = extract_prompt_from_rcp(rcp)
        expected = row.get("expected_answer", "")

        # Clean metadata: exclude bulky/internal fields that should not
        # be serialized into step logs.  The original row is kept in
        # self._dataset and looked up by problem_idx at verify time.
        meta = {
            k: v
            for k, v in row.items()
            if k not in ("responses_create_params", "expected_answer")
            and not isinstance(v, (list, dict))
            or k in ("verifier_metadata",)
        }

        # Forward tool definitions and reasoning config from responses_create_params
        # so that GymSolver can reconstruct the native request for agentic benchmarks.
        if "tools" in rcp:
            meta["tools"] = rcp["tools"]
        if "reasoning" in rcp:
            meta["reasoning"] = rcp["reasoning"]

        return SeedResult(
            prompt=prompt,
            expected_answer=expected,
            metadata=meta,
            messages=messages_from_rcp(rcp),
        )

    async def _seed_from_server(self, idx: int) -> SeedResult:
        c = await self._get_client()
        async with c.post(f"{self.endpoint}/seed_session", json={"idx": idx}) as r:
            r.raise_for_status()
            d = await r.json()
        sandbox_spec = None
        if d.get("sandbox_spec"):
            from nemo_evaluator.sandbox.base import SandboxSpec

            sandbox_spec = SandboxSpec(**d["sandbox_spec"])
        return SeedResult(
            prompt=d.get("prompt", ""),
            expected_answer=d.get("expected_answer", ""),
            metadata=d.get("metadata", {}),
            messages=d.get("messages"),
            system=d.get("system"),
            sandbox_spec=sandbox_spec,
        )

    # -- verify -------------------------------------------------------------

    async def verify(self, response: str, expected: str, sandbox: Sandbox | None = None, **meta: Any) -> VerifyResult:
        if self.protocol == "native":
            return await self._verify_native(response, expected, **meta)
        return await self._verify_evaluator(response, expected, **meta)

    async def _verify_evaluator(self, response: str, expected: str, **meta: Any) -> VerifyResult:
        c = await self._get_client()
        async with c.post(
            f"{self.endpoint}/verify",
            json={"response": response, "expected": expected, "metadata": meta},
        ) as r:
            r.raise_for_status()
            d = await r.json()
        return VerifyResult(
            reward=float(d.get("reward", 0.0)),
            extracted_answer=d.get("extracted_answer"),
            scoring_details=d.get("scoring_details", {}),
            metadata=d.get("metadata", {}),
        )

    async def _verify_native(self, response: str, expected: str, **meta: Any) -> VerifyResult:
        c = await self._get_client()

        # Look up the full original row from the dataset by problem_idx
        # to get responses_create_params and benchmark-specific fields
        # without smuggling them through SeedResult.metadata.
        problem_idx = meta.get("problem_idx")
        row: dict[str, Any] = {}
        if self._dataset is not None and problem_idx is not None:
            try:
                row = self._dataset[int(problem_idx)]
            except (IndexError, TypeError):
                pass

        rcp = row.get("responses_create_params")
        if rcp is None:
            rcp = wrap_text_as_responses_create_params(meta.get("prompt", ""))

        body: dict[str, Any] = {
            "responses_create_params": rcp,
            "response": wrap_text_as_gym_response(response),
        }
        # Forward benchmark-specific fields from the original row
        for k, v in row.items():
            if k in ("responses_create_params", "expected_answer"):
                continue
            body[k] = v

        # When no dataset row provided verifier_metadata, synthesize it
        # from the eval-loop metadata so gym servers can read their fields.
        if "verifier_metadata" not in body:
            _INTERNAL = {"source", "benchmark", "problem_idx", "prompt", "verifier_metadata"}
            body["verifier_metadata"] = {k: v for k, v in meta.items() if k not in _INTERNAL}

        # Also forward any metadata the eval loop passed that isn't already set
        for k, v in meta.items():
            if k not in body and k != "problem_idx":
                body[k] = v

        async with c.post(f"{self.endpoint}/verify", json=body) as r:
            r.raise_for_status()
            d = await r.json()
        return VerifyResult(
            reward=float(d.get("reward", 0.0)),
            extracted_answer=d.get("extracted_sql") or d.get("extracted_answer"),
            scoring_details={k: v for k, v in d.items() if k not in ("reward", "responses_create_params", "response")},
            metadata=d.get("metadata", {}),
        )

    # -- dataset_size -------------------------------------------------------

    async def dataset_size(self) -> int:
        if self._dataset is not None:
            return len(self._dataset)
        try:
            c = await self._get_client()
            async with c.get(f"{self.endpoint}/dataset_size") as r:
                r.raise_for_status()
                d = await r.json()
            return d.get("size", -1)
        except Exception:
            return -1

    async def close(self) -> None:
        if self._client and not self._client.closed:
            await self._client.close()
            self._client = None


# ---------------------------------------------------------------------------
# ManagedGymEnvironment -- subprocess lifecycle
# ---------------------------------------------------------------------------


class ManagedGymEnvironment(EvalEnvironment):
    """Starts a subprocess server, delegates to GymEnvironment, tears it down.

    Parameters
    ----------
    server_cmd / server_module / nel_benchmark:
        How to start the server (at least one required).
    protocol:
        Passed to the inner ``GymEnvironment``.
    dataset:
        Passed to the inner ``GymEnvironment``.
    """

    def __init__(
        self,
        server_cmd: str | None = None,
        server_module: str | None = None,
        nel_benchmark: str | None = None,
        host: str = "127.0.0.1",
        port: int | None = None,
        startup_timeout: float = 60.0,
        request_timeout: float = 120.0,
        protocol: Literal["evaluator", "native"] = "evaluator",
        dataset: GymDataset | None = None,
    ) -> None:
        super().__init__()
        self._server_cmd = server_cmd
        self._server_module = server_module
        self._nel_benchmark = nel_benchmark
        self._host = host
        self._port = port or _find_free_port()
        self._startup_timeout = startup_timeout
        self._request_timeout = request_timeout
        self._protocol = protocol
        self._dataset_obj = dataset
        self._process: subprocess.Popen | None = None
        self._inner: GymEnvironment | None = None
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
            cmd,
            shell=isinstance(cmd, str),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
        )
        self._wait_for_health()
        self._inner = GymEnvironment(
            self.endpoint,
            protocol=self._protocol,
            dataset=self._dataset_obj,
            timeout=self._request_timeout,
        )
        logger.info("Managed Gym server ready at %s (pid=%d)", self.endpoint, self._process.pid)

    def _build_cmd(self) -> list[str] | str:
        if self._server_cmd:
            return self._server_cmd
        if self._server_module:
            return [
                sys.executable,
                "-m",
                "uvicorn",
                f"{self._server_module}:app",
                "--host",
                self._host,
                "--port",
                str(self._port),
            ]
        if self._nel_benchmark:
            return [
                sys.executable,
                "-m",
                "nemo_evaluator.cli.main",
                "serve",
                "--benchmark",
                self._nel_benchmark,
                "--port",
                str(self._port),
                "--host",
                self._host,
            ]
        raise ValueError("ManagedGymEnvironment requires one of: server_cmd, server_module, or nel_benchmark")

    def _wait_for_health(self) -> None:
        """Wait for the server to become responsive.

        Tries ``/health`` first (evaluator servers).  Falls back to
        ``/openapi.json`` which every FastAPI app serves by default
        (native Gym resource servers don't expose ``/health``).
        """
        deadline = time.monotonic() + self._startup_timeout
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                output = ""
                if self._process.stdout:
                    output = self._process.stdout.read().decode(errors="replace")
                raise RuntimeError(
                    f"Server exited with code {self._process.returncode} during startup.\nOutput:\n{output}"
                )
            try:
                with urllib.request.urlopen(
                    f"{self.endpoint}/health",
                    timeout=2.0,
                ) as r:
                    if r.status == 200:
                        return
            except (urllib.error.URLError, OSError):
                pass
            else:
                try:
                    with urllib.request.urlopen(
                        f"{self.endpoint}/openapi.json",
                        timeout=2.0,
                    ) as r2:
                        if r2.status == 200:
                            return
                except (urllib.error.URLError, OSError):
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

    async def verify(self, response: str, expected: str, sandbox: Sandbox | None = None, **meta: Any) -> VerifyResult:
        if self._inner is None:
            self.start()
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
