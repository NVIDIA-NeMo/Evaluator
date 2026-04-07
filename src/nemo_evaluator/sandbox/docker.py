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
"""Docker-based per-problem sandbox.

Each sandbox is a Docker container on a bridge network (default), giving it
its own IP address to avoid port collisions when running many concurrently.
All operations are async via ``asyncio.create_subprocess_exec``.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Self
from uuid import uuid4

from nemo_evaluator.sandbox.base import ExecResult, OutsideEndpoint, SandboxSpec

logger = logging.getLogger(__name__)


class DockerSandbox:
    """Per-problem Docker container. Async, bridge network by default."""

    def __init__(
        self,
        spec: SandboxSpec,
        *,
        network: str = "bridge",
        memory: str = "4g",
        cpus: float = 2.0,
    ) -> None:
        self._spec = spec
        self._network = network
        self._memory = memory
        self._cpus = cpus
        self._container_id: str | None = None
        self._container_ip: str | None = None
        self._name = f"nel-sandbox-{uuid4().hex[:8]}"

    @property
    def spec(self) -> SandboxSpec:
        return self._spec

    async def start(self, *, outside_endpoints: list[OutsideEndpoint] | None = None) -> None:
        entrypoint = self._spec.entrypoint or "sleep infinity"

        cmd: list[str] = [
            "docker",
            "run",
            "-d",
            f"--name={self._name}",
            f"--network={self._network}",
            f"--memory={self._memory}",
            f"--cpus={self._cpus}",
            f"-w={self._spec.workdir}",
        ]

        for vol in self._spec.volumes:
            if vol.is_efs:
                continue
            flag = f"{vol.host_path}:{vol.container_path}"
            if vol.readonly:
                flag += ":ro"
            cmd.extend(["-v", flag])

        for k, v in self._spec.env.items():
            cmd.extend(["-e", f"{k}={v}"])
        if outside_endpoints:
            for ep in outside_endpoints:
                resolved = self.resolve_outside_endpoint(ep.url)
                cmd.extend(["-e", f"{ep.env_var}={resolved}"])

        cmd.extend([self._spec.image, "/bin/bash", "-c", entrypoint])

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"docker run failed: {stderr.decode()}")
        self._container_id = stdout.decode().strip()

        if self._network != "host":
            inspect = await asyncio.create_subprocess_exec(
                "docker",
                "inspect",
                "-f",
                "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}",
                self._container_id,
                stdout=asyncio.subprocess.PIPE,
            )
            ip_out, _ = await inspect.communicate()
            self._container_ip = ip_out.decode().strip() or None

        for local, remote in self._spec.files.items():
            await self.upload(Path(local), remote)

        logger.debug("sandbox started: %s image=%s ip=%s", self._name, self._spec.image, self._container_ip)

    async def stop(self) -> None:
        if self._container_id:
            proc = await asyncio.create_subprocess_exec(
                "docker",
                "rm",
                "-f",
                self._container_id,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.wait()
            logger.debug("sandbox stopped: %s", self._name)
            self._container_id = None

    async def exec(
        self,
        command: str,
        timeout_sec: float = 180,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        if not self._container_id:
            raise RuntimeError("Sandbox not started")
        cmd: list[str] = ["docker", "exec"]
        if user is not None:
            cmd.extend(["-u", str(user)])
        if cwd:
            cmd.extend(["-w", cwd])
        if env:
            for k, v in env.items():
                cmd.extend(["-e", f"{k}={v}"])
        cmd.extend([self._container_id, "/bin/bash", "-c", command])
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout_sec,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return ExecResult("", "timeout", -1)
        return ExecResult(stdout.decode(), stderr.decode(), proc.returncode or 0)

    async def upload(self, local_path: Path, remote_path: str) -> None:
        if not self._container_id:
            raise RuntimeError("Sandbox not started")
        proc = await asyncio.create_subprocess_exec(
            "docker",
            "cp",
            str(local_path),
            f"{self._container_id}:{remote_path}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"docker cp upload failed: {stderr.decode()}")

    async def download(self, remote_path: str, local_path: Path) -> None:
        if not self._container_id:
            raise RuntimeError("Sandbox not started")
        local_path.parent.mkdir(parents=True, exist_ok=True)
        proc = await asyncio.create_subprocess_exec(
            "docker",
            "cp",
            f"{self._container_id}:{remote_path}",
            str(local_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"docker cp download failed: {stderr.decode()}")

    def resolve_outside_endpoint(self, url: str) -> str:
        if self._network == "host":
            return url
        return url.replace("localhost", "host.docker.internal").replace("127.0.0.1", "host.docker.internal")

    @property
    def is_running(self) -> bool:
        return self._container_id is not None

    @property
    def container_ip(self) -> str | None:
        return self._container_ip

    async def __aenter__(self) -> Self:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.stop()
