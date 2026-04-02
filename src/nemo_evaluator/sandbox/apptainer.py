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
"""Apptainer-based per-problem sandbox using persistent instances.

Each sandbox is an Apptainer instance started from a SIF file.  Apptainer
shares the host network by default, so ``resolve_outside_endpoint`` is a
no-op and ``container_ip`` returns the node address (or ``localhost``).

File transfer uses a bind-mounted staging directory since Apptainer has no
equivalent to ``docker cp``.

Requires Apptainer >= 1.1 for ``instance start`` / ``instance stop``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Self
from urllib.parse import urlparse, urlunparse
from uuid import uuid4

from nemo_evaluator.sandbox.base import ExecResult, OutsideEndpoint, SandboxSpec

logger = logging.getLogger(__name__)

_MIN_VERSION = (1, 1)
_version_checked = False


def _check_apptainer_version() -> None:
    """Verify Apptainer >= 1.1 is available (checked once per process)."""
    global _version_checked
    if _version_checked:
        return
    try:
        proc = subprocess.run(
            ["apptainer", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        raw = proc.stdout.strip().split()[-1]
        parts = tuple(int(x) for x in raw.split(".")[:2])
        if parts < _MIN_VERSION:
            raise RuntimeError(
                f"ApptainerSandbox requires Apptainer >= {'.'.join(map(str, _MIN_VERSION))}, found {raw}"
            )
        _version_checked = True
    except FileNotFoundError:
        raise RuntimeError("apptainer binary not found on PATH")


class ApptainerSandbox:
    """Per-problem Apptainer persistent instance.

    Lifecycle:
        start  → ``apptainer instance start --writable-tmpfs {sif} {name}``
        exec   → ``apptainer exec instance://{name} bash -c {cmd}``
        stop   → ``apptainer instance stop {name}``

    File transfer uses a staging directory bind-mounted at ``/_staging``.
    """

    def __init__(
        self,
        spec: SandboxSpec,
        *,
        node: str | None = None,
        sif_cache_dir: str | None = None,
        memory_mb: int | None = None,
        het_group: int | None = None,
    ) -> None:
        _check_apptainer_version()
        self._spec = spec
        self._node = node
        self._sif_cache_dir = sif_cache_dir or os.environ.get("APPTAINER_CACHEDIR", "/tmp/nel_sif_cache")
        self._memory_mb = memory_mb
        self._het_group = het_group
        self._instance_name = f"nel-{uuid4().hex[:12]}"
        self._staging_dir: Path | None = None
        self._running = False

    @property
    def spec(self) -> SandboxSpec:
        return self._spec

    def _sif_path(self) -> str:
        """Resolve image to SIF path. If already a .sif path, use directly."""
        image = self._spec.image
        if image.endswith(".sif") or "/" in image and os.path.exists(image):
            return image
        safe = image.replace("/", "_").replace(":", "__")
        return str(Path(self._sif_cache_dir) / f"{safe}.sif")

    def _srun_prefix(self) -> list[str]:
        """When running on a SLURM node, prefix commands with srun."""
        if self._node:
            prefix = [
                "srun",
                "--overlap",
                f"--nodelist={self._node}",
                "--ntasks=1",
            ]
            if self._het_group is not None:
                prefix.append(f"--het-group={self._het_group}")
            return prefix
        return []

    async def start(
        self,
        *,
        outside_endpoints: list[OutsideEndpoint] | None = None,
    ) -> None:
        import tempfile

        staging_base = self._sif_cache_dir if self._node else None
        staging = tempfile.mkdtemp(prefix="nel_apptainer_staging_", dir=staging_base)
        self._staging_dir = Path(staging)

        sif = self._sif_path()
        entrypoint = self._spec.entrypoint or "sleep infinity"

        cmd: list[str] = [*self._srun_prefix()]
        cmd.extend(
            [
                "apptainer",
                "instance",
                "start",
                "--writable-tmpfs",
                "--cleanenv",
                "--no-home",
                "--bind",
                f"{staging}:/_staging",
            ]
        )

        for vol in self._spec.volumes:
            if vol.is_efs:
                continue
            bind = f"{vol.host_path}:{vol.container_path}"
            if vol.readonly:
                bind += ":ro"
            cmd.extend(["--bind", bind])

        if self._memory_mb:
            cmd.extend(["--memory", f"{self._memory_mb}M"])

        for k, v in self._spec.env.items():
            cmd.extend(["--env", f"{k}={v}"])
        if outside_endpoints:
            for ep in outside_endpoints:
                resolved = self.resolve_outside_endpoint(ep.url)
                cmd.extend(["--env", f"{ep.env_var}={resolved}"])

        cmd.extend([sif, self._instance_name])

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"apptainer instance start failed (rc={proc.returncode}): {stderr.decode()[:500]}")
        self._running = True

        if entrypoint != "sleep infinity":
            await self.exec(entrypoint, timeout_sec=30)

        for local, remote in self._spec.files.items():
            await self.upload(Path(local), remote)

        logger.debug(
            "apptainer sandbox started: %s sif=%s node=%s",
            self._instance_name,
            sif,
            self._node or "local",
        )

    async def stop(self) -> None:
        if not self._running:
            return
        cmd = [*self._srun_prefix(), "apptainer", "instance", "stop", self._instance_name]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.wait()
        self._running = False

        if self._staging_dir and self._staging_dir.exists():
            shutil.rmtree(self._staging_dir, ignore_errors=True)
            self._staging_dir = None

        logger.debug("apptainer sandbox stopped: %s", self._instance_name)

    async def exec(
        self,
        command: str,
        timeout_sec: float = 180,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        if not self._running:
            raise RuntimeError("Sandbox not started")
        if user is not None:
            logger.warning("ApptainerSandbox does not support user switching; ignoring user=%r", user)
        shell_cmd = command
        if env:
            exports = " ".join(f"{k}={v}" for k, v in env.items())
            shell_cmd = f"export {exports} && {shell_cmd}"
        if cwd:
            shell_cmd = f"cd {cwd} && {shell_cmd}"
        cmd = [
            *self._srun_prefix(),
            "apptainer",
            "exec",
            f"instance://{self._instance_name}",
            "/bin/bash",
            "-c",
            shell_cmd,
        ]
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
        return ExecResult(
            stdout.decode(),
            stderr.decode(),
            proc.returncode or 0,
        )

    async def upload(self, local_path: Path, remote_path: str) -> None:
        if not self._staging_dir:
            raise RuntimeError("Sandbox not started")
        staged = self._staging_dir / local_path.name
        shutil.copy2(str(local_path), str(staged))
        result = await self.exec(
            f"cp /_staging/{local_path.name} {remote_path}",
            timeout_sec=30,
        )
        if result.return_code != 0:
            raise RuntimeError(f"upload failed: {result.stderr[:300]}")

    async def download(self, remote_path: str, local_path: Path) -> None:
        if not self._staging_dir:
            raise RuntimeError("Sandbox not started")
        fname = Path(remote_path).name
        result = await self.exec(
            f"cp {remote_path} /_staging/{fname}",
            timeout_sec=30,
        )
        if result.return_code != 0:
            raise RuntimeError(f"download failed: {result.stderr[:300]}")
        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(self._staging_dir / fname), str(local_path))

    def resolve_outside_endpoint(self, url: str) -> str:
        if not self._node:
            return url
        parsed = urlparse(url)
        if parsed.hostname in ("localhost", "127.0.0.1", "::1"):
            evaluator_host = platform.node()
            port = parsed.port
            new_netloc = f"{evaluator_host}:{port}" if port else evaluator_host
            return urlunparse(parsed._replace(netloc=new_netloc))
        return url

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def container_ip(self) -> str | None:
        if self._node:
            return self._node
        return "localhost"

    async def __aenter__(self) -> Self:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.stop()
