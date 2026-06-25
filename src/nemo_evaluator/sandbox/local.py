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
"""Local sandbox: async subprocess in a temp directory, no container isolation.

Useful for development and testing where Docker is unavailable.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import signal
import tempfile
from pathlib import Path
from typing import Self

from nemo_evaluator.sandbox.base import ExecResult, OutsideEndpoint, SandboxSpec

logger = logging.getLogger(__name__)


class LocalSandbox:
    """No container isolation — async subprocess in a temp dir."""

    def __init__(self, spec: SandboxSpec) -> None:
        self._spec = spec
        self._workdir: Path | None = None
        self._outside_endpoints: list[OutsideEndpoint] = []

    @property
    def spec(self) -> SandboxSpec:
        return self._spec

    async def start(self, *, outside_endpoints: list[OutsideEndpoint] | None = None) -> None:
        self._outside_endpoints = outside_endpoints or []
        self._workdir = Path(tempfile.mkdtemp(prefix="nel-sandbox-"))
        for local, remote in self._spec.files.items():
            dest = self._workdir / Path(remote).name
            shutil.copy2(local, str(dest))

    async def stop(self) -> None:
        if self._workdir and self._workdir.exists():
            shutil.rmtree(self._workdir, ignore_errors=True)
            self._workdir = None

    async def exec(
        self,
        command: str,
        timeout_sec: float = 180,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        if not self._workdir:
            raise RuntimeError("Sandbox not started")
        if user is not None:
            logger.warning("LocalSandbox does not support user switching; ignoring user=%r", user)

        merged_env = dict(self._spec.env) if self._spec.env else None
        if env:
            merged_env = {**(merged_env or {}), **env}
        effective_cwd = Path(cwd) if cwd else self._workdir
        if not effective_cwd.exists():
            # The tool backend defaults cwd to the spec workdir ("/workspace"),
            # which does not exist for an in-container LocalSandbox. Fall back to
            # our tempdir workspace so bash/file tools run somewhere real. Only the
            # missing-path case falls back; a real-but-non-dir path still errors.
            effective_cwd = self._workdir
        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=effective_cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=merged_env,
            # New session => the shell and every child it spawns share a process
            # group (pgid == proc.pid). On timeout we SIGKILL the whole group, so
            # a wedged grandchild (e.g. a model-written `python3` infinite loop or
            # a process holding the stdout pipe open) cannot keep exec() blocked.
            start_new_session=True,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout_sec,
            )
        except asyncio.TimeoutError:
            await self._kill_process_group(proc)
            msg = f"Command exceeded the {timeout_sec:.0f}s execution time limit and was killed."
            # exit code 124 = conventional "timed out"; marks the tool result as an error.
            return ExecResult("", msg, 124)
        return ExecResult(stdout.decode(), stderr.decode(), proc.returncode or 0)

    @staticmethod
    async def _kill_process_group(proc: asyncio.subprocess.Process) -> None:
        """SIGKILL the process's whole group, then reap it without hanging.

        ``proc.kill()`` only signals the direct child (the shell); orphaned
        grandchildren survive and can keep the stdout pipe open, so a plain
        ``await proc.wait()`` may never return. Killing the group reaps them all;
        the wait is itself time-bounded as a final guard.
        """
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            # Group already gone, or we can't signal it — fall back to the child.
            try:
                proc.kill()
            except ProcessLookupError:
                pass
        try:
            await asyncio.wait_for(proc.wait(), timeout=10)
        except asyncio.TimeoutError:
            logger.warning("LocalSandbox: process %s did not reap within 10s of SIGKILL", proc.pid)

    async def upload(self, local_path: Path, remote_path: str) -> None:
        if not self._workdir:
            raise RuntimeError("Sandbox not started")
        dest = self._workdir / Path(remote_path).name
        shutil.copy2(str(local_path), str(dest))

    async def download(self, remote_path: str, local_path: Path) -> None:
        if not self._workdir:
            raise RuntimeError("Sandbox not started")
        src = self._workdir / Path(remote_path).name
        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(local_path))

    def resolve_outside_endpoint(self, url: str) -> str:
        return url

    def resolved_endpoint_url(self, env_var: str) -> str | None:
        for ep in self._outside_endpoints:
            if ep.env_var == env_var:
                return ep.url
        return None

    @property
    def is_running(self) -> bool:
        return self._workdir is not None and self._workdir.exists()

    @property
    def container_ip(self) -> str | None:
        return None

    async def __aenter__(self) -> Self:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.stop()
