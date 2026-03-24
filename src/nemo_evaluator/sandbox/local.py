"""Local sandbox: async subprocess in a temp directory, no container isolation.

Useful for development and testing where Docker is unavailable.
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
from pathlib import Path
from typing import Self

from nemo_evaluator.sandbox.base import ExecResult, OutsideEndpoint, SandboxSpec


class LocalSandbox:
    """No container isolation — async subprocess in a temp dir."""

    def __init__(self, spec: SandboxSpec) -> None:
        self._spec = spec
        self._workdir: Path | None = None

    @property
    def spec(self) -> SandboxSpec:
        return self._spec

    async def start(self, *, outside_endpoints: list[OutsideEndpoint] | None = None) -> None:
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
    ) -> ExecResult:
        if not self._workdir:
            raise RuntimeError("Sandbox not started")

        merged_env = dict(self._spec.env) if self._spec.env else None
        if env:
            merged_env = {**(merged_env or {}), **env}
        effective_cwd = Path(cwd) if cwd else self._workdir
        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=effective_cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=merged_env,
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
