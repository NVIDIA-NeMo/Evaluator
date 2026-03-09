"""SLURM-based per-problem sandbox using Pyxis/Enroot.

Each sandbox is a container on a SLURM node launched via ``srun --container-image``.
Multiple sandboxes (slots) can be multiplexed on a single node to avoid wasting
one full node per problem.
"""
from __future__ import annotations

import asyncio
import logging
import shutil
from pathlib import Path
from typing import Self

from nemo_evaluator.sandbox.base import ExecResult, OutsideEndpoint, SandboxSpec

logger = logging.getLogger(__name__)


class SlurmSandbox:
    """Per-problem container on a SLURM node via Pyxis/Enroot."""

    def __init__(
        self,
        spec: SandboxSpec,
        *,
        node: str,
        slot: int,
        shared_fs_root: str | None = None,
    ) -> None:
        self._spec = spec
        self._node = node
        self._slot = slot
        self._shared_fs = shared_fs_root
        self._container_name = f"nel-sandbox-{node}-{slot}"
        self._running = False

    @property
    def spec(self) -> SandboxSpec:
        return self._spec

    async def start(self, *, outside_endpoints: list[OutsideEndpoint] | None = None) -> None:
        env_args: list[str] = []
        for k, v in self._spec.env.items():
            env_args.extend(["--export", f"{k}={v}"])
        if outside_endpoints:
            for ep in outside_endpoints:
                resolved = self.resolve_outside_endpoint(ep.url)
                env_args.extend(["--export", f"{ep.env_var}={resolved}"])

        entrypoint = self._spec.entrypoint or "sleep infinity"

        cmd = [
            "srun", "--overlap",
            f"--nodelist={self._node}",
            "--ntasks=1",
            f"--container-image={self._spec.image}",
            f"--container-workdir={self._spec.workdir}",
            f"--container-name={self._container_name}",
            *env_args,
            "/bin/bash", "-c", entrypoint,
        ]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        # srun for background container: give it a moment to start
        # then check if it's still alive (non-blocking)
        await asyncio.sleep(2)
        if proc.returncode is not None and proc.returncode != 0:
            _, stderr = await proc.communicate()
            raise RuntimeError(f"srun sandbox start failed: {stderr.decode()}")

        self._running = True

        for local, remote in self._spec.files.items():
            await self.upload(Path(local), remote)

        logger.debug("slurm sandbox started: %s node=%s slot=%d image=%s",
                      self._container_name, self._node, self._slot, self._spec.image)

    async def stop(self) -> None:
        if not self._running:
            return
        cmd = [
            "srun", "--overlap",
            f"--nodelist={self._node}",
            "--ntasks=1",
            f"--container-name={self._container_name}",
            "kill", "1",
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.wait()
        self._running = False
        logger.debug("slurm sandbox stopped: %s", self._container_name)

    async def exec(self, command: str, timeout_sec: float = 180) -> ExecResult:
        if not self._running:
            raise RuntimeError("Sandbox not started")
        cmd = [
            "srun", "--overlap",
            f"--nodelist={self._node}",
            "--ntasks=1",
            f"--container-name={self._container_name}",
            "/bin/bash", "-c", command,
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout_sec,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return ExecResult("", "timeout", -1)
        return ExecResult(stdout.decode(), stderr.decode(), proc.returncode or 0)

    async def upload(self, local_path: Path, remote_path: str) -> None:
        if self._shared_fs:
            staging = Path(self._shared_fs) / self._container_name / Path(remote_path).name
            staging.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(local_path), str(staging))
            await self.exec(f"cp {staging} {remote_path}")
        else:
            await self.exec(f"cat > {remote_path}", timeout_sec=30)

    async def download(self, remote_path: str, local_path: Path) -> None:
        if self._shared_fs:
            staging = Path(self._shared_fs) / self._container_name / Path(remote_path).name
            staging.parent.mkdir(parents=True, exist_ok=True)
            await self.exec(f"cp {remote_path} {staging}")
            local_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(staging), str(local_path))
        else:
            result = await self.exec(f"cat {remote_path}", timeout_sec=30)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_text(result.stdout)

    def resolve_outside_endpoint(self, url: str) -> str:
        # SLURM nodes share the job's network — URL unchanged
        return url

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def container_ip(self) -> str | None:
        return self._node

    async def __aenter__(self) -> Self:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.stop()
