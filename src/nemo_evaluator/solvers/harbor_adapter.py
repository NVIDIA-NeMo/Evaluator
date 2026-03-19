"""Bridge between nel's Sandbox and Harbor's BaseEnvironment.

All Harbor imports are lazy so this module can be imported without
the harbor package installed -- the ImportError surfaces only when
a HarborSolver is actually instantiated.
"""
from __future__ import annotations

import logging
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nemo_evaluator.sandbox.base import Sandbox

logger = logging.getLogger(__name__)


class SandboxEnvironmentAdapter:
    """Wraps a nel ``Sandbox`` to satisfy Harbor's ``BaseEnvironment`` interface.

    We deliberately skip ``BaseEnvironment.__init__()`` and set the
    required attributes directly so nel stays in control of the
    container lifecycle.
    """

    def __init__(
        self,
        sandbox: Sandbox,
        *,
        logs_dir: Path,
        default_timeout: float = 600.0,
        persistent_env: dict[str, str] | None = None,
    ) -> None:
        from harbor.models.task.config import EnvironmentConfig as TaskEnvConfig
        from harbor.models.trial.paths import TrialPaths

        self._sandbox = sandbox
        self._default_timeout = default_timeout
        self.trial_paths = TrialPaths(trial_dir=logs_dir)
        self.trial_paths.mkdir()
        self.logger = logger
        self._persistent_env: dict[str, str] = dict(persistent_env or {})
        self.task_env_config = TaskEnvConfig()

    # -- Properties Harbor agents inspect ------------------------------------

    @property
    def is_mounted(self) -> bool:
        return False

    @property
    def supports_gpus(self) -> bool:
        return False

    @property
    def can_disable_internet(self) -> bool:
        return False

    # -- exec ----------------------------------------------------------------

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
    ):
        from harbor.environments.base import ExecResult as HarborExecResult

        merged_env = {**self._persistent_env, **(env or {})}
        nel_result = await self._sandbox.exec(
            command,
            timeout_sec=float(timeout_sec) if timeout_sec else self._default_timeout,
            cwd=cwd,
            env=merged_env or None,
        )
        return HarborExecResult(
            stdout=nel_result.stdout or None,
            stderr=nel_result.stderr or None,
            return_code=nel_result.return_code,
        )

    # -- File transfer -------------------------------------------------------

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        await self._sandbox.upload(Path(source_path), target_path)

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        await self._sandbox.download(source_path, Path(target_path))

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        src = Path(source_dir)
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as f:
            archive = Path(f.name)
        try:
            with tarfile.open(archive, "w:gz") as tar:
                for child in src.rglob("*"):
                    if child.is_file():
                        tar.add(str(child), arcname=str(child.relative_to(src)))
            remote_tar = f"/tmp/_nel_upload_{archive.stem}.tar.gz"
            await self._sandbox.upload(archive, remote_tar)
            await self._sandbox.exec(
                f"mkdir -p {target_dir} && tar xzf {remote_tar} -C {target_dir} && rm -f {remote_tar}",
                timeout_sec=120,
            )
        finally:
            archive.unlink(missing_ok=True)

    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        dst = Path(target_dir)
        dst.mkdir(parents=True, exist_ok=True)
        remote_tar = f"/tmp/_nel_download_{dst.name}.tar.gz"
        await self._sandbox.exec(
            f"tar czf {remote_tar} -C {source_dir} .",
            timeout_sec=120,
        )
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as f:
            local_tar = Path(f.name)
        try:
            await self._sandbox.download(remote_tar, local_tar)
            with tarfile.open(local_tar, "r:gz") as tar:
                tar.extractall(dst)
        finally:
            local_tar.unlink(missing_ok=True)
            await self._sandbox.exec(f"rm -f {remote_tar}", timeout_sec=10)

    # -- Lifecycle (no-ops: nel manages the container) -----------------------

    async def start(self, force_build: bool = False) -> None:
        pass

    async def stop(self, delete: bool = True) -> None:
        pass
