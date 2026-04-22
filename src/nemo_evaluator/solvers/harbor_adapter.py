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
"""Adapts :class:`~nemo_evaluator.sandbox.base.Sandbox` to Harbor's
``BaseEnvironment`` interface.

All Harbor imports are lazy so the module can be imported without the
``harbor`` package installed.
"""

from __future__ import annotations

import logging
import shlex
import tarfile
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nemo_evaluator.sandbox.base import Sandbox

logger = logging.getLogger(__name__)


class SandboxEnvironmentAdapter:
    """Wraps a :class:`Sandbox` to satisfy Harbor's ``BaseEnvironment`` interface.

    ``BaseEnvironment.__init__()`` is deliberately skipped; the required
    attributes are set directly so the evaluator retains control of the
    container lifecycle.
    """

    default_user: str | int | None

    def __init__(
        self,
        sandbox: Sandbox,
        *,
        session_id: str,
        logs_dir: Path,
        default_timeout: float = 600.0,
        persistent_env: dict[str, str] | None = None,
        override_env: dict[str, str] | None = None,
    ) -> None:
        from harbor.models.task.config import EnvironmentConfig as TaskEnvConfig
        from harbor.models.trial.paths import TrialPaths

        self._sandbox = sandbox
        self._default_timeout = default_timeout
        self.session_id = session_id
        self.trial_paths = TrialPaths(trial_dir=logs_dir)
        self.trial_paths.mkdir()
        self.logger = logger
        self._persistent_env: dict[str, str] = dict(persistent_env or {})
        self._override_env: dict[str, str] = dict(override_env or {})
        self.task_env_config = TaskEnvConfig()
        self.default_user = None

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

    # -- User / env resolution (mirrors BaseEnvironment) ---------------------

    def _resolve_user(self, user: str | int | None) -> str | int | None:
        return user if user is not None else self.default_user

    def _merge_env(self, env: dict[str, str] | None) -> dict[str, str] | None:
        """Merge environment variable layers for a single exec call.

        Precedence (last wins): ``persistent_env`` < *env* (caller) < ``override_env``.
        Override always wins so that per-session values (e.g. ``LLM_BASE_URL``)
        cannot be accidentally overwritten by agent code.
        """
        if not self._persistent_env and not self._override_env and not env:
            return None
        merged = {**self._persistent_env}
        if env:
            merged.update(env)
        merged.update(self._override_env)
        return merged

    # -- exec ----------------------------------------------------------------

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ):
        from harbor.environments.base import ExecResult as HarborExecResult

        user = self._resolve_user(user)
        env = self._merge_env(env)

        nel_result = await self._sandbox.exec(
            command,
            timeout_sec=float(timeout_sec) if timeout_sec is not None else self._default_timeout,
            cwd=cwd,
            env=env,
            user=user,
        )
        return HarborExecResult(
            stdout=nel_result.stdout or None,
            stderr=nel_result.stderr or None,
            return_code=nel_result.return_code,
        )

    # -- Filesystem queries --------------------------------------------------

    async def is_dir(self, path: str, user: str | int | None = None) -> bool:
        result = await self.exec(f"test -d {shlex.quote(path)}", timeout_sec=10, user=user)
        return result.return_code == 0

    async def is_file(self, path: str, user: str | int | None = None) -> bool:
        result = await self.exec(f"test -f {shlex.quote(path)}", timeout_sec=10, user=user)
        return result.return_code == 0

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
            remote_tar = f"/tmp/_eval_upload_{archive.stem}.tar.gz"
            await self._sandbox.upload(archive, remote_tar)
            q_dir = shlex.quote(target_dir)
            await self._sandbox.exec(
                f"mkdir -p {q_dir} && tar xzf {remote_tar} -C {q_dir} && rm -f {remote_tar}",
                timeout_sec=120,
            )
        finally:
            archive.unlink(missing_ok=True)

    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        dst = Path(target_dir)
        dst.mkdir(parents=True, exist_ok=True)
        remote_tar = f"/tmp/_eval_download_{dst.name}.tar.gz"
        await self._sandbox.exec(
            f"tar czf {remote_tar} -C {shlex.quote(source_dir)} .",
            timeout_sec=120,
        )
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as f:
            local_tar = Path(f.name)
        try:
            await self._sandbox.download(remote_tar, local_tar)
            with tarfile.open(local_tar, "r:gz") as tar:
                tar.extractall(dst, filter="data")
        finally:
            local_tar.unlink(missing_ok=True)
            await self._sandbox.exec(f"rm -f {remote_tar}", timeout_sec=10)

    # -- Lifecycle (no-ops: the evaluator manages the container) -------------

    async def start(self, force_build: bool = False) -> None:
        pass

    async def stop(self, delete: bool = True) -> None:
        pass
