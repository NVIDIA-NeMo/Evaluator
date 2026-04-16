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
"""WorkspaceTransfer — pluggable strategies for moving workspace state
between agent and verifier sandboxes in StatelessSandbox.

Strategies:
  - HostVolumeTransfer  — Docker, Slurm, Apptainer (host bind mounts)
  - EfsTransfer         — ECS Fargate (shared EFS filesystem)
  - LocalDirectTransfer — LocalSandbox (direct filesystem copies)
  - SandboxExecTransfer — generic fallback (exec + upload/download)
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import time
from dataclasses import replace as _dc_replace
from pathlib import Path
from typing import TYPE_CHECKING, Protocol
from uuid import uuid4

from nemo_evaluator.sandbox.base import SandboxSpec, VolumeMount

if TYPE_CHECKING:
    from nemo_evaluator.sandbox.base import Sandbox

logger = logging.getLogger(__name__)

_MAX_DIFF_PREVIEW = 100_000


# ── Protocol ─────────────────────────────────────────────────────────


class WorkspaceTransfer(Protocol):
    """Transfer workspace state between agent and verifier sandboxes."""

    def prepare_agent_spec(self, spec: SandboxSpec) -> SandboxSpec:
        """Add volume / mount config to agent spec before sandbox starts."""
        ...

    def prepare_verify_spec(self, spec: SandboxSpec) -> SandboxSpec:
        """Add volume / mount config to verifier spec before sandbox starts."""
        ...

    async def post_capture(self, source: Sandbox) -> None:
        """Hook after ``capture_cmd`` runs in agent. No-op for volume-based."""
        ...

    async def pre_restore(self, target: Sandbox) -> None:
        """Hook before ``apply_cmd`` runs in verifier. No-op for volume-based."""
        ...

    async def cleanup(self) -> None:
        """Delete staging dirs, EFS session dirs, etc."""
        ...


# ── Helpers ──────────────────────────────────────────────────────────


def _clone_spec_with_volume(spec: SandboxSpec, vol: VolumeMount) -> SandboxSpec:
    return _dc_replace(
        spec,
        env=dict(spec.env),
        files=dict(spec.files),
        volumes=list(spec.volumes) + [vol],
    )


# ── 1. HostVolumeTransfer ────────────────────────────────────────────


class HostVolumeTransfer:
    """Docker / Slurm / Apptainer — host-mounted shared directory.

    ``staging_base``: root for temp dirs.  ``None`` → ``/tmp`` (Docker).
    For Slurm / Apptainer-on-SLURM, pass the shared filesystem root so
    compute nodes can see the directory.
    """

    def __init__(self, staging_base: str | None = None) -> None:
        self._shared_dir = Path(
            tempfile.mkdtemp(
                prefix="nel_xfer_",
                dir=staging_base,
            )
        )
        self._shared_dir.chmod(0o700)

    def prepare_agent_spec(self, spec: SandboxSpec) -> SandboxSpec:
        return _clone_spec_with_volume(
            spec,
            VolumeMount(
                host_path=str(self._shared_dir),
                container_path="/output",
                readonly=False,
            ),
        )

    def prepare_verify_spec(self, spec: SandboxSpec) -> SandboxSpec:
        return _clone_spec_with_volume(
            spec,
            VolumeMount(
                host_path=str(self._shared_dir),
                container_path="/input",
                readonly=True,
            ),
        )

    async def post_capture(self, source: Sandbox) -> None:
        pass

    async def pre_restore(self, target: Sandbox) -> None:
        pass

    async def cleanup(self) -> None:
        shutil.rmtree(self._shared_dir, ignore_errors=True)


# ── 2. EfsTransfer ───────────────────────────────────────────────────


class EfsTransfer:
    """ECS Fargate — shared EFS filesystem.

    Each eval step gets a unique session directory on EFS, keyed by
    ``/{timestamp}_{full-uuid}``.  Both agent and verifier mount it.
    """

    def __init__(
        self,
        filesystem_id: str,
        access_point_id: str | None = None,
    ) -> None:
        self._fs_id = filesystem_id
        self._ap_id = access_point_id
        ts = int(time.time())
        uid = uuid4().hex
        self._session_path = f"/{ts}_{uid}"

    def _efs_volume(self, container_path: str, readonly: bool) -> VolumeMount:
        vol = VolumeMount(
            container_path=container_path,
            readonly=readonly,
            efs_filesystem_id=self._fs_id,
            efs_access_point_id=self._ap_id,
        )
        if not self._ap_id:
            vol = VolumeMount(
                container_path=container_path,
                readonly=readonly,
                efs_filesystem_id=self._fs_id,
                efs_root_directory=self._session_path,
            )
        return vol

    def prepare_agent_spec(self, spec: SandboxSpec) -> SandboxSpec:
        new_spec = _clone_spec_with_volume(
            spec,
            self._efs_volume("/output", readonly=False),
        )
        if self._ap_id:
            session = self._session_path.lstrip("/")
            new_spec.env = {**(new_spec.env or {}), "_NEL_EFS_SESSION": session}
        return new_spec

    def prepare_verify_spec(self, spec: SandboxSpec) -> SandboxSpec:
        new_spec = _clone_spec_with_volume(
            spec,
            self._efs_volume("/input", readonly=False),
        )
        if self._ap_id:
            session = self._session_path.lstrip("/")
            new_spec.env = {**(new_spec.env or {}), "_NEL_EFS_SESSION": session}
        return new_spec

    async def post_capture(self, source: Sandbox) -> None:
        logger.info(
            "EfsTransfer.post_capture: fs=%s ap=%s session=%s",
            self._fs_id,
            self._ap_id,
            self._session_path,
        )
        if self._ap_id:
            session = self._session_path.lstrip("/")
            result = await source.exec(
                f"ls -lh /output/{session}/workspace.tar 2>&1",
                timeout_sec=30,
            )
            found = result.return_code == 0 and "No such file" not in (result.stdout or "")
            if not found:
                logger.error(
                    "EfsTransfer.post_capture: session workspace MISSING at /output/%s/ — "
                    "capture_cmd may have written to wrong path: %s",
                    session,
                    (result.stdout or "").strip()[:300],
                )
            else:
                logger.info(
                    "EfsTransfer.post_capture: %s",
                    (result.stdout or "").strip()[:200],
                )
        else:
            result = await source.exec(
                "ls -lh /output/workspace.tar 2>&1",
                timeout_sec=10,
            )
            logger.info(
                "EfsTransfer.post_capture (no-ap): /output/workspace.tar → %s",
                (result.stdout or "").strip()[:200],
            )

    async def pre_restore(self, target: Sandbox) -> None:
        logger.info(
            "EfsTransfer.pre_restore: fs=%s ap=%s session=%s",
            self._fs_id,
            self._ap_id,
            self._session_path,
        )
        if self._ap_id:
            session = self._session_path.lstrip("/")
            result = await target.exec(
                f"ln -sf /input/{session}/workspace.tar /input/workspace.tar",
                timeout_sec=120,
            )
            if result.return_code != 0:
                logger.error(
                    "EfsTransfer.pre_restore: symlink failed rc=%d: %s",
                    result.return_code,
                    (result.stderr or "")[:300],
                )
        result = await target.exec(
            "ls -lh /input/workspace.tar 2>&1",
            timeout_sec=10,
        )
        logger.info(
            "EfsTransfer.pre_restore: /input/workspace.tar → %s",
            (result.stdout or "").strip()[:200],
        )

    async def cleanup(self) -> None:
        logger.debug("EfsTransfer: session %s — cleanup deferred to reaper", self._session_path)


# ── 3. LocalDirectTransfer ───────────────────────────────────────────


class LocalDirectTransfer:
    """LocalSandbox — no container isolation, direct filesystem ops.

    ``post_capture`` tars the agent workdir into a staging area.
    ``pre_restore`` untars into the verifier workdir.
    LocalSandbox ignores ``spec.volumes``, so no VolumeMount is added.
    """

    def __init__(self) -> None:
        self._staging = Path(tempfile.mkdtemp(prefix="nel_xfer_local_"))
        self._archive = self._staging / "workspace.tar.gz"

    def prepare_agent_spec(self, spec: SandboxSpec) -> SandboxSpec:
        return spec

    def prepare_verify_spec(self, spec: SandboxSpec) -> SandboxSpec:
        return spec

    async def post_capture(self, source: Sandbox) -> None:
        result = await source.exec(
            "tar czf /tmp/_nel_ws.tar.gz -C /output . 2>/dev/null || tar czf /tmp/_nel_ws.tar.gz -C . . 2>/dev/null",
            timeout_sec=120,
        )
        if result.return_code != 0:
            logger.warning("LocalDirectTransfer: tar failed (rc=%d): %s", result.return_code, result.stderr[:300])
            return
        try:
            await source.download("/tmp/_nel_ws.tar.gz", self._archive)
        except Exception:
            logger.warning("LocalDirectTransfer: download failed", exc_info=True)

    async def pre_restore(self, target: Sandbox) -> None:
        if not self._archive.exists():
            logger.warning("LocalDirectTransfer: no archive to restore")
            return
        try:
            await target.upload(self._archive, "/tmp/_nel_ws.tar.gz")
            result = await target.exec(
                "mkdir -p /input && tar xzf /tmp/_nel_ws.tar.gz -C /input",
                timeout_sec=120,
            )
            if result.return_code != 0:
                logger.warning("LocalDirectTransfer: untar failed (rc=%d): %s", result.return_code, result.stderr[:300])
        except Exception:
            logger.warning("LocalDirectTransfer: restore failed", exc_info=True)

    async def cleanup(self) -> None:
        shutil.rmtree(self._staging, ignore_errors=True)


# ── 4. SandboxExecTransfer ──────────────────────────────────────────


class SandboxExecTransfer:
    """Generic fallback — uses exec() + download() / upload() from the
    Sandbox protocol.  Works with any backend but slower since data
    travels through the orchestrator host.
    """

    def __init__(self) -> None:
        self._staging = Path(tempfile.mkdtemp(prefix="nel_xfer_exec_"))
        self._archive = self._staging / "workspace.tar.gz"

    def prepare_agent_spec(self, spec: SandboxSpec) -> SandboxSpec:
        return spec

    def prepare_verify_spec(self, spec: SandboxSpec) -> SandboxSpec:
        return spec

    async def post_capture(self, source: Sandbox) -> None:
        logger.info("SandboxExecTransfer.post_capture: starting tar+download")
        sz_check = await source.exec(
            "ls -lh /output/workspace.tar 2>&1 || echo 'NOT FOUND'",
            timeout_sec=10,
        )
        logger.info(
            "SandboxExecTransfer: /output/workspace.tar = %s",
            (sz_check.stdout or "").strip()[:200],
        )
        result = await source.exec(
            "tar czf /tmp/_ws.tar.gz -C /output . 2>/dev/null || tar czf /tmp/_ws.tar.gz -C /workspace . 2>/dev/null",
            timeout_sec=300,
        )
        if result.return_code != 0:
            logger.error("SandboxExecTransfer: tar FAILED (rc=%d): %s", result.return_code, (result.stderr or "")[:300])
            return
        gz_check = await source.exec("ls -lh /tmp/_ws.tar.gz", timeout_sec=10)
        logger.info(
            "SandboxExecTransfer: /tmp/_ws.tar.gz = %s",
            (gz_check.stdout or "").strip()[:200],
        )
        try:
            await source.download("/tmp/_ws.tar.gz", self._archive)
            logger.info(
                "SandboxExecTransfer: downloaded archive (%d bytes)",
                self._archive.stat().st_size if self._archive.exists() else 0,
            )
        except Exception:
            logger.error("SandboxExecTransfer: download FAILED", exc_info=True)

    async def pre_restore(self, target: Sandbox) -> None:
        if not self._archive.exists():
            logger.error(
                "SandboxExecTransfer: no local archive to restore — "
                "agent workspace will NOT be applied to verify container!"
            )
            return
        logger.info(
            "SandboxExecTransfer.pre_restore: uploading %d bytes",
            self._archive.stat().st_size,
        )
        try:
            await target.upload(self._archive, "/tmp/_ws.tar.gz")
            result = await target.exec(
                "mkdir -p /input && tar xzf /tmp/_ws.tar.gz -C /input",
                timeout_sec=300,
            )
            if result.return_code != 0:
                logger.error(
                    "SandboxExecTransfer: untar FAILED (rc=%d): %s", result.return_code, (result.stderr or "")[:300]
                )
            else:
                check = await target.exec(
                    "ls -lh /input/workspace.tar 2>&1",
                    timeout_sec=10,
                )
                logger.info(
                    "SandboxExecTransfer: restored — %s",
                    (check.stdout or "").strip()[:200],
                )
        except Exception:
            logger.error("SandboxExecTransfer: restore FAILED", exc_info=True)

    async def cleanup(self) -> None:
        shutil.rmtree(self._staging, ignore_errors=True)
