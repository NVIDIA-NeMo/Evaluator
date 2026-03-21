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
        self._shared_dir = Path(tempfile.mkdtemp(
            prefix="nel_xfer_", dir=staging_base,
        ))
        self._shared_dir.chmod(0o700)

    def prepare_agent_spec(self, spec: SandboxSpec) -> SandboxSpec:
        return _clone_spec_with_volume(spec, VolumeMount(
            host_path=str(self._shared_dir),
            container_path="/output",
            readonly=False,
        ))

    def prepare_verify_spec(self, spec: SandboxSpec) -> SandboxSpec:
        return _clone_spec_with_volume(spec, VolumeMount(
            host_path=str(self._shared_dir),
            container_path="/input",
            readonly=True,
        ))

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
        return _clone_spec_with_volume(
            spec, self._efs_volume("/output", readonly=False),
        )

    def prepare_verify_spec(self, spec: SandboxSpec) -> SandboxSpec:
        return _clone_spec_with_volume(
            spec, self._efs_volume("/input", readonly=True),
        )

    async def post_capture(self, source: Sandbox) -> None:
        if self._ap_id:
            session = self._session_path.lstrip("/")
            await source.exec(
                f"mkdir -p /output/{session} && "
                f"mv /output/workspace.tar /output/{session}/workspace.tar",
                timeout_sec=120,
            )

    async def pre_restore(self, target: Sandbox) -> None:
        if self._ap_id:
            session = self._session_path.lstrip("/")
            await target.exec(
                f"cp /input/{session}/workspace.tar /input/workspace.tar",
                timeout_sec=120,
            )

    async def cleanup(self) -> None:
        logger.debug("EfsTransfer: session %s — cleanup deferred to reaper",
                      self._session_path)


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
            "tar czf /tmp/_nel_ws.tar.gz -C /output . 2>/dev/null || "
            "tar czf /tmp/_nel_ws.tar.gz -C . . 2>/dev/null",
            timeout_sec=120,
        )
        if result.return_code != 0:
            logger.warning("LocalDirectTransfer: tar failed (rc=%d): %s",
                           result.return_code, result.stderr[:300])
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
                logger.warning("LocalDirectTransfer: untar failed (rc=%d): %s",
                               result.return_code, result.stderr[:300])
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
        result = await source.exec(
            "tar czf /tmp/_ws.tar.gz -C /output . 2>/dev/null || "
            "tar czf /tmp/_ws.tar.gz -C /workspace . 2>/dev/null",
            timeout_sec=120,
        )
        if result.return_code != 0:
            logger.warning("SandboxExecTransfer: tar failed (rc=%d): %s",
                           result.return_code, result.stderr[:300])
            return
        try:
            await source.download("/tmp/_ws.tar.gz", self._archive)
        except Exception:
            logger.warning("SandboxExecTransfer: download failed", exc_info=True)

    async def pre_restore(self, target: Sandbox) -> None:
        if not self._archive.exists():
            logger.warning("SandboxExecTransfer: no archive to restore")
            return
        try:
            await target.upload(self._archive, "/tmp/_ws.tar.gz")
            result = await target.exec(
                "mkdir -p /input && tar xzf /tmp/_ws.tar.gz -C /input",
                timeout_sec=120,
            )
            if result.return_code != 0:
                logger.warning("SandboxExecTransfer: untar failed (rc=%d): %s",
                               result.return_code, result.stderr[:300])
        except Exception:
            logger.warning("SandboxExecTransfer: restore failed", exc_info=True)

    async def cleanup(self) -> None:
        shutil.rmtree(self._staging, ignore_errors=True)
