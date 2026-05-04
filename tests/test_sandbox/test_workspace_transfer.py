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
"""Tests for WorkspaceTransfer implementations."""

from __future__ import annotations

import pytest

from nemo_evaluator.sandbox.base import SandboxSpec, VolumeMount
from nemo_evaluator.sandbox.transfer import (
    EfsTransfer,
    HostVolumeTransfer,
    LocalDirectTransfer,
    SandboxExecTransfer,
)

# ---------------------------------------------------------------------------
# HostVolumeTransfer
# ---------------------------------------------------------------------------


class TestHostVolumeTransfer:
    def test_shared_dir_created(self):
        t = HostVolumeTransfer()
        assert t._shared_dir.exists()
        assert t._shared_dir.stat().st_mode & 0o777 == 0o700

    def test_agent_spec_gets_output_volume(self):
        t = HostVolumeTransfer()
        spec = SandboxSpec(image="agent:latest")
        result = t.prepare_agent_spec(spec)
        vols = [v for v in result.volumes if v.container_path == "/output"]
        assert len(vols) == 1
        assert not vols[0].readonly
        assert vols[0].host_path == str(t._shared_dir)

    def test_verify_spec_gets_input_volume_readonly(self):
        t = HostVolumeTransfer()
        spec = SandboxSpec(image="verify:latest")
        result = t.prepare_verify_spec(spec)
        vols = [v for v in result.volumes if v.container_path == "/input"]
        assert len(vols) == 1
        assert vols[0].readonly

    def test_existing_volumes_preserved(self):
        t = HostVolumeTransfer()
        existing = VolumeMount(host_path="/data", container_path="/data")
        spec = SandboxSpec(image="agent:latest", volumes=[existing])
        result = t.prepare_agent_spec(spec)
        assert len(result.volumes) == 2
        assert result.volumes[0].container_path == "/data"
        assert result.volumes[1].container_path == "/output"

    @pytest.mark.asyncio
    async def test_cleanup_removes_dir(self):
        t = HostVolumeTransfer()
        d = t._shared_dir
        assert d.exists()
        await t.cleanup()
        assert not d.exists()

    def test_staging_base_parameterized(self, tmp_path):
        t = HostVolumeTransfer(staging_base=str(tmp_path))
        assert str(t._shared_dir).startswith(str(tmp_path))


# ---------------------------------------------------------------------------
# EfsTransfer
# ---------------------------------------------------------------------------


class TestEfsTransfer:
    def test_session_path_format(self):
        t = EfsTransfer(filesystem_id="fs-abc123")
        assert t._session_path.startswith("/")
        parts = t._session_path.lstrip("/").split("_", 1)
        assert len(parts) == 2
        assert parts[0].isdigit()
        assert len(parts[1]) == 32

    def test_agent_spec_gets_efs_volume(self):
        t = EfsTransfer(filesystem_id="fs-abc123", access_point_id="fsap-xyz")
        spec = SandboxSpec(image="agent:latest")
        result = t.prepare_agent_spec(spec)
        efs_vols = [v for v in result.volumes if v.is_efs]
        assert len(efs_vols) == 1
        vol = efs_vols[0]
        assert vol.container_path == "/output"
        assert vol.efs_filesystem_id == "fs-abc123"
        assert vol.efs_access_point_id == "fsap-xyz"
        assert vol.efs_root_directory is None
        assert not vol.readonly

    def test_verify_spec_gets_writable_efs_volume(self):
        t = EfsTransfer(filesystem_id="fs-abc123")
        spec = SandboxSpec(image="verify:latest")
        result = t.prepare_verify_spec(spec)
        efs_vols = [v for v in result.volumes if v.is_efs]
        assert len(efs_vols) == 1
        assert not efs_vols[0].readonly

    @pytest.mark.asyncio
    async def test_cleanup_does_not_raise(self):
        t = EfsTransfer(filesystem_id="fs-abc123")
        await t.cleanup()


# ---------------------------------------------------------------------------
# LocalDirectTransfer
# ---------------------------------------------------------------------------


class TestLocalDirectTransfer:
    def test_staging_dir_created(self):
        t = LocalDirectTransfer()
        assert t._staging.exists()

    def test_specs_unchanged(self):
        t = LocalDirectTransfer()
        spec = SandboxSpec(image="local:latest")
        assert t.prepare_agent_spec(spec) is spec
        assert t.prepare_verify_spec(spec) is spec

    @pytest.mark.asyncio
    async def test_cleanup_removes_staging(self):
        t = LocalDirectTransfer()
        d = t._staging
        assert d.exists()
        await t.cleanup()
        assert not d.exists()


# ---------------------------------------------------------------------------
# SandboxExecTransfer
# ---------------------------------------------------------------------------


class TestSandboxExecTransfer:
    def test_staging_dir_created(self):
        t = SandboxExecTransfer()
        assert t._staging.exists()

    def test_specs_unchanged(self):
        t = SandboxExecTransfer()
        spec = SandboxSpec(image="any:latest")
        assert t.prepare_agent_spec(spec) is spec
        assert t.prepare_verify_spec(spec) is spec

    @pytest.mark.asyncio
    async def test_cleanup_removes_staging(self):
        t = SandboxExecTransfer()
        d = t._staging
        assert d.exists()
        await t.cleanup()
        assert not d.exists()


# ---------------------------------------------------------------------------
# VolumeMount.is_efs
# ---------------------------------------------------------------------------


class TestVolumeMountIsEfs:
    def test_host_volume_not_efs(self):
        v = VolumeMount(host_path="/tmp/x", container_path="/data")
        assert not v.is_efs

    def test_efs_volume(self):
        v = VolumeMount(
            container_path="/data",
            efs_filesystem_id="fs-abc",
            efs_root_directory="/session/1",
        )
        assert v.is_efs

    def test_efs_volume_with_access_point(self):
        v = VolumeMount(
            container_path="/data",
            efs_filesystem_id="fs-abc",
            efs_access_point_id="fsap-xyz",
        )
        assert v.is_efs


# ---------------------------------------------------------------------------
# SandboxManager.get_transfer_strategy()
# ---------------------------------------------------------------------------


class TestManagerTransferStrategy:
    def _make_manager(self, backend="docker", **kwargs):
        from nemo_evaluator.sandbox.manager import SandboxManager

        return SandboxManager(backend=backend, **kwargs)

    def test_docker_returns_host_volume(self):
        mgr = self._make_manager(backend="docker")
        t = mgr.get_transfer_strategy()
        assert isinstance(t, HostVolumeTransfer)

    def test_local_returns_local_direct(self):
        mgr = self._make_manager(backend="local")
        t = mgr.get_transfer_strategy()
        assert isinstance(t, LocalDirectTransfer)

    def test_ecs_without_efs_returns_exec(self):
        from nemo_evaluator.sandbox.ecs_fargate import EcsFargateConfig

        cfg = EcsFargateConfig()
        mgr = self._make_manager(backend="ecs_fargate", ecs_config=cfg)
        t = mgr.get_transfer_strategy()
        assert isinstance(t, SandboxExecTransfer)

    def test_ecs_with_efs_returns_efs_transfer(self):
        from nemo_evaluator.sandbox.ecs_fargate import EcsFargateConfig

        cfg = EcsFargateConfig(efs_filesystem_id="fs-abc", efs_access_point_id="fsap-xyz")
        mgr = self._make_manager(backend="ecs_fargate", ecs_config=cfg)
        t = mgr.get_transfer_strategy()
        assert isinstance(t, EfsTransfer)
        assert t._fs_id == "fs-abc"

    def test_slurm_uses_shared_fs_root(self, tmp_path):
        shared_root = str(tmp_path / "lustre")
        (tmp_path / "lustre").mkdir()
        mgr = self._make_manager(
            backend="slurm",
            slurm_nodes=["node01"],
            shared_fs_root=shared_root,
        )
        t = mgr.get_transfer_strategy()
        assert isinstance(t, HostVolumeTransfer)
        assert str(t._shared_dir).startswith(shared_root)
