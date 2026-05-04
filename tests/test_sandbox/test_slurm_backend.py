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
"""Tests for SLURM sandbox backend — all subprocess calls mocked."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nemo_evaluator.sandbox.base import OutsideEndpoint, SandboxSpec, VolumeMount
from nemo_evaluator.sandbox.slurm import SlurmSandbox


# ── Helpers ───────────────────────────────────────────────────────────


def _make(node="node01", slot=0, shared_fs=None, het_group=None, **spec_kw):
    defaults = dict(image="python:3.12", workdir="/workspace")
    defaults.update(spec_kw)
    spec = SandboxSpec(**defaults)
    return SlurmSandbox(spec, node=node, slot=slot, shared_fs_root=shared_fs, het_group=het_group)


def _mock_proc(returncode=0, stdout=b"", stderr=b""):
    proc = AsyncMock()
    proc.communicate.return_value = (stdout, stderr)
    proc.returncode = returncode
    proc.wait = AsyncMock()
    return proc


# ── TestSlurmSandbox ──────────────────────────────────────────────────


class TestSlurmSandbox:
    def test_initial_state(self):
        sb = _make()
        assert not sb.is_running
        assert sb.container_ip == "node01"
        assert sb.spec.image == "python:3.12"

    @patch("nemo_evaluator.sandbox.slurm.asyncio.create_subprocess_exec")
    async def test_start_builds_correct_srun(self, mock_exec):
        proc = _mock_proc()
        proc.returncode = None
        mock_exec.return_value = proc

        sb = _make(env={"MY_VAR": "123"})
        await sb.start()

        assert sb.is_running
        cmd = mock_exec.call_args[0]
        assert cmd[0] == "srun"
        assert "--overlap" in cmd
        assert "--nodelist=node01" in cmd
        assert "--ntasks=1" in cmd
        assert "--container-image=python:3.12" in cmd
        assert "--container-workdir=/workspace" in cmd
        assert any("MY_VAR=123" in a for a in cmd)

    @patch("nemo_evaluator.sandbox.slurm.asyncio.create_subprocess_exec")
    async def test_start_with_het_group(self, mock_exec):
        proc = _mock_proc()
        proc.returncode = None
        mock_exec.return_value = proc

        sb = _make(het_group=2)
        await sb.start()

        cmd = mock_exec.call_args[0]
        assert "--het-group=2" in cmd

    @patch("nemo_evaluator.sandbox.slurm.asyncio.create_subprocess_exec")
    async def test_start_with_volumes(self, mock_exec):
        proc = _mock_proc()
        proc.returncode = None
        mock_exec.return_value = proc

        vols = [VolumeMount(host_path="/host/data", container_path="/data", readonly=True)]
        sb = _make(volumes=vols)
        await sb.start()

        cmd = mock_exec.call_args[0]
        assert any("--container-mounts=/host/data:/data:ro" in a for a in cmd)

    @patch("nemo_evaluator.sandbox.slurm.asyncio.create_subprocess_exec")
    async def test_start_skips_efs_volumes(self, mock_exec):
        proc = _mock_proc()
        proc.returncode = None
        mock_exec.return_value = proc

        vols = [VolumeMount(host_path="/host", container_path="/data", efs_filesystem_id="fs-123")]
        sb = _make(volumes=vols)
        await sb.start()

        cmd = mock_exec.call_args[0]
        assert not any("--container-mounts" in a for a in cmd)

    @patch("nemo_evaluator.sandbox.slurm.asyncio.create_subprocess_exec")
    async def test_start_with_outside_endpoints(self, mock_exec):
        proc = _mock_proc()
        proc.returncode = None
        mock_exec.return_value = proc

        sb = _make()
        eps = [OutsideEndpoint(url="http://localhost:4000/v1", env_var="MODEL_URL")]
        with patch("nemo_evaluator.sandbox.slurm.platform.node", return_value="slurm-head"):
            await sb.start(outside_endpoints=eps)

        cmd = mock_exec.call_args[0]
        assert any("MODEL_URL=http://slurm-head:4000/v1" in a for a in cmd)

    @patch("nemo_evaluator.sandbox.slurm.asyncio.create_subprocess_exec")
    async def test_start_failure_raises(self, mock_exec):
        proc = _mock_proc(returncode=1, stderr=b"srun: error: something bad")
        mock_exec.return_value = proc

        sb = _make()
        with pytest.raises(RuntimeError, match="srun sandbox start failed"):
            await sb.start()

    @patch("nemo_evaluator.sandbox.slurm.asyncio.create_subprocess_exec")
    async def test_stop(self, mock_exec):
        proc = _mock_proc()
        proc.returncode = None
        start_proc = _mock_proc()
        start_proc.returncode = None
        mock_exec.side_effect = [start_proc, proc]

        sb = _make()
        await sb.start()
        assert sb.is_running

        await sb.stop()
        assert not sb.is_running

        stop_cmd = mock_exec.call_args_list[-1][0]
        assert "kill" in stop_cmd
        assert "1" in stop_cmd

    @patch("nemo_evaluator.sandbox.slurm.asyncio.create_subprocess_exec")
    async def test_stop_idempotent(self, mock_exec):
        sb = _make()
        sb._running = False
        await sb.stop()
        mock_exec.assert_not_called()

    @patch("nemo_evaluator.sandbox.slurm.asyncio.create_subprocess_exec")
    async def test_exec_basic(self, mock_exec):
        proc = _mock_proc(stdout=b"hello world", stderr=b"")
        mock_exec.return_value = proc

        sb = _make()
        sb._running = True
        result = await sb.exec("echo hello world")

        assert result.stdout == "hello world"
        assert result.return_code == 0
        cmd = mock_exec.call_args[0]
        assert "echo hello world" in cmd

    @patch("nemo_evaluator.sandbox.slurm.asyncio.create_subprocess_exec")
    async def test_exec_with_cwd(self, mock_exec):
        proc = _mock_proc(stdout=b"ok")
        mock_exec.return_value = proc

        sb = _make()
        sb._running = True
        await sb.exec("ls", cwd="/app")

        cmd = mock_exec.call_args[0]
        shell_cmd = cmd[-1]
        assert "cd /app" in shell_cmd

    @patch("nemo_evaluator.sandbox.slurm.asyncio.create_subprocess_exec")
    async def test_exec_with_env(self, mock_exec):
        proc = _mock_proc(stdout=b"ok")
        mock_exec.return_value = proc

        sb = _make()
        sb._running = True
        await sb.exec("cmd", env={"FOO": "bar", "BAZ": "1"})

        cmd = mock_exec.call_args[0]
        shell_cmd = cmd[-1]
        assert "FOO=bar" in shell_cmd
        assert "BAZ=1" in shell_cmd

    async def test_exec_before_start_raises(self):
        sb = _make()
        with pytest.raises(RuntimeError, match="Sandbox not started"):
            await sb.exec("echo hi")

    @patch("nemo_evaluator.sandbox.slurm.asyncio.create_subprocess_exec")
    async def test_exec_timeout(self, mock_exec):
        proc = AsyncMock()

        async def hang():
            await asyncio.sleep(999)
            return (b"", b"")

        proc.communicate = hang
        proc.kill = MagicMock()
        proc.wait = AsyncMock()
        mock_exec.return_value = proc

        sb = _make()
        sb._running = True
        result = await sb.exec("sleep 999", timeout_sec=0.01)
        assert result.return_code == -1
        assert result.stderr == "timeout"


class TestSlurmResolveOutsideEndpoint:
    def test_localhost_rewritten(self):
        sb = _make(node="compute-42")
        with patch("nemo_evaluator.sandbox.slurm.platform.node", return_value="head-node"):
            result = sb.resolve_outside_endpoint("http://localhost:4000/v1")
        assert "head-node:4000" in result
        assert result.endswith("/v1")

    def test_127_0_0_1_rewritten(self):
        sb = _make()
        with patch("nemo_evaluator.sandbox.slurm.platform.node", return_value="ctrl"):
            result = sb.resolve_outside_endpoint("http://127.0.0.1:5000/api")
        assert "ctrl:5000" in result

    def test_non_local_unchanged(self):
        sb = _make()
        url = "https://api.nvidia.com/v1/chat"
        assert sb.resolve_outside_endpoint(url) == url


class TestSlurmResolvedEndpointUrl:
    def test_session_path_preserved(self):
        from nemo_evaluator.sandbox.base import OutsideEndpoint

        sb = _make(node="compute-1")
        sb._outside_endpoints = [
            OutsideEndpoint(url="http://localhost:4000/s/abc123", env_var="MODEL_BASE_URL"),
        ]
        with patch("nemo_evaluator.sandbox.slurm.platform.node", return_value="head-node"):
            result = sb.resolved_endpoint_url("MODEL_BASE_URL")
        assert "head-node:4000" in result
        assert "/s/abc123" in result

    def test_returns_none_for_unknown_var(self):
        sb = _make()
        assert sb.resolved_endpoint_url("NOPE") is None


class TestSlurmTransfer:
    @patch("nemo_evaluator.sandbox.slurm.asyncio.create_subprocess_exec")
    @patch("nemo_evaluator.sandbox.slurm.shutil.copy2")
    async def test_upload_shared_fs(self, mock_copy, mock_exec, tmp_path):
        proc = _mock_proc(stdout=b"ok")
        mock_exec.return_value = proc

        shared = tmp_path / "shared"
        shared.mkdir()

        sb = _make(shared_fs=str(shared))
        sb._running = True
        local = tmp_path / "data.txt"
        local.write_text("test data")

        await sb.upload(local, "/remote/data.txt")
        mock_copy.assert_called_once()
        mock_exec.assert_called_once()

    @patch("nemo_evaluator.sandbox.slurm.asyncio.create_subprocess_exec")
    @patch("nemo_evaluator.sandbox.slurm.shutil.copy2")
    async def test_download_shared_fs(self, mock_copy, mock_exec, tmp_path):
        proc = _mock_proc(stdout=b"ok")
        mock_exec.return_value = proc

        shared = tmp_path / "shared"
        shared.mkdir()

        sb = _make(shared_fs=str(shared))
        sb._running = True
        local = tmp_path / "local_out.txt"

        await sb.download("/remote/data.txt", local)
        mock_exec.assert_called_once()
        mock_copy.assert_called_once()

    @patch("nemo_evaluator.sandbox.slurm.asyncio.create_subprocess_exec")
    async def test_download_no_shared_fs(self, mock_exec, tmp_path):
        proc = _mock_proc(stdout=b"file content here")
        mock_exec.return_value = proc

        sb = _make(shared_fs=None)
        sb._running = True
        local = tmp_path / "out" / "data.txt"

        await sb.download("/remote/data.txt", local)
        assert local.read_text() == "file content here"


class TestSlurmContextManager:
    @patch("nemo_evaluator.sandbox.slurm.asyncio.create_subprocess_exec")
    async def test_aenter_aexit(self, mock_exec):
        start_proc = _mock_proc()
        start_proc.returncode = None
        stop_proc = _mock_proc()
        mock_exec.side_effect = [start_proc, stop_proc]

        sb = _make()
        async with sb:
            assert sb.is_running
        assert not sb.is_running
