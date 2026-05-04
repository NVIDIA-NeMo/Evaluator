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
"""Tests for sandbox backends (Docker, Local) — all subprocess calls mocked."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nemo_evaluator.sandbox.base import ExecResult, OutsideEndpoint, SandboxSpec, VolumeMount


# ── DockerSandbox ────────────────────────────────────────────────────────


class TestDockerSandbox:
    def _make(self, **kw):
        from nemo_evaluator.sandbox.docker import DockerSandbox

        spec = SandboxSpec(image="python:3.12", **kw)
        return DockerSandbox(spec)

    def test_not_running_initially(self):
        sb = self._make()
        assert not sb.is_running
        assert sb.container_ip is None

    @patch("nemo_evaluator.sandbox.docker.asyncio.create_subprocess_exec")
    async def test_start_runs_docker_run(self, mock_exec):
        proc = AsyncMock()
        proc.communicate.return_value = (b"abc123\n", b"")
        proc.returncode = 0
        inspect_proc = AsyncMock()
        inspect_proc.communicate.return_value = (b"172.17.0.5", b"")
        mock_exec.side_effect = [proc, inspect_proc]

        sb = self._make()
        await sb.start()
        assert sb.is_running
        assert sb.container_ip == "172.17.0.5"
        first_call = mock_exec.call_args_list[0]
        assert "docker" in first_call.args[0]
        assert "run" in first_call.args

    @patch("nemo_evaluator.sandbox.docker.asyncio.create_subprocess_exec")
    async def test_start_failure_raises(self, mock_exec):
        proc = AsyncMock()
        proc.communicate.return_value = (b"", b"error: no image")
        proc.returncode = 1
        mock_exec.return_value = proc

        sb = self._make()
        with pytest.raises(RuntimeError, match="docker run failed"):
            await sb.start()

    async def test_exec_without_start_raises(self):
        sb = self._make()
        with pytest.raises(RuntimeError, match="not started"):
            await sb.exec("echo hi")

    @patch("nemo_evaluator.sandbox.docker.asyncio.create_subprocess_exec")
    async def test_exec_returns_result(self, mock_exec):
        proc = AsyncMock()
        proc.communicate.return_value = (b"hello\n", b"")
        proc.returncode = 0
        mock_exec.return_value = proc

        sb = self._make()
        sb._container_id = "abc123"
        result = await sb.exec("echo hello")
        assert result.stdout == "hello\n"
        assert result.return_code == 0

    @patch("nemo_evaluator.sandbox.docker.asyncio.create_subprocess_exec")
    async def test_exec_with_cwd_and_env(self, mock_exec):
        proc = AsyncMock()
        proc.communicate.return_value = (b"", b"")
        proc.returncode = 0
        mock_exec.return_value = proc

        sb = self._make()
        sb._container_id = "abc123"
        await sb.exec("ls", cwd="/app", env={"FOO": "bar"})
        cmd_args = mock_exec.call_args.args
        assert "-w" in cmd_args
        assert "-e" in cmd_args

    @patch("nemo_evaluator.sandbox.docker.asyncio.create_subprocess_exec")
    async def test_exec_timeout(self, mock_exec):
        proc = AsyncMock()
        proc.communicate.side_effect = asyncio.TimeoutError()
        proc.kill = MagicMock()
        proc.wait = AsyncMock()
        mock_exec.return_value = proc

        sb = self._make()
        sb._container_id = "abc123"
        result = await sb.exec("sleep 999", timeout_sec=1)
        assert result.return_code == -1
        assert result.stderr == "timeout"

    @patch("nemo_evaluator.sandbox.docker.asyncio.create_subprocess_exec")
    async def test_stop_removes_container(self, mock_exec):
        proc = AsyncMock()
        proc.wait.return_value = 0
        mock_exec.return_value = proc

        sb = self._make()
        sb._container_id = "abc123"
        await sb.stop()
        assert not sb.is_running
        cmd_args = mock_exec.call_args.args
        assert "rm" in cmd_args
        assert "-f" in cmd_args

    @patch("nemo_evaluator.sandbox.docker.asyncio.create_subprocess_exec")
    async def test_upload_calls_docker_cp(self, mock_exec):
        proc = AsyncMock()
        proc.communicate.return_value = (b"", b"")
        proc.returncode = 0
        mock_exec.return_value = proc

        sb = self._make()
        sb._container_id = "abc123"
        await sb.upload(Path("/tmp/test.py"), "/workspace/test.py")
        cmd_args = mock_exec.call_args.args
        assert "cp" in cmd_args

    async def test_upload_without_start_raises(self):
        sb = self._make()
        with pytest.raises(RuntimeError, match="not started"):
            await sb.upload(Path("/tmp/x"), "/y")

    def test_resolve_endpoint_bridge(self):
        sb = self._make()
        assert "host.docker.internal" in sb.resolve_outside_endpoint("http://localhost:8000/v1")

    def test_resolve_endpoint_host_network(self):
        from nemo_evaluator.sandbox.docker import DockerSandbox

        spec = SandboxSpec(image="test")
        sb = DockerSandbox(spec, network="host")
        assert sb.resolve_outside_endpoint("http://localhost:8000") == "http://localhost:8000"

    def test_volumes_in_start_command(self):
        spec = SandboxSpec(
            image="test",
            volumes=[VolumeMount(host_path="/data", container_path="/mnt/data", readonly=True)],
        )
        from nemo_evaluator.sandbox.docker import DockerSandbox

        sb = DockerSandbox(spec)
        assert sb.spec.volumes[0].host_path == "/data"


# ── LocalSandbox ─────────────────────────────────────────────────────────


class TestLocalSandbox:
    def _make(self):
        from nemo_evaluator.sandbox.local import LocalSandbox

        return LocalSandbox(SandboxSpec(image="unused"))

    async def test_start_creates_workdir(self):
        sb = self._make()
        await sb.start()
        assert sb.is_running
        assert sb._workdir is not None
        assert sb._workdir.exists()
        await sb.stop()

    async def test_stop_removes_workdir(self):
        sb = self._make()
        await sb.start()
        wd = sb._workdir
        await sb.stop()
        assert not sb.is_running
        assert not wd.exists()

    async def test_exec_without_start_raises(self):
        sb = self._make()
        with pytest.raises(RuntimeError, match="not started"):
            await sb.exec("echo hi")

    @patch("nemo_evaluator.sandbox.local.asyncio.create_subprocess_shell")
    async def test_exec_runs_command(self, mock_shell):
        proc = AsyncMock()
        proc.communicate.return_value = (b"42\n", b"")
        proc.returncode = 0
        mock_shell.return_value = proc

        sb = self._make()
        await sb.start()
        result = await sb.exec("echo 42")
        assert result.stdout == "42\n"
        assert result.return_code == 0
        await sb.stop()

    @patch("nemo_evaluator.sandbox.local.asyncio.create_subprocess_shell")
    async def test_exec_timeout(self, mock_shell):
        proc = AsyncMock()
        proc.communicate.side_effect = asyncio.TimeoutError()
        proc.kill = MagicMock()
        proc.wait = AsyncMock()
        mock_shell.return_value = proc

        sb = self._make()
        await sb.start()
        result = await sb.exec("sleep 999", timeout_sec=0.01)
        assert result.return_code == -1
        await sb.stop()

    async def test_upload_download(self, tmp_path):
        sb = self._make()
        await sb.start()
        src = tmp_path / "hello.txt"
        src.write_text("world")
        await sb.upload(src, "hello.txt")
        dest = tmp_path / "out.txt"
        await sb.download("hello.txt", dest)
        assert dest.read_text() == "world"
        await sb.stop()

    def test_resolve_endpoint_passthrough(self):
        sb = self._make()
        assert sb.resolve_outside_endpoint("http://localhost:8000") == "http://localhost:8000"

    def test_container_ip_none(self):
        sb = self._make()
        assert sb.container_ip is None

    async def test_context_manager(self):
        from nemo_evaluator.sandbox.local import LocalSandbox

        async with LocalSandbox(SandboxSpec(image="x")) as sb:
            assert sb.is_running
        assert not sb.is_running


# ── SandboxSpec / ExecResult / VolumeMount ───────────────────────────────


class TestSandboxTypes:
    def test_sandbox_spec_defaults(self):
        spec = SandboxSpec(image="python:3.12")
        assert spec.workdir == "/workspace"
        assert spec.env == {}
        assert spec.files == {}

    def test_exec_result(self):
        r = ExecResult("out", "err", 1)
        assert r.stdout == "out"
        assert r.return_code == 1

    def test_volume_mount_efs(self):
        vm = VolumeMount(container_path="/efs", efs_filesystem_id="fs-123")
        assert vm.is_efs

    def test_volume_mount_host(self):
        vm = VolumeMount(host_path="/data", container_path="/mnt")
        assert not vm.is_efs

    def test_outside_endpoint(self):
        ep = OutsideEndpoint(url="http://localhost:5000", env_var="MODEL_URL")
        assert ep.url == "http://localhost:5000"
