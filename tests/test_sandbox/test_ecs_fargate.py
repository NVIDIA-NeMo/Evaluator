# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Tests for ECS Fargate sandbox — all AWS/SSH/network calls mocked."""

from __future__ import annotations

import threading
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nemo_evaluator.sandbox.base import ExecResult, OutsideEndpoint, SandboxSpec

_PATCH_PREFIX = "nemo_evaluator.sandbox.ecs_fargate"


# ── Helpers ───────────────────────────────────────────────────────────


def _sidecar(**overrides):
    from nemo_evaluator.sandbox.ecs_fargate import SshSidecarConfig

    defaults = dict(
        sshd_port=2222,
        public_key_secret_arn="arn:aws:secretsmanager:us-east-1:1234:secret:pub",
        private_key_secret_arn="arn:aws:secretsmanager:us-east-1:1234:secret:priv",
        exec_server_port=5000,
    )
    defaults.update(overrides)
    return SshSidecarConfig(**defaults)


def _cfg(**overrides):
    from nemo_evaluator.sandbox.ecs_fargate import EcsFargateConfig

    defaults = dict(
        region="us-west-2",
        cluster="test-cluster",
        subnets=["subnet-aaa"],
        security_groups=["sg-bbb"],
        assign_public_ip=True,
        execution_role_arn="arn:aws:iam::1234:role/ecsTaskExec",
        task_role_arn="arn:aws:iam::1234:role/ecsTask",
        ssh_sidecar=_sidecar(),
    )
    defaults.update(overrides)
    return EcsFargateConfig(**defaults)


def _make_sandbox(ecs_config=None, **spec_kw):
    from nemo_evaluator.sandbox.ecs_fargate import EcsFargateSandbox

    spec = SandboxSpec(image="python:3.12", **spec_kw)
    return EcsFargateSandbox(spec, ecs_config=ecs_config or _cfg())


async def _start_with_mocked_ecs(sb, outside_endpoints):
    tunnel = MagicMock()
    tunnel.local_port = 19000
    register_task_definition = MagicMock(return_value="task-def-arn")

    with (
        patch.object(sb, "_init_aws_clients"),
        patch.object(sb, "_resolve_image", return_value="python:3.12"),
        patch(f"{_PATCH_PREFIX}.download_secret_to_file", return_value="/tmp/key"),
        patch(f"{_PATCH_PREFIX}.download_secret_to_string", return_value="ssh-rsa fake"),
        patch(f"{_PATCH_PREFIX}.build_ssh_sidecar_container", return_value={"name": "ssh-tunnel"}),
        patch.object(sb, "_register_task_definition", register_task_definition),
        patch.object(sb, "_run_task", return_value="task-arn"),
        patch.object(sb, "_register_for_cleanup"),
        patch.object(sb, "_wait_for_running"),
        patch.object(sb, "_get_task_public_ip", return_value="10.0.0.10"),
        patch.object(sb, "_wait_for_ssh_ready"),
        patch(f"{_PATCH_PREFIX}._free_port", return_value=19001),
        patch(f"{_PATCH_PREFIX}.SshTunnel", return_value=tunnel) as tunnel_cls,
        patch(f"{_PATCH_PREFIX}.ExecClient"),
    ):
        await sb.start(outside_endpoints=outside_endpoints)

    return register_task_definition, tunnel_cls, tunnel


# ── TestResolveOutsideEndpoint ────────────────────────────────────────


class TestResolveOutsideEndpoint:
    """Endpoint resolution through the sandbox public interface."""

    def _sb(self):
        sb = _make_sandbox()
        return sb

    async def test_match_by_netloc_with_reverse_route(self):
        sb = self._sb()
        await _start_with_mocked_ecs(
            sb,
            [OutsideEndpoint(url="http://127.0.0.1:4000/v1/s/abc123", env_var="MODEL_BASE_URL")],
        )

        result = sb.resolve_outside_endpoint("http://127.0.0.1:4000/v1")
        assert result == "http://127.0.0.1:4000/v1"

    async def test_session_scoped_url_matches_base_url(self):
        """resolve_outside_endpoint preserves the caller's path (no session
        injection).  Use resolved_endpoint_url for session-scoped URLs."""
        sb = self._sb()
        await _start_with_mocked_ecs(
            sb,
            [OutsideEndpoint(url="http://127.0.0.1:4000/v1/s/session42", env_var="MODEL_BASE_URL")],
        )

        result = sb.resolve_outside_endpoint("http://127.0.0.1:4000/v1")
        assert "127.0.0.1:4000" in result
        assert result.endswith("/v1")

    async def test_path_preserved_through_rewrite(self):
        sb = self._sb()
        await _start_with_mocked_ecs(
            sb,
            [OutsideEndpoint(url="https://api.nvidia.com/v1/chat", env_var="LLM_URL")],
        )

        result = sb.resolve_outside_endpoint("https://api.nvidia.com/v1/chat/completions")
        assert result == "https://127.0.0.1:443/v1/chat/completions"

    async def test_scheme_rewritten_from_reverse_route(self):
        sb = self._sb()
        await _start_with_mocked_ecs(
            sb,
            [OutsideEndpoint(url="https://model.example.com:8443/v1", env_var="M")],
        )

        result = sb.resolve_outside_endpoint("https://model.example.com:8443/v1")
        assert result.startswith("https://127.0.0.1:8443")

    async def test_agent_server_fallback_to_ssh_tunnel_port(self):
        cfg = _cfg(
            container_port=8080,
            ssh_sidecar=_sidecar(exec_server_port=None),
        )
        sb = _make_sandbox(ecs_config=cfg)
        await _start_with_mocked_ecs(
            sb,
            [OutsideEndpoint(url="http://proxy.local:9999/root", env_var="MODEL_BASE_URL")],
        )

        result = sb.resolve_outside_endpoint("http://proxy.local:4000/v1")
        assert "127.0.0.1:9999" in result
        assert result.endswith("/v1")

    def test_no_match_raises_runtime_error(self):
        sb = self._sb()

        with pytest.raises(RuntimeError, match="requires SSH reverse tunnel"):
            sb.resolve_outside_endpoint("http://nowhere:1234/api")

    async def test_different_netloc_no_match(self):
        sb = self._sb()
        await _start_with_mocked_ecs(
            sb,
            [OutsideEndpoint(url="http://host-a:4000/v1", env_var="M")],
        )

        with pytest.raises(RuntimeError, match="requires SSH reverse tunnel"):
            sb.resolve_outside_endpoint("http://host-b:5000/v1")

    async def test_agent_server_resolves_any_url_through_single_endpoint_tunnel(self):
        cfg = _cfg(
            container_port=8080,
            ssh_sidecar=_sidecar(exec_server_port=None),
        )
        sb = _make_sandbox(ecs_config=cfg)
        await _start_with_mocked_ecs(
            sb,
            [OutsideEndpoint(url="http://127.0.0.1:7777/v1", env_var="MODEL_BASE_URL")],
        )

        result = sb.resolve_outside_endpoint("http://127.0.0.1:4000/v1")
        assert "127.0.0.1:7777" in result


class TestResolvedEndpointUrl:
    """Tests for the session-scoped resolved_endpoint_url method."""

    def _sb(self):
        return _make_sandbox()

    async def test_returns_session_path_via_reverse_route(self):
        sb = self._sb()
        await _start_with_mocked_ecs(
            sb,
            [OutsideEndpoint(url="http://localhost:4000/s/abc123", env_var="MODEL_BASE_URL")],
        )

        result = sb.resolved_endpoint_url("MODEL_BASE_URL")
        assert result == "http://127.0.0.1:4000/s/abc123"

    async def test_returns_session_path_via_ssh_tunnel(self):
        cfg = _cfg(
            container_port=8080,
            ssh_sidecar=_sidecar(exec_server_port=None),
        )
        sb = _make_sandbox(ecs_config=cfg)
        await _start_with_mocked_ecs(
            sb,
            [OutsideEndpoint(url="http://localhost:9999/s/xyz789", env_var="MODEL_BASE_URL")],
        )

        result = sb.resolved_endpoint_url("MODEL_BASE_URL")
        assert result == "http://127.0.0.1:9999/s/xyz789"

    async def test_returns_none_for_unknown_env_var(self):
        sb = self._sb()
        await _start_with_mocked_ecs(
            sb,
            [OutsideEndpoint(url="http://localhost:4000/s/abc", env_var="MODEL_BASE_URL")],
        )

        assert sb.resolved_endpoint_url("UNKNOWN_VAR") is None

    def test_returns_none_when_no_endpoints(self):
        sb = self._sb()
        assert sb.resolved_endpoint_url("MODEL_BASE_URL") is None

    async def test_harbor_scenario_base_url_gets_session_path(self):
        """The real bug: Harbor calls with base URL, needs session path back."""
        sb = self._sb()
        await _start_with_mocked_ecs(
            sb,
            [OutsideEndpoint(url="http://localhost:4000/s/session42", env_var="MODEL_BASE_URL")],
        )

        session_url = sb.resolved_endpoint_url("MODEL_BASE_URL")
        assert "/s/session42" in session_url
        assert "127.0.0.1:4000" in session_url

        plain_url = sb.resolve_outside_endpoint("http://localhost:4000")
        assert "/s/" not in plain_url


class TestReverseSpecs:
    """ECS reverse tunnel endpoint mapping through sandbox startup."""

    async def test_multiple_endpoints_get_distinct_reverse_ports(self):
        sb = _make_sandbox()
        eps = [
            OutsideEndpoint(url="http://host-a:4000/v1", env_var="A"),
            OutsideEndpoint(url="http://host-b:4000/form", env_var="B"),
        ]

        _, tunnel_cls, _ = await _start_with_mocked_ecs(sb, eps)

        specs = tunnel_cls.call_args.kwargs["reverses"]
        assert "4000:host-a:4000" in specs
        assert "20000:host-b:4000" in specs
        # OutsideEndpoint env vars are routed via RunTask containerOverrides
        # (per-invocation), so they don't leak into the cached task definition.
        assert sb._runtime_container_env["A"] == "http://127.0.0.1:4000/v1"
        assert sb._runtime_container_env["B"] == "http://127.0.0.1:20000/form"

    async def test_duplicate_target_reuses_reverse_port(self):
        sb = _make_sandbox()
        eps = [
            OutsideEndpoint(url="http://host-a:4000/v1", env_var="A"),
            OutsideEndpoint(url="http://host-a:4000/health", env_var="B"),
        ]

        _, tunnel_cls, _ = await _start_with_mocked_ecs(sb, eps)

        specs = tunnel_cls.call_args.kwargs["reverses"]
        assert specs == ["4000:host-a:4000"]
        assert sb._runtime_container_env["A"] == "http://127.0.0.1:4000/v1"
        assert sb._runtime_container_env["B"] == "http://127.0.0.1:4000/health"

    async def test_exec_server_port_is_reserved(self):
        sb = _make_sandbox()
        eps = [OutsideEndpoint(url="http://host-a:5000/v1", env_var="MODEL_BASE_URL")]

        _, tunnel_cls, _ = await _start_with_mocked_ecs(sb, eps)

        specs = tunnel_cls.call_args.kwargs["reverses"]
        assert specs == ["20000:host-a:5000"]
        assert sb.resolved_endpoint_url("MODEL_BASE_URL") == "http://127.0.0.1:20000/v1"

    async def test_env_vars_use_allocated_reverse_ports(self):
        sb = _make_sandbox()
        eps = [
            OutsideEndpoint(url="http://host-a:4000/v1", env_var="A"),
            OutsideEndpoint(url="http://host-b:4000/form", env_var="B"),
        ]

        await _start_with_mocked_ecs(sb, eps)

        assert sb._runtime_container_env["A"] == "http://127.0.0.1:4000/v1"
        assert sb._runtime_container_env["B"] == "http://127.0.0.1:20000/form"


class TestEcsFargateLifecycle:
    """Start/stop with mocked AWS clients and SSH tunnel."""

    async def test_start_happy_path(self):
        sb = _make_sandbox()

        with patch.object(sb, "_do_start") as mock_do_start:
            await sb.start(
                outside_endpoints=[OutsideEndpoint(url="http://127.0.0.1:4000/v1", env_var="MODEL_BASE_URL")]
            )

        assert sb.is_running
        assert sb._outside_endpoints[0].env_var == "MODEL_BASE_URL"
        mock_do_start.assert_called_once()

    async def test_start_failure_triggers_cleanup(self):
        sb = _make_sandbox()

        with (
            patch.object(sb, "_do_start", side_effect=RuntimeError("task failed")),
            patch.object(sb, "_cleanup") as mock_cleanup,
        ):
            with pytest.raises(RuntimeError, match="task failed"):
                await sb.start()

        assert not sb._started
        mock_cleanup.assert_called_once()

    async def test_start_idempotent(self):
        sb = _make_sandbox()
        sb._started = True

        await sb.start()
        assert sb._started

    async def test_start_accepts_multiple_outside_endpoints_in_exec_server_mode(self):
        sb = _make_sandbox()
        eps = [
            OutsideEndpoint(url="http://a:1/v1", env_var="A"),
            OutsideEndpoint(url="http://b:2/v1", env_var="B"),
        ]
        with patch.object(sb, "_do_start") as mock_do_start:
            await sb.start(outside_endpoints=eps)

        assert sb.is_running
        assert sb._outside_endpoints == eps
        mock_do_start.assert_called_once()

    async def test_start_rejects_multiple_outside_endpoints_in_agent_server_mode(self):
        cfg = _cfg(
            container_port=8080,
            ssh_sidecar=_sidecar(exec_server_port=None),
        )
        sb = _make_sandbox(ecs_config=cfg)
        eps = [
            OutsideEndpoint(url="http://a:1/v1", env_var="A"),
            OutsideEndpoint(url="http://b:2/v1", env_var="B"),
        ]

        with pytest.raises(ValueError, match="Agent-server mode supports only one OutsideEndpoint"):
            await sb.start(outside_endpoints=eps)

    async def test_start_rejects_outside_endpoint_without_hostname(self):
        sb = _make_sandbox()

        with (
            patch.object(sb, "_init_aws_clients"),
            patch.object(sb, "_resolve_image", return_value="python:3.12"),
            patch(f"{_PATCH_PREFIX}.download_secret_to_file", return_value="/tmp/key"),
            patch(f"{_PATCH_PREFIX}.download_secret_to_string", return_value="ssh-rsa fake"),
        ):
            with pytest.raises(ValueError, match="Cannot resolve hostname from OutsideEndpoint"):
                await sb.start(outside_endpoints=[OutsideEndpoint(url="http:///v1", env_var="MODEL_BASE_URL")])

    async def test_stop_calls_cleanup(self):
        sb = _make_sandbox()
        sb._started = True
        sb._stopped = False
        ecs_mock = MagicMock()
        sb._ecs = ecs_mock
        sb._task_arn = "arn:aws:ecs:us-west-2:1234:task/test/abc"
        sb._task_def_arn = "arn:aws:ecs:us-west-2:1234:task-definition/test:1"
        tunnel_mock = MagicMock()
        sb._ssh_tunnel = tunnel_mock
        sb._ssh_key_file = "/tmp/fake.key"
        exec_client = MagicMock()
        exec_client.close = AsyncMock(return_value=None)
        sb._exec_client = exec_client

        with patch("os.remove"):
            await sb.stop()

        assert sb._stopped
        tunnel_mock.close.assert_called_once()
        ecs_mock.stop_task.assert_called_once()
        ecs_mock.deregister_task_definition.assert_not_called()
        exec_client.close.assert_awaited_once()

    async def test_stop_idempotent(self):
        sb = _make_sandbox()
        sb._stopped = True
        sb._ecs = MagicMock()

        await sb.stop()
        sb._ecs.stop_task.assert_not_called()

    async def test_context_manager(self):
        sb = _make_sandbox()
        sb.start = AsyncMock()
        sb.stop = AsyncMock()

        async with sb:
            sb.start.assert_awaited_once()
        sb.stop.assert_awaited_once()


# ── TestEcsFargateExec ────────────────────────────────────────────────


class TestEcsFargateExec:
    """Exec/upload/download with mocked ExecClient (now async)."""

    def _started_sb(self):
        sb = _make_sandbox()
        sb._started = True
        client = MagicMock()
        client.exec = AsyncMock(return_value=ExecResult("", "", 0))
        client.upload = AsyncMock(return_value=None)
        client.download = AsyncMock(return_value=b"")
        client.close = AsyncMock(return_value=None)
        sb._exec_client = client
        return sb, client

    async def test_exec_delegates_to_client(self):
        sb, client = self._started_sb()
        client.exec.return_value = ExecResult("hello", "", 0)

        result = await sb.exec("echo hello")
        assert result.stdout == "hello"
        assert result.return_code == 0
        client.exec.assert_awaited_once()
        cmd_arg = client.exec.call_args[0][0]
        assert "echo hello" in cmd_arg

    async def test_exec_does_not_park_a_thread(self):
        """Regression for FEP-886. The sandbox.exec coroutine must await the
        ExecClient directly, not via the default executor — otherwise long
        agent calls park threads and starve the orchestrator at conc>200.

        Patches both ``asyncio.to_thread`` and ``loop.run_in_executor`` since
        the former is implemented as the latter; a future switch from one to
        the other would still reintroduce the same starvation pattern.
        """
        import asyncio as _asyncio

        sb, client = self._started_sb()
        client.exec.return_value = ExecResult("", "", 0)

        loop = _asyncio.get_running_loop()
        with (
            patch.object(_asyncio, "to_thread", side_effect=AssertionError("to_thread used")) as no_tt,
            patch.object(loop, "run_in_executor", side_effect=AssertionError("run_in_executor used")) as no_exec,
        ):
            await sb.exec("echo hi")
        no_tt.assert_not_called()
        no_exec.assert_not_called()

    async def test_exec_calls_run_concurrently(self):
        """`exec` calls must overlap — gather of N slow calls should finish in
        roughly the duration of one, not N × duration."""
        import asyncio as _asyncio
        import time as _time

        sb, client = self._started_sb()

        async def _slow(*args, **kwargs):
            await _asyncio.sleep(0.1)
            return ExecResult("", "", 0)

        client.exec.side_effect = _slow
        t0 = _time.monotonic()
        await _asyncio.gather(*(sb.exec(f"echo {i}") for i in range(20)))
        elapsed = _time.monotonic() - t0
        assert elapsed < 1.0, f"20 × 0.1s calls took {elapsed:.2f}s — serialized?"

    async def test_exec_wraps_cwd(self):
        sb, client = self._started_sb()
        await sb.exec("ls", cwd="/app")
        cmd = client.exec.call_args[0][0]
        assert "cd /app" in cmd

    async def test_exec_wraps_env(self):
        sb, client = self._started_sb()
        await sb.exec("cmd", env={"FOO": "bar"})
        cmd = client.exec.call_args[0][0]
        assert "FOO=bar" in cmd

    async def test_exec_wraps_user_string(self):
        sb, client = self._started_sb()
        await sb.exec("whoami", user="testuser")
        cmd = client.exec.call_args[0][0]
        assert "su -s /bin/bash" in cmd
        assert "testuser" in cmd

    async def test_exec_wraps_user_int(self):
        sb, client = self._started_sb()
        await sb.exec("id", user=1000)
        cmd = client.exec.call_args[0][0]
        assert "getent passwd 1000" in cmd

    async def test_exec_without_exec_client_raises(self):
        sb = _make_sandbox()
        sb._started = True
        sb._exec_client = None

        with pytest.raises(RuntimeError, match="exec-server mode"):
            await sb.exec("echo hi")

    async def test_upload_small_file(self, tmp_path):
        sb, client = self._started_sb()
        f = tmp_path / "small.txt"
        f.write_text("data")

        await sb.upload(f, "/remote/small.txt")
        client.upload.assert_awaited_once_with("/remote/small.txt", f)

    async def test_upload_large_file_via_s3(self, tmp_path):
        cfg = _cfg(s3_bucket="my-bucket", s3_prefix="sandbox")
        sb = _make_sandbox(ecs_config=cfg)
        sb._started = True
        client = MagicMock()
        client.exec = AsyncMock(return_value=ExecResult("", "", 0))
        sb._exec_client = client

        f = tmp_path / "big.bin"
        f.write_bytes(b"x" * (600 * 1024))

        with patch.object(sb, "_upload_via_s3", new=AsyncMock(return_value=None)) as mock_s3:
            await sb.upload(f, "/remote/big.bin")
        mock_s3.assert_awaited_once()

    async def test_upload_directory(self, tmp_path):
        sb, client = self._started_sb()
        d = tmp_path / "mydir"
        d.mkdir()
        (d / "a.txt").write_text("aaa")
        (d / "b.txt").write_text("bbb")

        await sb.upload(d, "/remote/mydir")
        assert client.upload.await_count == 2

    async def test_download(self, tmp_path):
        sb, client = self._started_sb()
        client.download.return_value = b"file contents"
        dest = tmp_path / "out" / "result.txt"

        await sb.download("/remote/result.txt", dest)
        assert dest.read_bytes() == b"file contents"
        client.download.assert_awaited_once_with("/remote/result.txt")

    async def test_exec_reconnect_closes_old_client(self):
        """On tunnel-dead reconnect, the replaced ExecClient must have its
        aiohttp session closed to avoid leaks across reconnects.
        """
        sb, client = self._started_sb()
        client.exec.side_effect = ConnectionError("tunnel down")

        tunnel = MagicMock()
        tunnel.is_open = False
        tunnel.local_port = 19002
        sb._ssh_tunnel = tunnel

        # The second-attempt exec (after reconnect) succeeds against the NEW client.
        new_client = MagicMock()
        new_client.exec = AsyncMock(return_value=ExecResult("ok", "", 0))
        new_client.close = AsyncMock(return_value=None)

        # Reconnect: open a new tunnel + create a new ExecClient.
        async def _reconnect():
            tunnel.is_open = True

        with (
            patch.object(sb, "reconnect_tunnel", side_effect=_reconnect),
            patch.object(sb, "_free_remote_reverse_ports") as mock_free,
            patch(f"{_PATCH_PREFIX}.ExecClient", return_value=new_client),
        ):
            tunnel.wait_ready = MagicMock()
            await sb.exec("echo hi")

        client.close.assert_awaited_once()
        # Reconnect must reclaim the stale reverse listen port before rebinding.
        mock_free.assert_called_once()

    async def test_concurrent_reconnect_happens_once(self):
        """A burst of concurrent execs hitting a dead tunnel must trigger a
        *single* serialized reconnect, not one `ssh -R` storm per failing exec
        (the 'bind: Address in use' pileup across task streams in the EVAL).
        """
        import asyncio as _asyncio

        sb, client = self._started_sb()
        client.exec.side_effect = ConnectionError("tunnel down")

        tunnel = MagicMock()
        tunnel.is_open = False
        tunnel.local_port = 19010
        tunnel.wait_ready = MagicMock()
        sb._ssh_tunnel = tunnel

        new_client = MagicMock()
        new_client.exec = AsyncMock(return_value=ExecResult("ok", "", 0))
        new_client.close = AsyncMock(return_value=None)

        reconnect_calls = 0

        async def _reconnect():
            nonlocal reconnect_calls
            reconnect_calls += 1
            await _asyncio.sleep(0.05)  # let the burst pile up on the lock
            tunnel.is_open = True

        with (
            patch.object(sb, "reconnect_tunnel", side_effect=_reconnect),
            patch.object(sb, "_free_remote_reverse_ports"),
            patch(f"{_PATCH_PREFIX}.ExecClient", return_value=new_client),
        ):
            results = await _asyncio.gather(*(sb.exec(f"echo {i}") for i in range(10)))

        assert all(r.stdout == "ok" for r in results)
        assert reconnect_calls == 1

    async def test_exec_reraises_original_error_when_reconnect_fails(self):
        """If recovery can't reopen the tunnel, the original ConnectionError
        propagates (so callers/Harbor still see the real failure)."""
        sb, client = self._started_sb()
        client.exec.side_effect = ConnectionError("tunnel down")

        tunnel = MagicMock()
        tunnel.is_open = False
        sb._ssh_tunnel = tunnel

        async def _boom():
            raise RuntimeError("cannot reopen")

        with (
            patch.object(sb, "reconnect_tunnel", side_effect=_boom),
            patch.object(sb, "_free_remote_reverse_ports"),
        ):
            with pytest.raises(ConnectionError, match="tunnel down"):
                await sb.exec("echo hi")

    def test_free_remote_reverse_ports_kills_stale_listener(self):
        from nemo_evaluator.sandbox.ecs_fargate import _OutsideEndpointRouting

        sb = _make_sandbox()
        sb._outside_endpoint_routing = _OutsideEndpointRouting.for_exec_server(
            [OutsideEndpoint(url="http://host-a:5000/v1", env_var="MODEL_BASE_URL")],
            _sidecar(),
        )
        sb._task_ip = "10.0.0.7"
        sb._ssh_key_file = "/tmp/key"

        with patch(f"{_PATCH_PREFIX}.subprocess.run", return_value=MagicMock(returncode=0)) as mock_run:
            sb._free_remote_reverse_ports()

        mock_run.assert_called_once()
        joined = " ".join(mock_run.call_args[0][0])
        assert "root@10.0.0.7" in joined
        assert "fuser" in joined
        # exec_server_port (5000) is reserved, so host-a:5000 maps to 20000.
        assert "20000" in joined

    def test_free_remote_reverse_ports_noop_without_ports(self):
        sb = _make_sandbox()
        sb._task_ip = "10.0.0.7"
        sb._ssh_key_file = "/tmp/key"

        with patch(f"{_PATCH_PREFIX}.subprocess.run") as mock_run:
            sb._free_remote_reverse_ports()

        mock_run.assert_not_called()

    async def test_reconnect_aborts_when_ecs_task_stopped(self):
        """Bug: when the sandbox task itself died, reconnect can't help. Record
        WHY (stop reason captured before ECS metadata expires) and abort with the
        original error — don't churn on a dead container (re-triggering the
        reverse-port bind pileup)."""
        sb, client = self._started_sb()
        client.exec.side_effect = ConnectionError("tunnel down")
        tunnel = MagicMock()
        tunnel.is_open = False
        sb._ssh_tunnel = tunnel
        sb._task_arn = "arn:aws:ecs:us-west-2:1234:task/test/dead"

        diag = "lastStatus=STOPPED; stopCode=EssentialContainerExited; stoppedReason='OOM'; container(main=STOPPED exit=137)"
        with (
            patch.object(sb, "_describe_task_diagnostics", return_value=("STOPPED", diag)),
            patch.object(sb, "reconnect_tunnel") as mock_reconnect,
            patch.object(sb, "_free_remote_reverse_ports") as mock_free,
        ):
            with pytest.raises(ConnectionError, match="tunnel down"):
                await sb.exec("echo hi")

        # Dead task → no reverse-port reclaim, no reconnect churn.
        mock_reconnect.assert_not_called()
        mock_free.assert_not_called()
        assert sb._task_confirmed_dead is True

    async def test_reconnect_short_circuits_after_task_confirmed_dead(self):
        """Once the task is known dead, later execs fail fast without re-probing ECS."""
        sb, client = self._started_sb()
        client.exec.side_effect = ConnectionError("tunnel down")
        tunnel = MagicMock()
        tunnel.is_open = False
        sb._ssh_tunnel = tunnel
        sb._task_confirmed_dead = True

        with patch.object(sb, "_describe_task_diagnostics") as mock_diag:
            with pytest.raises(ConnectionError):
                await sb.exec("echo hi")

        mock_diag.assert_not_called()

    async def test_reconnect_proceeds_when_task_running(self):
        """A transient tunnel death with the task still RUNNING reconnects normally."""
        sb, client = self._started_sb()
        client.exec.side_effect = ConnectionError("tunnel down")
        tunnel = MagicMock()
        tunnel.is_open = False
        tunnel.local_port = 19020
        tunnel.wait_ready = MagicMock()
        sb._ssh_tunnel = tunnel

        new_client = MagicMock()
        new_client.exec = AsyncMock(return_value=ExecResult("ok", "", 0))
        new_client.close = AsyncMock(return_value=None)

        async def _reconnect():
            tunnel.is_open = True

        with (
            patch.object(sb, "_describe_task_diagnostics", return_value=("RUNNING", "lastStatus=RUNNING")),
            patch.object(sb, "reconnect_tunnel", side_effect=_reconnect) as mock_reconnect,
            patch.object(sb, "_free_remote_reverse_ports"),
            patch(f"{_PATCH_PREFIX}.ExecClient", return_value=new_client),
        ):
            result = await sb.exec("echo hi")

        assert result.stdout == "ok"
        mock_reconnect.assert_called_once()
        assert sb._task_confirmed_dead is False

    def test_describe_task_diagnostics_captures_stop_reason(self):
        """Stop reason + container exit code must be surfaced for the post-mortem."""
        sb = _make_sandbox()
        sb._task_arn = "arn:aws:ecs:us-west-2:1234:task/test/x"
        ecs = MagicMock()
        ecs.describe_tasks.return_value = {
            "tasks": [
                {
                    "lastStatus": "STOPPED",
                    "stopCode": "EssentialContainerExited",
                    "stoppedReason": "Essential container in task exited",
                    "containers": [
                        {"name": "main", "lastStatus": "STOPPED", "exitCode": 137, "reason": "OutOfMemory"},
                        {"name": "ssh-tunnel", "lastStatus": "STOPPED", "exitCode": 0},
                    ],
                }
            ]
        }
        sb._ecs = ecs

        status, diag = sb._describe_task_diagnostics()

        assert status == "STOPPED"
        assert "stopCode=EssentialContainerExited" in diag
        assert "stoppedReason='Essential container in task exited'" in diag
        assert "exit=137" in diag
        assert "OutOfMemory" in diag

    def test_describe_task_diagnostics_handles_expired_metadata(self):
        sb = _make_sandbox()
        sb._task_arn = "arn:aws:ecs:us-west-2:1234:task/test/x"
        ecs = MagicMock()
        ecs.describe_tasks.return_value = {"tasks": []}
        sb._ecs = ecs

        status, diag = sb._describe_task_diagnostics()

        assert status == "GONE"
        assert "expired" in diag

    def test_describe_task_diagnostics_best_effort_on_api_error(self):
        sb = _make_sandbox()
        sb._task_arn = "arn:aws:ecs:us-west-2:1234:task/test/x"
        ecs = MagicMock()
        ecs.describe_tasks.side_effect = RuntimeError("throttled")
        sb._ecs = ecs

        status, diag = sb._describe_task_diagnostics()

        # UNKNOWN → caller still attempts reconnect (transient API hiccup).
        assert status == "UNKNOWN"
        assert "throttled" in diag


class TestExecClientAsync:
    """ExecClient is a true coroutine client, not a thread-pool blocker."""

    def test_methods_are_coroutines(self):
        """Async methods must be coroutine functions so callers can `await`
        them without `asyncio.to_thread` (the FEP-886 starvation pattern).
        """
        import asyncio as _asyncio

        from nemo_evaluator.sandbox.ecs_fargate import ExecClient

        c = ExecClient(port=1)
        assert _asyncio.iscoroutinefunction(c.exec)
        assert _asyncio.iscoroutinefunction(c.upload)
        assert _asyncio.iscoroutinefunction(c.download)
        assert _asyncio.iscoroutinefunction(c.health)
        assert _asyncio.iscoroutinefunction(c.close)

    async def test_close_without_session_is_noop(self):
        from nemo_evaluator.sandbox.ecs_fargate import ExecClient

        c = ExecClient(port=1)
        await c.close()  # no session created yet; must not raise


# ── TestSshTunnel ─────────────────────────────────────────────────────


class TestSshTunnel:
    """SSH tunnel with mocked subprocess and socket."""

    def _make_tunnel(self, **kw):
        from nemo_evaluator.sandbox.ecs_fargate import SshTunnel

        defaults = dict(host="1.2.3.4", port=2222, user="root", key_file="/tmp/key")
        defaults.update(kw)
        return SshTunnel(**defaults)

    def test_build_ssh_cmd_simple_forward(self):
        t = self._make_tunnel(forward_port=5000)
        t._local_port = 19000
        cmd = t._build_ssh_cmd()
        assert cmd[0] == "ssh"
        assert "-N" in cmd
        assert "StrictHostKeyChecking=no" in " ".join(cmd)
        assert "-i" in cmd
        assert "/tmp/key" in cmd
        fwd_idx = cmd.index("-L")
        assert "127.0.0.1:19000:127.0.0.1:5000" in cmd[fwd_idx + 1]
        assert cmd[-1] == "root@1.2.3.4"

    def test_build_ssh_cmd_with_forwards_and_reverses(self):
        t = self._make_tunnel(
            forwards=["127.0.0.1:8000:127.0.0.1:8000"],
            reverses=["4000:127.0.0.1:4000"],
        )
        cmd = t._build_ssh_cmd()
        assert "-L" in cmd
        assert "127.0.0.1:8000:127.0.0.1:8000" in cmd
        assert "-R" in cmd
        assert "4000:127.0.0.1:4000" in cmd

    @patch(f"{_PATCH_PREFIX}._free_port", return_value=11111)
    @patch(f"{_PATCH_PREFIX}.subprocess.Popen")
    def test_open_success(self, mock_popen, mock_free_port):
        proc = MagicMock()
        proc.poll.return_value = None
        proc.pid = 12345
        mock_popen.return_value = proc

        t = self._make_tunnel(forward_port=5000)
        with patch.object(t, "_wait_for_local_port"):
            with patch(f"{_PATCH_PREFIX}.time.sleep"):
                t.open(max_retries=1)

        assert t.is_open
        assert t.local_port == 11111
        mock_popen.assert_called_once()

    @patch(f"{_PATCH_PREFIX}._free_port", return_value=22222)
    @patch(f"{_PATCH_PREFIX}.subprocess.Popen")
    def test_open_retries_on_connection_refused(self, mock_popen, mock_free_port):
        fail_proc = MagicMock()
        fail_proc.poll.return_value = 255
        fail_proc.stderr = MagicMock()
        fail_proc.stderr.read.return_value = b"Connection refused"

        ok_proc = MagicMock()
        ok_proc.poll.return_value = None
        ok_proc.pid = 999

        mock_popen.side_effect = [fail_proc, ok_proc]

        t = self._make_tunnel(forward_port=5000)
        with patch.object(t, "_wait_for_local_port"):
            with patch(f"{_PATCH_PREFIX}.time.sleep"):
                t.open(max_retries=3)

        assert t.is_open
        assert mock_popen.call_count == 2

    @patch(f"{_PATCH_PREFIX}._free_port", return_value=44444)
    @patch(f"{_PATCH_PREFIX}.subprocess.Popen")
    def test_open_retries_on_remote_port_forwarding_failed(self, mock_popen, mock_free_port):
        """Regression: a reconnect whose reverse port is still held by the dead
        tunnel's stale listener must retry (the server frees it shortly) instead
        of aborting immediately — the EVAL connection-refused rollout failures.
        """
        fail_proc = MagicMock()
        fail_proc.poll.return_value = 255
        fail_proc.stderr = MagicMock()
        # ssh emits this (via error(), shown even at LogLevel=ERROR) when the
        # sandbox sshd can't rebind the reverse listen port (address in use).
        fail_proc.stderr.read.return_value = b"Warning: remote port forwarding failed for listen port 59579"

        ok_proc = MagicMock()
        ok_proc.poll.return_value = None
        ok_proc.pid = 777

        mock_popen.side_effect = [fail_proc, ok_proc]

        t = self._make_tunnel(forward_port=5000, reverses=["59579:host-a:5000"])
        with patch.object(t, "_wait_for_local_port"):
            with patch(f"{_PATCH_PREFIX}.time.sleep"):
                t.open(max_retries=3)

        assert t.is_open
        assert mock_popen.call_count == 2

    @patch(f"{_PATCH_PREFIX}._free_port", return_value=33333)
    @patch(f"{_PATCH_PREFIX}.subprocess.Popen")
    def test_open_raises_on_non_retryable_error(self, mock_popen, mock_free_port):
        proc = MagicMock()
        proc.poll.return_value = 1
        proc.stderr = MagicMock()
        proc.stderr.read.return_value = b"Permission denied"
        mock_popen.return_value = proc

        t = self._make_tunnel(forward_port=5000)
        with patch(f"{_PATCH_PREFIX}.time.sleep"):
            with pytest.raises(RuntimeError, match="SSH tunnel exited immediately"):
                t.open(max_retries=3)

    def test_close_kills_process(self):
        t = self._make_tunnel()
        proc = MagicMock()
        proc.pid = 111
        t._proc = proc

        t.close()
        proc.terminate.assert_called_once()
        assert t._proc is None

    def test_close_noop_when_no_process(self):
        t = self._make_tunnel()
        t._proc = None
        t.close()

    @patch("urllib.request.urlopen")
    def test_wait_ready_health_url(self, mock_urlopen):
        resp = MagicMock()
        resp.status = 200
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = resp

        t = self._make_tunnel()
        t._proc = MagicMock()
        t._proc.poll.return_value = None

        t.wait_ready(health_url="http://127.0.0.1:5000/health", timeout=5)
        mock_urlopen.assert_called_once()


# ── TestSshSidecarContainer ───────────────────────────────────────────


class TestSshSidecarContainer:
    """The generated sidecar must ship the tools the reconnect path relies on."""

    def test_sidecar_installs_fuser_for_reverse_port_reclaim(self):
        from nemo_evaluator.sandbox.ecs_fargate import build_ssh_sidecar_container

        container = build_ssh_sidecar_container(_sidecar(), public_key_value="ssh-rsa fake", max_lifetime_sec=0)
        cmd = container["command"][0]
        # psmisc provides `fuser`, used to free a stale reverse listen port.
        assert "psmisc" in cmd

    def test_sidecar_keeps_long_keepalive_tolerance(self):
        """EVAL-1143: keepalive stays long so tunnels survive ~2min container
        setup and multi-hour tasks. The reconnect fix must not regress this."""
        from nemo_evaluator.sandbox.ecs_fargate import build_ssh_sidecar_container

        container = build_ssh_sidecar_container(_sidecar(), public_key_value="ssh-rsa fake", max_lifetime_sec=0)
        cmd = container["command"][0]
        assert "ClientAliveInterval 30" in cmd
        assert "ClientAliveCountMax 20" in cmd


# ── TestImageBuilder ──────────────────────────────────────────────────


class TestImageBuilder:
    """ImageBuilder with mocked ECR, S3, CodeBuild."""

    def test_get_ecr_image_tag(self, tmp_path):
        from nemo_evaluator.sandbox.ecs_fargate import ImageBuilder

        env_dir = tmp_path / "env"
        env_dir.mkdir()
        (env_dir / "Dockerfile").write_text("FROM python:3.12")
        (env_dir / "setup.sh").write_text("echo hi")

        tag = ImageBuilder.get_ecr_image_tag(env_dir, "my-sandbox")
        assert tag.startswith("my-sandbox__")
        assert len(tag.split("__")[1]) == 8

    def test_get_ecr_image_tag_deterministic(self, tmp_path):
        from nemo_evaluator.sandbox.ecs_fargate import ImageBuilder

        env_dir = tmp_path / "env"
        env_dir.mkdir()
        (env_dir / "Dockerfile").write_text("FROM python:3.12")

        tag1 = ImageBuilder.get_ecr_image_tag(env_dir, "sb")
        tag2 = ImageBuilder.get_ecr_image_tag(env_dir, "sb")
        assert tag1 == tag2

    @patch(f"{_PATCH_PREFIX}._require_aws_sdks")
    def test_image_exists_in_ecr_true(self, mock_aws):
        from nemo_evaluator.sandbox.ecs_fargate import ImageBuilder

        ecr = MagicMock()
        ecr.describe_images.return_value = {"imageDetails": [{"imageDigest": "sha256:abc"}]}
        mock_boto3 = MagicMock()
        mock_boto3.client.return_value = ecr
        mock_aws.return_value = (
            mock_boto3,
            MagicMock(),
            type("CE", (Exception,), {"response": {"Error": {"Code": ""}}}),
        )

        assert ImageBuilder.image_exists_in_ecr("123.dkr.ecr.us-west-2.amazonaws.com/repo", "tag1") is True

    @patch(f"{_PATCH_PREFIX}._require_aws_sdks")
    def test_image_exists_in_ecr_false(self, mock_aws):
        from nemo_evaluator.sandbox.ecs_fargate import ImageBuilder

        class FakeClientError(Exception):
            def __init__(self):
                self.response = {"Error": {"Code": "ImageNotFoundException"}}

        ecr = MagicMock()
        ecr.describe_images.side_effect = FakeClientError()
        mock_boto3 = MagicMock()
        mock_boto3.client.return_value = ecr
        mock_aws.return_value = (mock_boto3, MagicMock(), FakeClientError)

        assert ImageBuilder.image_exists_in_ecr("123.dkr.ecr.us-west-2.amazonaws.com/repo", "missing") is False

    def test_ecr_region_extraction(self):
        from nemo_evaluator.sandbox.ecs_fargate import ImageBuilder

        assert ImageBuilder._ecr_region("123.dkr.ecr.us-west-2.amazonaws.com/repo") == "us-west-2"
        assert ImageBuilder._ecr_region("123.dkr.ecr.eu-west-1.amazonaws.com/repo") == "eu-west-1"
        assert ImageBuilder._ecr_region("my-repo", fallback="us-east-1") == "us-east-1"

    @patch(f"{_PATCH_PREFIX}.subprocess.run")
    def test_ecr_docker_login_success(self, mock_run):
        from nemo_evaluator.sandbox.ecs_fargate import ImageBuilder

        mock_run.return_value = MagicMock(returncode=0)
        ImageBuilder.ecr_docker_login("123.dkr.ecr.us-west-2.amazonaws.com/repo", region="us-west-2")
        mock_run.assert_called_once()
        assert "docker login" in mock_run.call_args[0][0]

    @patch(f"{_PATCH_PREFIX}.subprocess.run")
    def test_ecr_docker_login_failure(self, mock_run):
        from nemo_evaluator.sandbox.ecs_fargate import ImageBuilder

        mock_run.return_value = MagicMock(returncode=1, stderr="no credentials")
        with pytest.raises(RuntimeError, match="ECR docker login failed"):
            ImageBuilder.ecr_docker_login("123.dkr.ecr.us-west-2.amazonaws.com/repo")

    @patch(f"{_PATCH_PREFIX}.subprocess.run")
    def test_docker_push_to_ecr(self, mock_run):
        from nemo_evaluator.sandbox.ecs_fargate import ImageBuilder

        mock_run.side_effect = [
            MagicMock(returncode=0),
            MagicMock(returncode=0, stderr=""),
        ]
        url = ImageBuilder.docker_push_to_ecr("local:latest", "123.dkr.ecr.us-west-2.amazonaws.com/repo", "tag1")
        assert url == "123.dkr.ecr.us-west-2.amazonaws.com/repo:tag1"
        assert mock_run.call_count == 2

    @patch(f"{_PATCH_PREFIX}.ImageBuilder.image_exists_in_ecr", return_value=True)
    def test_ensure_image_built_cache_hit(self, mock_exists, tmp_path):
        from nemo_evaluator.sandbox.ecs_fargate import ImageBuilder

        env_dir = tmp_path / "env"
        env_dir.mkdir()
        (env_dir / "Dockerfile").write_text("FROM alpine")

        cfg = _cfg(ecr_repository="123.dkr.ecr.us-west-2.amazonaws.com/repo", environment_dir=str(env_dir))
        result = ImageBuilder.ensure_image_built(cfg=cfg, environment_name="test")

        assert "123.dkr.ecr.us-west-2.amazonaws.com/repo:" in result
        mock_exists.assert_called_once()


# ── TestTaskDefCache ──────────────────────────────────────────────────


@pytest.fixture
def clear_task_def_cache():
    from nemo_evaluator.sandbox import ecs_fargate

    ecs_fargate._task_def_cache.clear()
    ecs_fargate._task_def_inflight.clear()
    yield
    ecs_fargate._task_def_cache.clear()
    ecs_fargate._task_def_inflight.clear()


def _ssm_param_not_found_error():
    from botocore.exceptions import ClientError

    return ClientError({"Error": {"Code": "ParameterNotFound", "Message": "not found"}}, "GetParameter")


def _make_cache_sandbox():
    sb = _make_sandbox()
    sb._ecs = MagicMock()
    sb._ssm = MagicMock()
    sb._ssm.get_parameter.side_effect = _ssm_param_not_found_error()
    return sb


def _make_ssm_sandbox():
    sb = _make_sandbox()
    sb._ecs = MagicMock()
    sb._ssm = MagicMock()
    return sb


class TestTaskDefCache:
    def test_same_payload_reuses_cached_arn(self, clear_task_def_cache):
        sb = _make_cache_sandbox()
        sb._ecs.register_task_definition.return_value = {
            "taskDefinition": {"taskDefinitionArn": "arn:aws:ecs:us-east-1:1234:task-definition/test:1"}
        }
        payload = {
            "family": "test-abc",
            "cpu": "4096",
            "memory": "8192",
            "containerDefinitions": [{"name": "main", "image": "python:3.12"}],
        }

        arn1 = sb._do_register(payload)
        arn2 = sb._do_register({**payload, "family": "test-xyz"})

        assert arn1 == arn2
        sb._ecs.register_task_definition.assert_called_once()

    def test_different_payload_registers_new_task_def(self, clear_task_def_cache):
        sb = _make_cache_sandbox()
        sb._ecs.register_task_definition.side_effect = [
            {"taskDefinition": {"taskDefinitionArn": "arn:aws:ecs:us-east-1:1234:task-definition/test:1"}},
            {"taskDefinition": {"taskDefinitionArn": "arn:aws:ecs:us-east-1:1234:task-definition/test:2"}},
        ]
        payload_a = {"family": "test", "cpu": "4096", "containerDefinitions": []}
        payload_b = {"family": "test", "cpu": "8192", "containerDefinitions": []}

        arn1 = sb._do_register(payload_a)
        arn2 = sb._do_register(payload_b)

        assert arn1 != arn2
        assert sb._ecs.register_task_definition.call_count == 2

    def test_concurrent_identical_payloads_register_once(self, clear_task_def_cache):
        sb = _make_cache_sandbox()
        gate = threading.Event()

        def slow_register(**_kwargs):
            gate.wait(timeout=5.0)
            return {"taskDefinition": {"taskDefinitionArn": "arn:aws:ecs:us-east-1:1234:task-definition/test:1"}}

        sb._ecs.register_task_definition.side_effect = slow_register

        payload = {"family": "test", "cpu": "4096", "memory": "8192", "containerDefinitions": []}
        results: list[str] = []
        errors: list[BaseException] = []

        def call():
            try:
                results.append(sb._do_register(payload))
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=call) for _ in range(8)]
        for t in threads:
            t.start()
        # Poll until the elected registrant has entered slow_register (under the cache
        # lock it has already added the inflight entry the other threads will wait on).
        # Avoids a flaky time.sleep that may be too short under CI load.
        deadline = time.monotonic() + 5.0
        while sb._ecs.register_task_definition.call_count == 0:
            if time.monotonic() > deadline:
                pytest.fail("registrant never reached slow_register within 5s")
            time.sleep(0.005)
        gate.set()
        for t in threads:
            t.join(timeout=5.0)

        assert not errors, f"thread errors: {errors}"
        assert sb._ecs.register_task_definition.call_count == 1
        assert len(results) == 8
        assert all(r == results[0] for r in results)


# ── TestSsmTaskDefCache ───────────────────────────────────────────────


class TestSsmTaskDefCache:
    def test_ssm_hit_active_skips_register(self, clear_task_def_cache):
        sb = _make_ssm_sandbox()
        sb._ssm.get_parameter.return_value = {"Parameter": {"Value": "arn:cached:1"}}
        sb._ecs.describe_task_definition.return_value = {
            "taskDefinition": {"taskDefinitionArn": "arn:cached:1", "status": "ACTIVE"}
        }
        payload = {"family": "test", "cpu": "4096", "containerDefinitions": []}

        arn = sb._do_register(payload)

        assert arn == "arn:cached:1"
        sb._ssm.get_parameter.assert_called_once()
        sb._ecs.describe_task_definition.assert_called_once()
        sb._ecs.register_task_definition.assert_not_called()
        sb._ssm.put_parameter.assert_not_called()

    def test_ssm_hit_inactive_falls_through_to_register(self, clear_task_def_cache):
        sb = _make_ssm_sandbox()
        sb._ssm.get_parameter.return_value = {"Parameter": {"Value": "arn:stale:1"}}
        sb._ecs.describe_task_definition.return_value = {
            "taskDefinition": {"taskDefinitionArn": "arn:stale:1", "status": "INACTIVE"}
        }
        sb._ecs.register_task_definition.return_value = {"taskDefinition": {"taskDefinitionArn": "arn:fresh:2"}}
        payload = {"family": "test", "cpu": "4096", "containerDefinitions": []}

        arn = sb._do_register(payload)

        assert arn == "arn:fresh:2"
        sb._ecs.register_task_definition.assert_called_once()
        sb._ssm.put_parameter.assert_called_once()

    def test_ssm_miss_registers_and_writes(self, clear_task_def_cache):
        sb = _make_ssm_sandbox()
        sb._ssm.get_parameter.side_effect = _ssm_param_not_found_error()
        sb._ecs.register_task_definition.return_value = {"taskDefinition": {"taskDefinitionArn": "arn:new:1"}}
        payload = {"family": "test", "cpu": "4096", "containerDefinitions": []}

        arn = sb._do_register(payload)

        assert arn == "arn:new:1"
        sb._ecs.register_task_definition.assert_called_once()
        put_kwargs = sb._ssm.put_parameter.call_args.kwargs
        assert put_kwargs["Value"] == "arn:new:1"
        assert put_kwargs["Overwrite"] is True
        assert put_kwargs["Name"].startswith("/harbor/task-defs/")

    def test_ssm_read_error_falls_through_gracefully(self, clear_task_def_cache):
        from botocore.exceptions import ClientError

        sb = _make_ssm_sandbox()
        sb._ssm.get_parameter.side_effect = ClientError(
            {"Error": {"Code": "AccessDeniedException", "Message": "denied"}}, "GetParameter"
        )
        sb._ecs.register_task_definition.return_value = {"taskDefinition": {"taskDefinitionArn": "arn:fallback:1"}}
        payload = {"family": "test", "cpu": "4096", "containerDefinitions": []}

        arn = sb._do_register(payload)

        assert arn == "arn:fallback:1"
        sb._ecs.register_task_definition.assert_called_once()

    def test_ssm_write_error_does_not_fail_run(self, clear_task_def_cache):
        from botocore.exceptions import ClientError

        sb = _make_ssm_sandbox()
        sb._ssm.get_parameter.side_effect = _ssm_param_not_found_error()
        sb._ecs.register_task_definition.return_value = {"taskDefinition": {"taskDefinitionArn": "arn:ok:1"}}
        sb._ssm.put_parameter.side_effect = ClientError(
            {"Error": {"Code": "AccessDeniedException", "Message": "denied"}}, "PutParameter"
        )
        payload = {"family": "test", "cpu": "4096", "containerDefinitions": []}

        arn = sb._do_register(payload)

        assert arn == "arn:ok:1"
        sb._ecs.register_task_definition.assert_called_once()

    def test_ssm_hit_populates_local_cache(self, clear_task_def_cache):
        from nemo_evaluator.sandbox import ecs_fargate

        sb = _make_ssm_sandbox()
        sb._ssm.get_parameter.return_value = {"Parameter": {"Value": "arn:cached:1"}}
        sb._ecs.describe_task_definition.return_value = {
            "taskDefinition": {"taskDefinitionArn": "arn:cached:1", "status": "ACTIVE"}
        }
        payload = {"family": "test", "cpu": "4096", "containerDefinitions": []}

        arn1 = sb._do_register(payload)
        arn2 = sb._do_register(payload)

        assert arn1 == arn2 == "arn:cached:1"
        assert sb._ssm.get_parameter.call_count == 1
        assert sb._ecs.describe_task_definition.call_count == 1
        h = ecs_fargate._compute_task_def_hash(payload)
        assert ecs_fargate._task_def_cache[h] == "arn:cached:1"


# ── TestTaskDefHashStability ─────────────────────────────────────────


class TestTaskDefHashStability:
    """logConfiguration is stripped before hashing so cross-run cache hits work."""

    def test_different_log_stream_prefix_same_hash(self):
        from nemo_evaluator.sandbox.ecs_fargate import _compute_task_def_hash

        base_cd = [{"name": "main", "image": "python:3.12", "environment": []}]
        payload_a = {
            "family": "run-a",
            "cpu": "4096",
            "containerDefinitions": base_cd
            + [{"name": "ssh-tunnel", "logConfiguration": {"awslogs-stream-prefix": "run-A"}}],
        }
        payload_b = {
            "family": "run-b",
            "cpu": "4096",
            "containerDefinitions": base_cd
            + [{"name": "ssh-tunnel", "logConfiguration": {"awslogs-stream-prefix": "run-B"}}],
        }
        assert _compute_task_def_hash(payload_a) == _compute_task_def_hash(payload_b)

    def test_different_image_still_different_hash(self):
        from nemo_evaluator.sandbox.ecs_fargate import _compute_task_def_hash

        payload_a = {"family": "x", "containerDefinitions": [{"name": "main", "image": "python:3.12"}]}
        payload_b = {"family": "x", "containerDefinitions": [{"name": "main", "image": "python:3.13"}]}
        assert _compute_task_def_hash(payload_a) != _compute_task_def_hash(payload_b)


# ── TestPerInvocationEnvSplit ─────────────────────────────────────────


class TestPerInvocationEnvSplit:
    def test_split_routes_efs_session_and_outside_endpoint_overrides(self):
        from nemo_evaluator.sandbox.ecs_fargate import _OutsideEndpointRouting

        sb = _make_sandbox()
        sb._outside_endpoint_routing = _OutsideEndpointRouting.for_exec_server(
            [OutsideEndpoint(url="http://api.local:4000/s/abc123", env_var="MODEL_BASE_URL")],
            _sidecar(),
        )
        env = {
            "HARBOR_TASK_DIR": "harbor_datasets/terminal-bench@2.0/regex-chess",
            "_NEL_EFS_SESSION": "1779033306_abc",
            "MODEL_BASE_URL": "http://127.0.0.1:4000/s/abc123",
        }

        stable, runtime = sb._split_env(env)

        assert stable == {"HARBOR_TASK_DIR": "harbor_datasets/terminal-bench@2.0/regex-chess"}
        assert runtime == {
            "_NEL_EFS_SESSION": "1779033306_abc",
            "MODEL_BASE_URL": "http://127.0.0.1:4000/s/abc123",
        }

    def test_split_keeps_unrelated_env_in_stable(self):
        sb = _make_sandbox()
        env = {"FOO": "bar", "HARBOR_TASK_DIR": "x"}

        stable, runtime = sb._split_env(env)

        assert stable == env
        assert runtime == {}
