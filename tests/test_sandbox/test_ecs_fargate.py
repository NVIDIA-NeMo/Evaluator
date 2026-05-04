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
"""Tests for ECS Fargate sandbox — all AWS/SSH/network calls mocked."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nemo_evaluator.sandbox.base import ExecResult, OutsideEndpoint, SandboxSpec


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


# ── TestResolveOutsideEndpoint ────────────────────────────────────────


class TestResolveOutsideEndpoint:
    """Pure-logic tests — no AWS or SSH mocking needed."""

    def _sb(self):
        sb = _make_sandbox()
        return sb

    def test_match_by_netloc_with_reverse_port_map(self):
        sb = self._sb()
        sb._outside_endpoints = [OutsideEndpoint(url="http://127.0.0.1:4000/v1/s/abc123", env_var="MODEL_BASE_URL")]
        sb._reverse_port_map = {"MODEL_BASE_URL": (4000, "http")}

        result = sb.resolve_outside_endpoint("http://127.0.0.1:4000/v1")
        assert result == "http://127.0.0.1:4000/v1"

    def test_session_scoped_url_matches_base_url(self):
        """resolve_outside_endpoint preserves the caller's path (no session
        injection).  Use resolved_endpoint_url for session-scoped URLs."""
        sb = self._sb()
        sb._outside_endpoints = [OutsideEndpoint(url="http://127.0.0.1:4000/v1/s/session42", env_var="MODEL_BASE_URL")]
        sb._reverse_port_map = {"MODEL_BASE_URL": (4000, "http")}

        result = sb.resolve_outside_endpoint("http://127.0.0.1:4000/v1")
        assert "127.0.0.1:4000" in result
        assert result.endswith("/v1")

    def test_path_preserved_through_rewrite(self):
        sb = self._sb()
        sb._outside_endpoints = [OutsideEndpoint(url="https://api.nvidia.com/v1/chat", env_var="LLM_URL")]
        sb._reverse_port_map = {"LLM_URL": (443, "https")}

        result = sb.resolve_outside_endpoint("https://api.nvidia.com/v1/chat/completions")
        assert result == "https://127.0.0.1:443/v1/chat/completions"

    def test_scheme_rewritten_from_port_map(self):
        sb = self._sb()
        sb._outside_endpoints = [OutsideEndpoint(url="https://model.example.com:8443/v1", env_var="M")]
        sb._reverse_port_map = {"M": (8443, "https")}

        result = sb.resolve_outside_endpoint("https://model.example.com:8443/v1")
        assert result.startswith("https://127.0.0.1:8443")

    def test_fallback_to_ssh_tunnel_port(self):
        sb = self._sb()
        sb._ssh_tunnel_port = 9999
        sb._outside_endpoints = []
        sb._reverse_port_map = {}

        result = sb.resolve_outside_endpoint("http://proxy.local:4000/v1")
        assert "127.0.0.1:9999" in result
        assert result.endswith("/v1")

    def test_no_match_raises_runtime_error(self):
        sb = self._sb()
        sb._outside_endpoints = []
        sb._reverse_port_map = {}
        sb._ssh_tunnel_port = None

        with pytest.raises(RuntimeError, match="requires SSH reverse tunnel"):
            sb.resolve_outside_endpoint("http://nowhere:1234/api")

    def test_different_netloc_no_match(self):
        sb = self._sb()
        sb._outside_endpoints = [OutsideEndpoint(url="http://host-a:4000/v1", env_var="M")]
        sb._reverse_port_map = {"M": (4000, "http")}
        sb._ssh_tunnel_port = None

        with pytest.raises(RuntimeError, match="requires SSH reverse tunnel"):
            sb.resolve_outside_endpoint("http://host-b:5000/v1")

    def test_env_var_not_in_port_map_skipped(self):
        sb = self._sb()
        sb._outside_endpoints = [OutsideEndpoint(url="http://127.0.0.1:4000/v1", env_var="MISSING_KEY")]
        sb._reverse_port_map = {}
        sb._ssh_tunnel_port = 7777

        result = sb.resolve_outside_endpoint("http://127.0.0.1:4000/v1")
        assert "127.0.0.1:7777" in result


class TestResolvedEndpointUrl:
    """Tests for the session-scoped resolved_endpoint_url method."""

    def _sb(self):
        return _make_sandbox()

    def test_returns_session_path_via_reverse_port_map(self):
        sb = self._sb()
        sb._outside_endpoints = [OutsideEndpoint(url="http://localhost:4000/s/abc123", env_var="MODEL_BASE_URL")]
        sb._reverse_port_map = {"MODEL_BASE_URL": (4000, "http")}

        result = sb.resolved_endpoint_url("MODEL_BASE_URL")
        assert result == "http://127.0.0.1:4000/s/abc123"

    def test_returns_session_path_via_ssh_tunnel(self):
        sb = self._sb()
        sb._outside_endpoints = [OutsideEndpoint(url="http://localhost:4000/s/xyz789", env_var="MODEL_BASE_URL")]
        sb._reverse_port_map = {}
        sb._ssh_tunnel_port = 9999

        result = sb.resolved_endpoint_url("MODEL_BASE_URL")
        assert result == "http://127.0.0.1:9999/s/xyz789"

    def test_returns_none_for_unknown_env_var(self):
        sb = self._sb()
        sb._outside_endpoints = [OutsideEndpoint(url="http://localhost:4000/s/abc", env_var="MODEL_BASE_URL")]
        sb._reverse_port_map = {"MODEL_BASE_URL": (4000, "http")}

        assert sb.resolved_endpoint_url("UNKNOWN_VAR") is None

    def test_returns_none_when_no_endpoints(self):
        sb = self._sb()
        sb._outside_endpoints = []
        assert sb.resolved_endpoint_url("MODEL_BASE_URL") is None

    def test_harbor_scenario_base_url_gets_session_path(self):
        """The real bug: Harbor calls with base URL, needs session path back."""
        sb = self._sb()
        sb._outside_endpoints = [OutsideEndpoint(url="http://localhost:4000/s/session42", env_var="MODEL_BASE_URL")]
        sb._reverse_port_map = {"MODEL_BASE_URL": (4000, "http")}

        session_url = sb.resolved_endpoint_url("MODEL_BASE_URL")
        assert "/s/session42" in session_url
        assert "127.0.0.1:4000" in session_url

        plain_url = sb.resolve_outside_endpoint("http://localhost:4000")
        assert "/s/" not in plain_url


# ── TestEcsFargateLifecycle ───────────────────────────────────────────


_PATCH_PREFIX = "nemo_evaluator.sandbox.ecs_fargate"


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

    async def test_start_rejects_multiple_outside_endpoints(self):
        sb = _make_sandbox()
        eps = [
            OutsideEndpoint(url="http://a:1/v1", env_var="A"),
            OutsideEndpoint(url="http://b:2/v1", env_var="B"),
        ]
        with pytest.raises(ValueError, match="Only one OutsideEndpoint"):
            await sb.start(outside_endpoints=eps)

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

        with patch("os.remove"):
            await sb.stop()

        assert sb._stopped
        tunnel_mock.close.assert_called_once()
        ecs_mock.stop_task.assert_called_once()
        ecs_mock.deregister_task_definition.assert_called_once()

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
    """Exec/upload/download with mocked ExecClient."""

    def _started_sb(self):
        sb = _make_sandbox()
        sb._started = True
        client = MagicMock()
        sb._exec_client = client
        return sb, client

    async def test_exec_delegates_to_client(self):
        sb, client = self._started_sb()
        client.exec.return_value = ExecResult("hello", "", 0)

        result = await sb.exec("echo hello")
        assert result.stdout == "hello"
        assert result.return_code == 0
        client.exec.assert_called_once()
        cmd_arg = client.exec.call_args[0][0]
        assert "echo hello" in cmd_arg

    async def test_exec_wraps_cwd(self):
        sb, client = self._started_sb()
        client.exec.return_value = ExecResult("", "", 0)

        await sb.exec("ls", cwd="/app")
        cmd = client.exec.call_args[0][0]
        assert "cd /app" in cmd

    async def test_exec_wraps_env(self):
        sb, client = self._started_sb()
        client.exec.return_value = ExecResult("", "", 0)

        await sb.exec("cmd", env={"FOO": "bar"})
        cmd = client.exec.call_args[0][0]
        assert "FOO=bar" in cmd

    async def test_exec_wraps_user_string(self):
        sb, client = self._started_sb()
        client.exec.return_value = ExecResult("", "", 0)

        await sb.exec("whoami", user="testuser")
        cmd = client.exec.call_args[0][0]
        assert "su -s /bin/bash" in cmd
        assert "testuser" in cmd

    async def test_exec_wraps_user_int(self):
        sb, client = self._started_sb()
        client.exec.return_value = ExecResult("", "", 0)

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
        client.upload.assert_called_once_with("/remote/small.txt", f)

    async def test_upload_large_file_via_s3(self, tmp_path):
        cfg = _cfg(s3_bucket="my-bucket", s3_prefix="sandbox")
        sb = _make_sandbox(ecs_config=cfg)
        sb._started = True
        client = MagicMock()
        sb._exec_client = client

        f = tmp_path / "big.bin"
        f.write_bytes(b"x" * (600 * 1024))

        with patch.object(sb, "_upload_via_s3") as mock_s3:
            await sb.upload(f, "/remote/big.bin")
        mock_s3.assert_called_once()

    async def test_upload_directory(self, tmp_path):
        sb, client = self._started_sb()
        d = tmp_path / "mydir"
        d.mkdir()
        (d / "a.txt").write_text("aaa")
        (d / "b.txt").write_text("bbb")

        await sb.upload(d, "/remote/mydir")
        assert client.upload.call_count == 2

    async def test_download(self, tmp_path):
        sb, client = self._started_sb()
        client.download.return_value = b"file contents"
        dest = tmp_path / "out" / "result.txt"

        await sb.download("/remote/result.txt", dest)
        assert dest.read_bytes() == b"file contents"
        client.download.assert_called_once_with("/remote/result.txt")


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
