# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from __future__ import annotations

import importlib
import sys
from subprocess import CompletedProcess
from types import ModuleType

import pytest


def _purge_modules(prefix: str) -> None:
    for name in list(sys.modules.keys()):
        if name == prefix or name.startswith(prefix + "."):
            sys.modules.pop(name, None)


def test_sandbox_import_is_lightweight(monkeypatch: pytest.MonkeyPatch):
    """
    Importing `nemo_evaluator.sandbox` must not eagerly import optional AWS SDK deps.
    """

    # Ensure a fresh import so our spy sees any calls made during import-time.
    _purge_modules("nemo_evaluator.sandbox")

    real_import_module = importlib.import_module
    calls: list[str] = []

    def spy_import_module(name: str, package: str | None = None) -> ModuleType:
        calls.append(name)
        if name == "boto3" or name.startswith("botocore"):
            raise AssertionError(
                f"Optional AWS SDK dependency imported at import-time: {name}"
            )
        return real_import_module(name, package=package)

    monkeypatch.setattr(importlib, "import_module", spy_import_module)

    sandbox = real_import_module("nemo_evaluator.sandbox")

    # Ensure the module exposes the expected public surface (smoke check).
    assert hasattr(sandbox, "NemoEvaluatorSandbox")
    assert hasattr(sandbox, "EcsFargateSandbox")


def test_ecs_require_aws_sdks_raises_helpful_error(monkeypatch: pytest.MonkeyPatch):
    """
    If AWS SDK deps are missing, `_require_aws_sdks()` should raise a clear RuntimeError.
    """

    ecs_fargate = importlib.import_module("nemo_evaluator.sandbox.ecs_fargate")
    real_import_module = importlib.import_module

    def fail_import(name: str, package: str | None = None) -> ModuleType:
        if name in {"boto3", "botocore.config", "botocore.exceptions"}:
            raise ModuleNotFoundError(name)
        return real_import_module(name, package=package)

    monkeypatch.setattr(importlib, "import_module", fail_import)

    with pytest.raises(RuntimeError) as excinfo:
        ecs_fargate._require_aws_sdks()

    msg = str(excinfo.value)
    assert "requires AWS SDK dependencies" in msg
    assert "boto3" in msg


def test_ecs_sandbox_checks_for_aws_cli(monkeypatch: pytest.MonkeyPatch):
    """
    Instantiating the ECS sandbox should fail fast if required host tools are missing.
    """

    ecs_fargate = importlib.import_module("nemo_evaluator.sandbox.ecs_fargate")

    monkeypatch.setattr(ecs_fargate, "_which", lambda _name: None)

    cfg = ecs_fargate.EcsFargateConfig(
        region=None,
        cluster="dummy",
        task_definition="dummy",
        container_name="dummy",
        subnets=["subnet-123"],
        security_groups=["sg-123"],
    )

    with pytest.raises(ecs_fargate.AwsCliMissingError):
        ecs_fargate.EcsFargateSandbox(
            cfg=cfg,
            task_arn="dummy",
            run_id="dummy",
            task_id="dummy",
            trial_name="dummy",
        )


def test_parse_exec_markers_extracts_payload_and_strips_noise(monkeypatch: pytest.MonkeyPatch):
    ecs_fargate = importlib.import_module("nemo_evaluator.sandbox.ecs_fargate")

    # Avoid host-tool checks in constructor.
    monkeypatch.setattr(ecs_fargate, "_which", lambda _name: "/usr/bin/true")

    s = ecs_fargate.EcsFargateSandbox(
        cfg=ecs_fargate.EcsFargateConfig(
            region=None,
            cluster="c",
            task_definition="td",
            container_name="cn",
            subnets=["s"],
            security_groups=["sg"],
        ),
        task_arn="t",
        run_id="r",
        task_id="id",
        trial_name="trial",
    )

    cp = CompletedProcess(
        args=["aws", "ecs", "execute-command"],
        returncode=0,
        stdout=(
            "Starting session with SessionId: abc\n"
            "__TB_BEGIN__\n"
            "hello\n"
            "__TB_RC__=0\n"
            "Exiting session with sessionId: abc\n"
        ),
        stderr="The Session Manager plugin was installed successfully. Use the AWS CLI to start a session.\n",
    )

    assert s._parse_exec_markers(cp=cp, check=True) == "hello"


def test_parse_exec_markers_raises_when_rc_nonzero(monkeypatch: pytest.MonkeyPatch):
    ecs_fargate = importlib.import_module("nemo_evaluator.sandbox.ecs_fargate")
    monkeypatch.setattr(ecs_fargate, "_which", lambda _name: "/usr/bin/true")

    s = ecs_fargate.EcsFargateSandbox(
        cfg=ecs_fargate.EcsFargateConfig(
            region=None,
            cluster="c",
            task_definition="td",
            container_name="cn",
            subnets=["s"],
            security_groups=["sg"],
        ),
        task_arn="t",
        run_id="r",
        task_id="id",
        trial_name="trial",
    )

    cp = CompletedProcess(
        args=["aws", "ecs", "execute-command"],
        returncode=0,
        stdout="__TB_BEGIN__\nboom\n__TB_RC__=7\n",
        stderr="",
    )

    with pytest.raises(ecs_fargate.EcsExecError) as excinfo:
        s._parse_exec_markers(cp=cp, check=True)

    assert "rc=7" in str(excinfo.value)


def test_exec_capture_uses_s3_fallback_when_aws_reports_command_too_long(
    monkeypatch: pytest.MonkeyPatch,
):
    ecs_fargate = importlib.import_module("nemo_evaluator.sandbox.ecs_fargate")
    monkeypatch.setattr(ecs_fargate, "_which", lambda _name: "/usr/bin/true")

    s = ecs_fargate.EcsFargateSandbox(
        cfg=ecs_fargate.EcsFargateConfig(
            region=None,
            cluster="c",
            task_definition="td",
            container_name="cn",
            subnets=["s"],
            security_groups=["sg"],
            s3_bucket="bucket",
        ),
        task_arn="t",
        run_id="r",
        task_id="id",
        trial_name="trial",
    )

    called = {"s3": 0}

    def fake_s3(*, shell: str, timeout_sec: float, check: bool) -> str:
        assert "echo" in shell
        called["s3"] += 1
        return "via_s3"

    def fake_aws(*, command: str, timeout_sec: float):
        return CompletedProcess(args=["aws"], returncode=0, stdout=b"", stderr=b"COMMAND TOO LONG")

    monkeypatch.setattr(s, "_exec_capture_via_s3_script", fake_s3)
    monkeypatch.setattr(s, "_aws_ecs_execute_with_retry", fake_aws)

    assert s._exec_capture(cmd=["bash", "-lc", "echo hi"], timeout_sec=1.0) == "via_s3"
    assert called["s3"] == 1


def test_tmux_session_send_keys_large_payload_uses_paste_buffer(
    monkeypatch: pytest.MonkeyPatch,
):
    ecs_fargate = importlib.import_module("nemo_evaluator.sandbox.ecs_fargate")
    monkeypatch.setattr(ecs_fargate, "_which", lambda _name: "/usr/bin/true")

    s = ecs_fargate.EcsFargateSandbox(
        cfg=ecs_fargate.EcsFargateConfig(
            region=None,
            cluster="c",
            task_definition="td",
            container_name="cn",
            subnets=["s"],
            security_groups=["sg"],
        ),
        task_arn="t",
        run_id="r",
        task_id="id",
        trial_name="trial",
    )

    exec_calls: list[list[str]] = []
    paste_calls: list[str] = []

    def fake_exec_capture(*, cmd: list[str], timeout_sec: float, check: bool = True) -> str:
        _ = timeout_sec, check
        exec_calls.append(cmd)
        return ""

    def fake_paste(*, session_name: str, text: str, timeout_sec: float) -> None:
        _ = session_name, timeout_sec
        paste_calls.append(text)

    monkeypatch.setattr(s, "_exec_capture", fake_exec_capture)
    monkeypatch.setattr(s, "_tmux_paste_large_text", fake_paste)

    sess = ecs_fargate.EcsFargateTmuxSession(session_name="main", sandbox=s)
    big = "x" * (ecs_fargate.EcsFargateTmuxSession._LONG_TEXT_THRESHOLD + 10)

    sess.send_keys([big, "Enter"], block=False)

    assert paste_calls == [big]
    assert exec_calls, "expected tmux send-keys call"
    # Ensure we did NOT send the huge payload via `tmux send-keys`.
    assert all(big not in " ".join(c) for c in exec_calls)


def test_tmux_session_blocking_send_keys_waits_for_done(monkeypatch: pytest.MonkeyPatch):
    ecs_fargate = importlib.import_module("nemo_evaluator.sandbox.ecs_fargate")
    monkeypatch.setattr(ecs_fargate, "_which", lambda _name: "/usr/bin/true")

    s = ecs_fargate.EcsFargateSandbox(
        cfg=ecs_fargate.EcsFargateConfig(
            region=None,
            cluster="c",
            task_definition="td",
            container_name="cn",
            subnets=["s"],
            security_groups=["sg"],
        ),
        task_arn="t",
        run_id="r",
        task_id="id",
        trial_name="trial",
    )

    exec_calls: list[list[str]] = []

    def fake_exec_capture(*, cmd: list[str], timeout_sec: float, check: bool = True) -> str:
        _ = timeout_sec, check
        exec_calls.append(cmd)
        return ""

    monkeypatch.setattr(s, "_exec_capture", fake_exec_capture)

    sess = ecs_fargate.EcsFargateTmuxSession(session_name="main", sandbox=s)
    sess.send_keys(["echo hi", "Enter"], block=True, max_timeout_sec=5.0)

    assert exec_calls[0][:4] == ["tmux", "send-keys", "-t", "main"]
    assert ecs_fargate.EcsFargateTmuxSession._TMUX_COMPLETION_COMMAND in exec_calls[0]
    assert exec_calls[1][:3] == ["timeout", "5.0s", "tmux"]
    assert exec_calls[1][-2:] == ["wait", "done"]
