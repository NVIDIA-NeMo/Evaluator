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
from pathlib import Path
from types import ModuleType

import pytest


def _purge_modules(prefix: str) -> None:
    for name in list(sys.modules.keys()):
        if name == prefix or name.startswith(prefix + "."):
            sys.modules.pop(name, None)


# ── Import-time safety ──────────────────────────────────────────────


def test_sandbox_import_is_lightweight(monkeypatch: pytest.MonkeyPatch):
    """Importing ``nemo_evaluator.sandbox`` must not eagerly pull in boto3."""
    _purge_modules("nemo_evaluator.sandbox")

    real_import_module = importlib.import_module

    def spy_import_module(name: str, package: str | None = None) -> ModuleType:
        if name == "boto3" or name.startswith("botocore"):
            raise AssertionError(
                f"Optional AWS SDK dependency imported at import-time: {name}"
            )
        return real_import_module(name, package=package)

    monkeypatch.setattr(importlib, "import_module", spy_import_module)

    sandbox = real_import_module("nemo_evaluator.sandbox")

    assert hasattr(sandbox, "Sandbox")
    assert hasattr(sandbox, "ExecResult")
    assert hasattr(sandbox, "EcsFargateSandbox")


def test_ecs_require_aws_sdks_raises_helpful_error(monkeypatch: pytest.MonkeyPatch):
    """``_require_aws_sdks()`` should raise a clear RuntimeError when boto3 is absent."""
    ecs_fargate = importlib.import_module("nemo_evaluator.sandbox.ecs_fargate")
    real_import = (
        __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__
    )

    def fail_import(name, *args, **kwargs):
        if name in {"boto3", "botocore.config", "botocore.exceptions"}:
            raise ModuleNotFoundError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fail_import)

    with pytest.raises(RuntimeError) as excinfo:
        ecs_fargate._require_aws_sdks()

    msg = str(excinfo.value)
    assert "boto3" in msg.lower()


# ── Dataclass / Protocol smoke tests ────────────────────────────────


def test_exec_result_dataclass():
    from nemo_evaluator.sandbox import ExecResult

    r = ExecResult(stdout="ok", stderr="", return_code=0)
    assert r.stdout == "ok"
    assert r.return_code == 0


def test_sandbox_protocol_surface():
    from nemo_evaluator.sandbox.base import Sandbox

    required = {
        "start",
        "stop",
        "exec",
        "upload",
        "download",
        "is_running",
        "__enter__",
        "__exit__",
    }
    assert required.issubset(dir(Sandbox))


# ── EcsFargateConfig construction ───────────────────────────────────


def test_ecs_fargate_config_basic():
    from nemo_evaluator.sandbox.ecs_fargate import EcsFargateConfig

    cfg = EcsFargateConfig(
        region="us-east-1",
        cluster="test-cluster",
        subnets=["subnet-1"],
        security_groups=["sg-1"],
    )
    assert cfg.cluster == "test-cluster"
    assert cfg.region == "us-east-1"
    assert cfg.cpu == "4096"
    assert cfg.memory == "8192"


def test_ecs_fargate_config_from_dict():
    from nemo_evaluator.sandbox.ecs_fargate import EcsFargateConfig

    cfg = EcsFargateConfig.from_dict(
        {
            "region": "eu-west-1",
            "cluster": "my-cluster",
            "subnets": "subnet-a",
            "security_groups": ["sg-b"],
            "cpu": "2048",
            "memory": "4096",
        }
    )
    assert cfg.cluster == "my-cluster"
    assert cfg.subnets == ["subnet-a"]
    assert cfg.cpu == "2048"


def test_ssh_sidecar_config_defaults():
    from nemo_evaluator.sandbox.ecs_fargate import SshSidecarConfig

    sc = SshSidecarConfig()
    assert sc.sshd_port == 2222
    assert sc.ssh_ready_timeout_sec == 120.0
    assert sc.exec_server_port is None


# ── ImageBuilder (no AWS calls) ─────────────────────────────────────


def test_get_ecr_image_tag_deterministic(tmp_path: Path):
    from nemo_evaluator.sandbox.ecs_fargate import ImageBuilder

    (tmp_path / "Dockerfile").write_text("FROM alpine\n")
    (tmp_path / "app.py").write_text("print('hello')\n")

    tag1 = ImageBuilder.get_ecr_image_tag(tmp_path, "myenv")
    tag2 = ImageBuilder.get_ecr_image_tag(tmp_path, "myenv")
    assert tag1 == tag2
    assert tag1.startswith("myenv__")
    assert len(tag1.split("__")[1]) == 8


def test_get_ecr_image_tag_changes_on_content(tmp_path: Path):
    from nemo_evaluator.sandbox.ecs_fargate import ImageBuilder

    (tmp_path / "Dockerfile").write_text("FROM alpine\n")
    tag_before = ImageBuilder.get_ecr_image_tag(tmp_path, "env")

    (tmp_path / "Dockerfile").write_text("FROM ubuntu\n")
    tag_after = ImageBuilder.get_ecr_image_tag(tmp_path, "env")

    assert tag_before != tag_after


# ── ExecClient (no live server) ─────────────────────────────────────


def test_exec_client_constructs():
    from nemo_evaluator.sandbox.ecs_fargate import ExecClient

    client = ExecClient(port=12345)
    assert client._base == "http://127.0.0.1:12345"


def test_exec_client_health_returns_false_when_no_server():
    from nemo_evaluator.sandbox.ecs_fargate import ExecClient

    client = ExecClient(port=1, connect_timeout=0.5)
    assert client.health() is False
