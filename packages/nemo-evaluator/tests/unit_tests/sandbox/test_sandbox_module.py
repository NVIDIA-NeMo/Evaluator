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


def test_ecs_fargate_config_from_dict_coerces_strings():
    """from_dict must accept a bare string for list fields (subnets, security_groups)."""
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
    assert cfg.subnets == ["subnet-a"]
    assert cfg.security_groups == ["sg-b"]
    assert cfg.cpu == "2048"


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


def test_exec_client_health_returns_false_when_no_server():
    from nemo_evaluator.sandbox.ecs_fargate import ExecClient

    client = ExecClient(port=1, connect_timeout=0.5)
    assert client.health() is False


def test_retryable_message_matching_is_case_insensitive():
    from nemo_evaluator.sandbox.ecs_fargate import _is_retryable_error

    class FakeError(Exception):
        pass

    assert _is_retryable_error(FakeError("Capacity is unavailable right now"))
    assert _is_retryable_error(FakeError("CAPACITY IS UNAVAILABLE RIGHT NOW"))
    assert _is_retryable_error(FakeError("rate exceeded"))
    assert not _is_retryable_error(FakeError("completely unrelated error"))
