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
