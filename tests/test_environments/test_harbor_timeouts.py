# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for HarborEnvironment timeout handling — task.toml [verifier] timeout_sec plumbing."""

from __future__ import annotations

from pathlib import Path

import pytest

from nemo_evaluator.environments.harbor import DEFAULT_HARBOR_VERIFIER_TIMEOUT, HarborEnvironment
from tests.conftest import MockExecResult, MockSandbox


class _CapturingSandbox(MockSandbox):
    """MockSandbox that records the timeout_sec used for ``bash /tests/test.sh`` invocations."""

    def __init__(self, exec_results: dict[str, MockExecResult] | None = None) -> None:
        super().__init__(exec_results=exec_results)
        self.captured_timeouts: list[float] = []

    async def exec(
        self,
        command: str,
        timeout_sec: float = 180,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        user: str | int | None = None,
    ) -> MockExecResult:
        if "bash /tests/test.sh" in command:
            self.captured_timeouts.append(timeout_sec)
        return await super().exec(command, timeout_sec, cwd=cwd, env=env, user=user)


def _make_task_dir(root: Path, name: str, task_toml_body: str) -> Path:
    task_dir = root / name
    (task_dir / "tests").mkdir(parents=True)
    (task_dir / "instruction.md").write_text("test instruction")
    (task_dir / "task.toml").write_text(task_toml_body)
    (task_dir / "tests" / "test.sh").write_text("#!/bin/bash\nexit 0\n")
    return task_dir


@pytest.mark.asyncio
async def test_seed_extracts_verifier_timeout_from_task_toml(tmp_path):
    dataset = tmp_path / "ds"
    _make_task_dir(
        dataset,
        "t1",
        "[agent]\ntimeout_sec = 900\n[verifier]\ntimeout_sec = 1800\n",
    )
    env = HarborEnvironment(dataset_path=str(dataset))
    seed = await env.seed(0)
    assert seed.metadata["agent_timeout_sec"] == 900
    assert seed.metadata["verifier_timeout_sec"] == 1800


@pytest.mark.asyncio
async def test_seed_falls_back_to_default_when_task_omits_verifier_timeout(tmp_path):
    dataset = tmp_path / "ds"
    _make_task_dir(dataset, "t1", "[agent]\ntimeout_sec = 900\n")
    env = HarborEnvironment(dataset_path=str(dataset))
    seed = await env.seed(0)
    assert seed.metadata["verifier_timeout_sec"] == DEFAULT_HARBOR_VERIFIER_TIMEOUT


@pytest.mark.asyncio
async def test_verify_defaults_to_600_when_metadata_missing(tmp_path):
    dataset = tmp_path / "ds"
    task_dir = _make_task_dir(dataset, "t1", "")
    env = HarborEnvironment(dataset_path=str(dataset))

    sb = _CapturingSandbox()
    await sb.start()
    await env.verify(response="", expected="", sandbox=sb, task_dir=str(task_dir))
    assert sb.captured_timeouts == [float(DEFAULT_HARBOR_VERIFIER_TIMEOUT)]


@pytest.mark.asyncio
async def test_verify_uses_1800_when_task_declares_it(tmp_path):
    dataset = tmp_path / "ds"
    _make_task_dir(
        dataset,
        "t1",
        "[agent]\ntimeout_sec = 900\n[verifier]\ntimeout_sec = 1800\n",
    )
    env = HarborEnvironment(dataset_path=str(dataset))

    sb = _CapturingSandbox()
    await sb.start()
    seed = await env.seed(0)
    await env.verify(response="", expected="", sandbox=sb, **seed.metadata)
    assert sb.captured_timeouts == [1800.0]
