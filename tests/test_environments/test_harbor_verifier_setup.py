# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for HarborEnvironment task.toml [verifier] setup_script plumbing."""

from __future__ import annotations

from pathlib import Path

import pytest

from nemo_evaluator.environments.harbor import HarborEnvironment
from tests.conftest import MockSandbox


def _make_task_dir(root: Path, name: str, task_toml_body: str) -> Path:
    task_dir = root / name
    (task_dir / "tests").mkdir(parents=True)
    (task_dir / "instruction.md").write_text("test instruction")
    (task_dir / "task.toml").write_text(task_toml_body)
    (task_dir / "tests" / "test.sh").write_text("#!/bin/bash\nexit 0\n")
    return task_dir


_TOML_WITH_SETUP = "[verifier]\ntimeout_sec = 600\nsetup_script = '/opt/local-httpbin/pre-verify.sh'\n"
_TOML_WITHOUT_SETUP = "[verifier]\ntimeout_sec = 600\n"


@pytest.mark.asyncio
async def test_seed_extracts_verifier_setup_script(tmp_path):
    dataset = tmp_path / "ds"
    _make_task_dir(dataset, "t1", _TOML_WITH_SETUP)
    env = HarborEnvironment(dataset_path=str(dataset))
    seed = await env.seed(0)
    assert seed.metadata["verifier_setup_script"] == "/opt/local-httpbin/pre-verify.sh"


@pytest.mark.asyncio
async def test_seed_omits_setup_script_when_absent(tmp_path):
    dataset = tmp_path / "ds"
    _make_task_dir(dataset, "t1", _TOML_WITHOUT_SETUP)
    env = HarborEnvironment(dataset_path=str(dataset))
    seed = await env.seed(0)
    assert "verifier_setup_script" not in seed.metadata


@pytest.mark.asyncio
async def test_verify_prepends_setup_script(tmp_path):
    dataset = tmp_path / "ds"
    _make_task_dir(dataset, "t1", _TOML_WITH_SETUP)
    env = HarborEnvironment(dataset_path=str(dataset))
    seed = await env.seed(0)

    sandbox = MockSandbox()
    await sandbox.start()
    await env.verify("response", "", sandbox=sandbox, **seed.metadata)

    test_cmds = [c for c in sandbox._exec_log if "bash /tests/test.sh" in c]
    assert len(test_cmds) == 1
    assert "bash /opt/local-httpbin/pre-verify.sh && bash /tests/test.sh" in test_cmds[0]


@pytest.mark.asyncio
async def test_verify_runs_test_sh_directly_without_setup_script(tmp_path):
    dataset = tmp_path / "ds"
    _make_task_dir(dataset, "t1", _TOML_WITHOUT_SETUP)
    env = HarborEnvironment(dataset_path=str(dataset))
    seed = await env.seed(0)

    sandbox = MockSandbox()
    await sandbox.start()
    await env.verify("response", "", sandbox=sandbox, **seed.metadata)

    test_cmds = [c for c in sandbox._exec_log if "bash /tests/test.sh" in c]
    assert len(test_cmds) == 1
    assert "pre-verify" not in test_cmds[0]
    assert "&& bash /tests/test.sh" in test_cmds[0]
