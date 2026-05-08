# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

from nemo_evaluator.environments.base import SeedResult
from nemo_evaluator.solvers.base import ErrorKind
from nemo_evaluator.solvers.oracle import OracleSolver
from tests.conftest import MockExecResult, MockSandbox


def _seed(task_dir: str, **metadata) -> SeedResult:
    md = {"source": "harbor", "task_id": "test-task", "task_dir": task_dir}
    md.update(metadata)
    return SeedResult(prompt="", expected_answer="", metadata=md)


@pytest.mark.asyncio
async def test_missing_sandbox_returns_infra_error(tmp_path):
    (tmp_path / "solution").mkdir()
    (tmp_path / "solution" / "solve.sh").write_text("#!/bin/bash\necho ok\n")
    solver = OracleSolver(timeout=30)
    result = await solver.solve(_seed(str(tmp_path)))
    assert result.error_kind == ErrorKind.INFRA
    assert "sandbox" in (result.error or "").lower()


@pytest.mark.asyncio
async def test_missing_task_dir_returns_system_error(mock_sandbox):
    solver = OracleSolver(timeout=30)
    seed = SeedResult(prompt="", expected_answer="", metadata={"source": "harbor", "task_id": "x"})
    result = await solver.solve(seed, sandbox=mock_sandbox)
    assert result.error_kind == ErrorKind.SYSTEM


@pytest.mark.asyncio
async def test_missing_solve_sh_returns_system_error(tmp_path, mock_sandbox):
    # task_dir exists but solution/solve.sh does not
    solver = OracleSolver(timeout=30)
    result = await solver.solve(_seed(str(tmp_path)), sandbox=mock_sandbox)
    assert result.error_kind == ErrorKind.SYSTEM
    assert "solve.sh" in (result.error or "")


@pytest.mark.asyncio
async def test_happy_path_uploads_and_execs(tmp_path, mock_sandbox):
    (tmp_path / "solution").mkdir()
    (tmp_path / "solution" / "solve.sh").write_text("#!/bin/bash\necho hello\n")

    solver = OracleSolver(timeout=30)
    result = await solver.solve(_seed(str(tmp_path)), sandbox=mock_sandbox)

    assert result.error is None
    assert result.error_kind == ErrorKind.NONE
    assert len(result.trajectory) == 1
    entry = result.trajectory[0]
    assert entry["type"] == "oracle"
    assert entry["return_code"] == 0
    # upload happened (solve.sh bytes landed at /solution/solve.sh in mock FS)
    assert mock_sandbox._files.get("/solution/solve.sh", b"").startswith(b"#!/bin/bash")
    # mkdir + chmod + bash all dispatched
    assert any("mkdir -p /solution" in c for c in mock_sandbox._exec_log)
    assert any("chmod +x /solution/solve.sh" in c for c in mock_sandbox._exec_log)
    assert any("bash /solution/solve.sh" in c for c in mock_sandbox._exec_log)


@pytest.mark.asyncio
async def test_uploads_sibling_files_to_solution_dir(tmp_path, mock_sandbox):
    """Multi-file solution/ dirs (e.g. headless-terminal) need every file at /solution/."""
    (tmp_path / "solution").mkdir()
    (tmp_path / "solution" / "solve.sh").write_text("#!/bin/bash\ncp /solution/helper.py /app/\n")
    (tmp_path / "solution" / "helper.py").write_text("print('hi')\n")

    solver = OracleSolver(timeout=30)
    result = await solver.solve(_seed(str(tmp_path)), sandbox=mock_sandbox)

    assert result.error is None
    # Both files ended up at /solution/ in the sandbox
    assert mock_sandbox._files.get("/solution/solve.sh", b"").startswith(b"#!/bin/bash")
    assert mock_sandbox._files.get("/solution/helper.py", b"").startswith(b"print")


@pytest.mark.asyncio
async def test_timeout_strategy_max_picks_larger(tmp_path, mock_sandbox):
    (tmp_path / "solution").mkdir()
    (tmp_path / "solution" / "solve.sh").write_text("#!/bin/bash\nexit 0\n")

    solver = OracleSolver(timeout=900, run_timeout=600, timeout_strategy="max")
    # task.toml says 7200s
    seed = _seed(str(tmp_path), agent_timeout_sec=7200.0)
    result = await solver.solve(seed, sandbox=mock_sandbox)
    # Just confirm no error; logger will have printed the 7200s resolution.
    assert result.error is None


@pytest.mark.asyncio
async def test_nonzero_exit_propagates_return_code(tmp_path, mock_sandbox_factory):
    (tmp_path / "solution").mkdir()
    (tmp_path / "solution" / "solve.sh").write_text("#!/bin/bash\nexit 42\n")

    sb = mock_sandbox_factory({"bash /solution/solve.sh": MockExecResult(return_code=42, stderr="boom")})
    solver = OracleSolver(timeout=30)
    result = await solver.solve(_seed(str(tmp_path)), sandbox=sb)

    assert result.error is None  # non-zero is not solver error; verify phase decides pass/fail
    assert result.trajectory[0]["return_code"] == 42
    assert result.trajectory[0]["stderr"] == "boom"


@pytest.fixture
def mock_sandbox_factory():
    def _make(exec_results):
        return MockSandbox(exec_results=exec_results)

    return _make
