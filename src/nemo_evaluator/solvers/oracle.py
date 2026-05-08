# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Gold-trajectory replay solver for Harbor-style tasks.

Reads the task's gold trajectory (``solution/solve.sh``) and executes it inside
the sandbox that the Harbor environment has prepared.  Used to measure the
infrastructure ceiling — what fraction of tasks pass when the "agent" already
knows the answer — independent of LLM capability.

Mirrors upstream harbor ``OracleAgent``: uploads the full ``solution/``
directory to ``/solution/`` inside the sandbox and runs
``/solution/solve.sh``.  This supports tasks whose solve.sh references sibling
files (e.g. ``cp /solution/helper.py``).

No LLM service is needed; the config schema omits ``service``.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

from nemo_evaluator.environments.base import SeedResult
from nemo_evaluator.solvers.base import ErrorKind, SolveResult
from nemo_evaluator.solvers.harbor import _resolve_agent_timeout

if TYPE_CHECKING:
    from nemo_evaluator.sandbox.base import Sandbox


logger = logging.getLogger(__name__)


class OracleSolver:
    """Execute the task's own ``solution/solve.sh`` inside the sandbox.

    The task directory is read from ``SeedResult.metadata["task_dir"]``
    (populated by :class:`HarborEnvironment`).  Timeout resolution mirrors
    :class:`HarborSolver` so ``override``/``task``/``max`` strategies behave
    identically and ``max_agent_timeout`` caps runaway per-task budgets.
    """

    def __init__(
        self,
        *,
        timeout: float = 1800.0,
        run_timeout: float | None = None,
        timeout_strategy: str = "override",
        max_agent_timeout: float | None = None,
    ) -> None:
        self._timeout = timeout
        self._run_timeout = run_timeout if run_timeout is not None else timeout
        self._timeout_strategy = timeout_strategy
        self._max_agent_timeout = max_agent_timeout

    async def solve(self, task: SeedResult, sandbox: Sandbox | None = None) -> SolveResult:
        if sandbox is None:
            return SolveResult(
                response="",
                trajectory=[],
                reward=0.0,
                error="OracleSolver requires a sandbox",
                error_kind=ErrorKind.INFRA,
            )

        task_dir_str = task.metadata.get("task_dir")
        if not task_dir_str:
            return SolveResult(
                response="",
                trajectory=[],
                reward=0.0,
                error="task_dir missing from seed metadata (needed to locate solution/solve.sh)",
                error_kind=ErrorKind.SYSTEM,
            )

        task_id = task.metadata.get("task_id", "?")
        solution_dir = Path(task_dir_str) / "solution"
        solve_sh = solution_dir / "solve.sh"
        if not solve_sh.exists():
            return SolveResult(
                response="",
                trajectory=[{"error": f"no solve.sh at {solve_sh}"}],
                reward=0.0,
                error=f"solution/solve.sh missing for task {task_id}",
                error_kind=ErrorKind.SYSTEM,
            )

        task_timeout: float | None = None
        raw = task.metadata.get("agent_timeout_sec")
        if isinstance(raw, (int, float)):
            task_timeout = float(raw)
        effective = _resolve_agent_timeout(
            self._timeout_strategy,
            self._run_timeout,
            task_timeout,
            self._max_agent_timeout,
        )

        # Match upstream harbor OracleAgent: upload the full solution/ dir to
        # /solution in the sandbox so solve.sh can reference sibling files
        # (e.g. `cp /solution/helper.py /app/`).
        remote_dir = "/solution"
        remote_solve = f"{remote_dir}/solve.sh"
        await sandbox.exec(f"mkdir -p {remote_dir}", timeout_sec=10)
        for child in sorted(solution_dir.iterdir()):
            if child.is_file():
                await sandbox.upload(child, f"{remote_dir}/{child.name}")
        await sandbox.exec(f"chmod +x {remote_solve}", timeout_sec=10)

        logger.info(
            "OracleSolver: running %s (timeout=%.0fs, strategy=%s)",
            solve_sh,
            effective,
            self._timeout_strategy,
        )
        t0 = time.monotonic()
        try:
            result = await sandbox.exec(f"bash {remote_solve}", timeout_sec=effective)
        except Exception as e:
            elapsed = time.monotonic() - t0
            logger.warning(
                "OracleSolver: sandbox.exec raised %s after %.1fs on %s: %s",
                type(e).__name__,
                elapsed,
                task_id,
                e,
            )
            return SolveResult(
                response="",
                trajectory=[{"type": "oracle", "script_path": str(solve_sh), "elapsed_sec": elapsed, "error": str(e)}],
                reward=0.0,
                error=f"sandbox.exec raised {type(e).__name__}: {e}",
                error_kind=ErrorKind.INFRA,
            )
        elapsed = time.monotonic() - t0

        trajectory = [
            {
                "type": "oracle",
                "script_path": str(solve_sh),
                "return_code": result.return_code,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "elapsed_sec": elapsed,
            }
        ]

        if result.return_code != 0:
            logger.warning(
                "OracleSolver: solve.sh for %s exited %d after %.1fs",
                task_id,
                result.return_code,
                elapsed,
            )

        return SolveResult(
            response=result.stdout or "",
            trajectory=trajectory,
        )
