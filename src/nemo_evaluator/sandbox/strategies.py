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
"""Sandbox lifecycle strategies for the eval loop.

Three strategies control how sandboxes are managed during a single eval step:

- ``NoSandbox``: No container at all (MMLU, GSM8K, etc.).
- ``StatefulSandbox``: One sandbox shared by solve and verify — verification
  sees all agent side-effects directly (HumanEval, current Harbor behavior).
- ``StatelessSandbox``: Agent and verification run in separate sandboxes.
  State is transferred via a pluggable ``WorkspaceTransfer`` strategy
  (host volumes, EFS, direct copy, or exec-based tar/untar).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from nemo_evaluator.sandbox.base import OutsideEndpoint, Sandbox, SandboxSpec
    from nemo_evaluator.sandbox.manager import SandboxManager
    from nemo_evaluator.sandbox.transfer import WorkspaceTransfer

logger = logging.getLogger(__name__)


# -- Context passed to lifecycle constructors --------------------------------


@dataclass
class LifecycleContext:
    sandbox_mgr: SandboxManager
    agent_spec: SandboxSpec | None
    verify_spec: SandboxSpec
    capture_cmd: str | None
    apply_cmd: str | None
    verify_timeout: float = 600.0
    outside_endpoints: list[OutsideEndpoint] = field(default_factory=list)


# -- Protocol ----------------------------------------------------------------


class SandboxLifecycle(Protocol):
    """Manages sandbox(es) for a single eval step."""

    async def setup(self) -> None: ...

    async def get_agent_sandbox(self) -> Sandbox | None: ...

    async def transition_to_verify(
        self,
        response_text: str,
        solver_modified: bool,
    ) -> None: ...

    async def get_verify_sandbox(self) -> Sandbox | None: ...

    async def teardown(self) -> None: ...


# -- Strategies --------------------------------------------------------------


class NoSandbox:
    """No container needed (text-only benchmarks)."""

    async def setup(self) -> None:
        pass

    async def get_agent_sandbox(self) -> None:
        return None

    async def transition_to_verify(
        self,
        response_text: str,
        solver_modified: bool,
    ) -> None:
        pass

    async def get_verify_sandbox(self) -> None:
        return None

    async def teardown(self) -> None:
        pass


class StatefulSandbox:
    """One sandbox for both solve and verify.

    Verification sees all agent side-effects directly.  The sandbox is
    started lazily on the first call to either ``get_agent_sandbox`` or
    ``get_verify_sandbox``, so external solvers that skip the agent phase
    still get a sandbox for verification (e.g. HumanEval + ChatSolver).
    """

    def __init__(
        self,
        sandbox_mgr: SandboxManager,
        spec: SandboxSpec,
        outside_endpoints: list[OutsideEndpoint] | None = None,
    ) -> None:
        self._mgr = sandbox_mgr
        self._spec = spec
        self._outside_eps = outside_endpoints or []
        self._sandbox: Sandbox | None = None

    async def setup(self) -> None:
        pass

    async def _ensure_sandbox(self) -> Sandbox:
        if self._sandbox is None:
            self._sandbox = await self._mgr.acquire(
                self._spec,
                outside_endpoints=self._outside_eps,
            )
        return self._sandbox

    async def get_agent_sandbox(self) -> Sandbox:
        return await self._ensure_sandbox()

    async def transition_to_verify(
        self,
        response_text: str,
        solver_modified: bool,
    ) -> None:
        pass

    async def get_verify_sandbox(self) -> Sandbox:
        return await self._ensure_sandbox()

    async def teardown(self) -> None:
        if self._sandbox is not None:
            try:
                await self._mgr.release(self._sandbox)
            except Exception:
                logger.debug("StatefulSandbox: release failed", exc_info=True)
            self._sandbox = None


class StatelessSandbox:
    """Agent and verification run in separate sandboxes.

    Workspace state is transferred via a pluggable ``WorkspaceTransfer``
    strategy:

    1. Agent phase — ``get_agent_sandbox()`` starts an agent container
       with the transfer's volume config applied.
    2. Transition — ``transition_to_verify()`` runs the benchmark's
       ``capture_cmd`` inside the agent container, calls
       ``transfer.post_capture()``, then releases the agent container.
    3. Verify phase — ``get_verify_sandbox()`` starts a *fresh* container,
       calls ``transfer.pre_restore()``, runs ``apply_cmd``, and returns
       the sandbox to the scorer.
    """

    def __init__(self, ctx: LifecycleContext, transfer: WorkspaceTransfer) -> None:
        self._ctx = ctx
        self._transfer = transfer
        self._agent_sandbox: Sandbox | None = None
        self._verify_sandbox: Sandbox | None = None
        self._agent_exec_env: dict[str, str] = {}
        self._lock = asyncio.Lock()
        self._torn_down = False

    async def setup(self) -> None:
        pass

    async def get_agent_sandbox(self) -> Sandbox:
        if self._agent_sandbox is not None:
            return self._agent_sandbox
        if self._ctx.agent_spec is None:
            raise RuntimeError("StatelessSandbox: no agent_spec but get_agent_sandbox called")
        spec = self._transfer.prepare_agent_spec(self._ctx.agent_spec)
        self._agent_exec_env = dict(spec.env)
        self._agent_sandbox = await self._ctx.sandbox_mgr.acquire(
            spec,
            outside_endpoints=self._ctx.outside_endpoints,
        )
        wd = self._ctx.agent_spec.workdir
        await self._agent_sandbox.exec(
            f"_h=$(cd {wd} 2>/dev/null && git rev-parse HEAD 2>/dev/null) && echo $_h > /tmp/_nel_base_commit || true",
            timeout_sec=10,
        )
        return self._agent_sandbox

    async def transition_to_verify(
        self,
        response_text: str,
        solver_modified: bool,
    ) -> None:
        async with self._lock:
            if self._torn_down:
                return
            if self._agent_sandbox and self._ctx.capture_cmd and solver_modified:
                try:
                    await self._agent_sandbox.exec(
                        "mkdir -p /output /input",
                        timeout_sec=10,
                    )
                    try:
                        _wd = self._ctx.agent_spec.workdir
                        pre_diag = await self._agent_sandbox.exec(
                            "echo '=== base commit ===' && cat /tmp/_nel_base_commit 2>/dev/null || echo 'NOT SET'; "
                            f"echo '=== current HEAD ===' && cd {_wd} 2>/dev/null && git log --oneline -1 2>/dev/null; "
                            "echo '=== git status ===' && git status --short 2>/dev/null | head -30; "
                            "echo '=== cwd ===' && pwd",
                            timeout_sec=15,
                        )
                        logger.info(
                            "StatelessSandbox: pre-capture agent diagnostics:\n%s",
                            (pre_diag.stdout or "")[:2000],
                        )
                    except Exception:
                        logger.debug("StatelessSandbox: pre-capture diagnostics failed", exc_info=True)
                    cap_result = await self._agent_sandbox.exec(
                        self._ctx.capture_cmd,
                        timeout_sec=300,
                        env=self._agent_exec_env or None,
                    )
                    if cap_result.return_code != 0:
                        logger.error(
                            "StatelessSandbox: capture_cmd FAILED (rc=%d):\nSTDERR: %s\nSTDOUT: %s",
                            cap_result.return_code,
                            (cap_result.stderr or "")[:1000],
                            (cap_result.stdout or "")[:500],
                        )
                    else:
                        verify_result = await self._agent_sandbox.exec(
                            "ls -lh /output/workspace.tar 2>&1 || echo 'MISSING'",
                            timeout_sec=10,
                        )
                        logger.info(
                            "StatelessSandbox: capture complete — %s",
                            (verify_result.stdout or "").strip()[:200],
                        )
                        await self._transfer.post_capture(self._agent_sandbox)
                except Exception:
                    logger.error(
                        "StatelessSandbox: capture_cmd / post_capture FAILED",
                        exc_info=True,
                    )
            elif solver_modified:
                logger.error(
                    "StatelessSandbox: solver_modified=True but no agent "
                    "sandbox — workspace will NOT be transferred to verify!",
                )
            if self._agent_sandbox is not None:
                try:
                    await self._ctx.sandbox_mgr.release(self._agent_sandbox)
                except Exception:
                    logger.debug("StatelessSandbox: agent release failed", exc_info=True)
                self._agent_sandbox = None

    async def get_verify_sandbox(self) -> Sandbox:
        spec = self._transfer.prepare_verify_spec(self._ctx.verify_spec)
        self._verify_sandbox = await self._ctx.sandbox_mgr.acquire(spec)
        await self._verify_sandbox.exec("mkdir -p /output /input", timeout_sec=10)
        await self._transfer.pre_restore(self._verify_sandbox)
        check = await self._verify_sandbox.exec(
            "ls -lh /input/workspace.tar 2>&1",
            timeout_sec=10,
        )
        ws_present = check.return_code == 0
        logger.info(
            "StatelessSandbox: verify workspace check — %s — %s",
            "PRESENT" if ws_present else "MISSING",
            (check.stdout or "").strip()[:200],
        )
        if not ws_present:
            logger.error(
                "StatelessSandbox: /input/workspace.tar MISSING in verify "
                "container — agent changes will NOT be applied! "
                "Verify results will be against unmodified base image.",
            )
        if self._ctx.apply_cmd:
            try:
                _vwd = self._ctx.verify_spec.workdir
                diag = await self._verify_sandbox.exec(
                    f"echo '=== git HEAD ===' && cd {_vwd} && git log --oneline -1 2>/dev/null; "
                    "echo '=== patch stat ===' && "
                    "git apply --stat /tmp/_nel_ws/_nel_patch.diff 2>&1 | head -30; "
                    "echo '=== patch header ===' && head -40 /tmp/_nel_ws/_nel_patch.diff 2>/dev/null; "
                    "echo '=== patch files vs working tree ===' && "
                    r"grep '^diff --git' /tmp/_nel_ws/_nel_patch.diff 2>/dev/null | "
                    r"sed 's|diff --git a/\(.*\) b/.*|\1|' | while read f; do "
                    '[ -f "$f" ] && echo "  OK  $f" || echo "  MISS $f"; done || true',
                    timeout_sec=30,
                )
                if diag.return_code == 0 or diag.stdout:
                    logger.info(
                        "StatelessSandbox: pre-apply diagnostics:\n%s",
                        (diag.stdout or "")[:3000],
                    )
            except Exception:
                logger.debug("StatelessSandbox: pre-apply diagnostics failed", exc_info=True)

            result = await self._verify_sandbox.exec(
                self._ctx.apply_cmd,
                timeout_sec=self._ctx.verify_timeout,
            )
            if result.return_code != 0:
                logger.error(
                    "StatelessSandbox: apply_cmd FAILED (rc=%d):\nSTDERR: %s\nSTDOUT: %s",
                    result.return_code,
                    (result.stderr or "")[:2000],
                    (result.stdout or "")[:1000],
                )
        return self._verify_sandbox

    async def teardown(self) -> None:
        async with self._lock:
            if self._torn_down:
                return
            self._torn_down = True
        for sb in (self._agent_sandbox, self._verify_sandbox):
            if sb is not None:
                try:
                    await self._ctx.sandbox_mgr.release(sb)
                except Exception:
                    pass
        self._agent_sandbox = None
        self._verify_sandbox = None
        await self._transfer.cleanup()


# -- Factory -----------------------------------------------------------------


def pick_lifecycle(
    seed: Any,
    sandbox_mgr: SandboxManager | None,
    *,
    outside_endpoints: list[OutsideEndpoint] | None = None,
    config_capture_cmd: str | None = None,
    verify_timeout: float = 600.0,
) -> SandboxLifecycle:
    """Select the appropriate lifecycle strategy for *seed*.

    - ``verify_sandbox_spec`` set → ``StatelessSandbox``
    - sandbox_mgr can resolve a spec → ``StatefulSandbox``
    - otherwise → ``NoSandbox``
    """
    eps = outside_endpoints or []

    if seed.verify_sandbox_spec is not None and sandbox_mgr is not None:
        agent_spec = sandbox_mgr.resolve_spec(seed) or seed.sandbox_spec
        verify_spec = (
            sandbox_mgr.resolve_spec(
                seed,
                base_override=seed.verify_sandbox_spec,
            )
            or seed.verify_sandbox_spec
        )

        transfer = sandbox_mgr.get_transfer_strategy()

        return StatelessSandbox(
            LifecycleContext(
                sandbox_mgr=sandbox_mgr,
                agent_spec=agent_spec,
                verify_spec=verify_spec,
                capture_cmd=config_capture_cmd or seed.capture_cmd,
                apply_cmd=seed.apply_cmd,
                verify_timeout=verify_timeout,
                outside_endpoints=eps,
            ),
            transfer=transfer,
        )

    if sandbox_mgr is not None:
        spec = sandbox_mgr.resolve_spec(seed)
        if spec is not None:
            return StatefulSandbox(sandbox_mgr, spec, eps)

    return NoSandbox()
