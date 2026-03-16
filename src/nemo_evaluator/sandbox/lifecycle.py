"""Sandbox lifecycle strategies for the eval loop.

Three strategies control how sandboxes are managed during a single eval step:

- ``NoSandbox``: No container at all (MMLU, GSM8K, etc.).
- ``StatefulSandbox``: One sandbox shared by solve and verify — verification
  sees all agent side-effects directly (HumanEval, current Harbor behavior).
- ``StatelessSandbox``: Agent and verification run in separate sandboxes.
  State is transferred via a shared host volume and a benchmark-defined
  ``capture_cmd`` / ``apply_cmd`` pair.  This is the two-container model
  that enables solver-agnostic verification.
"""
from __future__ import annotations

import logging
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from nemo_evaluator.sandbox.base import OutsideEndpoint, Sandbox, SandboxSpec, VolumeMount
    from nemo_evaluator.sandbox.manager import SandboxManager

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
        self, response_text: str, solver_modified: bool,
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
        self, response_text: str, solver_modified: bool,
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
                self._spec, outside_endpoints=self._outside_eps,
            )
        return self._sandbox

    async def get_agent_sandbox(self) -> Sandbox:
        return await self._ensure_sandbox()

    async def transition_to_verify(
        self, response_text: str, solver_modified: bool,
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

    State is transferred via a host-mounted shared volume:

    1. Agent phase — ``get_agent_sandbox()`` starts an agent container with
       ``/output`` mounted read-write.  Only called when the solver actually
       needs a sandbox (lazy).
    2. Transition — ``transition_to_verify()`` runs the benchmark's
       ``capture_cmd`` inside the agent container (for in-container solvers)
       or writes the solver's text response to the shared volume (for
       external solvers).  The agent container is then released.
    3. Verify phase — ``get_verify_sandbox()`` starts a *fresh* container
       with ``/input`` mounted read-only, runs ``apply_cmd`` to restore the
       agent's changes, and returns the sandbox to the scorer.
    """

    def __init__(self, ctx: LifecycleContext) -> None:
        self._ctx = ctx
        self._shared_dir: Path | None = None
        self._agent_sandbox: Sandbox | None = None
        self._verify_sandbox: Sandbox | None = None

    async def setup(self) -> None:
        d = tempfile.mkdtemp(prefix="nel_shared_")
        Path(d).chmod(0o700)
        self._shared_dir = Path(d)

    async def get_agent_sandbox(self) -> Sandbox:
        if self._agent_sandbox is not None:
            return self._agent_sandbox
        if self._ctx.agent_spec is None:
            raise RuntimeError(
                "StatelessSandbox: no agent_spec but get_agent_sandbox called"
            )
        spec = _attach_volume(
            self._ctx.agent_spec, self._shared_dir, "/output", readonly=False,
        )
        self._agent_sandbox = await self._ctx.sandbox_mgr.acquire(
            spec, outside_endpoints=self._ctx.outside_endpoints,
        )
        return self._agent_sandbox

    async def transition_to_verify(
        self, response_text: str, solver_modified: bool,
    ) -> None:
        if self._agent_sandbox and self._ctx.capture_cmd and solver_modified:
            try:
                await self._agent_sandbox.exec(
                    self._ctx.capture_cmd, timeout_sec=120,
                )
            except Exception:
                logger.warning(
                    "StatelessSandbox: capture_cmd failed, "
                    "falling back to response text",
                    exc_info=True,
                )
                self._write_response(response_text)
        else:
            self._write_response(response_text)

        if self._agent_sandbox is not None:
            try:
                await self._ctx.sandbox_mgr.release(self._agent_sandbox)
            except Exception:
                logger.debug("StatelessSandbox: agent release failed", exc_info=True)
            self._agent_sandbox = None

    async def get_verify_sandbox(self) -> Sandbox:
        spec = _attach_volume(
            self._ctx.verify_spec, self._shared_dir, "/input", readonly=True,
        )
        self._verify_sandbox = await self._ctx.sandbox_mgr.acquire(spec)
        if self._ctx.apply_cmd:
            result = await self._verify_sandbox.exec(
                self._ctx.apply_cmd,
                timeout_sec=self._ctx.verify_timeout,
            )
            if result.return_code != 0:
                logger.warning(
                    "StatelessSandbox: apply_cmd exited %d: %s",
                    result.return_code, result.stderr[:500],
                )
        return self._verify_sandbox

    async def teardown(self) -> None:
        for sb in (self._agent_sandbox, self._verify_sandbox):
            if sb is not None:
                try:
                    await self._ctx.sandbox_mgr.release(sb)
                except Exception:
                    pass
        self._agent_sandbox = None
        self._verify_sandbox = None

        if self._shared_dir is not None:
            shutil.rmtree(self._shared_dir, ignore_errors=True)
            self._shared_dir = None

    def _write_response(self, text: str) -> None:
        if self._shared_dir is not None:
            (self._shared_dir / "response.txt").write_text(text or "")


# -- Helpers -----------------------------------------------------------------

def _attach_volume(
    spec: SandboxSpec,
    host_dir: Path | None,
    container_path: str,
    *,
    readonly: bool,
) -> SandboxSpec:
    """Return a copy of *spec* with an extra volume mount appended."""
    from nemo_evaluator.sandbox.base import SandboxSpec as SS, VolumeMount

    if host_dir is None:
        return spec

    vol = VolumeMount(
        host_path=str(host_dir),
        container_path=container_path,
        readonly=readonly,
    )
    return SS(
        image=spec.image,
        workdir=spec.workdir,
        env=dict(spec.env),
        files=dict(spec.files),
        entrypoint=spec.entrypoint,
        volumes=list(spec.volumes) + [vol],
    )


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
        return StatelessSandbox(LifecycleContext(
            sandbox_mgr=sandbox_mgr,
            agent_spec=seed.sandbox_spec,
            verify_spec=seed.verify_sandbox_spec,
            capture_cmd=config_capture_cmd or seed.capture_cmd,
            apply_cmd=seed.apply_cmd,
            verify_timeout=verify_timeout,
            outside_endpoints=eps,
        ))

    if sandbox_mgr is not None:
        spec = sandbox_mgr.resolve_spec(seed)
        if spec is not None:
            return StatefulSandbox(sandbox_mgr, spec, eps)

    return NoSandbox()
