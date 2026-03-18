"""HarborSolver: runs Harbor-compatible agents inside a nel Sandbox."""
from __future__ import annotations

import contextlib
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from nemo_evaluator.observability.types import ModelResponse
from nemo_evaluator.solvers.base import SolveResult

if TYPE_CHECKING:
    from nemo_evaluator.environments.base import SeedResult
    from nemo_evaluator.sandbox.base import Sandbox

logger = logging.getLogger(__name__)


def _extract_response(context: Any) -> str:
    """Extract text response from agent context (metadata > last rollout > empty)."""
    if context.metadata and isinstance(context.metadata.get("response"), str):
        return context.metadata["response"]
    if context.rollout_details:
        last = context.rollout_details[-1]
        content = last.get("content") if isinstance(last, dict) else getattr(last, "content", None)
        if isinstance(content, str):
            return content
    return ""


def _check_harbor_installed() -> None:
    try:
        import harbor  # noqa: F401
    except ImportError:
        raise ImportError(
            "Harbor agent integration requires the harbor package. "
            "Install with: pip install nemo-evaluator[harbor]"
        ) from None


@contextlib.contextmanager
def _inject_env(api_key: str | None, model_url: str | None, model_id: str | None):
    """Temporarily set LLM_* env vars that Harbor built-in agents expect."""
    overrides: dict[str, str] = {}
    if api_key and "LLM_API_KEY" not in os.environ:
        overrides["LLM_API_KEY"] = api_key
    if model_url and "LLM_BASE_URL" not in os.environ:
        overrides["LLM_BASE_URL"] = model_url
    if model_id and "LLM_MODEL" not in os.environ:
        overrides["LLM_MODEL"] = model_id

    prev = {k: os.environ.get(k) for k in overrides}
    os.environ.update(overrides)
    try:
        yield
    finally:
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class HarborSolver:
    """Runs any Harbor agent inside a nel Sandbox.

    Agent resolution (``harbor_agent`` parameter):
      - Built-in name (e.g. ``"openhands"``) -> ``AgentFactory.create_agent_from_name()``
      - Import path  (e.g. ``"my_pkg:MyAgent"``) -> ``AgentFactory.create_agent_from_import_path()``
      - Directory path (e.g. ``"./agents/my-agent/"``) -> ``ByobInstalledAgent``
    """

    def __init__(
        self,
        *,
        harbor_agent: str,
        harbor_agent_kwargs: dict[str, Any] | None = None,
        model_url: str = "",
        model_id: str = "",
        timeout: float = 1800.0,
        api_key: str | None = None,
    ) -> None:
        _check_harbor_installed()

        self._harbor_agent = harbor_agent
        self._harbor_agent_kwargs = harbor_agent_kwargs or {}
        self._model_url = model_url
        self._model_id = model_id
        self._timeout = timeout
        self._api_key = api_key

    def _create_agent(self, logs_dir: Path) -> Any:
        from harbor.agents.factory import AgentFactory

        kwargs = dict(self._harbor_agent_kwargs)
        if "model_name" not in kwargs and self._model_id:
            kwargs["model_name"] = self._model_id
        if "model_url" not in kwargs and self._model_url:
            kwargs["model_url"] = self._model_url
        if "api_key" not in kwargs and self._api_key:
            kwargs["api_key"] = self._api_key

        name = self._harbor_agent

        if ":" in name:
            return AgentFactory.create_agent_from_import_path(
                name, logs_dir=logs_dir, **kwargs,
            )

        agent_path = Path(name)
        if agent_path.is_dir():
            from nemo_evaluator.solvers.byob_agent import ByobInstalledAgent
            return ByobInstalledAgent(
                agent_dir=agent_path, logs_dir=logs_dir, **kwargs,
            )

        return AgentFactory.create_agent_from_name(
            name, logs_dir=logs_dir, **kwargs,
        )

    async def solve(
        self, task: SeedResult, sandbox: Sandbox | None = None,
    ) -> SolveResult:
        if sandbox is None:
            raise RuntimeError(
                "HarborSolver requires a sandbox. Make sure the benchmark "
                "provides a sandbox_spec and a SandboxManager is configured."
            )

        from harbor.models.agent.context import AgentContext

        from nemo_evaluator.solvers.harbor_adapter import SandboxEnvironmentAdapter

        t0 = time.monotonic()
        logs_dir = Path(tempfile.mkdtemp(prefix="nel_harbor_"))

        try:
            adapter = SandboxEnvironmentAdapter(sandbox, logs_dir=logs_dir)

            with _inject_env(self._api_key, self._model_url, self._model_id):
                agent = self._create_agent(logs_dir)

                await sandbox.exec(
                    "mkdir -p /logs/agent /logs/verifier /logs/artifacts",
                    timeout_sec=10,
                )

                await agent.setup(adapter)

                context = AgentContext()
                await agent.run(task.prompt, adapter, context)

            if not adapter.is_mounted:
                try:
                    await adapter.download_dir("/logs/agent", logs_dir / "agent")
                except Exception:
                    logger.debug("Failed to download agent logs", exc_info=True)

            if context.is_empty() and hasattr(agent, "populate_context_post_run"):
                try:
                    agent.populate_context_post_run(context)
                except Exception:
                    logger.debug("populate_context_post_run failed", exc_info=True)

            trajectory = []
            if context.rollout_details:
                trajectory = [d.model_dump() for d in context.rollout_details]
            elif context.metadata:
                trajectory = [context.metadata]

            response = _extract_response(context)

            latency_ms = (time.monotonic() - t0) * 1000
            return SolveResult(
                response=response,
                model_response=ModelResponse(
                    content=response,
                    model=self._model_id,
                    total_tokens=(context.n_input_tokens or 0) + (context.n_output_tokens or 0),
                    completion_tokens=context.n_output_tokens or 0,
                    latency_ms=round(latency_ms, 2),
                ),
                trajectory=trajectory,
            )

        except Exception:
            logger.exception("HarborSolver.solve() failed")
            latency_ms = (time.monotonic() - t0) * 1000
            return SolveResult(
                response="",
                model_response=ModelResponse(
                    content="", model=self._model_id,
                    total_tokens=0, completion_tokens=0,
                    latency_ms=round(latency_ms, 2),
                ),
                trajectory=[{"error": "harbor_solver_error"}],
            )

    async def close(self) -> None:
        pass
