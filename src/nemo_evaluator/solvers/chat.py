"""ChatSolver: single model call. Default for standard benchmarks."""
from __future__ import annotations

from typing import Any

from nemo_evaluator.environments.base import SeedResult

from .base import SolveResult
from .trajectory_util import build_single_turn_atif


class ChatSolver:
    """Single model call. Default for standard benchmarks."""

    def __init__(self, client: Any, system_prompt: str | None = None) -> None:
        self._client = client
        self._system = system_prompt

    async def solve(self, task: SeedResult) -> SolveResult:
        effective_system = self._system or task.system
        if task.messages:
            resp = await self._client.chat(messages=task.messages)
        else:
            resp = await self._client.chat(task.prompt, system=effective_system)
        trajectory = build_single_turn_atif(
            task.prompt, resp.content, system=effective_system,
            model_name=getattr(resp, "model", None),
            prompt_tokens=getattr(resp, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(resp, "completion_tokens", 0) or 0,
        )
        return SolveResult(response=resp.content, model_response=resp, trajectory=trajectory)

    async def close(self) -> None:
        await self._client.close()
