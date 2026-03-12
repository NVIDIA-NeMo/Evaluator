"""ChatSolver: single model call. Default for standard benchmarks."""
from __future__ import annotations

from typing import Any

from nemo_evaluator.environments.base import SeedResult

from .base import SolveResult
from .trajectory_util import _single_turn_trajectory


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
        trajectory = _single_turn_trajectory(task.prompt, resp.content, effective_system)
        return SolveResult(response=resp.content, model_response=resp, trajectory=trajectory)

    async def close(self) -> None:
        await self._client.close()
