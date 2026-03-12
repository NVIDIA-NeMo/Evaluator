"""VLMSolver: vision-language model solver."""
from __future__ import annotations

from typing import Any

from nemo_evaluator.environments.base import SeedResult

from .base import SolveResult
from .trajectory_util import _single_turn_trajectory


class VLMSolver:
    """Vision-language model solver. Uses vlm_chat() when images are present,
    falls back to regular chat() for text-only tasks."""

    def __init__(self, client: Any, system_prompt: str | None = None,
                 image_detail: str = "auto") -> None:
        self._client = client
        self._system = system_prompt
        self._detail = image_detail

    async def solve(self, task: SeedResult) -> SolveResult:
        effective_system = self._system or task.system
        if task.images:
            resp = await self._client.vlm_chat(
                prompt=task.prompt, images=task.images,
                system=effective_system, detail=self._detail,
            )
        elif task.messages:
            resp = await self._client.chat(messages=task.messages)
        else:
            resp = await self._client.chat(task.prompt, system=effective_system)
        trajectory = _single_turn_trajectory(task.prompt, resp.content, effective_system)
        return SolveResult(response=resp.content, model_response=resp, trajectory=trajectory)

    async def close(self) -> None:
        await self._client.close()
