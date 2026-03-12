"""EmbeddingSolver: returns embeddings as JSON-encoded response."""
from __future__ import annotations

from typing import Any

from nemo_evaluator.environments.base import SeedResult

from .base import SolveResult


class EmbeddingSolver:
    """Solver for embedding tasks. Returns embeddings as JSON-encoded response."""

    def __init__(self, client: Any) -> None:
        self._client = client

    async def solve(self, task: SeedResult) -> SolveResult:
        import json
        embedding = await self._client.embed(task.prompt)
        return SolveResult(response=json.dumps(embedding))

    async def close(self) -> None:
        await self._client.close()
