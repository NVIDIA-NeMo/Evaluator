"""EmbeddingSolver: returns embeddings as JSON-encoded response."""
from __future__ import annotations

import time
from typing import Any

from nemo_evaluator.environments.base import SeedResult
from nemo_evaluator.observability.types import ModelResponse

from .base import SolveResult


class EmbeddingSolver:
    """Solver for embedding tasks. Returns embeddings as JSON-encoded response."""

    def __init__(self, client: Any) -> None:
        self._client = client

    async def solve(self, task: SeedResult) -> SolveResult:
        import json

        t0 = time.monotonic()
        embedding = await self._client.embed(task.prompt)
        latency = (time.monotonic() - t0) * 1000

        text = json.dumps(embedding)
        prompt_tokens = len(task.prompt) // 4 if task.prompt else 0
        model_resp = ModelResponse(
            content=text,
            model="embedding",
            prompt_tokens=prompt_tokens,
            total_tokens=prompt_tokens,
            latency_ms=round(latency, 2),
        )
        return SolveResult(response=text, model_response=model_resp)

    async def close(self) -> None:
        await self._client.close()
