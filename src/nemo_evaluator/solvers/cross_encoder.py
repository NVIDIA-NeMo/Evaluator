"""CrossEncoderSolver: cross-encoder/reranking tasks."""
from __future__ import annotations

import time

from nemo_evaluator.environments.base import SeedResult
from nemo_evaluator.observability.types import ModelResponse

from .base import SolveResult
from .trajectory_util import _single_turn_trajectory


class CrossEncoderSolver:
    """Solver for cross-encoder/reranking tasks. Sends query-document pairs
    to a /v1/rerank endpoint and returns the score."""

    def __init__(self, base_url: str, model: str, api_key: str | None = None) -> None:
        from nemo_evaluator.runner.model_client import ModelClient
        self._model_client = ModelClient(
            base_url=base_url.rstrip("/"), model=model, api_key=api_key,
        )
        self._url = f"{base_url.rstrip('/')}/rerank"
        self._model = model

    async def solve(self, task: SeedResult) -> SolveResult:
        import json

        query = task.metadata.get("query", task.prompt)
        documents = task.metadata.get("documents", [task.prompt])

        payload = {"model": self._model, "query": query, "documents": documents}
        t0 = time.monotonic()
        data = await self._model_client._post_with_retry(self._url, payload)
        latency = (time.monotonic() - t0) * 1000

        text = json.dumps(data.get("results", []))
        prompt_tokens = (len(query) + sum(len(d) for d in documents)) // 4
        model_resp = ModelResponse(
            content=text,
            model=self._model,
            prompt_tokens=prompt_tokens,
            total_tokens=prompt_tokens,
            latency_ms=round(latency, 2),
        )
        trajectory = _single_turn_trajectory(query, text)
        return SolveResult(response=text, model_response=model_resp, trajectory=trajectory)

    async def close(self) -> None:
        await self._model_client.close()
