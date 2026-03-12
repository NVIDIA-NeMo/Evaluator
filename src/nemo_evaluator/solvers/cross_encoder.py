"""CrossEncoderSolver: cross-encoder/reranking tasks."""
from __future__ import annotations

from nemo_evaluator.environments.base import SeedResult

from .base import SolveResult


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
        data = await self._model_client._post_with_retry(self._url, payload)
        return SolveResult(response=json.dumps(data.get("results", [])))

    async def close(self) -> None:
        await self._model_client.close()
