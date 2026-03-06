from __future__ import annotations

import logging
from typing import Any

import httpx

from nemo_evaluator.adapters.base import EnvironmentAdapter
from nemo_evaluator.environments.base import SeedResult, VerifyResult

logger = logging.getLogger(__name__)


class GymAdapter(EnvironmentAdapter):
    """Consumes a remote environment server via seed_session/verify REST calls.

    Works with nel serve endpoints regardless of gym_compat mode.
    Evaluator owns the model call between seed and verify, giving full
    trajectory capture.
    """

    def __init__(self, endpoint: str, timeout: float = 60.0) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout
        self.name = f"remote@{self.endpoint}"
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def seed(self, idx: int) -> SeedResult:
        c = await self._get_client()
        r = await c.post(f"{self.endpoint}/seed_session", json={"idx": idx})
        r.raise_for_status()
        d = r.json()
        return SeedResult(
            prompt=d.get("prompt", ""), expected_answer=d.get("expected_answer", ""),
            metadata=d.get("metadata", {}),
            messages=d.get("messages"), system=d.get("system"),
        )

    async def verify(self, response: str, expected: str, **meta: Any) -> VerifyResult:
        c = await self._get_client()
        r = await c.post(
            f"{self.endpoint}/verify",
            json={"response": response, "expected": expected, "metadata": meta},
        )
        r.raise_for_status()
        d = r.json()
        return VerifyResult(
            reward=float(d.get("reward", 0.0)),
            extracted_answer=d.get("extracted_answer"),
            scoring_details=d.get("scoring_details", {}),
            metadata=d.get("metadata", {}),
        )

    async def dataset_size(self) -> int:
        try:
            c = await self._get_client()
            r = await c.get(f"{self.endpoint}/dataset_size")
            r.raise_for_status()
            return r.json().get("size", -1)
        except Exception:
            return -1

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
