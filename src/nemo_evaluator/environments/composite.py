"""Composite environment: seed from one env, verify with another.

Enables BYOB benchmarks (which handle data loading) to delegate
verification to a remote environment (e.g. a Gym server running
a judge model) without either side knowing about the other's format.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from nemo_evaluator.environments.base import EvalEnvironment, SeedResult, VerifyResult

if TYPE_CHECKING:
    from nemo_evaluator.sandbox.base import Sandbox

logger = logging.getLogger(__name__)


class CompositeEnvironment(EvalEnvironment):
    """Compose ``seed()`` from one environment with ``verify()`` from another.

    Parameters
    ----------
    seed_env:
        Provides data via ``seed()`` and ``dataset_size()``.
    verify_env:
        Scores responses via ``verify()``.
    """

    def __init__(self, seed_env: EvalEnvironment, verify_env: EvalEnvironment) -> None:
        super().__init__()
        self._seed_env = seed_env
        self._verify_env = verify_env
        self.name = seed_env.name

    async def seed(self, idx: int) -> SeedResult:
        return await self._seed_env.seed(idx)

    async def verify(
        self, response: str, expected: str,
        sandbox: Sandbox | None = None, **metadata: Any,
    ) -> VerifyResult:
        return await self._verify_env.verify(
            response, expected, sandbox=sandbox, **metadata,
        )

    async def dataset_size(self) -> int:
        return await self._seed_env.dataset_size()

    async def close(self) -> None:
        try:
            await self._seed_env.close()
        finally:
            await self._verify_env.close()
