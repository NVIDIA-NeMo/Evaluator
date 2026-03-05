from __future__ import annotations

from abc import ABC, abstractmethod

from nemo_evaluator.environments.base import SeedResult, VerifyResult


class EnvironmentAdapter(ABC):
    name: str = "unnamed"

    @abstractmethod
    async def seed(self, idx: int) -> SeedResult: ...

    @abstractmethod
    async def verify(self, response: str, expected: str, **meta) -> VerifyResult: ...

    async def dataset_size(self) -> int:
        return -1
