"""Base classes for evaluation environments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SeedResult:
    prompt: str
    expected_answer: str
    metadata: dict[str, Any] = field(default_factory=dict)
    messages: list[dict[str, str]] | None = None
    system: str | None = None


@dataclass
class VerifyResult:
    reward: float
    extracted_answer: str | None = None
    scoring_details: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class EvalEnvironment(ABC):
    """Base for all evaluation environments -- BYOB, harness wrappers, remote, etc."""

    name: str = "unnamed"

    def __init__(self) -> None:
        self._dataset: list[dict[str, Any]] = []

    @property
    def dataset(self) -> list[dict[str, Any]]:
        return self._dataset

    @dataset.setter
    def dataset(self, value: list[dict[str, Any]]) -> None:
        self._dataset = value

    def __len__(self) -> int:
        return len(self._dataset)

    async def dataset_size(self) -> int:
        return len(self._dataset)

    @abstractmethod
    async def seed(self, idx: int) -> SeedResult: ...

    @abstractmethod
    async def verify(self, response: str, expected: str, **metadata: Any) -> VerifyResult: ...

    async def close(self) -> None:
        """Clean up resources. Override for environments that hold connections."""


