"""EvalEnvironment base class – the core abstraction for all benchmarks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SeedResult:
    """Returned by EvalEnvironment.seed(). Superset of what any consumer needs."""

    prompt: str
    expected_answer: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VerifyResult:
    """Returned by EvalEnvironment.verify(). Reward is the minimum; rest is rich observability."""

    reward: float
    extracted_answer: str | None = None
    scoring_details: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class EvalEnvironment(ABC):
    """Abstract base for all Evaluator benchmarks.

    Subclasses implement two methods:
      - seed(idx)   -> SeedResult   (prepare problem)
      - verify(response, expected, **metadata) -> VerifyResult  (score answer)

    The dataset is loaded once at construction time. The environment is stateless
    across problems (each seed/verify pair is independent).
    """

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

    @abstractmethod
    def seed(self, idx: int) -> SeedResult:
        """Return prompt, expected_answer, and metadata for problem at index *idx*."""

    @abstractmethod
    def verify(self, response: str, expected: str, **metadata: Any) -> VerifyResult:
        """Score *response* against *expected*. Return reward + scoring details."""
