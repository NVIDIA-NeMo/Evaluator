"""Environment base classes for single-turn and multi-turn evaluation."""

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
    messages: list[dict[str, str]] | None = None
    system: str | None = None


@dataclass
class VerifyResult:
    """Returned by EvalEnvironment.verify(). Reward is the minimum; rest is rich observability."""

    reward: float
    extracted_answer: str | None = None
    scoring_details: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Observation:
    """Returned by StepEnvironment.reset() and step(). Drives multi-turn loops."""

    content: str
    done: bool = False
    reward: float = 0.0
    tools: list[dict[str, Any]] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class EvalEnvironment(ABC):
    """Abstract base for single-turn benchmarks.

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


class StepEnvironment(ABC):
    """Abstract base for multi-turn / interactive BYOB benchmarks.

    Unlike EvalEnvironment (single-turn: seed -> model -> verify), StepEnvironment
    drives an interactive loop: reset -> (model generates action) -> step -> ... -> done.

    The evaluator owns the outer loop (n-repeats, metrics, artifacts).
    StepEnvironment owns the environment state and reward computation.
    The model/agent handles the inner loop (deciding what action to take).

    Example BYOB multi-turn benchmark:

        @register("my_agent_bench")
        class MyAgentBench(StepEnvironment):
            def __init__(self):
                self._tasks = load_tasks()

            def __len__(self):
                return len(self._tasks)

            def reset(self, idx):
                self._state = initial_state(self._tasks[idx])
                return Observation(content=self._tasks[idx]["instruction"],
                                   tools=[{"name": "execute", "schema": {...}}])

            def step(self, action):
                result = execute_action(self._state, action)
                done = is_solved(self._state)
                return Observation(content=result, done=done,
                                   reward=1.0 if done else 0.0)

            @property
            def max_steps(self):
                return 20
    """

    name: str = "unnamed"

    @abstractmethod
    def __len__(self) -> int:
        """Number of problems in the dataset."""

    @abstractmethod
    def reset(self, idx: int) -> Observation:
        """Initialize environment for problem idx. Returns first observation."""

    @abstractmethod
    def step(self, action: str) -> Observation:
        """Apply agent's action, return next observation. Set done=True when finished."""

    @property
    def max_steps(self) -> int:
        """Max steps before forced termination. Override for per-benchmark limits."""
        return 50
