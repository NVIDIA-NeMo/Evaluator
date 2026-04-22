"""Types used by the evaluator resilience control plane."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Deque, Protocol


class FailureClass(StrEnum):
    """Failure taxonomy for retry + adaptation decisions."""

    HARD_OVERLOAD = "hard_overload"
    SOFT_OVERLOAD = "soft_overload"
    TRANSIENT = "transient_non_overload"
    FATAL = "fatal"


class Clock(Protocol):
    """Monotonic clock abstraction for deterministic tests."""

    def monotonic(self) -> float:
        """Return monotonic time in seconds."""
        ...


class SystemClock:
    """System monotonic clock implementation."""

    def monotonic(self) -> float:
        return time.monotonic()


@dataclass
class OperationCounters:
    """Common operation counters used for scheduler and endpoint metrics."""

    operations_started: int = 0
    operations_completed: int = 0
    retries_scheduled: int = 0
    limit_decreases: int = 0
    limit_increases: int = 0
    cancellations: int = 0


@dataclass
class EndpointState:
    """Mutable per-endpoint resilience state."""

    key: str
    limit: int
    min_limit: int
    max_limit: int
    cooldown_until: float = 0.0
    success_streak: int = 0
    overload_hard_count: int = 0
    overload_soft_count: int = 0
    latency_ewma: float | None = None
    retry_budget_tokens: float = 0.0
    retry_budget_last_refill: float = 0.0
    last_seen: float = 0.0
    soft_overload_events: Deque[float] = field(default_factory=deque)
    queued: int = 0
    inflight: int = 0
    max_inflight_seen: int = 0
    saw_failure: bool = False
    counters: OperationCounters = field(default_factory=OperationCounters)


@dataclass(frozen=True)
class ClassifierResult:
    """Classification output for one raised exception."""

    failure_class: FailureClass
    retryable: bool
    retry_after_seconds: float | None
    status_code: int | None = None
    error_type: str | None = None


@dataclass(frozen=True)
class RetryContext:
    """Inputs for retry wait calculation."""

    attempt_number: int
    retry_after_seconds: float | None
    pressure: float
    cooldown_remaining_seconds: float
