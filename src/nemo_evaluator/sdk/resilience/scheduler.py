"""Process-local resilience scheduler for admission control and retries."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Deque, ParamSpec, TypeVar

import anyio
from nemo_evaluator.sdk.resilience.classifier import classify_exception
from nemo_evaluator.sdk.resilience.config import ResilienceConfig
from nemo_evaluator.sdk.resilience.policy import (
    additive_increase,
    cooldown_for_failure_class,
    multiplicative_decrease,
    pressure,
    retry_wait_seconds,
)
from nemo_evaluator.sdk.resilience.types import (
    ClassifierResult,
    Clock,
    EndpointState,
    FailureClass,
    OperationCounters,
    RetryContext,
    SystemClock,
)

_logger = logging.getLogger(__name__)
_T = TypeVar("_T")
_P = ParamSpec("_P")


class ResilienceError(RuntimeError):
    """Base class for resilience runtime failures."""

    def __init__(self, message: str, *, endpoint_key: str, attempt: int, cause: BaseException | None = None) -> None:
        super().__init__(message)
        self.endpoint_key = endpoint_key
        self.attempt = attempt
        self.cause = cause


class ResilienceQueueFullError(ResilienceError):
    """Raised when scheduler queue bounds are exceeded."""

    def __init__(self, message: str, *, endpoint_key: str, reason: str, attempt: int = 0) -> None:
        super().__init__(message, endpoint_key=endpoint_key, attempt=attempt)
        self.reason = reason


class ResilienceDeadlineExceededError(ResilienceError):
    """Raised when retry would violate deadline."""


class ResilienceMaxAttemptsExceededError(ResilienceError):
    """Raised when retry attempts are exhausted."""


class ResilienceCancelledError(ResilienceError):
    """Raised when operation is cancelled during scheduler flow."""


@dataclass
class _Controller:
    state: EndpointState
    limiter: anyio.CapacityLimiter
    lock: asyncio.Lock


class ResilienceScheduler:
    """Central process-local scheduler for retries and adaptive admission.

    A scheduler instance is intended to be session-scoped (for example one job or
    one live-eval request). Its global limiter enforces bounds within that session,
    not across the entire evaluator process.
    """

    def __init__(self, config: ResilienceConfig, *, clock: Clock | None = None) -> None:
        self._config = config
        self._clock = clock or SystemClock()
        self._global_limiter = anyio.CapacityLimiter(config.global_limit)
        self._controllers: dict[str, _Controller] = {}
        self._lock = asyncio.Lock()
        self._global_queued = 0
        self._shutdown = False
        self._metrics = OperationCounters()

    def now(self) -> float:
        """Return scheduler monotonic time."""
        return self._clock.monotonic()

    async def shutdown(self) -> None:
        """Mark scheduler as shutting down."""
        async with self._lock:
            self._shutdown = True

    async def run_with_resilience(
        self,
        endpoint_key: str,
        operation: Callable[_P, Awaitable[_T]],
        *args: _P.args,
        max_attempts: int,
        deadline_at: float | None,
        **kwargs: _P.kwargs,
    ) -> _T:
        """Execute operation with adaptive scheduling and retry policy."""
        last_failure_class: FailureClass | None = None
        _logger.debug(
            "Resilience operation started",
            extra={"endpoint_key": endpoint_key, "max_attempts": max_attempts, "deadline_at": deadline_at},
        )
        controller = await self._get_controller(endpoint_key)
        async with controller.lock:
            controller.state.counters.operations_started += 1
        async with self._lock:
            self._metrics.operations_started += 1
        for attempt in range(1, max_attempts + 1):
            try:
                result = await self._run_once(endpoint_key, operation, *args, attempt=attempt, **kwargs)
            except ResilienceCancelledError:
                raise
            except Exception as exc:
                classified = classify_exception(exc)
                last_failure_class = classified.failure_class
                controller = await self._get_controller(endpoint_key)
                await self._record_failure(controller, classified)

                if not classified.retryable:
                    raise
                if attempt >= max_attempts:
                    raise ResilienceMaxAttemptsExceededError(
                        f"Retry attempts exhausted: {attempt} out of {max_attempts}",
                        endpoint_key=endpoint_key,
                        attempt=attempt,
                        cause=exc,
                    ) from exc

                now = self.now()
                cooldown_remaining = max(0.0, controller.state.cooldown_until - now)
                wait_seconds = retry_wait_seconds(
                    RetryContext(
                        attempt_number=attempt,
                        retry_after_seconds=classified.retry_after_seconds,
                        pressure=pressure(controller.state.limit, controller.state.max_limit),
                        cooldown_remaining_seconds=cooldown_remaining,
                    ),
                    self._config,
                )
                if deadline_at is not None and now + wait_seconds > deadline_at:
                    raise ResilienceDeadlineExceededError(
                        "Retry deadline exceeded",
                        endpoint_key=endpoint_key,
                        attempt=attempt,
                        cause=exc,
                    ) from exc
                _logger.info(
                    "Resilience retry scheduled",
                    extra={
                        "endpoint_key": endpoint_key,
                        "attempt": attempt,
                        "failure_class": classified.failure_class.value,
                        "wait_seconds": wait_seconds,
                        "retry_after_seconds": classified.retry_after_seconds,
                        "pressure": pressure(controller.state.limit, controller.state.max_limit),
                        "cooldown_remaining_seconds": cooldown_remaining,
                        "endpoint_limit": controller.state.limit,
                    },
                )
                async with controller.lock, self._lock:
                    self._metrics.retries_scheduled += 1
                    controller.state.counters.retries_scheduled += 1
                await asyncio.sleep(wait_seconds)
            else:
                await self.record_success(endpoint_key)
                controller = await self._get_controller(endpoint_key)
                _logger.debug(
                    "Resilience operation completed",
                    extra={
                        "endpoint_key": endpoint_key,
                        "attempts_used": attempt,
                        "last_failure_class": last_failure_class.value if last_failure_class else None,
                        "endpoint_limit": controller.state.limit,
                        "endpoint_max_inflight_seen": controller.state.max_inflight_seen,
                    },
                )
                async with self._lock:
                    self._metrics.operations_completed += 1
                async with controller.lock:
                    controller.state.counters.operations_completed += 1
                return result

        raise RuntimeError("Unreachable: retry loop exhausted without terminal outcome")

    async def _run_once(
        self,
        endpoint_key: str,
        operation: Callable[_P, Awaitable[_T]],
        *args: _P.args,
        attempt: int,
        **kwargs: _P.kwargs,
    ) -> _T:
        controller = await self._get_controller(endpoint_key)
        await self._enqueue_or_raise(controller.state, endpoint_key)

        endpoint_acquired = False
        global_acquired = False
        dispatched = False
        started = self.now()
        try:
            await controller.limiter.acquire()
            endpoint_acquired = True
            await self._global_limiter.acquire()
            global_acquired = True
            await self._on_dispatch(controller.state, attempt=attempt)
            dispatched = True
            return await operation(*args, **kwargs)
        except asyncio.CancelledError as exc:
            async with controller.lock, self._lock:
                self._metrics.cancellations += 1
                controller.state.counters.cancellations += 1
            raise ResilienceCancelledError(
                "Operation cancelled",
                endpoint_key=endpoint_key,
                attempt=attempt,
                cause=exc,
            ) from exc
        finally:
            elapsed = max(0.0, self.now() - started)
            await self._on_complete(controller.state, dispatched=dispatched, elapsed_seconds=elapsed)
            if global_acquired:
                self._global_limiter.release()
            if endpoint_acquired:
                controller.limiter.release()

    async def _get_controller(self, endpoint_key: str) -> _Controller:
        """Get or create endpoint controller state and refresh its last-seen time."""
        now = self.now()
        async with self._lock:
            self._evict_stale_endpoints(now)
            controller = self._controllers.get(endpoint_key)
            if controller is None:
                state = EndpointState(
                    key=endpoint_key,
                    limit=max(
                        self._config.endpoint_min_limit,
                        min(self._config.endpoint_initial_limit, self._config.endpoint_max_limit),
                    ),
                    min_limit=self._config.endpoint_min_limit,
                    max_limit=self._config.endpoint_max_limit,
                    retry_budget_tokens=self._config.retry_budget_burst,
                    retry_budget_last_refill=now,
                    last_seen=now,
                )
                controller = _Controller(state=state, limiter=anyio.CapacityLimiter(state.limit), lock=asyncio.Lock())
                self._controllers[endpoint_key] = controller
            controller.state.last_seen = now
            return controller

    def _evict_stale_endpoints(self, now: float) -> None:
        """Evict endpoint state by TTL first, then by least-recently-seen capacity pressure."""
        stale_keys = [
            key
            for key, controller in self._controllers.items()
            if now - controller.state.last_seen > self._config.endpoint_state_ttl_seconds
        ]
        for key in stale_keys:
            self._controllers.pop(key, None)
        if len(self._controllers) <= self._config.endpoint_state_max_entries:
            return
        victims = sorted(self._controllers.items(), key=lambda item: item[1].state.last_seen)[
            : len(self._controllers) - self._config.endpoint_state_max_entries
        ]
        for key, _ in victims:
            self._controllers.pop(key, None)

    async def _enqueue_or_raise(self, state: EndpointState, endpoint_key: str) -> None:
        """Account queued work or raise typed queue/shutdown errors before dispatch."""
        async with self._lock:
            if self._shutdown:
                raise ResilienceCancelledError("Scheduler is shutting down", endpoint_key=endpoint_key, attempt=0)
            if self._global_queued >= self._config.global_max_queued:
                raise ResilienceQueueFullError(
                    f"Global queue is full: {self._global_queued} out of {self._config.global_max_queued}",
                    endpoint_key=endpoint_key,
                    reason="global_queue_full",
                )
            if state.queued >= self._config.endpoint_max_queued:
                raise ResilienceQueueFullError(
                    f"Endpoint queue is full for {endpoint_key}: {state.queued} out of {self._config.endpoint_max_queued}",
                    endpoint_key=endpoint_key,
                    reason="endpoint_queue_full",
                )
            self._global_queued += 1
            state.queued += 1

    async def _on_dispatch(self, state: EndpointState, *, attempt: int) -> None:
        """Move one queued item to inflight and apply retry-budget checks."""
        now = self.now()
        async with self._lock:
            self._refill_retry_budget(state, now)
            if attempt > 1:
                if state.retry_budget_tokens < 1.0:
                    raise ResilienceQueueFullError(
                        "Retry budget exhausted",
                        endpoint_key=state.key,
                        reason="retry_budget_exhausted",
                    )
                state.retry_budget_tokens -= 1.0
            self._global_queued = max(0, self._global_queued - 1)
            state.queued = max(0, state.queued - 1)
            state.inflight += 1
            state.max_inflight_seen = max(state.max_inflight_seen, state.inflight)

    async def _on_complete(self, state: EndpointState, *, dispatched: bool, elapsed_seconds: float) -> None:
        """Reconcile queued/inflight counters and update latency EWMA on completion."""
        async with self._lock:
            if not dispatched:
                self._global_queued = max(0, self._global_queued - 1)
                state.queued = max(0, state.queued - 1)
            else:
                state.inflight = max(0, state.inflight - 1)
                if state.latency_ewma is None:
                    state.latency_ewma = elapsed_seconds
                else:
                    state.latency_ewma = (0.2 * elapsed_seconds) + (0.8 * state.latency_ewma)

    async def _record_failure(self, controller: _Controller, classified: ClassifierResult) -> None:
        """Apply failure feedback to endpoint state (AIMD and timeout escalation)."""
        failure_class = classified.failure_class
        now = self.now()
        async with controller.lock:
            state = controller.state
            state.success_streak = 0
            state.saw_failure = True
            previous_limit = state.limit
            if failure_class == FailureClass.HARD_OVERLOAD:
                state.overload_hard_count += 1
                state.limit = multiplicative_decrease(state.limit, state.min_limit, self._config.beta_hard_overload)
                state.cooldown_until = max(
                    state.cooldown_until, now + cooldown_for_failure_class(failure_class, self._config)
                )
            elif failure_class == FailureClass.SOFT_OVERLOAD:
                state.overload_soft_count += 1
                state.soft_overload_events.append(now)
                state.limit = multiplicative_decrease(state.limit, state.min_limit, self._config.beta_soft_overload)
                state.cooldown_until = max(
                    state.cooldown_until, now + cooldown_for_failure_class(failure_class, self._config)
                )
                self._trim_events(state.soft_overload_events, now, self._config.escalation_window_seconds)
                if len(state.soft_overload_events) >= self._config.timeout_soft_overload_escalation_count:
                    state.limit = multiplicative_decrease(state.limit, state.min_limit, self._config.beta_hard_overload)
                    state.cooldown_until = max(state.cooldown_until, now + self._config.cooldown_seconds_hard)
                    state.overload_hard_count += 1
            if failure_class in {FailureClass.HARD_OVERLOAD, FailureClass.SOFT_OVERLOAD}:
                controller.limiter.total_tokens = state.limit
                if state.limit != previous_limit:
                    state.counters.limit_decreases += 1
                    async with self._lock:
                        self._metrics.limit_decreases += 1
                    _logger.info(
                        "Resilience endpoint limit decreased",
                        extra={
                            "endpoint_key": state.key,
                            "failure_class": failure_class.value,
                            "previous_limit": previous_limit,
                            "new_limit": state.limit,
                            "cooldown_until": state.cooldown_until,
                            "overload_soft_count": state.overload_soft_count,
                            "overload_hard_count": state.overload_hard_count,
                            "cause_status_code": classified.status_code,
                            "cause_error_type": classified.error_type,
                        },
                    )

    async def record_success(self, endpoint_key: str) -> None:
        """Record success feedback for callers that need explicit signaling."""
        controller = await self._get_controller(endpoint_key)
        now = self.now()
        async with controller.lock:
            state = controller.state
            state.success_streak += 1
            if now < state.cooldown_until:
                return
            if state.success_streak >= self._config.success_window:
                state.success_streak = 0
                previous_limit = state.limit
                # Fast-start before the first failure: double capacity each window.
                # After any failure, revert to conservative +1 additive increase.
                if state.saw_failure:
                    state.limit = additive_increase(state.limit, state.max_limit)
                else:
                    state.limit = min(state.max_limit, max(state.min_limit, state.limit * 2))
                controller.limiter.total_tokens = state.limit
                if state.limit != previous_limit:
                    state.counters.limit_increases += 1
                    async with self._lock:
                        self._metrics.limit_increases += 1
                    _logger.info(
                        "Resilience endpoint limit increased",
                        extra={
                            "endpoint_key": state.key,
                            "previous_limit": previous_limit,
                            "new_limit": state.limit,
                            "success_window": self._config.success_window,
                        },
                    )

    def _refill_retry_budget(self, state: EndpointState, now: float) -> None:
        """Refill endpoint retry tokens using elapsed monotonic time."""
        elapsed = max(0.0, now - state.retry_budget_last_refill)
        state.retry_budget_last_refill = now
        state.retry_budget_tokens = min(
            self._config.retry_budget_burst,
            state.retry_budget_tokens + elapsed * self._config.retry_budget_tokens_per_sec,
        )

    @staticmethod
    def _trim_events(events: Deque[float], now: float, window_seconds: float) -> None:
        """Drop soft-overload timestamps older than the escalation window."""
        cutoff = now - max(0.0, window_seconds)
        while events and events[0] < cutoff:
            events.popleft()

    async def summary(self) -> dict[str, object]:
        """Return aggregate session metrics for resilience diagnostics."""
        async with self._lock:
            max_inflight_seen = max((c.state.max_inflight_seen for c in self._controllers.values()), default=0)
            per_endpoint: dict[str, dict[str, int | float]] = {}
            for endpoint_key, controller in self._controllers.items():
                state = controller.state
                per_endpoint[endpoint_key] = {
                    "operations_started": state.counters.operations_started,
                    "operations_completed": state.counters.operations_completed,
                    "retries_scheduled": state.counters.retries_scheduled,
                    "limit_decreases": state.counters.limit_decreases,
                    "limit_increases": state.counters.limit_increases,
                    "cancellations": state.counters.cancellations,
                    "overload_hard_count": state.overload_hard_count,
                    "overload_soft_count": state.overload_soft_count,
                    "current_limit": state.limit,
                    "max_inflight_seen": state.max_inflight_seen,
                }
            return {
                "operations_started": self._metrics.operations_started,
                "operations_completed": self._metrics.operations_completed,
                "retries_scheduled": self._metrics.retries_scheduled,
                "limit_decreases": self._metrics.limit_decreases,
                "limit_increases": self._metrics.limit_increases,
                "cancellations": self._metrics.cancellations,
                "endpoints_tracked": len(self._controllers),
                "max_endpoint_inflight_seen": max_inflight_seen,
                "per_endpoint": per_endpoint,
            }
