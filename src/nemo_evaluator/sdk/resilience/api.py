"""Public API boundary for evaluator resilience scheduling."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import ParamSpec, TypeVar, cast

from nemo_evaluator.sdk.resilience.config import ResilienceConfig
from nemo_evaluator.sdk.resilience.scheduler import ResilienceScheduler

_T = TypeVar("_T")
_P = ParamSpec("_P")
_logger = logging.getLogger(__name__)

_current_scheduler: ContextVar[ResilienceScheduler | None] = ContextVar("resilience_scheduler", default=None)
_default_scheduler = ResilienceScheduler(ResilienceConfig())


def _active_scheduler() -> ResilienceScheduler:
    scheduler = _current_scheduler.get()
    if scheduler is not None:
        return scheduler
    return _default_scheduler


@asynccontextmanager
async def use_resilience_session(
    *,
    global_limit: int | None = None,
    endpoint_max_limit: int | None = None,
) -> AsyncIterator[None]:
    """Override active scheduler for the current async context.

    This uses a ContextVar-backed session boundary so concurrent requests/jobs can
    each have their own scheduler instance and concurrency cap.

    Note:
        Limits are global only within this session. They are not process-wide caps
        shared across concurrent evaluator jobs. Service-level coordination remains
        out of scope for this process-local V2 design.
    """
    base = ResilienceConfig()
    scheduler = ResilienceScheduler(
        ResilienceConfig(
            global_limit=max(1, global_limit) if global_limit is not None else base.global_limit,
            endpoint_max_limit=max(1, endpoint_max_limit)
            if endpoint_max_limit is not None
            else base.endpoint_max_limit,
        )
    )
    token = _current_scheduler.set(scheduler)
    try:
        yield
    finally:
        summary = await scheduler.summary()
        _logger.info("Resilience session summary", extra=summary)
        await scheduler.shutdown()
        _current_scheduler.reset(token)


async def run_with_resilience(
    endpoint_key: str,
    operation: Callable[_P, Awaitable[_T]],
    *args: _P.args,
    max_attempts: int,
    deadline_at: float | None = None,
    **kwargs: _P.kwargs,
) -> _T:
    """Execute an outbound attempt with scheduler-managed retries/admission."""
    scheduler = _active_scheduler()
    effective_attempts = max(1, max_attempts)
    return await scheduler.run_with_resilience(
        endpoint_key,
        operation,
        *args,
        max_attempts=effective_attempts,
        deadline_at=deadline_at,
        **kwargs,
    )


async def run_indexed_tasks(
    indices: Sequence[int],
    worker: Callable[[int], Awaitable[_T]],
    *,
    parallelism: int,
) -> list[_T]:
    """Execute index-keyed work with bounded in-flight task dispatch.

    Callers provide item indices and an async worker; this helper schedules up to
    `parallelism` tasks at a time, preserving result ordering by input position.

    Notes:
        `parallelism` is a hard cap on active row-level tasks for a session.
        Scheduler admission still applies inside each task, so endpoint/global
        shedding can reduce actual outbound call concurrency below this value.
        Keeping this outer cap bounds task memory/CPU overhead while retaining
        adaptive network pressure control in the scheduler.
    """
    if not indices:
        return []

    results: list[_T | None] = [None] * len(indices)
    max_inflight = min(len(indices), max(1, parallelism))
    _logger.debug(
        "Resilience indexed task execution started",
        extra={"task_count": len(indices), "worker_parallelism": parallelism, "max_inflight_workers": max_inflight},
    )
    next_position = 0
    in_flight: dict[asyncio.Task[_T], int] = {}

    while next_position < len(indices) or in_flight:
        try:
            while next_position < len(indices) and len(in_flight) < max_inflight:
                position = next_position
                index = indices[position]
                in_flight[asyncio.create_task(worker(index))] = position
                next_position += 1

            done, _ = await asyncio.wait(in_flight.keys(), return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                position = in_flight.pop(task)
                results[position] = task.result()
        except Exception:
            for task in in_flight:
                task.cancel()
            if in_flight:
                await asyncio.gather(*in_flight.keys(), return_exceptions=True)
            raise

    _logger.info(
        "Resilience indexed task execution completed",
        extra={"task_count": len(indices), "worker_parallelism": parallelism, "max_inflight_workers": max_inflight},
    )
    return cast(list[_T], results)
