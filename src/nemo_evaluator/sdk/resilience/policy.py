"""Policy math for resilience retry waits and endpoint adaptation."""

from __future__ import annotations

import math
import random

from nemo_evaluator.sdk.resilience.config import ResilienceConfig
from nemo_evaluator.sdk.resilience.types import FailureClass, RetryContext


def pressure(limit: int, max_limit: int) -> float:
    """Compute normalized pressure (0..1) from endpoint capacity state."""
    if max_limit <= 0:
        return 1.0
    return max(0.0, min(1.0, 1.0 - (limit / max_limit)))


def retry_wait_seconds(context: RetryContext, config: ResilienceConfig) -> float:
    """Compute retry delay with jitter, pressure scaling, and overload floors."""
    exp = min(
        config.backoff_cap_ms / 1000.0, (config.backoff_initial_ms / 1000.0) * (2 ** max(0, context.attempt_number - 1))
    )
    jittered = random.uniform(0.0, exp)
    pressure_mult = 1.0 + (config.pressure_gain_k * context.pressure)
    computed = jittered * pressure_mult
    server_floor = context.retry_after_seconds or 0.0
    return max(server_floor, context.cooldown_remaining_seconds, computed)


def additive_increase(limit: int, max_limit: int) -> int:
    """Increase endpoint limit by one within configured bounds."""
    return min(max_limit, limit + 1)


def multiplicative_decrease(limit: int, min_limit: int, beta: float) -> int:
    """Reduce endpoint limit by multiplicative factor within bounds."""
    return max(min_limit, math.ceil(limit * beta))


def cooldown_for_failure_class(failure_class: FailureClass, config: ResilienceConfig) -> float:
    """Return cooldown period in seconds for a failure class."""
    if failure_class == FailureClass.HARD_OVERLOAD:
        return config.cooldown_seconds_hard
    if failure_class == FailureClass.SOFT_OVERLOAD:
        return config.cooldown_seconds_soft
    return 0.0
