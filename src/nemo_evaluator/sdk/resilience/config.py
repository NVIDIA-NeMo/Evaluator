"""Internal resilience policy configuration for evaluator outbound calls."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ResilienceConfig:
    """Tuning knobs for retry + admission control.

    These settings are internal to evaluator and are not user-facing API.

    Attributes:
        global_limit: Hard cap on concurrent outbound attempts for one resilience
            session (job/request context), not a process-wide service cap.
        endpoint_initial_limit: Starting per-endpoint concurrency limit.
        endpoint_min_limit: Lower bound for per-endpoint adaptive concurrency.
        endpoint_max_limit: Upper bound for per-endpoint adaptive concurrency.
        success_window: Number of successes needed before additive increase (+1) is applied.
        beta_hard_overload: Multiplicative decrease factor for hard overload failures.
        beta_soft_overload: Multiplicative decrease factor for soft overload failures (timeouts).
        cooldown_seconds_hard: Cooldown duration after hard overload feedback.
        cooldown_seconds_soft: Cooldown duration after soft overload feedback.
        timeout_soft_overload_escalation_count: Soft overload events needed to escalate to hard behavior.
        escalation_window_seconds: Sliding window used for soft-overload escalation counting.
        backoff_initial_ms: Initial retry backoff in milliseconds.
        backoff_cap_ms: Maximum retry backoff in milliseconds.
        pressure_gain_k: Pressure multiplier gain for retry wait inflation.
        max_attempts_default: Default maximum attempts if callsites do not override.
        task_deadline_seconds_default: Default per-task retry deadline.
        global_max_queued: Max queued operations across all endpoints before rejecting work.
        endpoint_max_queued: Max queued operations per endpoint before rejecting work.
        retry_budget_tokens_per_sec: Retry token refill rate per endpoint.
        retry_budget_burst: Retry token burst capacity per endpoint.
        shutdown_grace_seconds: Grace period for graceful scheduler shutdown.
        endpoint_state_max_entries: Maximum endpoint states retained in memory.
        endpoint_state_ttl_seconds: TTL for endpoint states before eviction.
    """

    global_limit: int = 64
    endpoint_initial_limit: int = 4
    endpoint_min_limit: int = 1
    endpoint_max_limit: int = 64

    success_window: int = 20
    beta_hard_overload: float = 0.5
    beta_soft_overload: float = 0.7
    cooldown_seconds_hard: float = 3.0
    cooldown_seconds_soft: float = 1.5
    timeout_soft_overload_escalation_count: int = 3
    escalation_window_seconds: float = 10.0

    backoff_initial_ms: float = 250.0
    backoff_cap_ms: float = 15_000.0
    pressure_gain_k: float = 1.0
    max_attempts_default: int = 3
    task_deadline_seconds_default: float = 60.0

    global_max_queued: int = 8_192
    endpoint_max_queued: int = 1_024
    retry_budget_tokens_per_sec: float = 5.0
    retry_budget_burst: float = 10.0

    shutdown_grace_seconds: float = 15.0

    endpoint_state_max_entries: int = 4_096
    endpoint_state_ttl_seconds: float = 300.0
