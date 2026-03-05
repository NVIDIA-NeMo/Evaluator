"""Bootstrap confidence intervals and normal approximation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ConfidenceInterval:
    mean: float
    ci_lower: float
    ci_upper: float
    confidence: float
    method: str


def bootstrap_ci(
    scores: list[float],
    confidence: float = 0.95,
    n_bootstrap: int = 10_000,
    seed: int | None = 42,
) -> ConfidenceInterval:
    """Compute bootstrap confidence interval for the mean of *scores*."""
    arr = np.array(scores, dtype=np.float64)
    mean = float(arr.mean())

    if len(arr) < 2:
        return ConfidenceInterval(mean=mean, ci_lower=mean, ci_upper=mean, confidence=confidence, method="bootstrap")

    rng = np.random.default_rng(seed)
    boot_means = np.array([
        rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_bootstrap)
    ])

    alpha = 1.0 - confidence
    lower = float(np.percentile(boot_means, 100 * alpha / 2))
    upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))

    return ConfidenceInterval(mean=mean, ci_lower=lower, ci_upper=upper, confidence=confidence, method="bootstrap")


def normal_ci(scores: list[float], confidence: float = 0.95) -> ConfidenceInterval:
    """Normal approximation confidence interval for the mean."""
    arr = np.array(scores, dtype=np.float64)
    mean = float(arr.mean())
    n = len(arr)

    if n < 2:
        return ConfidenceInterval(mean=mean, ci_lower=mean, ci_upper=mean, confidence=confidence, method="normal")

    from scipy import stats

    se = float(arr.std(ddof=1) / np.sqrt(n))
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    return ConfidenceInterval(
        mean=mean,
        ci_lower=mean - z * se,
        ci_upper=mean + z * se,
        confidence=confidence,
        method="normal",
    )
