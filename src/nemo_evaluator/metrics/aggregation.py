"""Per-category breakdowns and summary statistics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nemo_evaluator.metrics.confidence import ConfidenceInterval, bootstrap_ci


@dataclass
class CategoryResult:
    category: str
    n_samples: int
    mean_reward: float
    ci: ConfidenceInterval


def category_breakdown(
    results: list[dict[str, Any]],
    category_key: str = "category",
) -> list[CategoryResult]:
    """Group results by a metadata category and compute per-group statistics.

    Each result dict must have 'reward' (float) and 'metadata' (dict) keys.
    """
    groups: dict[str, list[float]] = {}
    for r in results:
        cat = r.get("metadata", {}).get(category_key, "unknown")
        groups.setdefault(cat, []).append(float(r["reward"]))

    breakdown = []
    for cat in sorted(groups):
        scores = groups[cat]
        ci = bootstrap_ci(scores)
        breakdown.append(CategoryResult(
            category=cat,
            n_samples=len(scores),
            mean_reward=ci.mean,
            ci=ci,
        ))
    return breakdown


def summary_stats(rewards: list[float]) -> dict[str, float]:
    """Compute basic summary statistics over a list of reward values."""
    import numpy as np

    arr = np.array(rewards, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "median": float(np.median(arr)),
        "n": len(arr),
    }
