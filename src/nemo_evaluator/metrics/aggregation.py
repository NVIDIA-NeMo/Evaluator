# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
        breakdown.append(
            CategoryResult(
                category=cat,
                n_samples=len(scores),
                mean_reward=ci.mean,
                ci=ci,
            )
        )
    return breakdown


def scoring_details_breakdown(
    results: list[dict[str, Any]],
    *,
    max_cardinality: int = 50,
) -> dict[str, list[CategoryResult]]:
    """Auto-discover groupable fields in scoring_details and compute per-group stats.

    Only considers string/bool fields with 2..max_cardinality distinct values
    (filters out unique IDs and constant fields).  Returns a dict mapping
    field name to a list of CategoryResult.
    """
    if not results:
        return {}

    # Collect candidate fields and their values
    field_values: dict[str, list[tuple[str, float]]] = {}
    for r in results:
        sd = r.get("scoring_details", {})
        reward = float(r.get("reward", 0.0))
        for k, v in sd.items():
            if isinstance(v, bool):
                field_values.setdefault(k, []).append((str(v), reward))
            elif isinstance(v, str) and len(v) < 100:
                field_values.setdefault(k, []).append((v, reward))

    breakdowns: dict[str, list[CategoryResult]] = {}
    for field_name, pairs in field_values.items():
        if len(pairs) != len(results):
            continue
        distinct = {v for v, _ in pairs}
        if len(distinct) < 2 or len(distinct) > max_cardinality:
            continue

        groups: dict[str, list[float]] = {}
        for val, reward in pairs:
            groups.setdefault(val, []).append(reward)

        cats = []
        for cat in sorted(groups):
            scores = groups[cat]
            ci = bootstrap_ci(scores)
            cats.append(
                CategoryResult(
                    category=cat,
                    n_samples=len(scores),
                    mean_reward=ci.mean,
                    ci=ci,
                )
            )
        breakdowns[field_name] = cats

    return breakdowns


def summary_stats(rewards: list[float]) -> dict[str, float]:
    """Compute basic summary statistics over a list of reward values."""
    if not rewards:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "median": 0.0, "n": 0}

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
