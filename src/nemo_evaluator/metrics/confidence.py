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
"""Confidence intervals: bootstrap, normal (CLT), and clustered standard errors.

The clustered SE implementation follows Evan Miller, "Adding Error Bars to
Evals" (arXiv 2411.00640, Anthropic, 2024): when benchmark questions are
grouped (e.g. MMLU subjects, DROP passages), naive SE underestimates
uncertainty by ~3x because within-group scores are correlated.
"""

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
    se: float | None = None  # standard error, when available


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
    boot_means = np.array([rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_bootstrap)])

    alpha = 1.0 - confidence
    lower = float(np.percentile(boot_means, 100 * alpha / 2))
    upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))

    return ConfidenceInterval(mean=mean, ci_lower=lower, ci_upper=upper, confidence=confidence, method="bootstrap")


def normal_ci(scores: list[float], confidence: float = 0.95) -> ConfidenceInterval:
    """Normal (CLT) confidence interval for the mean.

    Recommended over bootstrap for binary/bounded eval scores — equally
    accurate and 10,000x faster (Miller 2024).
    """
    arr = np.array(scores, dtype=np.float64)
    mean = float(arr.mean())
    n = len(arr)

    if n < 2:
        return ConfidenceInterval(
            mean=mean, ci_lower=mean, ci_upper=mean, confidence=confidence, method="normal", se=0.0
        )

    from scipy import stats

    se = float(arr.std(ddof=1) / np.sqrt(n))
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    return ConfidenceInterval(
        mean=mean,
        ci_lower=mean - z * se,
        ci_upper=mean + z * se,
        confidence=confidence,
        method="normal",
        se=se,
    )


def clustered_ci(
    scores: list[float],
    clusters: list[str | int],
    confidence: float = 0.95,
) -> ConfidenceInterval:
    """Clustered standard error CI for grouped evaluation data.

    When benchmark questions share structure (e.g. MMLU subjects, DROP
    passages, MGSM languages), scores within a group are correlated.
    Naive SE treats all N scores as independent, underestimating
    uncertainty by up to 3x (Miller 2024, Table 1).

    Clustered SE accounts for within-group correlation by computing
    variance at the cluster level::

        SE_clustered = sqrt( G / ((G-1) * N^2) * sum_g(e_g^2) )

    where G = number of clusters, N = total observations, and
    e_g = sum of residuals (score_i - mean) within cluster g.

    Parameters
    ----------
    scores:
        Per-sample scores (rewards).
    clusters:
        Cluster assignment for each score (e.g. category name, passage ID).
        Must be same length as scores.
    confidence:
        Confidence level (default 0.95).

    Returns
    -------
    ConfidenceInterval with method="clustered".
    """
    if len(scores) != len(clusters):
        raise ValueError(f"scores ({len(scores)}) and clusters ({len(clusters)}) must have same length")

    arr = np.array(scores, dtype=np.float64)
    mean = float(arr.mean())
    n = len(arr)

    if n < 2:
        return ConfidenceInterval(
            mean=mean, ci_lower=mean, ci_upper=mean, confidence=confidence, method="clustered", se=0.0
        )

    # Group residuals by cluster
    residuals = arr - mean
    cluster_sums: dict[str | int, float] = {}
    for r, c in zip(residuals, clusters, strict=True):
        cluster_sums[c] = cluster_sums.get(c, 0.0) + float(r)

    g = len(cluster_sums)
    if g < 2:
        # Single cluster — fall back to normal SE
        return normal_ci(scores, confidence)

    # Clustered variance: (G / (G-1)) * (1/N^2) * sum(e_g^2)
    sum_sq = sum(e**2 for e in cluster_sums.values())
    clustered_var = (g / (g - 1)) * sum_sq / (n**2)
    se = float(np.sqrt(clustered_var))

    from scipy import stats

    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    return ConfidenceInterval(
        mean=mean,
        ci_lower=mean - z * se,
        ci_upper=mean + z * se,
        confidence=confidence,
        method="clustered",
        se=se,
    )
