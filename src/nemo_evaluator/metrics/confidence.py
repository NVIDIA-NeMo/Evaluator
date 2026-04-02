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
    boot_means = np.array([rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_bootstrap)])

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
