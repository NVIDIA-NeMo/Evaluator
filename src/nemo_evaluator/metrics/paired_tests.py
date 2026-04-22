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
"""Pure statistical tests for paired regression analysis.

Public API
----------
- ``mcnemar_test``       -- one-sided McNemar's exact test for degradation (binary)
- ``sign_test``          -- one-sided sign test for degradation (continuous)
- ``permutation_test``   -- permutation test on paired differences (continuous)
- ``detect_test``        -- auto-detect appropriate test from data shape
- ``mde_estimate``       -- minimum detectable effect at 80% power

Dataclasses
-----------
- ``McNemarResult``
- ``SignTestResult``
- ``PermutationResult``
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

logger = logging.getLogger(__name__)

SIGNIFICANCE_THRESHOLD = 0.05
POWER_80_FACTOR = 2.8  # z_alpha + z_beta for alpha=0.05 one-sided, 80% power


@dataclass
class McNemarResult:
    p_value: float | None = None
    significant: bool | None = None
    method: str | None = None
    n_discordant: int = 0
    effect_size: float | None = None  # (b - c) / n_paired
    ci_lower: float | None = None  # 95% CI on regression rate difference
    ci_upper: float | None = None
    hypothesis: str = "one-sided (degradation)"

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class SignTestResult:
    """Result of one-sided sign test on paired differences."""

    n_positive: int = 0  # baseline > candidate (regressions)
    n_negative: int = 0  # candidate > baseline (improvements)
    n_ties: int = 0
    p_value: float | None = None
    significant: bool | None = None
    method: str = "sign_test"

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class PermutationResult:
    """Result of permutation test on paired differences."""

    observed_mean_diff: float = 0.0
    p_value: float | None = None
    significant: bool | None = None
    n_permutations: int = 10_000
    effect_size: float | None = None  # Cohen's d paired
    method: str = "permutation"

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


def mcnemar_test(
    contingency: dict[str, int],
    n_paired: int = 0,
    alpha: float = SIGNIFICANCE_THRESHOLD,
) -> McNemarResult:
    """One-sided McNemar's test for degradation (H1: regressions > improvements).

    Uses exact binomial test for all sample sizes.
    """
    b = contingency["baseline_only_correct"]  # regressions
    c = contingency["candidate_only_correct"]  # improvements
    n_discordant = b + c

    effect_size = round((b - c) / n_paired, 6) if n_paired > 0 else None

    ci_lower = ci_upper = None
    if n_paired > 0 and n_discordant > 0:
        p_b = b / n_paired
        p_c = c / n_paired
        se = math.sqrt((p_b + p_c - (p_b - p_c) ** 2) / n_paired) if n_paired > 1 else 0
        if se > 0:
            ci_lower = round((p_b - p_c) - 1.96 * se, 6)
            ci_upper = round((p_b - p_c) + 1.96 * se, 6)

    result = McNemarResult(
        n_discordant=n_discordant,
        effect_size=effect_size,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
    )

    if n_discordant == 0:
        result.p_value = 1.0
        result.significant = False
        result.method = "exact"
        return result

    try:
        from scipy.stats import binomtest

        res = binomtest(b, n_discordant, 0.5, alternative="greater")
        result.p_value = round(float(res.pvalue), 6)
        result.method = "exact_binomial"
        result.significant = result.p_value < alpha
    except ImportError:
        logger.debug("scipy not installed; skipping McNemar test (pip install nemo-evaluator[stats])")

    return result


def sign_test(
    paired_deltas: list[float],
    alpha: float = SIGNIFICANCE_THRESHOLD,
) -> SignTestResult:
    """One-sided sign test for degradation (H1: baseline > candidate).

    Counts positive vs negative paired differences, ignoring ties (zeros).
    This is the correct generalization of McNemar to continuous data —
    when N>1 repeats are averaged, scores become continuous and McNemar
    no longer applies.
    """
    n_positive = sum(1 for d in paired_deltas if d > 0)  # regressions
    n_negative = sum(1 for d in paired_deltas if d < 0)  # improvements
    n_ties = sum(1 for d in paired_deltas if d == 0)

    result = SignTestResult(
        n_positive=n_positive,
        n_negative=n_negative,
        n_ties=n_ties,
    )

    n_non_tied = n_positive + n_negative
    if n_non_tied == 0:
        result.p_value = 1.0
        result.significant = False
        return result

    try:
        from scipy.stats import binomtest

        res = binomtest(n_positive, n_non_tied, 0.5, alternative="greater")
        result.p_value = round(float(res.pvalue), 6)
        result.significant = result.p_value < alpha
    except ImportError:
        logger.debug("scipy not installed; skipping sign test (pip install nemo-evaluator[stats])")

    return result


def permutation_test(
    paired_deltas: list[float],
    n_permutations: int = 10_000,
    seed: int = 42,
    alpha: float = SIGNIFICANCE_THRESHOLD,
) -> PermutationResult:
    """One-sided permutation test on paired differences (H1: mean diff > 0 = baseline better).

    Under H0, the signs of the paired differences are exchangeable.
    Randomly flip signs n_permutations times, compare observed mean to
    the permutation distribution. Uses only numpy (no scipy required).
    """
    deltas = np.array(paired_deltas, dtype=np.float64)
    observed_mean = float(deltas.mean())
    n = len(deltas)

    if n == 0:
        return PermutationResult(observed_mean_diff=0.0, p_value=1.0, significant=False)

    std = float(deltas.std(ddof=1)) if n > 1 else 0.0
    effect_size = round(observed_mean / std, 6) if std > 0 else None

    rng = np.random.default_rng(seed)
    signs = rng.choice([-1, 1], size=(n_permutations, n))
    perm_means = (signs * deltas).mean(axis=1)

    p_value = float((perm_means >= observed_mean).sum() + 1) / (n_permutations + 1)

    return PermutationResult(
        observed_mean_diff=round(observed_mean, 6),
        p_value=round(p_value, 6),
        significant=p_value < alpha,
        n_permutations=n_permutations,
        effect_size=effect_size,
    )


def detect_test(
    base_records: dict[tuple[int, int], dict[str, Any]],
    cand_records: dict[tuple[int, int], dict[str, Any]],
) -> Literal["mcnemar", "permutation"]:
    """Auto-detect the appropriate statistical test based on data shape.

    - All rewards are 0.0 or 1.0 -> mcnemar (paired binary)
    - Any non-binary rewards -> permutation (continuous deltas)

    Note: ``"sign"`` test is available via explicit ``test="sign"`` but is
    never auto-selected; it is a manual option for callers who prefer it.
    """
    paired_keys = set(base_records) & set(cand_records)
    for key in paired_keys:
        b_reward = float(base_records[key].get("reward", 0))
        c_reward = float(cand_records[key].get("reward", 0))
        if b_reward not in (0.0, 1.0) or c_reward not in (0.0, 1.0):
            return "permutation"
    return "mcnemar"


def mde_estimate(n_discordant: int) -> float:
    """Minimum detectable effect at 80% power for one-sided binomial."""
    if n_discordant <= 0:
        return 1.0
    return POWER_80_FACTOR / math.sqrt(max(1, n_discordant))
