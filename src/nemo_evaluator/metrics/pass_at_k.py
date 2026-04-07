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
"""pass@k metric from the Codex paper (Chen et al., 2021)."""

from __future__ import annotations

import math


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator of pass@k.

    Args:
        n: total attempts per problem
        c: number of correct attempts
        k: the k in pass@k

    Returns:
        Probability that at least one of k random draws from n attempts is correct.
    """
    if n < k:
        raise ValueError(f"n ({n}) must be >= k ({k})")
    if c < 0 or c > n:
        raise ValueError(f"c ({c}) must be in [0, n={n}]")
    if c == 0:
        return 0.0
    if c >= n:
        return 1.0

    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def aggregate_pass_at_k(problem_results: list[tuple[int, int]], k: int) -> float:
    """Average pass@k across problems.

    Args:
        problem_results: list of (n_attempts, n_correct) per problem
        k: the k in pass@k

    Returns:
        Mean pass@k across all problems.
    """
    if not problem_results:
        return 0.0
    values = [pass_at_k(n, c, k) for n, c in problem_results]
    return sum(values) / len(values)
