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
"""Headline scoring metrics shared by single-shard and merge paths.

Factored out of :mod:`nemo_evaluator.engine.eval_loop` and
:mod:`nemo_evaluator.engine.sharding` so both callsites compute the
bundle's primary ``scores`` entries the same way.

Decision rule: a run is "fractional" if any sample reward is outside
``{0.0, 1.0}``. Fractional runs emit ``mean_reward`` (bootstrap CI over
per-problem means); binary runs emit ``pass@1`` (and ``pass@n_repeats``
when ``n_repeats > 1``), each with a sample-level CI for ``k == 1`` and
a bootstrap CI over per-problem ``pass@k`` values.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping, Sequence
from typing import Any

from nemo_evaluator.metrics.confidence import bootstrap_ci, sample_level_ci
from nemo_evaluator.metrics.pass_at_k import aggregate_pass_at_k, pass_at_k

_BINARY_REWARDS: frozenset[float | int] = frozenset({0, 0.0, 1, 1.0})


def is_fractional(rewards: Iterable[float]) -> bool:
    """Return True iff any reward is outside ``{0.0, 1.0}``.

    A single fractional sample flips the whole run to ``mean_reward``
    semantics; callers may record a tail-sample flag on an observer and
    promote it through to :func:`headline_score_metrics`.
    """
    return any(r not in _BINARY_REWARDS for r in rewards)


def headline_score_metrics(
    per_problem_rewards: Mapping[Hashable, Sequence[float]],
    n_repeats: int,
) -> dict[str, Any]:
    """Compute ``mean_reward`` or ``pass@k`` entries for a ``scores`` dict.

    Args:
        per_problem_rewards: mapping from problem id to the list of
            reward values observed for that problem (one per repeat or
            per shard replay).
        n_repeats: configured repeats per problem. For the merge path
            actual per-problem repeat counts can be lower (partial
            shards); those are handled by filtering ``(n, c)`` tuples
            with ``n >= k`` for ``pass@k``.

    Returns:
        Dict with either a single ``mean_reward`` entry (fractional
        rewards) or ``pass@1`` and optionally ``pass@n_repeats`` entries
        (binary rewards). Empty input yields ``{}``.
    """
    if not per_problem_rewards:
        return {}

    all_rewards: list[float] = [float(r) for rewards in per_problem_rewards.values() for r in rewards]
    if not all_rewards:
        return {}

    metrics: dict[str, Any] = {}
    if is_fractional(all_rewards):
        per_problem_means = [sum(rs) / len(rs) for rs in per_problem_rewards.values() if rs]
        if per_problem_means:
            ci = bootstrap_ci(per_problem_means)
            metrics["mean_reward"] = {
                "value": round(ci.mean, 4),
                "ci_lower": round(ci.ci_lower, 4),
                "ci_upper": round(ci.ci_upper, 4),
            }
        return metrics

    problem_list: list[tuple[int, int]] = [
        (len(rewards), sum(1 for r in rewards if r > 0)) for rewards in per_problem_rewards.values() if rewards
    ]
    ks = [1] + ([n_repeats] if n_repeats > 1 else [])
    for k in ks:
        valid = [(n, c) for n, c in problem_list if n >= k]
        if not valid:
            continue
        entry: dict[str, Any] = {"value": round(aggregate_pass_at_k(valid, k), 4)}
        if k == 1:
            sci = sample_level_ci(valid)
            if sci is not None:
                entry["ci_lower"] = round(sci.ci_lower, 4)
                entry["ci_upper"] = round(sci.ci_upper, 4)
        bci = bootstrap_ci([pass_at_k(n, c, k) for n, c in valid])
        entry["bootstrap_ci_lower"] = round(bci.ci_lower, 4)
        entry["bootstrap_ci_upper"] = round(bci.ci_upper, 4)
        metrics[f"pass@{k}"] = entry

    return metrics
