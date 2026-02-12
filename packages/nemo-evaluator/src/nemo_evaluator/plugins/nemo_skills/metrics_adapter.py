# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Metrics adapter: translates nemo-skills metrics to EvaluationResult."""

from typing import Any, Dict, Optional

from nemo_evaluator.api.api_dataclasses import (
    EvaluationResult,
    GroupResult,
    MetricResult,
    Score,
    ScoreStats,
    TaskResult,
)
from nemo_evaluator.logging import get_logger

logger = get_logger(__name__)

# Count fields to exclude per INV-006
COUNT_FIELDS = {"num_entries", "num_prompts", "num_instructions"}


def translate(
    ns_metrics: Dict[str, Any],
    benchmark_name: str,
    config: Optional[Dict[str, Any]] = None,
) -> EvaluationResult:
    """Translates nemo-skills metrics to EvaluationResult.

    Per C-032: maps "_all_" subsets to TaskResult entries keyed by benchmark_name.
    Maps named subsets to nested GroupResult entries. Maps aggregation modes to
    distinct MetricResult entries. Skips count fields per INV-006.

    Per INV-005: Score.value stores as-is (already percentage), ScoreStats fields
    are scaled to percentage (0-100). ScoreStats.count is never scaled.

    Args:
        ns_metrics: nemo-skills metrics dictionary
        benchmark_name: Benchmark identifier
        config: Optional configuration (unused)

    Returns:
        EvaluationResult with tasks and optional groups

    Raises:
        ValueError: If ns_metrics does not contain "_all_" key or it's empty
    """
    if "_all_" not in ns_metrics or not ns_metrics["_all_"]:
        raise ValueError("ns_metrics must contain non-empty '_all_' key")

    result = EvaluationResult(tasks={}, groups={})

    # Process _all_ metrics -> TaskResult
    all_metrics = ns_metrics["_all_"]
    task_result = TaskResult(metrics={})

    for agg_mode, agg_data in all_metrics.items():
        if not isinstance(agg_data, dict):
            continue

        metric_result = _translate_scores(agg_data, agg_mode)
        task_result.metrics[agg_mode] = metric_result

    result.tasks[benchmark_name] = task_result

    # Process named subsets -> GroupResult
    for subset_name, subset_data in ns_metrics.items():
        if subset_name in ["_all_", "benchmark_name", "config"]:
            continue
        if not isinstance(subset_data, dict):
            continue

        # Create GroupResult for this subset
        if benchmark_name not in result.groups:
            result.groups[benchmark_name] = GroupResult(groups={}, metrics={})

        subset_group = GroupResult(groups={}, metrics={})
        for agg_mode, agg_data in subset_data.items():
            if not isinstance(agg_data, dict):
                continue
            metric_result = _translate_scores(agg_data, agg_mode)
            subset_group.metrics[agg_mode] = metric_result

        result.groups[benchmark_name].groups[subset_name] = subset_group

    return result


def _translate_scores(agg_data: Dict[str, Any], agg_mode: str) -> MetricResult:
    """Translates a single aggregation mode's data to MetricResult.

    Per INV-005: Score.value is stored as-is (nemo-skills outputs percentages).
    ScoreStats fields (mean, stderr, stddev) are scaled to percentage if they
    are in fraction scale (0.0-1.0). Count is never scaled.

    Per INV-006: Skips count fields, dict values, and non-numeric values.

    Args:
        agg_data: Aggregation data dictionary
        agg_mode: Aggregation mode name (e.g., "greedy", "pass@1")

    Returns:
        MetricResult with scores
    """
    metric_result = MetricResult(scores={})
    count_value = agg_data.get("num_entries")

    for key, value in agg_data.items():
        # Skip count fields per INV-006
        if key in COUNT_FIELDS:
            continue
        # Skip dict values
        if isinstance(value, dict):
            continue
        # Skip non-numeric values
        if not isinstance(value, (int, float)):
            continue

        # Check for statistics sub-dict
        stats_key = f"{key}_statistics"
        if stats_key in agg_data and isinstance(agg_data[stats_key], dict):
            stats_data = agg_data[stats_key]

            # Extract statistics with scaling per INV-005
            mean = stats_data.get("avg")
            stderr = stats_data.get("std_err_across_runs")
            stddev = stats_data.get("std_dev_across_runs")

            # Scale if in fraction range (0.0-1.0)
            if mean is not None and 0.0 <= mean <= 1.0:
                mean = mean * 100
            if stderr is not None and 0.0 <= stderr <= 1.0:
                stderr = stderr * 100
            if stddev is not None and 0.0 <= stddev <= 1.0:
                stddev = stddev * 100

            stats = ScoreStats(
                count=count_value,
                sum=None,
                sum_squared=None,
                min=None,
                max=None,
                mean=mean,
                variance=None,
                stddev=stddev,
                stderr=stderr,
            )
        else:
            # No statistics dict - create minimal stats with count only
            stats = ScoreStats(count=count_value)

        # Create Score with stats
        score = Score(value=float(value), stats=stats)

        metric_result.scores[key] = score

    return metric_result
