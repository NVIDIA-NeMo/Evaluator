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

"""Score aggregation logic for BYOB benchmarks.

Extracted from runner.py to break the circular dependency with eval_logic.py
and to provide a single, well-tested aggregation path for both subprocess
and native execution modes.
"""

import math
from typing import Dict, List


def aggregate_scores(scores: List[Dict], benchmark_name: str) -> Dict:
    """Aggregate per-sample scores into summary statistics.

    This is the core aggregation logic:
    1. Collect all numeric keys (bool, int, float) across all sample dicts.
    2. For each key:
       - Convert booleans to 0.0/1.0.
       - Compute: mean, population variance (/n), stddev, stderr = stddev / sqrt(n).
       - If n <= 1: variance=0, stderr=0.
       - Detect binary: is_binary = all(v in (0.0, 1.0) for v in values).
       - Binary metrics: scale display values by 100 (percentage).
       - Round all values to 4 decimal places.
    3. Output structure conforms to engine expectations.

    Args:
        scores: List of score dictionaries from scorer function.
        benchmark_name: Name of benchmark (used as task name in output).

    Returns:
        Nested dict with structure:
        {
            "tasks": {
                "<benchmark_name>": {
                    "metrics": {
                        "pass@1": {
                            "scores": {
                                "<key>": {
                                    "value": N,
                                    "count": N,
                                    "mean": N,
                                    "stderr": N,
                                    "stddev": N
                                }
                            }
                        }
                    }
                }
            }
        }
    """
    if not scores:
        return {}

    # Collect all numeric keys across all samples
    all_keys = set()
    for score_dict in scores:
        for key, value in score_dict.items():
            if isinstance(value, (bool, int, float)):
                all_keys.add(key)

    # Compute statistics for each key
    aggregated_scores = {}
    for key in sorted(all_keys):
        # Extract values for this key, converting booleans to 0.0/1.0
        values = []
        for score_dict in scores:
            if key in score_dict:
                val = score_dict[key]
                if isinstance(val, bool):
                    values.append(1.0 if val else 0.0)
                elif isinstance(val, (int, float)):
                    values.append(float(val))

        if not values:
            continue

        n = len(values)
        mean_val = sum(values) / n

        # Population variance (divide by n, not n-1)
        if n <= 1:
            variance = 0.0
            stddev = 0.0
            stderr = 0.0
        else:
            variance = sum((v - mean_val) ** 2 for v in values) / n
            stddev = math.sqrt(variance)
            stderr = stddev / math.sqrt(n)

        # Detect binary (all values in {0.0, 1.0})
        is_binary = all(v in (0.0, 1.0) for v in values)

        # Scale binary metrics to percentage (0-100)
        if is_binary:
            display_value = mean_val * 100
            display_mean = mean_val * 100
            display_stderr = stderr * 100
            display_stddev = stddev * 100
        else:
            display_value = mean_val
            display_mean = mean_val
            display_stderr = stderr
            display_stddev = stddev

        # Round to 4 decimal places
        aggregated_scores[key] = {
            "value": round(display_value, 4),
            "count": n,
            "mean": round(display_mean, 4),
            "stderr": round(display_stderr, 4),
            "stddev": round(display_stddev, 4),
        }

    return {
        "tasks": {
            benchmark_name: {"metrics": {"pass@1": {"scores": aggregated_scores}}}
        }
    }
