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

"""Tests for nemo-skills metrics adapter."""

import pytest

from nemo_evaluator.api.api_dataclasses import EvaluationResult, GroupResult, TaskResult
from nemo_evaluator.plugins.nemo_skills.metrics_adapter import translate


class TestTranslationLogic:
    """Tests for translate function (T-049 through T-054, T-099 through T-101)."""

    def test_t049_translate_basic_all_mapping(self):
        """T-049: Basic _all_ mapping to TaskResult with percentage-scale values (AC-019, INV-005, OQR-002)."""
        ns_metrics = {
            "_all_": {
                "greedy": {
                    "symbolic_correct": 85.0,
                    "num_entries": 100,
                }
            }
        }
        result = translate(ns_metrics, "gsm8k")
        assert isinstance(result, EvaluationResult)
        assert "gsm8k" in result.tasks
        task_result = result.tasks["gsm8k"]
        assert "greedy" in task_result.metrics
        metric = task_result.metrics["greedy"]
        assert "symbolic_correct" in metric.scores
        assert metric.scores["symbolic_correct"].value == 85.0

    def test_t050_translate_statistics_scaling(self):
        """T-050: Statistics fields scaled by 100 to percentage scale (AC-019, INV-005, OQR-002)."""
        ns_metrics = {
            "_all_": {
                "greedy": {
                    "symbolic_correct": 85.0,
                    "symbolic_correct_statistics": {
                        "avg": 0.85,
                        "std_err_across_runs": 0.02,
                        "std_dev_across_runs": 0.05,
                    },
                    "num_entries": 100,
                }
            }
        }
        result = translate(ns_metrics, "gsm8k")
        score = result.tasks["gsm8k"].metrics["greedy"].scores["symbolic_correct"]
        assert score.value == 85.0
        assert score.stats.mean == 85.0  # 0.85 * 100
        assert score.stats.stderr == 2.0  # 0.02 * 100
        assert score.stats.stddev == 5.0  # 0.05 * 100

    def test_t051_translate_scaling_invariant(self):
        """T-051: Complete scaling invariant: all stats in percentage scale, count not scaled (AC-020, INV-005)."""
        ns_metrics = {
            "_all_": {
                "greedy": {
                    "symbolic_correct": 85.0,
                    "symbolic_correct_statistics": {
                        "avg": 0.85,
                        "std_err_across_runs": 0.02,
                        "std_dev_across_runs": 0.05,
                    },
                    "num_entries": 100,
                }
            }
        }
        result = translate(ns_metrics, "gsm8k")
        score = result.tasks["gsm8k"].metrics["greedy"].scores["symbolic_correct"]
        # Stats in percentage scale
        assert score.stats.mean == 85.0
        assert score.stats.stderr == 2.0
        assert score.stats.stddev == 5.0
        # Count NOT scaled
        assert score.stats.count == 100

    def test_t052_translate_named_subsets_to_group_result(self):
        """T-052: Named subsets map to GroupResult (AC-021, DM-007)."""
        ns_metrics = {
            "_all_": {
                "greedy": {
                    "symbolic_correct": 85.0,
                    "num_entries": 100,
                }
            },
            "algebra": {
                "greedy": {
                    "symbolic_correct": 90.0,
                    "num_entries": 50,
                }
            },
        }
        result = translate(ns_metrics, "gsm8k")
        assert "gsm8k" in result.groups
        assert "algebra" in result.groups["gsm8k"].groups
        algebra_group = result.groups["gsm8k"].groups["algebra"]
        assert "greedy" in algebra_group.metrics
        assert algebra_group.metrics["greedy"].scores["symbolic_correct"].value == 90.0

    def test_t053_translate_count_field_exclusion(self):
        """T-053: Count fields (num_entries, num_prompts, num_instructions) excluded from scores (INV-006, R-023)."""
        ns_metrics = {
            "_all_": {
                "greedy": {
                    "symbolic_correct": 85.0,
                    "num_entries": 100,
                    "num_prompts": 100,
                    "num_instructions": 200,
                }
            }
        }
        result = translate(ns_metrics, "gsm8k")
        scores = result.tasks["gsm8k"].metrics["greedy"].scores
        assert "num_entries" not in scores
        assert "num_prompts" not in scores
        assert "num_instructions" not in scores
        assert "symbolic_correct" in scores

    def test_t054_translate_dict_and_non_numeric_exclusion(self):
        """T-054: Dict-typed and non-numeric values excluded from scores (INV-006)."""
        ns_metrics = {
            "_all_": {
                "greedy": {
                    "symbolic_correct": 85.0,
                    "some_dict_value": {"nested": "data"},
                    "some_string_value": "not a number",
                    "num_entries": 100,
                }
            }
        }
        result = translate(ns_metrics, "gsm8k")
        scores = result.tasks["gsm8k"].metrics["greedy"].scores
        assert "some_dict_value" not in scores
        assert "some_string_value" not in scores
        assert "symbolic_correct" in scores

    def test_t099_translate_missing_all_key_raises_value_error(self):
        """T-099: Missing '_all_' key raises ValueError with descriptive message."""
        ns_metrics = {"algebra": {"greedy": {"score": 90.0}}}
        with pytest.raises(ValueError, match="_all_"):
            translate(ns_metrics, "gsm8k")

    def test_t100_translate_empty_all_raises_value_error(self):
        """T-100: Empty '_all_' dict raises ValueError."""
        ns_metrics = {"_all_": {}}
        with pytest.raises(ValueError, match="_all_"):
            translate(ns_metrics, "gsm8k")

    def test_t101_translate_multiple_aggregation_modes(self):
        """T-101: Multiple aggregation modes (pass@1, majority@8) map to distinct MetricResult entries."""
        ns_metrics = {
            "_all_": {
                "pass@1": {
                    "symbolic_correct": 70.0,
                    "num_entries": 100,
                },
                "majority@8": {
                    "symbolic_correct": 85.0,
                    "num_entries": 100,
                },
            }
        }
        result = translate(ns_metrics, "gsm8k")
        metrics = result.tasks["gsm8k"].metrics
        assert "pass@1" in metrics
        assert "majority@8" in metrics
        assert metrics["pass@1"].scores["symbolic_correct"].value == 70.0
        assert metrics["majority@8"].scores["symbolic_correct"].value == 85.0
