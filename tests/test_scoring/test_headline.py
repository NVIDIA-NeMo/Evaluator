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
"""Tests for :mod:`nemo_evaluator.metrics.headline`."""

import pytest

from nemo_evaluator.metrics.headline import headline_score_metrics, is_fractional


class TestIsFractional:
    def test_all_zero_one(self):
        assert not is_fractional([0.0, 1.0, 0, 1, 0.0, 1.0])

    def test_empty(self):
        assert not is_fractional([])

    def test_single_fractional(self):
        # A single 0.5 is enough to flip the whole run to fractional.
        assert is_fractional([0.0, 1.0, 0.5, 0.0])

    def test_all_fractional(self):
        assert is_fractional([0.1, 0.2, 0.3])


class TestHeadlineScoreMetrics:
    def test_empty_input(self):
        assert headline_score_metrics({}, n_repeats=1) == {}

    def test_all_empty_problem_lists(self):
        assert headline_score_metrics({0: [], 1: []}, n_repeats=1) == {}

    def test_binary_rewards_emit_pass_at_k(self):
        # 4 problems, 2 repeats, pass counts = [2, 1, 0, 2]
        per_problem = {0: [1.0, 1.0], 1: [1.0, 0.0], 2: [0.0, 0.0], 3: [1.0, 1.0]}
        metrics = headline_score_metrics(per_problem, n_repeats=2)

        assert "mean_reward" not in metrics
        assert "pass@1" in metrics
        assert "pass@2" in metrics
        # pass@1 = mean over problems of c/n: (2/2 + 1/2 + 0/2 + 2/2) / 4 = 0.625
        assert metrics["pass@1"]["value"] == pytest.approx(0.625, abs=1e-4)
        # pass@2 with n=c=2 ⇒ 1.0; with c=1, n=2 ⇒ 1 - C(1,2)/C(2,2) = undefined,
        # but pass_at_k returns 1.0 when c>=1 and n==k because 1 - 0/1.
        # Problems: (2,2)->1, (2,1)->1, (2,0)->0, (2,2)->1 ⇒ mean = 0.75
        assert metrics["pass@2"]["value"] == pytest.approx(0.75, abs=1e-4)

    def test_fractional_rewards_emit_mean_reward(self):
        # Per-problem means: {0.75, 0.55, 0.5, 0.35} ⇒ overall mean 0.5375
        per_problem = {0: [0.7, 0.8], 1: [0.5, 0.6], 2: [0.0, 1.0], 3: [0.3, 0.4]}
        metrics = headline_score_metrics(per_problem, n_repeats=2)

        assert "pass@1" not in metrics
        assert "pass@2" not in metrics
        assert metrics["mean_reward"]["value"] == pytest.approx(0.5375, abs=1e-4)
        assert "ci_lower" in metrics["mean_reward"]
        assert "ci_upper" in metrics["mean_reward"]

    def test_single_fractional_sample_flips_the_run(self):
        # 99 binary samples + 1 fractional ⇒ whole run reported as mean_reward.
        # Protects PinchBench-style rubrics from the inflated pass@1 that the
        # ``reward > 0`` threshold would produce.
        per_problem: dict[int, list[float]] = {i: [1.0, 0.0] for i in range(49)}
        per_problem[49] = [1.0, 0.5]
        metrics = headline_score_metrics(per_problem, n_repeats=2)

        assert "mean_reward" in metrics
        assert "pass@1" not in metrics
        assert "pass@2" not in metrics

    def test_partial_shard_with_fewer_attempts_than_k(self):
        # Simulates partial shards: one problem only has 1 attempt but
        # n_repeats=2. pass@2 would need n>=2; that problem is filtered out
        # of the pass@2 computation but still contributes to pass@1.
        per_problem = {0: [1.0, 0.0], 1: [1.0], 2: [0.0, 1.0]}
        metrics = headline_score_metrics(per_problem, n_repeats=2)

        assert "pass@1" in metrics
        assert "pass@2" in metrics
        # pass@1 across all 3 problems: (1/2 + 1/1 + 1/2) / 3 ≈ 0.6667
        assert metrics["pass@1"]["value"] == pytest.approx(2.0 / 3.0, abs=1e-4)

    def test_single_repeat_only_emits_pass_at_1(self):
        per_problem = {0: [1.0], 1: [0.0], 2: [1.0]}
        metrics = headline_score_metrics(per_problem, n_repeats=1)

        assert "pass@1" in metrics
        assert "pass@2" not in metrics
        assert metrics["pass@1"]["value"] == pytest.approx(2.0 / 3.0, abs=1e-4)
