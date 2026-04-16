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
"""Tests for sample_level_ci — within-problem binomial variance CI for pass@1."""

import math

import numpy as np

from nemo_evaluator.metrics.confidence import sample_level_ci


class TestSampleLevelCIReturnsNone:
    """Cases where the CI cannot be computed."""

    def test_empty_list(self):
        assert sample_level_ci([]) is None

    def test_single_repeat_all_problems(self):
        """With n=1 per problem, within-problem variance is not estimable."""
        results = [(1, 1), (1, 0), (1, 1), (1, 0)]
        assert sample_level_ci(results) is None

    def test_single_repeat_single_problem(self):
        assert sample_level_ci([(1, 0)]) is None

    def test_single_repeat_large_set(self):
        results = [(1, 1)] * 250 + [(1, 0)] * 250
        assert sample_level_ci(results) is None


class TestSampleLevelCIBasic:
    """Basic correctness checks."""

    def test_returns_confidence_interval(self):
        results = [(3, 1), (3, 2), (3, 0), (3, 3)]
        ci = sample_level_ci(results)
        assert ci is not None
        assert ci.method == "sample_binomial"
        assert ci.confidence == 0.95

    def test_mean_is_pass_at_1(self):
        """Mean should equal the average of c_i/n_i across problems."""
        results = [(3, 1), (3, 2), (3, 0), (3, 3)]
        ci = sample_level_ci(results)
        expected_mean = (1 / 3 + 2 / 3 + 0 / 3 + 3 / 3) / 4
        assert ci is not None
        assert abs(ci.mean - expected_mean) < 1e-10

    def test_ci_ordering(self):
        results = [(5, 2)] * 100 + [(5, 3)] * 100
        ci = sample_level_ci(results)
        assert ci is not None
        assert ci.ci_lower <= ci.mean <= ci.ci_upper

    def test_ci_contains_mean(self):
        results = [(3, 1), (3, 2), (3, 3), (3, 0)] * 50
        ci = sample_level_ci(results)
        assert ci is not None
        assert ci.ci_lower < ci.mean < ci.ci_upper


class TestSampleLevelCIZeroVariance:
    """Problems that are all-pass or all-fail contribute zero within-problem variance."""

    def test_all_correct_all_problems(self):
        """Every problem solved on every attempt — SE is zero, CI collapses."""
        results = [(3, 3)] * 100
        ci = sample_level_ci(results)
        assert ci is not None
        assert ci.mean == 1.0
        assert ci.ci_lower == 1.0
        assert ci.ci_upper == 1.0

    def test_all_wrong_all_problems(self):
        results = [(3, 0)] * 100
        ci = sample_level_ci(results)
        assert ci is not None
        assert ci.mean == 0.0
        assert ci.ci_lower == 0.0
        assert ci.ci_upper == 0.0

    def test_mix_of_deterministic_problems(self):
        """All problems are either 0/3 or 3/3 — no within-problem variance."""
        results = [(3, 3)] * 60 + [(3, 0)] * 40
        ci = sample_level_ci(results)
        assert ci is not None
        assert ci.mean == 0.6
        assert ci.ci_lower == ci.ci_upper == ci.mean


class TestSampleLevelCIMath:
    """Verify the exact SE formula against hand-computed values."""

    def test_uniform_problems_exact_se(self):
        """All problems have same (n, c) so we can compute SE analytically."""
        n, c, n_problems = 5, 2, 200
        results = [(n, c)] * n_problems

        p_hat = c / n
        # Unbiased var per problem: p_hat*(1-p_hat) / (n-1)
        var_per_problem = p_hat * (1 - p_hat) / (n - 1)
        expected_se = math.sqrt(n_problems * var_per_problem) / n_problems

        ci = sample_level_ci(results)
        assert ci is not None

        # Recover SE from the CI width: CI = mean ± z*SE, z_{0.025} ≈ 1.96
        z = 1.959964
        actual_se = (ci.ci_upper - ci.ci_lower) / (2 * z)
        assert abs(actual_se - expected_se) < 1e-8

    def test_two_problem_types_exact(self):
        """50 problems with (4, 1) and 50 with (4, 3). Verify SE."""
        results = [(4, 1)] * 50 + [(4, 3)] * 50
        n_problems = 100

        # p_hat values: 0.25 and 0.75
        var_1 = 0.25 * 0.75 / 3  # (n-1)=3
        var_2 = 0.75 * 0.25 / 3
        total_var = 50 * var_1 + 50 * var_2
        expected_se = math.sqrt(total_var) / n_problems

        ci = sample_level_ci(results)
        assert ci is not None

        z = 1.959964
        actual_se = (ci.ci_upper - ci.ci_lower) / (2 * z)
        assert abs(actual_se - expected_se) < 1e-8


class TestSampleLevelCIScaling:
    """CI should shrink with more repeats and more problems."""

    def test_more_repeats_tighter_ci(self):
        """Doubling repeats per problem should shrink the CI."""
        rng = np.random.default_rng(42)
        true_p = rng.uniform(0.2, 0.8, size=200)

        def make_results(n_repeats):
            return [(n_repeats, int(round(p * n_repeats))) for p in true_p]

        ci_3 = sample_level_ci(make_results(3))
        ci_10 = sample_level_ci(make_results(10))
        assert ci_3 is not None and ci_10 is not None

        width_3 = ci_3.ci_upper - ci_3.ci_lower
        width_10 = ci_10.ci_upper - ci_10.ci_lower
        assert width_10 < width_3

    def test_more_problems_tighter_ci(self):
        """More problems with the same per-problem stats should shrink the CI."""
        base = [(5, 2)]
        ci_50 = sample_level_ci(base * 50)
        ci_500 = sample_level_ci(base * 500)
        assert ci_50 is not None and ci_500 is not None

        width_50 = ci_50.ci_upper - ci_50.ci_lower
        width_500 = ci_500.ci_upper - ci_500.ci_lower
        # Should scale as 1/sqrt(n_problems), so ~3.16x tighter
        assert width_500 < width_50
        ratio = width_50 / width_500
        assert abs(ratio - math.sqrt(10)) < 0.5  # sqrt(500/50) ≈ 3.16

    def test_scaling_with_repeats_follows_sqrt_law(self):
        """SE should scale as 1/sqrt(r) for fixed per-problem difficulty."""
        results_3 = [(3, 1)] * 300
        results_12 = [(12, 4)] * 300  # Same p_hat=1/3, 4x repeats

        ci_3 = sample_level_ci(results_3)
        ci_12 = sample_level_ci(results_12)
        assert ci_3 is not None and ci_12 is not None

        width_3 = ci_3.ci_upper - ci_3.ci_lower
        width_12 = ci_12.ci_upper - ci_12.ci_lower
        ratio = width_3 / width_12
        # Expected: sqrt(12/3) * sqrt((12-1)/(3-1)) correction ≈ 2 * sqrt(5.5/2)
        # Simpler: var_3 = p(1-p)/(3-1), var_12 = p(1-p)/(12-1)
        # ratio of SE = sqrt((3-1)/(12-1)) = sqrt(2/11) → ratio of widths = sqrt(11/2) ≈ 2.35
        expected_ratio = math.sqrt(11 / 2)
        assert abs(ratio - expected_ratio) < 0.1


class TestSampleLevelCIHeterogeneous:
    """Problems with different numbers of attempts (partial shards, etc.)."""

    def test_mixed_n_values(self):
        """Some problems have 3 attempts, some have 5, some have 1."""
        results = [(3, 1), (5, 3), (1, 0), (3, 2), (5, 1), (1, 1)]
        ci = sample_level_ci(results)
        assert ci is not None

        # n=1 problems should be skipped for variance, but included in mean
        expected_mean = (1 / 3 + 3 / 5 + 0 / 1 + 2 / 3 + 1 / 5 + 1 / 1) / 6
        assert abs(ci.mean - expected_mean) < 1e-10

    def test_single_estimable_problem(self):
        """Only one problem with n>1 — still computable."""
        results = [(1, 0), (1, 1), (3, 1), (1, 0)]
        ci = sample_level_ci(results)
        assert ci is not None
        assert ci.ci_lower < ci.mean < ci.ci_upper


class TestSampleLevelCIConfidenceLevel:
    """Custom confidence levels."""

    def test_wider_at_99_than_95(self):
        results = [(5, 2)] * 200
        ci_95 = sample_level_ci(results, confidence=0.95)
        ci_99 = sample_level_ci(results, confidence=0.99)
        assert ci_95 is not None and ci_99 is not None

        width_95 = ci_95.ci_upper - ci_95.ci_lower
        width_99 = ci_99.ci_upper - ci_99.ci_lower
        assert width_99 > width_95

    def test_narrower_at_90_than_95(self):
        results = [(5, 2)] * 200
        ci_90 = sample_level_ci(results, confidence=0.90)
        ci_95 = sample_level_ci(results, confidence=0.95)
        assert ci_90 is not None and ci_95 is not None

        width_90 = ci_90.ci_upper - ci_90.ci_lower
        width_95 = ci_95.ci_upper - ci_95.ci_lower
        assert width_90 < width_95


class TestSampleLevelCIvsBootstrap:
    """Sample-level CI should be tighter than bootstrap CI for pass@1."""

    def test_sample_ci_tighter_than_bootstrap(self):
        """On a heterogeneous problem set, sample CI should be tighter because
        it doesn't include between-problem variance."""
        from nemo_evaluator.metrics.confidence import bootstrap_ci
        from nemo_evaluator.metrics.pass_at_k import pass_at_k

        rng = np.random.default_rng(99)
        n_problems, n_repeats = 500, 3
        true_p = rng.uniform(0.0, 1.0, size=n_problems)
        results = [(n_repeats, int(rng.binomial(n_repeats, p))) for p in true_p]

        sci = sample_level_ci(results)
        bci = bootstrap_ci([pass_at_k(n, c, 1) for n, c in results])

        assert sci is not None
        sample_width = sci.ci_upper - sci.ci_lower
        bootstrap_width = bci.ci_upper - bci.ci_lower
        assert sample_width < bootstrap_width


class TestSampleLevelCIRealistic:
    """Realistic SWE-bench-like scenario."""

    def test_swebench_500_problems_3_repeats(self):
        """Simulate a SWE-bench run: 500 problems, 3 repeats, ~65% pass rate."""
        rng = np.random.default_rng(2026)
        n_repeats = 3
        # Bimodal: some easy (p~0.9), some hard (p~0.1), some medium (p~0.5)
        true_p = np.concatenate(
            [
                rng.beta(9, 1, size=200),  # easy: mean ~0.9
                rng.beta(1, 9, size=150),  # hard: mean ~0.1
                rng.beta(5, 5, size=150),  # medium: mean ~0.5
            ]
        )
        results = [(n_repeats, int(rng.binomial(n_repeats, p))) for p in true_p]

        ci = sample_level_ci(results)
        assert ci is not None

        # CI width should be roughly 2-4pp (much less than bootstrap ~8pp)
        width = ci.ci_upper - ci.ci_lower
        assert 0.01 < width < 0.06  # sanity: 1-6pp

        # Mean should be reasonable
        assert 0.3 < ci.mean < 0.7


class TestSampleLevelCIMonteCarloCoverage:
    """Gold-standard validation: simulate many runs, check that the 95% CI
    covers the true mean ~95% of the time."""

    def test_coverage_uniform_difficulty(self):
        """500 problems, 3 repeats, uniform difficulty spread."""
        rng = np.random.default_rng(12345)
        n_problems, n_repeats, n_simulations = 200, 3, 2000
        confidence = 0.95

        true_p = rng.uniform(0.1, 0.9, size=n_problems)
        true_mean = true_p.mean()

        covers = 0
        for _ in range(n_simulations):
            results = [(n_repeats, int(rng.binomial(n_repeats, p))) for p in true_p]
            ci = sample_level_ci(results, confidence=confidence)
            assert ci is not None
            if ci.ci_lower <= true_mean <= ci.ci_upper:
                covers += 1

        coverage = covers / n_simulations
        # With 2000 sims, SE of coverage estimate ≈ sqrt(0.95*0.05/2000) ≈ 0.5%
        # Accept 93%–97% (±2pp from 95%).
        assert 0.93 <= coverage <= 0.97, f"Coverage {coverage:.3f} outside [0.93, 0.97]"

    def test_coverage_bimodal_difficulty(self):
        """SWE-bench-like bimodal problem difficulty."""
        rng = np.random.default_rng(54321)
        n_repeats, n_simulations = 5, 2000
        confidence = 0.95

        true_p = np.concatenate(
            [
                rng.beta(8, 2, size=100),  # easy
                rng.beta(2, 8, size=100),  # hard
            ]
        )
        true_mean = true_p.mean()

        covers = 0
        for _ in range(n_simulations):
            results = [(n_repeats, int(rng.binomial(n_repeats, p))) for p in true_p]
            ci = sample_level_ci(results, confidence=confidence)
            assert ci is not None
            if ci.ci_lower <= true_mean <= ci.ci_upper:
                covers += 1

        coverage = covers / n_simulations
        assert 0.93 <= coverage <= 0.97, f"Coverage {coverage:.3f} outside [0.93, 0.97]"
