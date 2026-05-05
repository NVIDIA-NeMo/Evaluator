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
import numpy as np
import pytest

from nemo_evaluator.metrics.confidence import bootstrap_ci, clustered_ci, normal_ci


class TestBootstrapCI:
    def test_all_ones_collapses_ci(self):
        ci = bootstrap_ci([1.0] * 100)
        assert ci.mean == 1.0
        assert ci.ci_lower == 1.0
        assert ci.ci_upper == 1.0

    def test_all_zeros_collapses_ci(self):
        ci = bootstrap_ci([0.0] * 100)
        assert ci.mean == 0.0
        assert ci.ci_lower == 0.0
        assert ci.ci_upper == 0.0

    def test_mixed_scores_produce_nontrivial_interval(self):
        ci = bootstrap_ci([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
        assert ci.mean == 0.5
        assert ci.ci_lower < ci.mean < ci.ci_upper

    def test_deterministic_with_seed(self):
        scores = [0.5, 0.7, 0.3, 0.6, 0.4]
        ci1 = bootstrap_ci(scores, seed=42)
        ci2 = bootstrap_ci(scores, seed=42)
        assert ci1.ci_lower == ci2.ci_lower
        assert ci1.ci_upper == ci2.ci_upper

    def test_single_sample_degenerates_to_point(self):
        ci = bootstrap_ci([0.5])
        assert ci.mean == 0.5
        assert ci.ci_lower == ci.ci_upper == 0.5

    def test_ci_respects_ordering(self):
        ci = bootstrap_ci([0.8, 0.7, 0.9, 0.85, 0.75, 0.6, 0.95])
        assert ci.ci_lower <= ci.mean <= ci.ci_upper


class TestNormalCI:
    def test_basic(self):
        ci = normal_ci([1.0, 0.0, 1.0, 0.0, 1.0])
        assert ci.mean == 0.6
        assert ci.se is not None
        assert ci.se > 0
        assert ci.ci_lower < ci.mean < ci.ci_upper
        assert ci.method == "normal"

    def test_single_sample(self):
        ci = normal_ci([0.5])
        assert ci.ci_lower == ci.ci_upper == 0.5


class TestClusteredCI:
    def test_clustered_wider_than_naive(self):
        """Clustered SE should be >= naive SE when within-cluster scores correlate."""
        # All 1s in cluster A, all 0s in cluster B — perfectly correlated within cluster
        scores = [1.0] * 50 + [0.0] * 50
        clusters = ["A"] * 50 + ["B"] * 50

        naive = normal_ci(scores)
        clustered = clustered_ci(scores, clusters)

        assert clustered.se >= naive.se
        assert (clustered.ci_upper - clustered.ci_lower) >= (naive.ci_upper - naive.ci_lower)

    def test_no_clustering_effect_when_random(self):
        """When cluster assignment is random (no real structure), clustered SE ≈ naive SE."""
        rng = np.random.default_rng(42)
        scores = list(rng.choice([0.0, 1.0], size=200))
        clusters = [f"c{i % 50}" for i in range(200)]  # 50 clusters of 4

        naive = normal_ci(scores)
        clustered = clustered_ci(scores, clusters)

        # Should be within 2x of each other for random assignment
        assert clustered.se < naive.se * 2.5

    def test_single_cluster_falls_back_to_normal(self):
        scores = [1.0, 0.0, 1.0]
        clusters = ["A", "A", "A"]
        ci = clustered_ci(scores, clusters)
        assert ci.method == "normal"  # falls back

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            clustered_ci([1.0, 0.0], ["A"])

    def test_se_field_populated(self):
        scores = [1.0, 0.0, 1.0, 0.0]
        clusters = ["A", "A", "B", "B"]
        ci = clustered_ci(scores, clusters)
        assert ci.se is not None
        assert ci.method == "clustered"

    def test_mmlu_style_grouping(self):
        """Simulate MMLU-like structure: 4 subjects, 25 questions each.
        Within each subject, scores are correlated (all high or all low).

        Uses index-based seeding (not ``hash(subj)``) for determinism —
        Python's ``hash(str)`` is randomized per-process via PYTHONHASHSEED,
        which made this test flake on the SE-ratio assertion below.
        """
        scores = []
        clusters = []
        for i, (subj, acc) in enumerate([("math", 0.9), ("history", 0.6), ("science", 0.8), ("art", 0.5)]):
            rng = np.random.default_rng(42 + i)
            scores.extend(list(rng.choice([0.0, 1.0], size=25, p=[1 - acc, acc])))
            clusters.extend([subj] * 25)

        naive = normal_ci(scores)
        clustered = clustered_ci(scores, clusters)

        # With 4 diverse subjects, clustered SE should be meaningfully larger
        assert clustered.se > naive.se * 1.2
        assert clustered.ci_lower <= clustered.mean <= clustered.ci_upper
