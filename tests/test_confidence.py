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
from nemo_evaluator.metrics.confidence import bootstrap_ci


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
