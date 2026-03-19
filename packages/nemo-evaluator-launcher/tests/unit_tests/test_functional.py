# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Tests for the functional module."""

import pytest
from omegaconf import OmegaConf

from nemo_evaluator_launcher.api.functional import filter_tasks


def _cfg(tasks):
    """Build a minimal RunConfig from a list of task dicts.

    Each dict must have at least ``name`` and ``container`` so tests can
    verify which specific instance was selected.
    """
    return OmegaConf.create({"evaluation": {"tasks": tasks}})


class TestFilterTasks:
    """Tests for filter_tasks function."""

    @pytest.mark.parametrize(
        "tasks, filters, expected_tasks",
        [
            pytest.param(
                [
                    {"name": "mmlu", "container": "eval:v1"},
                    {"name": "gsm8k", "container": "eval:v1"},
                ],
                [],
                [
                    {"name": "mmlu", "container": "eval:v1"},
                    {"name": "gsm8k", "container": "eval:v1"},
                ],
                id="empty_filters_returns_all",
            ),
            pytest.param(
                [
                    {"name": "mmlu", "container": "eval:v1"},
                    {"name": "gsm8k", "container": "eval:v1"},
                    {"name": "ifeval", "container": "eval:v1"},
                ],
                ["mmlu", "ifeval"],
                [
                    {"name": "mmlu", "container": "eval:v1"},
                    {"name": "ifeval", "container": "eval:v1"},
                ],
                id="plain_name",
            ),
            pytest.param(
                [
                    {"name": "mmlu", "container": "eval:v1"},
                    {"name": "gsm8k", "container": "eval:v1"},
                    {"name": "mmlu", "container": "eval:v2"},
                ],
                ["mmlu"],
                [
                    {"name": "mmlu", "container": "eval:v1"},
                    {"name": "mmlu", "container": "eval:v2"},
                ],
                id="plain_name_selects_all_duplicates",
            ),
            pytest.param(
                [
                    {"name": "mmlu", "container": "eval:v1"},
                    {"name": "gsm8k", "container": "eval:v1"},
                    {"name": "mmlu", "container": "eval:v2"},
                ],
                ["mmlu.2"],
                [
                    {"name": "mmlu", "container": "eval:v2"},
                ],
                id="unique_name_selects_single_instance",
            ),
            pytest.param(
                [
                    {"name": "mmlu", "container": "eval:v1"},
                    {"name": "gsm8k", "container": "eval:v1"},
                    {"name": "mmlu", "container": "eval:v2"},
                ],
                ["mmlu.0"],
                [
                    {"name": "mmlu", "container": "eval:v1"},
                ],
                id="unique_name_first_instance",
            ),
            pytest.param(
                [
                    {"name": "mmlu", "container": "eval:v1"},
                    {"name": "gsm8k", "container": "eval:v1"},
                    {"name": "mmlu", "container": "eval:v2"},
                    {"name": "ifeval", "container": "eval:v1"},
                ],
                ["gsm8k", "mmlu.2"],
                [
                    {"name": "gsm8k", "container": "eval:v1"},
                    {"name": "mmlu", "container": "eval:v2"},
                ],
                id="mixed_plain_and_unique",
            ),
            pytest.param(
                [
                    {"name": "c", "container": "eval:c"},
                    {"name": "a", "container": "eval:a"},
                    {"name": "b", "container": "eval:b"},
                ],
                ["b", "a"],
                [
                    {"name": "a", "container": "eval:a"},
                    {"name": "b", "container": "eval:b"},
                ],
                id="preserves_original_order",
            ),
            # Plain name takes priority over unique name to avoid shadowing.
            # "mmlu.0" as unique name would resolve to index 0 (the "mmlu"
            # task), but as plain name it matches index 2 — plain name wins.
            pytest.param(
                [
                    {"name": "mmlu", "container": "eval:v1"},
                    {"name": "gsm8k", "container": "eval:v1"},
                    {"name": "mmlu.0", "container": "eval:v2"},
                ],
                ["mmlu.0"],
                [
                    {"name": "mmlu.0", "container": "eval:v2"},
                ],
                id="plain_name_priority_over_unique_name",
            ),
            # "mmlu.1" as unique name would resolve to index 1 (the "mmlu"
            # task), but as plain name it matches index 2 — plain name wins.
            pytest.param(
                [
                    {"name": "gsm8k", "container": "eval:v1"},
                    {"name": "mmlu", "container": "eval:v1"},
                    {"name": "mmlu.1", "container": "eval:v2"},
                ],
                ["mmlu.1"],
                [
                    {"name": "mmlu.1", "container": "eval:v2"},
                ],
                id="plain_name_priority_different_indices",
            ),
        ],
    )
    def test_filter_tasks(self, tasks, filters, expected_tasks):
        cfg = _cfg(tasks)
        result = filter_tasks(cfg, filters)
        actual = OmegaConf.to_container(result.evaluation.tasks)
        assert actual == expected_tasks

    def test_no_tasks_raises(self):
        cfg = OmegaConf.create({"evaluation": {"tasks": []}})
        with pytest.raises(ValueError, match="No tasks defined"):
            filter_tasks(cfg, ["mmlu"])

    def test_unmatched_filter_raises(self):
        cfg = _cfg([{"name": "mmlu"}, {"name": "gsm8k"}])
        with pytest.raises(ValueError, match="nonexistent"):
            filter_tasks(cfg, ["nonexistent"])

    def test_does_not_mutate_input(self):
        cfg = _cfg([{"name": "mmlu"}, {"name": "gsm8k"}])
        filter_tasks(cfg, ["mmlu"])
        assert len(cfg.evaluation.tasks) == 2
