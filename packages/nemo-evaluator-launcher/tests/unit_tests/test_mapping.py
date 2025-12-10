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
"""Tests for the mapping module."""

import copy

import pytest

from nemo_evaluator_launcher.common.mapping import (
    _process_mapping,
    get_task_from_mapping,
    load_tasks_mapping,
)


def test_process_mapping():
    """Test mapping processing logic."""
    mapping_toml = {
        "harness1": {
            "container": "test-container:latest",
            "tasks": {
                "chat": {
                    "task1": {},
                },
                "completions": {
                    "task2": {},
                },
            },
        },
        "harness2": {
            "container": "test-container2:latest",
            "tasks": {
                "chat": {
                    "task3": {},
                },
            },
        },
    }

    result = _process_mapping(mapping_toml)

    expected = {
        ("harness1", "task1"): {
            "task": "task1",
            "harness": "harness1",
            "container": "test-container:latest",
            "endpoint_type": "chat",
        },
        ("harness1", "task2"): {
            "task": "task2",
            "harness": "harness1",
            "container": "test-container:latest",
            "endpoint_type": "completions",
        },
        ("harness2", "task3"): {
            "task": "task3",
            "harness": "harness2",
            "container": "test-container2:latest",
            "endpoint_type": "chat",
        },
    }
    assert result == expected


def test_get_task_from_mapping():
    """Test retrieving tasks from mapping via query logic."""
    tasks_mapping = {
        ("harnessA", "task1"): {
            "task": "task1",
            "harness": "harnessA",
            "container": "test-container:latest",
            "endpoint_type": "chat",
        },
        ("harnessA", "task2"): {
            "task": "task2",
            "harness": "harnessA",
            "container": "test-container:latest",
            "endpoint_type": "completions",
        },
        ("harnessB", "task1"): {
            "task": "task1",
            "harness": "harnessB",
            "container": "test-container2:latest",
            "endpoint_type": "chat",
        },
    }

    expected1 = copy.deepcopy(tasks_mapping[("harnessA", "task2")])
    expected2 = r"there are multiple tasks named 'task1' in the mapping, please select one of \['harnessA\.task1', 'harnessB\.task1'\]"
    expected3 = copy.deepcopy(tasks_mapping[("harnessA", "task1")])
    expected4 = copy.deepcopy(tasks_mapping[("harnessB", "task1")])

    result1 = get_task_from_mapping("task2", tasks_mapping)
    assert result1 == expected1

    with pytest.raises(ValueError, match=expected2):
        _ = get_task_from_mapping("task1", tasks_mapping)

    result3 = get_task_from_mapping("harnessA.task1", tasks_mapping)
    assert result3 == expected3

    result4 = get_task_from_mapping("harnessB.task1", tasks_mapping)
    assert result4 == expected4


def test_load_tasks_mapping():
    tasks_mapping = load_tasks_mapping()
    assert isinstance(tasks_mapping, dict)
    assert len(tasks_mapping) > 0
    for key, value in tasks_mapping.items():
        assert isinstance(key, tuple)
        assert len(key) == 2
        assert isinstance(key[0], str)
        assert isinstance(key[1], str)
        assert isinstance(value, dict)
        assert "task" in value
        assert "harness" in value
        assert "container" in value
        assert "endpoint_type" in value
        assert key[0] == value["harness"]
        assert key[1] == value["task"]
