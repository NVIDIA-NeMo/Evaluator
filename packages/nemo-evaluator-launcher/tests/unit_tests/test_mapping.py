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

from nemo_evaluator_launcher.common.container_metadata.intermediate_repr import (
    TaskIntermediateRepresentation,
)
from nemo_evaluator_launcher.common.mapping import (
    get_task_definition_for_job,
    get_task_from_mapping,
    load_tasks_mapping,
)


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


def test_load_tasks_mapping_from_container(monkeypatch):
    """If from_container is provided, we should load tasks via container-metadata."""

    container = "example.com/my-image:latest"

    def _fake_load_tasks_from_container(arg):
        assert arg == container
        return [
            TaskIntermediateRepresentation(
                name="task_a",
                description="desc",
                harness="harness_x",
                container=container,
                container_digest="sha256:deadbeef",
                defaults={"config": {"supported_endpoint_types": ["chat"]}},
            )
        ]

    # If we accidentally fall back to packaged resources, fail fast.
    monkeypatch.setattr(
        "nemo_evaluator_launcher.common.mapping.load_tasks_from_tasks_file",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("load_tasks_from_tasks_file should not be called")
        ),
    )
    monkeypatch.setattr(
        "nemo_evaluator_launcher.common.container_metadata.load_tasks_from_container",
        _fake_load_tasks_from_container,
    )

    mapping = load_tasks_mapping(from_container=container)
    assert ("harness_x", "task_a") in mapping
    assert mapping[("harness_x", "task_a")]["container"] == container
    assert mapping[("harness_x", "task_a")]["endpoint_type"] == "chat"


def test_get_task_definition_for_job_container_missing_task_warns(monkeypatch, caplog):
    """If task is missing in provided container, we warn and proceed."""

    container = "example.com/my-image:latest"
    base_mapping = {("base", "base_task"): {"task": "base_task", "harness": "base"}}

    # Force container mapping to not include the task.
    monkeypatch.setattr(
        "nemo_evaluator_launcher.common.mapping.load_tasks_mapping",
        lambda *args, **kwargs: {},
    )

    td = get_task_definition_for_job(
        task_query="ruler-1m-chat",
        base_mapping=base_mapping,
        container=container,
    )

    assert td["task"] == "ruler-1m-chat"
    assert td["container"] == container
    assert td["endpoint_type"] == "chat"
    assert "Task not found in any mapping" in caplog.text


def test_get_task_from_mapping_allow_missing_returns_none():
    """Test that allow_missing=True returns None instead of raising ValueError."""
    tasks_mapping = {
        ("harnessA", "task1"): {
            "task": "task1",
            "harness": "harnessA",
            "container": "test-container:latest",
            "endpoint_type": "chat",
        },
    }

    # Task not found with allow_missing=False should raise
    with pytest.raises(ValueError, match="does not exist in the mapping"):
        get_task_from_mapping("nonexistent", tasks_mapping)

    # Task not found with allow_missing=True should return None
    result = get_task_from_mapping("nonexistent", tasks_mapping, allow_missing=True)
    assert result is None

    # Task not found with harness.task format and allow_missing=True
    result = get_task_from_mapping(
        "unknown.nonexistent", tasks_mapping, allow_missing=True
    )
    assert result is None

    # Task found should still return the task (regardless of allow_missing)
    result = get_task_from_mapping("task1", tasks_mapping, allow_missing=True)
    assert result == tasks_mapping[("harnessA", "task1")]


def test_get_task_definition_for_job_unlisted_task_has_flag(monkeypatch, caplog):
    """Test that unlisted tasks have is_unlisted_task=True flag."""
    container = "example.com/my-image:latest"
    base_mapping = {}

    # Force container mapping to also be empty (task truly unlisted).
    monkeypatch.setattr(
        "nemo_evaluator_launcher.common.mapping.load_tasks_mapping",
        lambda *args, **kwargs: {},
    )

    td = get_task_definition_for_job(
        task_query="harness.unlisted_task",
        base_mapping=base_mapping,
        container=container,
    )

    assert td["task"] == "unlisted_task"
    assert td["harness"] == "harness"
    assert td["container"] == container
    assert td.get("is_unlisted_task") is True


def test_get_task_definition_for_job_listed_task_no_flag():
    """Test that listed tasks do not have is_unlisted_task flag."""
    base_mapping = {
        ("harness", "listed_task"): {
            "task": "listed_task",
            "harness": "harness",
            "container": "test:latest",
            "endpoint_type": "chat",
        }
    }

    td = get_task_definition_for_job(
        task_query="harness.listed_task",
        base_mapping=base_mapping,
        container=None,
    )

    assert td["task"] == "listed_task"
    assert "is_unlisted_task" not in td or td.get("is_unlisted_task") is False


def test_get_task_definition_for_job_unlisted_requires_container():
    """Test that unlisted tasks without container raise ValueError."""
    base_mapping = {}

    with pytest.raises(ValueError, match="does not exist in the mapping"):
        get_task_definition_for_job(
            task_query="harness.unlisted_task",
            base_mapping=base_mapping,
            container=None,  # No container provided
        )


def test_get_task_definition_for_job_container_override_for_listed_task():
    """Test that listed tasks with container override use the override container."""
    base_mapping = {
        ("harness", "listed_task"): {
            "task": "listed_task",
            "harness": "harness",
            "container": "original:latest",
            "endpoint_type": "chat",
        }
    }

    td = get_task_definition_for_job(
        task_query="harness.listed_task",
        base_mapping=base_mapping,
        container="override:v2",
    )

    assert td["task"] == "listed_task"
    assert td["container"] == "override:v2"
    assert "is_unlisted_task" not in td or td.get("is_unlisted_task") is False
