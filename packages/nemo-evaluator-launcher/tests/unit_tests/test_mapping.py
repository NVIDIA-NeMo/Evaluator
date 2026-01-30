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
    assert "Task not found in provided container" in caplog.text


def test_get_task_definition_for_job_unlisted_task_sets_flag(monkeypatch, caplog):
    """When task is not in container mapping, is_unlisted=True and task_query is preserved."""

    container = "example.com/my-image:latest"
    base_mapping = {}

    # Force container mapping to not include the task.
    monkeypatch.setattr(
        "nemo_evaluator_launcher.common.mapping.load_tasks_mapping",
        lambda *args, **kwargs: {},
    )

    td = get_task_definition_for_job(
        task_query="lm-evaluation-harness.polemo2",
        base_mapping=base_mapping,
        container=container,
    )

    assert td["is_unlisted"] is True
    assert td["task_query"] == "lm-evaluation-harness.polemo2"
    assert td["harness"] == "lm-evaluation-harness"
    assert td["task"] == "polemo2"
    assert td["container"] == container


def test_get_task_definition_for_job_listed_task_not_unlisted(monkeypatch):
    """When task is found in mapping, is_unlisted=False."""

    container = "example.com/my-image:latest"
    base_mapping = {}
    container_mapping = {
        ("test-harness", "known-task"): {
            "task": "known-task",
            "harness": "test-harness",
            "container": container,
            "endpoint_type": "chat",
        }
    }

    monkeypatch.setattr(
        "nemo_evaluator_launcher.common.mapping.load_tasks_mapping",
        lambda *args, **kwargs: container_mapping,
    )

    td = get_task_definition_for_job(
        task_query="test-harness.known-task",
        base_mapping=base_mapping,
        container=container,
    )

    assert td["is_unlisted"] is False
    assert td["task"] == "known-task"


def test_get_task_definition_for_job_harness_mismatch_raises(monkeypatch):
    """When harness in query doesn't match container's harnesses, raise error."""

    container = "example.com/my-image:latest"
    base_mapping = {}
    container_mapping = {
        ("simple_evals", "gpqa"): {
            "task": "gpqa",
            "harness": "simple_evals",
            "container": container,
            "endpoint_type": "chat",
        }
    }

    monkeypatch.setattr(
        "nemo_evaluator_launcher.common.mapping.load_tasks_mapping",
        lambda *args, **kwargs: container_mapping,
    )

    with pytest.raises(
        ValueError, match="Harness 'lm-evaluation-harness' does not match container"
    ):
        get_task_definition_for_job(
            task_query="lm-evaluation-harness.some_task",
            base_mapping=base_mapping,
            container=container,
        )


def test_get_task_definition_for_job_default_container_from_harness(monkeypatch):
    """When no container provided, use harness default container."""

    from nemo_evaluator_launcher.common.container_metadata import (
        HarnessIntermediateRepresentation,
    )

    harness_container = "nvcr.io/nvidia/eval-factory/lm-evaluation-harness:25.11"
    base_mapping = {}

    # Mock harness lookup
    def fake_load_harnesses(*args, **kwargs):
        harnesses = {
            "lm-evaluation-harness": HarnessIntermediateRepresentation(
                name="lm-evaluation-harness",
                description="LM Evaluation Harness",
                full_name="EleutherAI/lm-evaluation-harness",
                url="https://github.com/EleutherAI/lm-evaluation-harness",
                container=harness_container,
                container_digest="sha256:abc123",
            )
        }
        return harnesses, [], []

    # Mock container mapping that will be loaded after harness lookup
    container_mapping = {
        ("lm-evaluation-harness", "polemo2"): {
            "task": "polemo2",
            "harness": "lm-evaluation-harness",
            "container": harness_container,
            "endpoint_type": "chat",
        }
    }

    load_counter = {"count": 0}

    def fake_load_tasks_mapping(*args, from_container=None, **kwargs):
        load_counter["count"] += 1
        if from_container == harness_container:
            return container_mapping
        return {}

    monkeypatch.setattr(
        "nemo_evaluator_launcher.common.container_metadata.load_harnesses_and_tasks_from_tasks_file",
        fake_load_harnesses,
    )
    monkeypatch.setattr(
        "nemo_evaluator_launcher.common.mapping.load_tasks_mapping",
        fake_load_tasks_mapping,
    )

    td = get_task_definition_for_job(
        task_query="lm-evaluation-harness.polemo2",
        base_mapping=base_mapping,
        container=None,
    )

    assert td["task"] == "polemo2"
    assert td["harness"] == "lm-evaluation-harness"
    assert td["container"] == harness_container
    assert td["is_unlisted"] is False


def test_get_task_definition_for_job_unknown_harness_raises(monkeypatch):
    """When harness not found in supported harnesses, raise error."""

    from nemo_evaluator_launcher.common.container_metadata import (
        HarnessIntermediateRepresentation,
    )

    base_mapping = {}

    def fake_load_harnesses(*args, **kwargs):
        # Return harnesses that don't include the one we're looking for
        return (
            {
                "other-harness": HarnessIntermediateRepresentation(
                    name="other-harness",
                    description="Other",
                    full_name="other/harness",
                    url="https://example.com",
                    container="other:latest",
                    container_digest="sha256:def456",
                )
            },
            [],
            [],
        )

    monkeypatch.setattr(
        "nemo_evaluator_launcher.common.container_metadata.load_harnesses_and_tasks_from_tasks_file",
        fake_load_harnesses,
    )

    with pytest.raises(
        ValueError, match="Harness 'unknown-harness' not found in supported harnesses"
    ):
        get_task_definition_for_job(
            task_query="unknown-harness.some_task",
            base_mapping=base_mapping,
            container=None,
        )
