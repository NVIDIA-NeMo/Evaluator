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
"""Tests for Pydantic config validation — extra fields forbidden."""

import copy
from typing import Any, Dict, Optional

import pytest
from omegaconf import OmegaConf
from pydantic import ValidationError

from nemo_evaluator_launcher.api.functional import _validate_config_sections
from nemo_evaluator_launcher.common.config_models import (
    EvaluationModel,
    MountsModel,
    TaskModel,
)

# Minimal valid sections for _validate_config_sections integration tests.
MINIMAL_EVALUATION: Dict[str, Any] = {
    "tasks": [{"name": "lm-evaluation-harness.ifeval"}],
}
MINIMAL_EXECUTION: Dict[str, Any] = {}


def make_config(
    evaluation: Optional[Dict[str, Any]] = None,
    execution: Optional[Dict[str, Any]] = None,
):
    """Build config with evaluation and execution; uses minimal section when omitted."""
    return OmegaConf.create(
        {
            "evaluation": copy.deepcopy(evaluation or MINIMAL_EVALUATION),
            "execution": copy.deepcopy(execution or MINIMAL_EXECUTION),
        }
    )


# ---------------------------------------------------------------------------
# MountsModel
# ---------------------------------------------------------------------------

VALID_MOUNTS = [
    pytest.param(
        {"deployment": {"/host/model": "/model"}, "mount_home": False},
        {
            "deployment": {"/host/model": "/model"},
            "evaluation": {},
            "mount_home": False,
        },
        id="deployment_only_no_home",
    ),
    pytest.param(
        {"evaluation": {"/host/data": "/data"}, "mount_home": True},
        {"deployment": {}, "evaluation": {"/host/data": "/data"}, "mount_home": True},
        id="evaluation_only_with_home",
    ),
    pytest.param(
        {"deployment": {"/a": "/b"}, "evaluation": {"/c": "/d"}, "mount_home": False},
        {"deployment": {"/a": "/b"}, "evaluation": {"/c": "/d"}, "mount_home": False},
        id="both_mounts_no_home",
    ),
    pytest.param(
        {},
        {"deployment": {}, "evaluation": {}, "mount_home": True},
        id="empty_uses_defaults",
    ),
]


class TestMountsModel:
    @pytest.mark.parametrize("kwargs, expected_fields", VALID_MOUNTS)
    def test_valid(self, kwargs, expected_fields):
        model = MountsModel(**kwargs)
        for field, expected in expected_fields.items():
            assert getattr(model, field) == expected

    def test_extra_field_forbidden(self):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            MountsModel(typo_field="oops")


# ---------------------------------------------------------------------------
# TaskModel
# ---------------------------------------------------------------------------

VALID_TASKS = [
    pytest.param(
        {
            "name": "lm-evaluation-harness.ifeval",
            "env_vars": {"HF_TOKEN": "host:HF_TOKEN"},
            "nemo_evaluator_config": {"config": {"params": {"parallelism": 4}}},
        },
        {
            "name": "lm-evaluation-harness.ifeval",
            "env_vars": {"HF_TOKEN": "host:HF_TOKEN"},
            "nemo_evaluator_config": {"config": {"params": {"parallelism": 4}}},
            "pre_cmd": "",
            "dataset_mount_path": "/datasets",
        },
        id="task_with_env_vars_and_params",
    ),
    pytest.param(
        {"name": "lm-evaluation-harness.ifeval"},
        {
            "name": "lm-evaluation-harness.ifeval",
            "env_vars": {},
            "nemo_evaluator_config": {},
            "container": None,
            "pre_cmd": "",
        },
        id="task_name_only_defaults",
    ),
    pytest.param(
        {
            "name": "lm-evaluation-harness.ifeval",
            "container": "nvcr.io/nvidia/eval-factory/lm-evaluation-harness:26.01",
            "pre_cmd": "echo hello",
            "dataset_dir": "/host/datasets",
            "dataset_mount_path": "/mnt/data",
        },
        {
            "name": "lm-evaluation-harness.ifeval",
            "container": "nvcr.io/nvidia/eval-factory/lm-evaluation-harness:26.01",
            "pre_cmd": "echo hello",
            "dataset_dir": "/host/datasets",
            "dataset_mount_path": "/mnt/data",
        },
        id="task_full_fields",
    ),
]


class TestTaskModel:
    @pytest.mark.parametrize("kwargs, expected_fields", VALID_TASKS)
    def test_valid(self, kwargs, expected_fields):
        model = TaskModel(**kwargs)
        for field, expected in expected_fields.items():
            assert getattr(model, field) == expected

    def test_extra_field_forbidden(self):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            TaskModel(name="some.task", unknown_key="bad")


# ---------------------------------------------------------------------------
# EvaluationModel
# ---------------------------------------------------------------------------

VALID_EVALUATIONS = [
    pytest.param(
        {
            "tasks": [{"name": "lm-evaluation-harness.ifeval"}],
            "env_vars": {"HF_TOKEN": "host:HF_TOKEN"},
        },
        {
            "env_vars": {"HF_TOKEN": "host:HF_TOKEN"},
            "pre_cmd": "",
            "nemo_evaluator_config": {},
        },
        1,  # expected number of tasks
        id="evaluation_with_env_vars",
    ),
    pytest.param(
        {
            "tasks": [
                {"name": "lm-evaluation-harness.ifeval"},
                {"name": "lm-evaluation-harness.ifeval", "pre_cmd": "echo hi"},
            ],
            "nemo_evaluator_config": {"config": {"params": {"parallelism": 8}}},
        },
        {
            "env_vars": {},
            "nemo_evaluator_config": {"config": {"params": {"parallelism": 8}}},
            "pre_cmd": "",
        },
        2,
        id="evaluation_two_tasks_global_params",
    ),
    pytest.param(
        {},
        {"tasks": [], "env_vars": {}, "nemo_evaluator_config": {}, "pre_cmd": ""},
        0,
        id="evaluation_empty_defaults",
    ),
]


class TestEvaluationModel:
    @pytest.mark.parametrize(
        "kwargs, expected_fields, expected_task_count", VALID_EVALUATIONS
    )
    def test_valid(self, kwargs, expected_fields, expected_task_count):
        model = EvaluationModel(**kwargs)
        assert len(model.tasks) == expected_task_count
        for field, expected in expected_fields.items():
            assert getattr(model, field) == expected

    def test_extra_field_forbidden(self):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            EvaluationModel(typo_field="oops")

    def test_task_with_extra_field_forbidden(self):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            EvaluationModel(tasks=[{"name": "some.task", "bad_key": "value"}])


# ---------------------------------------------------------------------------
# Integration: _validate_config_sections via OmegaConf
# ---------------------------------------------------------------------------


class TestValidateConfigSections:
    """Integration tests: _validate_config_sections against OmegaConf configs."""

    def test_valid_evaluation_passes(self):
        cfg = make_config(
            evaluation={
                "tasks": [{"name": "lm-evaluation-harness.ifeval"}],
                "env_vars": {},
            }
        )
        _validate_config_sections(cfg)

    def test_unknown_evaluation_key_raises(self):
        cfg = make_config(
            evaluation={
                "tasks": [{"name": "lm-evaluation-harness.ifeval"}],
                "typo_key": "oops",
            }
        )
        with pytest.raises(ValueError, match="Invalid 'evaluation' config"):
            _validate_config_sections(cfg)

    def test_unknown_task_key_raises(self):
        cfg = make_config(evaluation={"tasks": [{"name": "some.task", "bad_field": 1}]})
        with pytest.raises(ValueError, match="Invalid 'evaluation' config"):
            _validate_config_sections(cfg)

    def test_valid_mounts_passes(self):
        cfg = make_config(
            execution={
                "mounts": {"deployment": {}, "evaluation": {}, "mount_home": False}
            }
        )
        _validate_config_sections(cfg)

    def test_unknown_mounts_key_raises(self):
        cfg = make_config(execution={"mounts": {"deployment": {}, "typo": "oops"}})
        with pytest.raises(ValueError, match="Invalid 'execution.mounts' config"):
            _validate_config_sections(cfg)
