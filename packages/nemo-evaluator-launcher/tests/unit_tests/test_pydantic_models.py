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

import pytest
from omegaconf import OmegaConf
from pydantic import ValidationError

from nemo_evaluator_launcher.api.functional import _validate_config_sections
from nemo_evaluator_launcher.common.pydantic_models import (
    EvaluationModel,
    MountsModel,
    TaskModel,
)


class TestMountsModel:
    def test_valid(self):
        MountsModel(deployment={"/host/model": "/model"}, mount_home=False)

    def test_extra_field_forbidden(self):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            MountsModel(typo_field="oops")


class TestTaskModel:
    def test_valid(self):
        TaskModel(
            name="lm-evaluation-harness.ifeval",
            env_vars={"HF_TOKEN": "host:HF_TOKEN"},
            nemo_evaluator_config={"config": {"params": {"parallelism": 4}}},
        )

    def test_extra_field_forbidden(self):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            TaskModel(name="some.task", unknown_key="bad")


class TestEvaluationModel:
    def test_valid(self):
        EvaluationModel(
            tasks=[{"name": "lm-evaluation-harness.ifeval"}],
            env_vars={"HF_TOKEN": "host:HF_TOKEN"},
        )

    def test_extra_field_forbidden(self):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            EvaluationModel(typo_field="oops")

    def test_task_with_extra_field_forbidden(self):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            EvaluationModel(tasks=[{"name": "some.task", "bad_key": "value"}])


class TestValidateConfigSections:
    """Integration tests: _validate_config_sections against OmegaConf configs."""

    def test_valid_evaluation_passes(self):
        cfg = OmegaConf.create(
            {
                "evaluation": {
                    "tasks": [{"name": "lm-evaluation-harness.ifeval"}],
                    "env_vars": {},
                }
            }
        )
        _validate_config_sections(cfg)

    def test_unknown_evaluation_key_raises(self):
        cfg = OmegaConf.create({"evaluation": {"tasks": [], "typo_key": "oops"}})
        with pytest.raises(ValueError, match="Invalid 'evaluation' config"):
            _validate_config_sections(cfg)

    def test_unknown_task_key_raises(self):
        cfg = OmegaConf.create(
            {"evaluation": {"tasks": [{"name": "some.task", "bad_field": 1}]}}
        )
        with pytest.raises(ValueError, match="Invalid 'evaluation' config"):
            _validate_config_sections(cfg)

    def test_valid_mounts_passes(self):
        cfg = OmegaConf.create(
            {
                "execution": {
                    "mounts": {"deployment": {}, "evaluation": {}, "mount_home": False}
                }
            }
        )
        _validate_config_sections(cfg)

    def test_unknown_mounts_key_raises(self):
        cfg = OmegaConf.create(
            {"execution": {"mounts": {"deployment": {}, "typo": "oops"}}}
        )
        with pytest.raises(ValueError, match="Invalid 'execution.mounts' config"):
            _validate_config_sections(cfg)
