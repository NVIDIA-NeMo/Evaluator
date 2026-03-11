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
"""Config validation tests — parametrized valid/invalid configs.

Structural validation (_validate_config_sections) and nemo_evaluator_config
param validation (_validate_nemo_evaluator_config_params).
"""

import textwrap

import pytest
import yaml
from omegaconf import OmegaConf

from nemo_evaluator_launcher.api.functional import (
    _validate_config_sections,
    _validate_nemo_evaluator_config_params,
)


def _from_yaml(s: str):
    return OmegaConf.create(yaml.safe_load(textwrap.dedent(s)))


# ---------------------------------------------------------------------------
# Structural validation — _validate_config_sections
# ---------------------------------------------------------------------------

VALID_STRUCTURAL_CONFIGS = [
    pytest.param(
        """\
        evaluation:
          tasks:
            - name: lm-evaluation-harness.ifeval
              env_vars:
                HF_TOKEN: host:HF_TOKEN
              nemo_evaluator_config:
                config:
                  params:
                    parallelism: 4
        """,
        id="evaluation_task_with_known_fields",
    ),
    pytest.param(
        """\
        evaluation:
          env_vars:
            HF_TOKEN: host:HF_TOKEN
            SOME_SECRET: lit:my-secret-value
          tasks:
            - name: lm-evaluation-harness.ifeval
              env_vars:
                TASK_TOKEN: host:TASK_TOKEN
        """,
        id="evaluation_env_vars_at_both_levels",
    ),
    pytest.param(
        """\
        execution:
          mounts:
            deployment:
              /host/model: /model
            evaluation:
              /host/data: /data
            mount_home: false
        """,
        id="mounts_all_known_fields",
    ),
    pytest.param(
        """\
        execution:
          mounts:
            deployment: {}
        """,
        id="mounts_only_deployment_field",
    ),
    pytest.param(
        """\
        execution:
          mounts:
            mount_home: true
        """,
        id="mounts_only_mount_home_field",
    ),
]

INVALID_STRUCTURAL_CONFIGS = [
    pytest.param(
        """\
        evaluation:
          tasks:
            - name: lm-evaluation-harness.ifeval
          extra_setting: should_not_exist
        """,
        "Invalid 'evaluation' config",
        id="evaluation_unknown_top_level_key",
    ),
    pytest.param(
        """\
        evaluation:
          tasks:
            - name: lm-evaluation-harness.ifeval
              non_existing_field: true
        """,
        "Invalid 'evaluation' config",
        id="evaluation_task_non_existing_field",
    ),
    pytest.param(
        """\
        evaluation:
          env_vars:
            HF_TOKEN: host:HF_TOKEN
          tasks:
            - name: lm-evaluation-harness.ifeval
              env_vars:
                nested_mistake:
                  key: value
        """,
        "Invalid 'evaluation' config",
        id="evaluation_task_env_vars_value_not_a_string",
    ),
    pytest.param(
        """\
        evaluation:
          tasks:
            - name: lm-evaluation-harness.ifeval
              env_var:
                HF_TOKEN: host:HF_TOKEN
        """,
        "Invalid 'evaluation' config",
        id="evaluation_task_typo_env_var_vs_env_vars",
    ),
    pytest.param(
        """\
        evaluation:
          tasks:
            - name: lm-evaluation-harness.ifeval
              precmd: echo hi
        """,
        "Invalid 'evaluation' config",
        id="evaluation_task_typo_precmd_vs_pre_cmd",
    ),
    pytest.param(
        """\
        execution:
          mounts:
            deployment: {}
            non_existing_mount_option: true
        """,
        "Invalid 'execution.mounts' config",
        id="mounts_unknown_key",
    ),
    pytest.param(
        """\
        execution:
          mounts:
            deployment:
              /host: {}
        """,
        "Invalid 'execution.mounts' config",
        id="mounts_invalid_value",
    ),
    pytest.param(
        """\
        execution:
          mounts:
            evaluation:
              /host: {}
        """,
        "Invalid 'execution.mounts' config",
        id="mounts_evaluation_invalid_value",
    ),
]


class TestStructuralValidation:
    @pytest.mark.parametrize("raw_yaml", VALID_STRUCTURAL_CONFIGS)
    def test_valid_config_passes(self, raw_yaml):
        # Given a config with only known fields
        cfg = _from_yaml(raw_yaml)
        # When validated
        # Then no error is raised
        _validate_config_sections(cfg)

    @pytest.mark.parametrize("raw_yaml, expected_error", INVALID_STRUCTURAL_CONFIGS)
    def test_invalid_config_raises(self, raw_yaml, expected_error):
        # Given a config with an unknown or removed field
        cfg = _from_yaml(raw_yaml)
        # When validated
        # Then ValueError is raised naming the problematic section
        with pytest.raises(ValueError, match=expected_error):
            _validate_config_sections(cfg)


# ---------------------------------------------------------------------------
# nemo_evaluator_config param validation — real packaged IRs.
#
# lm-evaluation-harness.ifeval  container: nvcr.io/nvidia/eval-factory/lm-evaluation-harness:26.01
#   params: parallelism, request_timeout, limit_samples, max_retries
#   extras: num_fewshot, tokenizer, tokenizer_backend, ...
# ---------------------------------------------------------------------------

VALID_PARAM_CONFIGS = [
    pytest.param(
        "lm-evaluation-harness.ifeval",
        {"parallelism": 4, "limit_samples": 10},
        id="ifeval_known_standard_params",
    ),
    pytest.param(
        "lm-evaluation-harness.ifeval",
        {"parallelism": 1, "request_timeout": 3600, "max_retries": 3},
        id="ifeval_known_standard_params_multiple",
    ),
    pytest.param(
        "lm-evaluation-harness.ifeval",
        {"extra": {"num_fewshot": 0}},
        id="ifeval_known_extra_param",
    ),
    pytest.param(
        "lm-evaluation-harness.ifeval",
        {"temperature": 0.7, "top_p": 0.9, "max_new_tokens": 1024},
        id="ifeval_known_generation_params",
    ),
    pytest.param(
        "lm-evaluation-harness.ifeval",
        {
            "parallelism": 8,
            "request_timeout": 60,
            "extra": {"tokenizer_backend": "huggingface"},
        },
        id="ifeval_known_extra_tokenizer_param",
    ),
]

INVALID_PARAM_CONFIGS = [
    pytest.param(
        "lm-evaluation-harness.ifeval",
        {"non_existing_param": 42},
        "non_existing_param",
        id="ifeval_completely_unknown_param",
    ),
    pytest.param(
        "lm-evaluation-harness.ifeval",
        {"parallelism": 1, "totally_made_up_param_xyz": 0.9},
        "totally_made_up_param_xyz",
        id="ifeval_unsupported_fake_param",
    ),
    pytest.param(
        "lm-evaluation-harness.ifeval",
        {"extra": {"totally_made_up_extra_xyz": 1}},
        "totally_made_up_extra_xyz",
        id="ifeval_unsupported_fake_extra_param",
    ),
    pytest.param(
        "lm-evaluation-harness.ifeval",
        {"parallelism": 1, "bad_param_a": 1, "bad_param_b": 2},
        "bad_param",
        id="ifeval_multiple_unknown_params",
    ),
]


_IFEVAL_CONTAINER = "nvcr.io/nvidia/eval-factory/lm-evaluation-harness:26.01"


class TestNemoEvaluatorParamValidation:
    def _make_cfg(self, task_name: str, params: dict):
        return OmegaConf.create(
            {
                "evaluation": {
                    "tasks": [
                        {
                            "name": task_name,
                            "container": _IFEVAL_CONTAINER,
                            "nemo_evaluator_config": {"config": {"params": params}},
                        }
                    ],
                    "nemo_evaluator_config": {},
                }
            }
        )

    @pytest.mark.parametrize("task_name, params", VALID_PARAM_CONFIGS)
    def test_valid_params_no_warning(self, task_name, params, caplog):
        # Given a config using params that exist in the task's command template
        cfg = self._make_cfg(task_name, params)
        # When validated against real packaged IRs
        _validate_nemo_evaluator_config_params(cfg)
        # Then no warning about unused params
        assert "not used in the command" not in caplog.text

    @pytest.mark.parametrize(
        "task_name, params, expected_in_warning", INVALID_PARAM_CONFIGS
    )
    def test_invalid_params_emit_warning(
        self, task_name, params, expected_in_warning, caplog
    ):
        # Given a config with params not referenced in the task's command template
        cfg = self._make_cfg(task_name, params)
        # When validated against real packaged IRs
        _validate_nemo_evaluator_config_params(cfg)
        # Then a warning is emitted naming the unknown param (run is not blocked)
        assert expected_in_warning in caplog.text

    def test_global_valid_task_invalid_emits_warning(self, caplog):
        # Given global nemo_evaluator_config has valid params
        # but task-level adds an invalid param on top
        cfg = OmegaConf.create(
            {
                "evaluation": {
                    "tasks": [
                        {
                            "name": "lm-evaluation-harness.ifeval",
                            "container": _IFEVAL_CONTAINER,
                            "nemo_evaluator_config": {
                                "config": {"params": {"bad_task_param": 1}}
                            },
                        }
                    ],
                    "nemo_evaluator_config": {"config": {"params": {"parallelism": 4}}},
                }
            }
        )
        # When validated
        _validate_nemo_evaluator_config_params(cfg)
        # Then warning fired for the task-level bad param
        assert "bad_task_param" in caplog.text

    def test_global_invalid_task_valid_emits_warning(self, caplog):
        # Given global nemo_evaluator_config has an invalid param
        # but task-level only sets valid params
        cfg = OmegaConf.create(
            {
                "evaluation": {
                    "tasks": [
                        {
                            "name": "lm-evaluation-harness.ifeval",
                            "container": _IFEVAL_CONTAINER,
                            "nemo_evaluator_config": {
                                "config": {"params": {"parallelism": 4}}
                            },
                        }
                    ],
                    "nemo_evaluator_config": {
                        "config": {"params": {"bad_global_param": 99}}
                    },
                }
            }
        )
        # When validated
        _validate_nemo_evaluator_config_params(cfg)
        # Then warning fired for the global bad param (survives merge)
        assert "bad_global_param" in caplog.text
