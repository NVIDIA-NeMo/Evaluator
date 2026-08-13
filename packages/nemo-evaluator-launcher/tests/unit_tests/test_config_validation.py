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
param validation (via _validate_config_sections).
"""

import copy
import textwrap
from typing import Any, Dict, Optional
from unittest.mock import patch

import pytest
import yaml
from omegaconf import OmegaConf

from nemo_evaluator_launcher.api.functional import (
    _validate_config_sections,
    _validate_pd_deployment,
)

# Minimal valid sections; use make_config() to build full configs.
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
        execution: {}
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
        execution: {}
        """,
        id="evaluation_env_vars_at_both_levels",
    ),
    pytest.param(
        """\
        evaluation:
          tasks:
            - name: lm-evaluation-harness.ifeval
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
        evaluation:
          tasks:
            - name: lm-evaluation-harness.ifeval
        execution:
          mounts:
            deployment: {}
        """,
        id="mounts_only_deployment_field",
    ),
    pytest.param(
        """\
        evaluation:
          tasks:
            - name: lm-evaluation-harness.ifeval
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
        execution: {}
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
        execution: {}
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
        execution: {}
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
        execution: {}
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
        execution: {}
        """,
        "Invalid 'evaluation' config",
        id="evaluation_task_typo_precmd_vs_pre_cmd",
    ),
    pytest.param(
        """\
        evaluation:
          tasks:
            - name: lm-evaluation-harness.ifeval
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
        evaluation:
          tasks:
            - name: lm-evaluation-harness.ifeval
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
        evaluation:
          tasks:
            - name: lm-evaluation-harness.ifeval
        execution:
          mounts:
            evaluation:
              /host: {}
        """,
        "Invalid 'execution.mounts' config",
        id="mounts_evaluation_invalid_value",
    ),
    pytest.param(
        """\
        evaluation:
          tasks:
            - name: lm-evaluation-harness.ifeval
        execution:
          env_vars:
            deployment:
              HF_TOKEN: host:HF_TOKEN
        """,
        "execution.env_vars.*is not supported",
        id="execution_env_vars_deployment",
    ),
    pytest.param(
        """\
        evaluation:
          tasks:
            - name: lm-evaluation-harness.ifeval
        execution:
          env_vars:
            HF_TOKEN: host:HF_TOKEN
        """,
        "execution.env_vars.*is not supported",
        id="execution_env_vars_flat",
    ),
    pytest.param(
        """\
        evaluation:
          tasks:
            - name: lm-evaluation-harness.ifeval
        execution:
          env_vars: {}
        """,
        "execution.env_vars.*is not supported",
        id="execution_env_vars_empty",
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


class TestVllmPdDeploymentValidation:
    @pytest.mark.parametrize(
        "prefill_nodes,decode_nodes,num_nodes,num_instances,n_tasks,error",
        [
            (1, 1, 2, 1, 2, None),
            (0, 1, 1, 1, 1, "prefill_nodes must be a positive integer"),
            (1, 0, 1, 1, 1, "decode_nodes must be a positive integer"),
            (1, 1, 3, 1, 3, r"must equal prefill_nodes \+ decode_nodes"),
            (1, 1, 2, 2, 2, "requires execution.num_instances == 1"),
            (1, 1, 2, 1, 1, "execution.deployment.n_tasks must equal"),
            (1, 1, "2", "1", "2", None),
            (1, 1, 2, 1.5, 2, "execution.num_instances must be 1"),
            (1, 1, 2.5, 1, 2, "execution.num_nodes must be an integer"),
            (1, 1, 2, 1, 2.5, "execution.deployment.n_tasks must be an integer"),
        ],
        ids=[
            "valid",
            "zero_prefill",
            "zero_decode",
            "node_count_mismatch",
            "multiple_instances",
            "task_count_mismatch",
            "stringified_execution_values",
            "fractional_instance_count",
            "fractional_node_count",
            "fractional_task_count",
        ],
    )
    def test_topology_validation(
        self,
        prefill_nodes,
        decode_nodes,
        num_nodes,
        num_instances,
        n_tasks,
        error,
    ):
        cfg = OmegaConf.create(
            {
                "deployment": {
                    "type": "vllm_pd",
                    "prefill_nodes": prefill_nodes,
                    "decode_nodes": decode_nodes,
                },
                "execution": {
                    "type": "slurm",
                    "num_nodes": num_nodes,
                    "num_instances": num_instances,
                    "deployment": {"n_tasks": n_tasks},
                },
            }
        )

        if error is None:
            _validate_pd_deployment(cfg)
        else:
            with pytest.raises(ValueError, match=error):
                _validate_pd_deployment(cfg)

    def test_defaults_tasks_to_nodes_when_execution_deployment_is_null(self):
        cfg = OmegaConf.create(
            {
                "deployment": {
                    "type": "vllm_pd",
                    "prefill_nodes": 1,
                    "decode_nodes": 1,
                },
                "execution": {
                    "type": "slurm",
                    "num_nodes": 2,
                    "num_instances": 1,
                    "deployment": None,
                },
            }
        )

        _validate_pd_deployment(cfg)

    @pytest.mark.parametrize(
        "port_config,error",
        [
            (
                {"port": 8001, "prefill_port": 8001},
                "deployment.port must differ from deployment.prefill_port",
            ),
            (
                {"port": 0},
                "deployment.port must be an integer between 1 and 65535",
            ),
            (
                {"decode_nixl_side_channel_port": 65536},
                "deployment.decode_nixl_side_channel_port must be an integer between 1 and 65535",
            ),
        ],
        ids=[
            "router_prefill_port_collision",
            "zero_router_port",
            "oversized_nixl_port",
        ],
    )
    def test_rejects_invalid_ports(self, port_config, error):
        cfg = OmegaConf.create(
            {
                "deployment": {
                    "type": "vllm_pd",
                    "prefill_nodes": 1,
                    "decode_nodes": 1,
                    **port_config,
                },
                "execution": {
                    "type": "slurm",
                    "num_nodes": 2,
                    "num_instances": 1,
                    "deployment": {"n_tasks": 2},
                },
            }
        )

        with pytest.raises(ValueError, match=error):
            _validate_pd_deployment(cfg)

    @pytest.mark.parametrize(
        "runtime_image_sha256,error",
        [
            ("", None),
            ("a" * 64, None),
            ("sha256:" + "a" * 64, "must be a 64-character SHA-256 digest"),
            ("not-a-digest", "must be a 64-character SHA-256 digest"),
        ],
        ids=["unset", "valid", "prefixed", "invalid"],
    )
    def test_runtime_image_digest(self, runtime_image_sha256, error):
        cfg = OmegaConf.create(
            {
                "deployment": {
                    "type": "vllm_pd",
                    "prefill_nodes": 1,
                    "decode_nodes": 1,
                    "runtime_image_sha256": runtime_image_sha256,
                },
                "execution": {
                    "type": "slurm",
                    "num_nodes": 2,
                    "num_instances": 1,
                    "deployment": {"n_tasks": 2},
                },
            }
        )

        if error is None:
            _validate_pd_deployment(cfg)
        else:
            with pytest.raises(ValueError, match=error):
                _validate_pd_deployment(cfg)


# ---------------------------------------------------------------------------
# nemo_evaluator_config param validation — real packaged IRs.
#
# lm-evaluation-harness.ifeval  container: nvcr.io/nvidia/eval-factory/lm-evaluation-harness:26.03
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


_IFEVAL_CONTAINER = "nvcr.io/nvidia/eval-factory/lm-evaluation-harness:26.03"
SIMPLE_EVALS_CONTAINER = "nvcr.io/nvidia/eval-factory/simple-evals:26.03"


class TestNemoEvaluatorParamValidation:
    def _make_cfg(self, task_name: str, params: dict, container: Optional[str] = None):
        """Build config for param validation. Omit container to use packaged IRs (default mapping)."""
        task = {
            "name": task_name,
            "nemo_evaluator_config": {"config": {"params": params}},
        }
        if container is not None:
            task["container"] = container
        return make_config(
            evaluation={
                "tasks": [task],
                "nemo_evaluator_config": {},
            }
        )

    @pytest.mark.parametrize("task_name, params", VALID_PARAM_CONFIGS)
    def test_valid_params_no_warning(self, task_name, params, caplog):
        # Given a config using params that exist in the task's command template
        cfg = self._make_cfg(task_name, params)
        # When validated against real packaged IRs (includes param validation)
        _validate_config_sections(cfg)
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
        # When validated against real packaged IRs (includes param validation)
        _validate_config_sections(cfg)
        # Then a warning is emitted naming the unknown param (run is not blocked)
        assert expected_in_warning in caplog.text

    def test_global_valid_task_invalid_emits_warning(self, caplog):
        # Given global nemo_evaluator_config has valid params
        # but task-level adds an invalid param on top (no container -> use packaged IRs)
        cfg = make_config(
            evaluation={
                "tasks": [
                    {
                        "name": "lm-evaluation-harness.ifeval",
                        "nemo_evaluator_config": {
                            "config": {"params": {"bad_task_param": 1}}
                        },
                    }
                ],
                "nemo_evaluator_config": {"config": {"params": {"parallelism": 4}}},
            }
        )
        # When validated (includes param validation)
        _validate_config_sections(cfg)
        # Then warning fired for the task-level bad param
        assert "bad_task_param" in caplog.text

    def test_global_invalid_task_valid_emits_warning(self, caplog):
        # Given global nemo_evaluator_config has an invalid param
        # but task-level only sets valid params (no container -> use packaged IRs)
        cfg = make_config(
            evaluation={
                "tasks": [
                    {
                        "name": "lm-evaluation-harness.ifeval",
                        "nemo_evaluator_config": {
                            "config": {"params": {"parallelism": 4}}
                        },
                    }
                ],
                "nemo_evaluator_config": {
                    "config": {"params": {"bad_global_param": 99}}
                },
            }
        )
        # When validated (includes param validation)
        _validate_config_sections(cfg)
        # Then warning fired for the global bad param (survives merge)
        assert "bad_global_param" in caplog.text

    @pytest.mark.parametrize(
        "container_image",
        [_IFEVAL_CONTAINER, SIMPLE_EVALS_CONTAINER],
        ids=["ifeval", "simple_evals"],
    )
    def test_task_with_container_calls_get_task_definition_with_that_image(
        self, container_image
    ):
        """When a task has a container override, get_task_definition_for_job is called with that image."""
        cfg = self._make_cfg(
            "lm-evaluation-harness.ifeval",
            {"parallelism": 4},
            container=container_image,
        )
        with patch(
            "nemo_evaluator_launcher.api.functional.get_task_definition_for_job"
        ) as mock_get_task_def:
            mock_get_task_def.return_value = {
                "command": "echo {{ parallelism }}",
                "harness": "lm-evaluation-harness",
            }
            _validate_config_sections(cfg)
        mock_get_task_def.assert_called_once()
        assert mock_get_task_def.call_args.kwargs["container"] == container_image

    def test_task_with_custom_container_unresolvable_emits_warning(self, caplog):
        """When a task overrides container (e.g. simple-evals) and definition cannot be resolved, a warning is logged."""
        cfg = make_config(
            evaluation={
                "tasks": [
                    {
                        "name": "lm-evaluation-harness.ifeval",
                        "container": SIMPLE_EVALS_CONTAINER,
                        "nemo_evaluator_config": {
                            "config": {"params": {"parallelism": 4}}
                        },
                    }
                ],
                "nemo_evaluator_config": {},
            }
        )
        with patch(
            "nemo_evaluator_launcher.api.functional.get_task_definition_for_job",
            side_effect=ValueError("Task not found in container mapping"),
        ) as mock_get_task_def:
            _validate_config_sections(cfg)
        assert "skipping param validation" in caplog.text
        mock_get_task_def.assert_called_once()
        assert mock_get_task_def.call_args.kwargs["container"] == SIMPLE_EVALS_CONTAINER
