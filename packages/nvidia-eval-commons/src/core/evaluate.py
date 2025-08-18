# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import importlib
import json
import os

import structlog
import yaml
from nvidia_eval_commons.adapters.server import AdapterServerProcess
from nvidia_eval_commons.api.api_dataclasses import (
    Evaluation,
    EvaluationConfig,
    EvaluationResult,
    EvaluationTarget,
)
from nvidia_eval_commons.core.input import (
    prepare_output_directory,
    validate_configuration,
)
from nvidia_eval_commons.core.resources import monitor_memory_usage
from nvidia_eval_commons.core.utils import run_command

logger = structlog.get_logger(__name__)


def parse_output(evaluation: Evaluation) -> EvaluationResult:
    # create a module name that is importable
    output_module = importlib.import_module(f"core_evals.{evaluation.pkg_name}.output")
    return output_module.parse_output(evaluation.config.output_dir)


def evaluate(
    eval_cfg: EvaluationConfig, target_cfg: EvaluationTarget
) -> EvaluationResult:
    run_config = {
        "config": eval_cfg.model_dump(),
        "target": target_cfg.model_dump(),
    }
    evaluation = validate_configuration(run_config)
    prepare_output_directory(evaluation)

    def run_evaluation_core():
        with AdapterServerProcess(evaluation):
            cmd = evaluation.render_command()

            run_command(cmd, verbose=True, propagate_errors=True)

            evaluation_result = parse_output(evaluation)
            return evaluation_result

    evaluation_result, metrics = monitor_memory_usage(
        run_evaluation_core, interval_ms=100
    )

    metrics_path = os.path.join(
        evaluation.config.output_dir, "eval_factory_metrics.json"
    )
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    evaluation_result_dict = {
        "git_hash": os.getenv("CORE_EVALS_GIT_HASH"),
        "command": evaluation.render_command(),
        **run_config,
        "results": evaluation_result.model_dump(exclude_none=True),
    }

    logger.info(yaml.dump(evaluation_result_dict))

    with open(os.path.join(evaluation.config.output_dir, "results.yml"), "w") as f:
        yaml.dump(evaluation_result_dict, f)

    return evaluation_result
