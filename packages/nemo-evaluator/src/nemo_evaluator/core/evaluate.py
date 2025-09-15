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

import yaml

from nemo_evaluator.adapters.server import AdapterServerProcess
from nemo_evaluator.api.api_dataclasses import (
    Evaluation,
    EvaluationConfig,
    EvaluationResult,
    EvaluationTarget,
)
from nemo_evaluator.core.input import prepare_output_directory, validate_configuration
from nemo_evaluator.core.resources import monitor_memory_usage
from nemo_evaluator.core.utils import run_command
from nemo_evaluator.logging import get_logger

logger = get_logger(__name__)


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

    # Get cache directory from caching interceptor configuration
    cache_dir = None
    if (
        target_cfg.api_endpoint
        and target_cfg.api_endpoint.adapter_config
        and target_cfg.api_endpoint.adapter_config.interceptors
    ):
        for interceptor in target_cfg.api_endpoint.adapter_config.interceptors:
            if (
                interceptor.name == "caching"
                and interceptor.enabled
                and interceptor.config
                and interceptor.config.get("cache_dir")
            ):
                cache_dir = interceptor.config["cache_dir"]
                logger.info(f"Using caching interceptor cache_dir: {cache_dir}")
                break

    if not cache_dir:
        logger.info("No cache directory configured, token usage will not be collected")

    evaluation_result, metrics = monitor_memory_usage(
        run_evaluation_core, interval_ms=100, cache_dir=cache_dir
    )

    metrics_path = os.path.join(
        evaluation.config.output_dir, "eval_factory_metrics.json"
    )

    # Read existing metrics if file exists
    existing_metrics = {}
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r") as f:
                existing_metrics = json.load(f)
        except (json.JSONDecodeError, IOError):
            pass  # Start fresh if file is corrupted

    # Merge with existing metrics, using "evaluation" as the key
    # If evaluation key already exists, merge the metrics instead of overwriting
    if "evaluation" in existing_metrics:
        # Aggregate existing evaluation metrics with new ones
        existing_eval = existing_metrics["evaluation"]
        if isinstance(existing_eval, dict) and isinstance(metrics, dict):
            # Merge dictionaries with appropriate aggregation strategy
            merged_eval = existing_eval.copy()
            for key, value in metrics.items():
                if (
                    key in merged_eval
                    and isinstance(merged_eval[key], (int, float))
                    and isinstance(value, (int, float))
                ):
                    if key in ["runtime_seconds"]:
                        merged_eval[key] += value
                    elif key in ["peak_memory_bytes", "peak_tree_memory_bytes"]:
                        merged_eval[key] = max(merged_eval[key], value)
                    else:
                        merged_eval[key] += value
                elif key == "end_time":
                    merged_eval[key] = value
                elif key == "start_time":
                    merged_eval[key] = value
                else:
                    merged_eval[key] = value
            merged_metrics = {**existing_metrics, "evaluation": merged_eval}
        else:
            merged_metrics = {**existing_metrics, "evaluation": metrics}
    else:
        merged_metrics = {**existing_metrics, "evaluation": metrics}

    # Write merged metrics to file
    with open(metrics_path, "w") as f:
        json.dump(merged_metrics, f, indent=2)

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
