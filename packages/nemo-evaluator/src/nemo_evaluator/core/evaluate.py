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
import signal
import sys

import psutil
import yaml

from nemo_evaluator.adapters.server import AdapterServerProcess
from nemo_evaluator.api.api_dataclasses import (
    Evaluation,
    EvaluationConfig,
    EvaluationResult,
    EvaluationTarget,
)
from nemo_evaluator.core.input import prepare_output_directory, validate_configuration
from nemo_evaluator.core.resources import (
    aggregate_runtime_metrics,
    monitor_memory_usage,
)
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
    """
    Run an evaluation using configuration objects.

    Args:
        eval_cfg: Evaluation configuration object containing output directory,
                  parameters, and evaluation type
        target_cfg: Target configuration object containing API endpoint details
                    and adapter configuration

    Returns:
        EvaluationResult: Evaluation results and metadata
    """
    run_config = {
        "config": eval_cfg.model_dump(),
        "target": target_cfg.model_dump(),
    }
    evaluation = validate_configuration(run_config)
    prepare_output_directory(evaluation)

    def kill_all(signum=None, frame=None):
        """Kill all processes and exit."""
        logger.critical("FATAL: Terminating all processes...")

        parent = psutil.Process(os.getpid())  # current process
        children = parent.children(recursive=True)
        for child in children:
            if signum == signal.SIGINT:
                # Send SIGINT to children for immediate termination (skip post-eval hooks)
                child.send_signal(signal.SIGINT)
            else:
                # Send SIGTERM to children for graceful termination (run post-eval hooks)
                child.terminate()

        # Use faster timeout for keyboard interrupt (SIGINT)
        timeout = 1 if signum == signal.SIGINT else 5
        gone, alive = psutil.wait_procs(children, timeout=timeout)
        for child in alive:
            logger.warning(f"Force killing child process {child.pid}")
            child.kill()

        sys.exit(1)

    # Set up signal handlers
    signal.signal(signal.SIGTERM, kill_all)
    signal.signal(signal.SIGINT, kill_all)

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
        run_evaluation_core,
        interval_ms=100,
        cache_dir=cache_dir,
        output_dir=evaluation.config.output_dir,
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

    # Aggregate all run data from run_times directory
    aggregated_metrics = aggregate_runtime_metrics(evaluation.config.output_dir)

    if aggregated_metrics:
        runtime = aggregated_metrics.get("runtime_seconds", 0)
        inference_time = aggregated_metrics.get("inference_time_seconds", 0)
        scoring_time = aggregated_metrics.get("scoring_time_seconds", 0)
        logger.info(
            "Aggregated metrics",
            runtime_seconds=runtime,
            inference_time_seconds=inference_time,
            scoring_time_seconds=scoring_time,
            peak_memory_bytes=aggregated_metrics.get("peak_memory_bytes", 0),
            total_runs=aggregated_metrics.get("total_runs", 0),
        )

    # Use aggregated metrics if available, otherwise use current metrics
    final_metrics = aggregated_metrics if aggregated_metrics else metrics

    # Merge with existing metrics, using "evaluation" as the key
    # If evaluation key already exists, merge the metrics instead of overwriting
    if "evaluation" in existing_metrics:
        # Aggregate existing evaluation metrics with new ones
        existing_eval = existing_metrics["evaluation"]
        if isinstance(existing_eval, dict) and isinstance(final_metrics, dict):
            # Merge dictionaries with appropriate aggregation strategy
            merged_eval = existing_eval.copy()
            for key, value in final_metrics.items():
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
            merged_metrics = {**existing_metrics, "evaluation": final_metrics}
    else:
        merged_metrics = {**existing_metrics, "evaluation": final_metrics}

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
