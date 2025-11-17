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

import argparse
import os
import pkgutil
import sys

import yaml

# Import logging to ensure centralized logging is configured
from nemo_evaluator import logging  # noqa: F401
from nemo_evaluator.adapters.adapter_config import AdapterConfig
from nemo_evaluator.api.api_dataclasses import (
    EvaluationConfig,
    EvaluationMetadata,
    EvaluationTarget,
)
from nemo_evaluator.core.evaluate import evaluate
from nemo_evaluator.core.input import (
    _get_framework_evaluations,
    _parse_cli_args,
    load_run_config,
    validate_configuration,
)

__all__ = []

from .utils import deep_update


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug", action="store_true", help="Debug the core_evals script"
    )
    subparsers = parser.add_subparsers(help="Functions")
    parser_ls = subparsers.add_parser("ls", help="List available evaluation types")
    parser_ls.set_defaults(command="ls")

    parser_run = subparsers.add_parser("run_eval", help="Run the evaluation")
    parser_run.add_argument("--eval_type", type=str, help="Run config.: task name")
    parser_run.add_argument("--model_id", type=str, help="Run config.: model name")
    parser_run.add_argument(
        "--model_type",
        type=str,
        help="Run config.: endpoint type",
        choices=["chat", "completions", "vlm", "embedding"],
    )
    parser_run.add_argument("--model_url", type=str, help="Run config.: model URI")
    parser_run.add_argument(
        "--output_dir", type=str, help="Run config.: results output dir."
    )
    parser_run.add_argument(
        "--api_key_name",
        type=str,
        help="Run config.: API key env variable name (optional)",
        default=None,
    )
    parser_run.add_argument(
        "--run_config",
        type=str,
        help="Load the run configuration from the YAML file (optional and overridden by the cli arguments)",
        default=None,
    )
    parser_run.add_argument(
        "--overrides",
        type=str,
        help="Comma-separated dot-style parameters to override config values (overriding values from run_config and CLI args)",
        default=None,
    )
    parser_run.add_argument(
        "--dry_run",
        action="store_true",
        help="Shows rendered config and command instead of running",
        default=False,
    )
    parser_run.set_defaults(command="run_eval")
    return parser


def get_args() -> argparse.Namespace:
    parser = get_parser()

    args = parser.parse_args()

    if args.debug:
        # Override with debug level if --debug flag is used
        from nemo_evaluator.logging import get_logger

        logger = get_logger(__name__)
        logger.warning(
            "This flag is deprecated and will be removed in the future, please set environment variable NEMO_EVALUATOR_LOG_LEVEL=DEBUG instead!"
        )
        logger.warning("Setting NEMO_EVALUATOR_LOG_LEVEL=DEBUG")
        os.environ["NEMO_EVALUATOR_LOG_LEVEL"] = "DEBUG"

    if "command" not in args:
        parser.print_help()
        sys.exit(0)
    return args


def show_available_tasks() -> None:
    """
    Prints all available evaluations in the format of::

        {harness1}:
            * benchmark A
            * benchmark B
        {harness2}:
            * benchmark A
            * benchmark B
        ...

    .. important:: Only evaluations from installed wheels are being displayed.
    """
    try:
        import core_evals

        core_evals_pkg = list(pkgutil.iter_modules(core_evals.__path__))
    except ImportError:
        core_evals_pkg = []

    if not core_evals_pkg:
        print("NO evaluation packages are installed.")

    for pkg in core_evals_pkg:
        framework_eval_mapping, *_ = _get_framework_evaluations(
            os.path.join(pkg.module_finder.path, pkg.name, "framework.yml")
        )
        for ind_pkg in framework_eval_mapping.keys():
            print(f"{ind_pkg}: ")
            for task in framework_eval_mapping[ind_pkg].keys():
                print(f"  * {task}")


def run(args) -> None:
    run_config = load_run_config(args.run_config) if args.run_config else {}
    # CLI args take precedence over YAML run config
    run_config = deep_update(run_config, _parse_cli_args(args), skip_nones=True)
    if args.dry_run:
        evaluation = validate_configuration(run_config)
        print("Rendered config:\n")
        config = evaluation.model_dump()
        print(yaml.dump(config, sort_keys=False, default_flow_style=False, indent=2))
        print("\nRendered command:\n")
        cmd = evaluation.render_command()
        print(cmd)
        exit(0)

    metadata_cfg: EvaluationMetadata | None = run_config.get("metadata")
    adapter_config = AdapterConfig.get_validated_config(run_config)
    eval_cfg = EvaluationConfig(**run_config["config"])
    target_cfg = EvaluationTarget(**run_config["target"])
    target_cfg.api_endpoint.adapter_config = adapter_config

    evaluate(eval_cfg=eval_cfg, target_cfg=target_cfg, metadata=metadata_cfg)


def run_eval() -> None:
    """
    CLI entry point for running evaluations.

    This function parses command line arguments and executes evaluations.
    It does not take parameters directly - all configuration is passed via CLI arguments.

    CLI Arguments:
        --eval_type: Type of evaluation to run (e.g., "mmlu_pro", "gsm8k")
        --model_id: Model identifier (e.g "meta/llama-3.1-8b-instruct")
        --model_url: API endpoint URL (e.g "https://integrate.api.nvidia.com/v1/chat/completions" for chat endpoint type)
        --model_type: Endpoint type ("chat", "completions", "vlm", "embedding")
        --api_key_name: Environment variable name for API key integration with endpoints (optional)
        --output_dir: Output directory for results
        --run_config: Path to YAML Run Configuration file (optional)
        --overrides: Comma-separated dot-style parameter overrides (optional)
        --dry_run: Show rendered config without running (optional)
        --debug: Enable debug logging (optional, deprecated, use NV_LOG_LEVEL=DEBUG env var)

    Usage:
        run_eval()  # Parses sys.argv automatically
    """
    args = get_args()

    if sys.argv[0].endswith("eval-factory"):
        from nemo_evaluator.logging import get_logger

        logger = get_logger(__name__)
        logger.warning(
            "You appear to be using a deprecated eval_factory command. Please use nemo-evaluator instead with the same arguments. eval-factory command is going to be removed before 25.12 containers are released."
        )

    if args.command == "ls":
        show_available_tasks()
    elif args.command == "run_eval":
        run(args)


if __name__ == "__main__":
    run_eval()
