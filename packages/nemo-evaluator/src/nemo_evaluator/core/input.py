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

import os
import pkgutil
from typing import Optional

import yaml

from nemo_evaluator.adapters.adapter_config import AdapterConfig
from nemo_evaluator.api.api_dataclasses import (
    Evaluation,
    EvaluationConfig,
    EvaluationTarget,
)
from nemo_evaluator.core.utils import (
    MisconfigurationError,
    deep_update,
    dotlist_to_dict,
)
from nemo_evaluator.logging import get_logger

__all__ = ["get_evaluation", "get_available_evaluations"]


_logger = get_logger(__name__)


def load_run_config(yaml_file: str) -> dict:
    """Load the run configuration from the YAML file.

    NOTE: The YAML config allows to override all the run configuration parameters.
    """
    with open(yaml_file, "r") as file:
        config = yaml.safe_load(file)
    return config


def _parse_cli_args(args) -> dict:
    """Parse CLI arguments into the run configuration format.

    NOTE: The CLI args allow to override a subset of the run configuration parameters.
    """
    config = {
        "config": {},
        "target": {
            "api_endpoint": {},
        },
    }

    if args.eval_type:
        config["config"]["type"] = args.eval_type
    if args.output_dir:
        config["config"]["output_dir"] = args.output_dir
    if args.api_key_name:
        config["target"]["api_endpoint"]["api_key"] = args.api_key_name
    if args.model_id:
        config["target"]["api_endpoint"]["model_id"] = args.model_id
    if args.model_type:
        config["target"]["api_endpoint"]["type"] = args.model_type
    if args.model_url:
        config["target"]["api_endpoint"]["url"] = args.model_url

    overrides = parse_override_params(args.overrides)
    # "--overrides takes precedence over other CLI args (e.g. --model_id)"
    config = deep_update(config, overrides, skip_nones=True)
    return config


def parse_override_params(override_params_str: Optional[str] = None) -> dict:
    if not override_params_str:
        return {}

    # Split the string into key-value pairs, handling commas inside quotes
    pairs = []
    current_pair = ""
    in_quotes = False
    quote_char = None

    for char in override_params_str:
        if char in ('"', "'") and not in_quotes:
            in_quotes = True
            quote_char = char
            current_pair += char
        elif char == quote_char and in_quotes:
            in_quotes = False
            quote_char = None
            current_pair += char
        elif char == "," and not in_quotes:
            pairs.append(current_pair.strip())
            current_pair = ""
        else:
            current_pair += char

    if current_pair:
        pairs.append(current_pair.strip())

    return dotlist_to_dict(pairs)


def get_framework_evaluations(filepath: str) -> tuple[str, dict, dict[str, Evaluation]]:
    framework = {}
    with open(filepath, "r") as f:
        framework = yaml.safe_load(f)

        framework_name = framework["framework"]["name"]
        pkg_name = framework["framework"]["pkg_name"]
        run_config_framework_defaults = framework["defaults"]
    run_config_framework_defaults["framework_name"] = framework_name
    run_config_framework_defaults["pkg_name"] = pkg_name
    evaluations = dict()
    for evaluation_dict in framework["evaluations"]:
        # Apply run config evaluation defaults onto the framework defaults
        run_config_task_defaults = deep_update(
            run_config_framework_defaults, evaluation_dict["defaults"], skip_nones=True
        )

        evaluation = Evaluation(
            **run_config_task_defaults,
        )
        evaluations[evaluation_dict["defaults"]["config"]["type"]] = evaluation

    return framework_name, run_config_framework_defaults, evaluations


# improve typing
def _get_framework_evaluations(
    def_file: str,
) -> tuple[dict[str, dict[str, Evaluation]], dict[str, dict], dict[str, Evaluation]]:
    # we should decide if this should raise at this point.
    # Probably not because this function is used with task invocation that might
    # be from different harness
    if not os.path.exists(def_file):
        raise ValueError(f"Framework Definition File does not exists at {def_file}")

    framework_eval_mapping = {}  # framework name -> set of tasks   | used in 'framework.task' invocation
    eval_name_mapping = {}  # task name      -> set of tasks   | used in 'task' invocation

    _logger.debug("Loading task definitions", filepath=def_file)
    (
        framework_name,
        framework_defaults,
        framework_evaluations,
    ) = get_framework_evaluations(def_file)
    framework_eval_mapping[framework_name] = framework_evaluations
    eval_name_mapping.update(framework_evaluations)
    framework_defaults = {framework_name: framework_defaults}

    return framework_eval_mapping, framework_defaults, eval_name_mapping


def merge_dicts(dict1, dict2):
    merged = {}
    all_keys = set(dict1.keys()) | set(dict2.keys())

    for key in all_keys:
        v1 = dict1.get(key)
        v2 = dict2.get(key)

        if key in dict1 and key in dict2:
            result = []
            # Handle case where value is a list or not
            if isinstance(v1, list):
                result.extend(v1)
            elif v1 is not None:
                result.append(v1)
            if isinstance(v2, list):
                result.extend(v2)
            elif v2 is not None:
                result.append(v2)
            merged[key] = result
        elif key in dict1:
            merged[key] = v1
        else:
            merged[key] = v2
    return merged


def get_available_evaluations() -> tuple[
    dict[str, dict[str, Evaluation]], dict[str, Evaluation], dict
]:
    """
    Returns all available evaluations in Evaluation objects
    .. important:: Only evaluations from installed wheels are being returned.

    Returns:
        tuple[ dict[str, dict[str, Evaluation]], dict[str, Evaluation], dict ]:
        Tuple with the following elements:
        1. Mapping: harness name -> tasks (dict)
        2. Mapping: harness name -> default configs (for non exposed tasks). Returned Evaluation should serve as a blueprint
        3. Mapping: task name -> list of Evaluations
    """

    all_framework_eval_mappings = {}
    all_framework_defaults = {}
    all_eval_name_mapping = {}
    try:
        import core_evals

        core_evals_pkg = list(pkgutil.iter_modules(core_evals.__path__))
    except ImportError:
        core_evals_pkg = []

    for pkg in core_evals_pkg:
        (
            framework_eval_mapping,
            framework_defaults,
            eval_name_mapping,
        ) = _get_framework_evaluations(
            os.path.join(pkg.module_finder.path, pkg.name, "framework.yml")
        )
        all_framework_eval_mappings.update(framework_eval_mapping)
        all_framework_defaults.update(framework_defaults)
        all_eval_name_mapping = merge_dicts(all_eval_name_mapping, eval_name_mapping)

    return (
        all_framework_eval_mappings,
        all_framework_defaults,
        all_eval_name_mapping,
    )


def check_task_invocation(run_config: dict):
    """
    Checks if task invocation is formatted correctly and a harness or task is available:


    Args:
        run_config (dict): _description_

    Raises:
        MisconfigurationError: if eval type does not follow specified format
        MisconfigurationError: if provided framework is not available
        MisconfigurationError: if provided task is not available
    """
    # evaluation type can be either 'framework.task' or 'task'

    eval_type_components = run_config["config"]["type"].split(".")
    if len(eval_type_components) == 2:  # framework.task invocation
        framework_name, evaluation_name = eval_type_components
    elif len(eval_type_components) == 1:  # task invocation
        framework_name, evaluation_name = None, eval_type_components[0]
    else:
        raise MisconfigurationError(
            "eval_type must follow 'framework_name.evaluation_name'. No additional dots are allowed."
        )

    framework_evals_mapping, _, all_evals_mapping = get_available_evaluations()

    # framework.task invocation
    if framework_name:
        try:
            framework_evals_mapping[framework_name]
        except KeyError:
            raise MisconfigurationError(
                f"Unknown framework {framework_name}. Frameworks available: {', '.join(framework_evals_mapping.keys())}"
            )
    else:
        try:
            all_evals_mapping[evaluation_name]
        except KeyError:
            raise MisconfigurationError(
                f"Unknown evaluation {evaluation_name}. Evaluations available: {', '.join(all_evals_mapping.keys())}"
            )


def check_required_default_missing(run_config: dict):
    if run_config["config"].get("type") is None:
        raise MisconfigurationError(
            "Missing required argument: config.type (cli: --eval_type)"
        )
    if run_config["config"].get("output_dir") is None:
        raise MisconfigurationError(
            "Missing required argument: config.output_dir (cli: --output_dir)"
        )


def check_adapter_config(run_config):
    adapter_config: AdapterConfig | None = AdapterConfig.get_validated_config(
        run_config
    )

    if adapter_config:
        if run_config["target"].get("api_endpoint") is None:
            raise MisconfigurationError(
                "You need to define target.api_endpoint in order to use an adapter (cli: --model_id, --model_url, --model_type)"
            )
        if run_config["target"]["api_endpoint"].get("url") is None:
            raise MisconfigurationError(
                "You need to define target.api_endpoint.url in order to use an adapter (cli: --model_url)"
            )


def get_evaluation(
    evaluation_config: EvaluationConfig, target_config: EvaluationTarget
) -> Evaluation:  # type: ignore
    """Infers harness information from evaluation config and wraps it
    into Evaluation

    Args:
        evaluation_config (EvaluationConfig): _description_

    Returns:
        Evaluation: EvalConfig
    """
    eval_type_components = evaluation_config.type.split(".")
    if len(eval_type_components) == 2:  # framework.task invocation
        framework_name, evaluation_name = eval_type_components
    elif len(eval_type_components) == 1:  # task invocation
        framework_name, evaluation_name = None, eval_type_components[0]
    else:
        raise

    all_framework_eval_mappings, all_framework_defaults, all_eval_name_mapping = (
        get_available_evaluations()
    )

    # First, get default Evaluation
    # "framework.task" invocation
    if framework_name:
        try:
            default_evaluation = all_framework_eval_mappings[framework_name][
                evaluation_name
            ]
        except KeyError:
            default_evaluation = Evaluation(**all_framework_defaults[framework_name])
            evaluation_config.type = evaluation_name
            default_evaluation.config.params.task = evaluation_name
    else:
        if isinstance(all_eval_name_mapping[evaluation_name], list):
            framework_handlers = [
                evaluation.framework_name
                for evaluation in all_eval_name_mapping[evaluation_name]
            ]
            raise MisconfigurationError(
                f"{evaluation_name} is available in multiple frameworks: {','.join(framework_handlers)}. \
Please indicate which implementation you would like to choose by using 'framework.task' invocation. \
For example: {framework_handlers[0]}.{evaluation_name}. "
            )
        default_evaluation = all_eval_name_mapping[evaluation_name]

    default_configuration = default_evaluation.model_dump(exclude_none=True)
    user_configuration = {
        "config": evaluation_config.model_dump(),
        "target": target_config.model_dump(),
    }
    merged_configuration = deep_update(
        default_configuration, user_configuration, skip_nones=True
    )
    return Evaluation(**merged_configuration)


def check_type_compatibility(evaluation: Evaluation):
    if (
        evaluation.config.supported_endpoint_types is not None
        and evaluation.target.api_endpoint.type
        not in evaluation.config.supported_endpoint_types
    ):
        if evaluation.target.api_endpoint.type is None:
            raise MisconfigurationError(
                "target.api_endpoint.type should be defined and match one of the endpoint "
                f"types supported by the benchmark: '{evaluation.config.supported_endpoint_types}'",
            )
        if (
            evaluation.target.api_endpoint.type
            not in evaluation.config.supported_endpoint_types
        ):
            raise MisconfigurationError(
                f"The benchmark '{evaluation.config.type}' does not support the model type '{evaluation.target.api_endpoint.type}'. "
                f"The benchmark supports '{evaluation.config.supported_endpoint_types}'."
            )

    if evaluation.target.api_endpoint.type:
        # Check this only if the model is really required (to accomodate for non-model evals)
        if evaluation.target.api_endpoint.url is None:
            raise MisconfigurationError(
                "target.api_endpoint.url (CLI: --model_url) should be defined to run model evaluation!"
            )
        if evaluation.target.api_endpoint.model_id is None:
            raise MisconfigurationError(
                "target.api_endpoint.model_id (CLI: --model_id) should be defined to run model evaluation!"
            )


def prepare_output_directory(evaluation: Evaluation):
    try:
        os.makedirs(evaluation.config.output_dir, exist_ok=True)
    except OSError as error:
        print(f"An error occurred while creating output directory: {error}")

    with open(os.path.join(evaluation.config.output_dir, "run_config.yml"), "w") as f:
        yaml.dump(evaluation.model_dump(), f)


def validate_configuration(run_config: dict) -> Evaluation:
    """Validates requested task through a dataclass. Additionally,
    handles creation of task folowing the logic:

    - evaluation type can be either 'framework.task' or 'task'
    - FDF stands for Framework Definition File


    Args:
        run_config_cli_overrides (dict): run configuration merged from config file and CLI

    Raises:

    """
    check_required_default_missing(run_config)
    check_task_invocation(run_config)
    check_adapter_config(run_config)
    evaluation = get_evaluation(
        EvaluationConfig(**run_config["config"]),
        EvaluationTarget(**run_config["target"]),
    )
    check_type_compatibility(evaluation)
    _logger.info(f"User-invoked config: \n{yaml.dump(evaluation.model_dump())}")
    return evaluation
