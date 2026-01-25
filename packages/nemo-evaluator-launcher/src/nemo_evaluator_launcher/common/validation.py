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
"""Comprehensive pre-submission validation for nemo-evaluator-launcher.

This module provides validation that runs BEFORE job submission to catch errors
early and avoid wasting cluster resources. The validation focuses on:
- Executor-specific required fields (hostname, account, output_dir, etc.)
- Deployment-specific required fields (checkpoint, model_name, etc.)
- Task existence verification
- Environment variable checks
- Common configuration mistakes

Usage:
    from nemo_evaluator_launcher.common.validation import validate_config

    result = validate_config(cfg)
    if not result.valid:
        result.print_errors()
        sys.exit(1)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from difflib import get_close_matches
from typing import Any

from omegaconf import DictConfig, OmegaConf

from nemo_evaluator_launcher.common.metadata import (
    DEPLOYMENT_REQUIRED_CONFIG_FIELDS,
    EXECUTOR_REQUIRED_CONFIG_FIELDS,
)


@dataclass
class ValidationError:
    """A single validation error with context and suggestions."""

    path: str
    message: str
    suggestion: str | None = None
    severity: str = "error"  # "error" or "warning"


@dataclass
class ValidationResult:
    """Result of configuration validation."""

    errors: list[ValidationError] = field(default_factory=list)
    warnings: list[ValidationError] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        """Return True if there are no errors (warnings are OK)."""
        return len(self.errors) == 0

    def add_error(
        self,
        path: str,
        message: str,
        suggestion: str | None = None,
    ) -> None:
        """Add an error to the result."""
        self.errors.append(
            ValidationError(path=path, message=message, suggestion=suggestion)
        )

    def add_warning(
        self,
        path: str,
        message: str,
        suggestion: str | None = None,
    ) -> None:
        """Add a warning to the result."""
        self.warnings.append(
            ValidationError(
                path=path, message=message, suggestion=suggestion, severity="warning"
            )
        )

    def print_errors(self) -> None:
        """Print validation errors and warnings in a formatted way."""
        # Import here to avoid circular imports and keep validation lightweight
        from nemo_evaluator_launcher.common.printing_utils import (
            bold,
            cyan,
            grey,
            red,
            yellow,
        )

        if not self.errors and not self.warnings:
            return

        # Print header
        print()
        print(bold(red("Configuration Validation Failed")))
        print()

        # Print errors
        if self.errors:
            print(red(f"Errors ({len(self.errors)}):"))
            print()
            for err in self.errors:
                print(f"  {red('x')} {bold(err.path)}")
                print(f"    {err.message}")
                if err.suggestion:
                    print(f"    {cyan('Hint:')} {err.suggestion}")
                print()

        # Print warnings
        if self.warnings:
            print(yellow(f"Warnings ({len(self.warnings)}):"))
            print()
            for warn in self.warnings:
                print(f"  {yellow('!')} {bold(warn.path)}")
                print(f"    {warn.message}")
                if warn.suggestion:
                    print(f"    {cyan('Hint:')} {warn.suggestion}")
                print()

        print(grey("Run with --help to see all available options."))
        print()


# Required fields per executor type (imported from metadata module)
EXECUTOR_REQUIRED_FIELDS = EXECUTOR_REQUIRED_CONFIG_FIELDS

# Required fields per deployment type (imported from metadata module)
DEPLOYMENT_REQUIRED_FIELDS = DEPLOYMENT_REQUIRED_CONFIG_FIELDS

# CLI flag suggestions for common fields
CLI_FLAG_SUGGESTIONS: dict[str, str] = {
    # Executor fields
    "execution.hostname": "--slurm-hostname <hostname>",
    "execution.account": "--slurm-account <account>",
    "execution.output_dir": "--output-dir <path>",
    "execution.partition": "--slurm-partition <partition>",
    "execution.walltime": "--slurm-walltime <HH:MM:SS>",
    # Deployment fields
    "deployment.checkpoint_path": "--checkpoint <path>",
    "deployment.served_model_name": "--model-name <name>",
    "deployment.model_name": "--nim-model <name>",
    "deployment.tensor_parallel_size": "--tp <size>",
    "deployment.data_parallel_size": "--dp <size>",
    # Target fields
    "target.api_endpoint.model_id": "--model <model-id>",
    "target.api_endpoint.url": "--url <url>",
}


def _get_nested(obj: Any, path: str, default: Any = None) -> Any:
    """Get a nested value from a dict/DictConfig using dot notation."""
    keys = path.split(".")
    current = obj
    for key in keys:
        if isinstance(current, (dict, DictConfig)):
            if key not in current:
                return default
            current = current[key]
        else:
            return default
    return current


def _is_missing(obj: Any, path: str) -> bool:
    """Check if a value at path is missing (??? in OmegaConf or None/missing)."""
    keys = path.split(".")
    current = obj

    for i, key in enumerate(keys):
        if isinstance(current, DictConfig):
            if key not in current:
                return True
            # Check for OmegaConf MISSING marker
            if OmegaConf.is_missing(current, key):
                return True
            current = current[key]
        elif isinstance(current, dict):
            if key not in current:
                return True
            current = current[key]
        else:
            return True

    # Also treat None and empty string as "missing" for required fields
    return current is None or current == ""


def _find_similar_tasks(query: str, valid_tasks: set[str], n: int = 3) -> list[str]:
    """Find similar task names using fuzzy matching."""
    matches = get_close_matches(query, list(valid_tasks), n=n, cutoff=0.5)
    return matches


def _get_cli_suggestion(path: str) -> str | None:
    """Get CLI flag suggestion for a config path."""
    return CLI_FLAG_SUGGESTIONS.get(path)


def validate_executor(cfg: Any, result: ValidationResult) -> None:
    """Validate executor-specific configuration."""
    executor_type = _get_nested(cfg, "execution.type", "local")

    required_fields = EXECUTOR_REQUIRED_FIELDS.get(executor_type, [])

    for field_name in required_fields:
        path = f"execution.{field_name}"
        if _is_missing(cfg, path):
            suggestion = _get_cli_suggestion(path)
            result.add_error(
                path=path,
                message=f"Required for '{executor_type}' executor",
                suggestion=suggestion,
            )


def validate_deployment(cfg: Any, result: ValidationResult) -> None:
    """Validate deployment-specific configuration."""
    deployment_type = _get_nested(cfg, "deployment.type", "none")

    required_fields = DEPLOYMENT_REQUIRED_FIELDS.get(deployment_type, [])

    for field_name in required_fields:
        path = f"deployment.{field_name}"
        if _is_missing(cfg, path):
            suggestion = _get_cli_suggestion(path)
            result.add_error(
                path=path,
                message=f"Required for '{deployment_type}' deployment",
                suggestion=suggestion,
            )

    # Check for conflicting target.api_endpoint when deployment is configured
    if deployment_type != "none":
        if not _is_missing(cfg, "target.api_endpoint.url"):
            result.add_warning(
                path="target.api_endpoint.url",
                message="URL is set but deployment will auto-configure the endpoint",
                suggestion="Remove target.api_endpoint.url when using a deployment",
            )
        if not _is_missing(cfg, "target.api_endpoint.model_id"):
            result.add_warning(
                path="target.api_endpoint.model_id",
                message="model_id is set but deployment will auto-configure the endpoint",
                suggestion="Remove target.api_endpoint.model_id when using a deployment",
            )


def validate_target_api_endpoint(cfg: Any, result: ValidationResult) -> None:
    """Validate target API endpoint configuration when deployment is 'none'."""
    deployment_type = _get_nested(cfg, "deployment.type", "none")

    if deployment_type != "none":
        # Deployment handles the target configuration
        return

    # For deployment=none, need either:
    # 1. target.api_endpoint.model_id (required for NVIDIA API Catalog)
    # 2. target.api_endpoint.url (optional, has default)
    if _is_missing(cfg, "target.api_endpoint.model_id"):
        result.add_error(
            path="target.api_endpoint.model_id",
            message="Required when not using a deployment",
            suggestion="--model <model-id> (e.g., meta/llama-3.2-3b-instruct)",
        )


def validate_tasks(cfg: Any, result: ValidationResult) -> None:
    """Validate that specified tasks exist in the mapping."""
    # Handle both formats:
    # 1. evaluation.tasks: [{name: task1}, {name: task2}]  (preferred)
    # 2. evaluation: [{name: task1}, {name: task2}]  (legacy format)
    tasks = _get_nested(cfg, "evaluation.tasks", None)
    if tasks is None:
        # Try legacy format where evaluation is directly a list
        evaluation = _get_nested(cfg, "evaluation", [])
        if isinstance(evaluation, (list, tuple)):
            tasks = evaluation
        else:
            tasks = []

    if not tasks:
        result.add_error(
            path="evaluation.tasks",
            message="No tasks specified",
            suggestion="--task <task-name> (e.g., --task ifeval --task gsm8k)",
        )
        return

    # Try to load the tasks mapping to validate task names
    try:
        from nemo_evaluator_launcher.common.mapping import load_tasks_mapping

        mapping = load_tasks_mapping()
        valid_tasks = {key[1] for key in mapping.keys()}  # Extract task names

        for task in tasks:
            task_name = task.name if hasattr(task, "name") else task.get("name", "")
            if task_name and task_name not in valid_tasks:
                similar = _find_similar_tasks(task_name, valid_tasks)
                if similar:
                    suggestion = f"Did you mean: {', '.join(similar)}?"
                else:
                    suggestion = "Run 'nel ls tasks' to see available tasks"

                result.add_error(
                    path=f"evaluation.tasks[{task_name}]",
                    message=f"Task '{task_name}' not found",
                    suggestion=suggestion,
                )
    except Exception:
        # If we can't load the mapping, skip task validation
        # The error will be caught later during execution
        pass


def validate_env_vars(cfg: Any, result: ValidationResult) -> None:
    """Validate that required environment variables are set."""
    deployment_type = _get_nested(cfg, "deployment.type", "none")

    # Check NGC_API_KEY for NVIDIA API Catalog (deployment=none)
    if deployment_type == "none":
        api_key_env = _get_nested(cfg, "target.api_endpoint.api_key_name", "NGC_API_KEY")
        if api_key_env and not os.environ.get(api_key_env):
            result.add_warning(
                path="environment",
                message=f"Environment variable '{api_key_env}' is not set",
                suggestion=f"export {api_key_env}=<your-api-key>",
            )

    # Check task-specific env vars
    # Handle both formats (evaluation.tasks and legacy evaluation list)
    tasks = _get_nested(cfg, "evaluation.tasks", None)
    if tasks is None:
        evaluation = _get_nested(cfg, "evaluation", [])
        if isinstance(evaluation, (list, tuple)):
            tasks = evaluation
        else:
            tasks = []
    for task in tasks:
        task_name = task.name if hasattr(task, "name") else task.get("name", "")
        env_vars = task.env_vars if hasattr(task, "env_vars") else task.get("env_vars", {})

        if env_vars:
            for env_name, env_value in env_vars.items():
                if isinstance(env_value, str) and not os.environ.get(env_value):
                    result.add_warning(
                        path=f"evaluation.tasks[{task_name}].env_vars.{env_name}",
                        message=f"Environment variable '{env_value}' is not set",
                        suggestion=f"export {env_value}=<value>",
                    )


def validate_missing_values(cfg: Any, result: ValidationResult) -> None:
    """Check for ??? (MISSING) values in the configuration."""

    def _check_missing(obj: Any, path: str = "") -> None:
        if OmegaConf.is_dict(obj):
            # Use keys() to iterate without resolving values (which would throw on MISSING)
            for key in obj.keys():
                current_path = f"{path}.{key}" if path else str(key)
                if OmegaConf.is_missing(obj, key):
                    suggestion = _get_cli_suggestion(current_path)
                    result.add_error(
                        path=current_path,
                        message="Value is required (marked as ???)",
                        suggestion=suggestion,
                    )
                else:
                    # Only access the value if it's not missing
                    value = obj[key]
                    _check_missing(value, current_path)
        elif OmegaConf.is_list(obj):
            for i, value in enumerate(obj):
                current_path = f"{path}[{i}]"
                _check_missing(value, current_path)

    if OmegaConf.is_config(cfg):
        _check_missing(cfg)


def validate_config(cfg: Any, skip_task_validation: bool = False) -> ValidationResult:
    """Comprehensive configuration validation.

    This validates the configuration BEFORE job submission to catch errors early.
    All validation is performed in a single pass to provide complete feedback.

    Args:
        cfg: The configuration object (OmegaConf DictConfig or dict).
        skip_task_validation: If True, skip task existence validation.
            Useful when tasks will be loaded from a custom container.

    Returns:
        ValidationResult with all errors and warnings.

    Example:
        result = validate_config(cfg)
        if not result.valid:
            result.print_errors()
            sys.exit(1)
    """
    result = ValidationResult()

    # 1. Check for MISSING (???) values
    validate_missing_values(cfg, result)

    # 2. Validate executor requirements
    validate_executor(cfg, result)

    # 3. Validate deployment requirements
    validate_deployment(cfg, result)

    # 4. Validate target API endpoint (when deployment=none)
    validate_target_api_endpoint(cfg, result)

    # 5. Validate tasks exist
    if not skip_task_validation:
        validate_tasks(cfg, result)

    # 6. Check environment variables
    validate_env_vars(cfg, result)

    return result
