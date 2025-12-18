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
import pathlib
from typing import Any

import yaml

from nemo_evaluator_launcher.common.container_metadata import (
    TaskIntermediateRepresentation,
    load_tasks_from_tasks_file,
)
from nemo_evaluator_launcher.common.logging_utils import logger


def _load_packaged_resource(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
    """Deprecated: mapping.toml support was removed in favor of packaged IRs."""
    raise RuntimeError(
        "mapping.toml is no longer supported. Use packaged IRs (all_tasks_irs.yaml) instead."
    )


def _process_mapping(mapping_toml: dict) -> dict:
    """Process the raw mapping TOML into the expected format.

    Args:
        mapping_toml: Raw mapping TOML data.
    Returns:
        dict: Processed mapping in the expected format.
    """
    mapping = {}
    for harness_name, harness_data in mapping_toml.items():
        # Skip entries that don't have the expected structure
        if not isinstance(harness_data, dict):
            logger.warning(
                "Skipping invalid harness entry",
                harness_name=harness_name,
                reason="harness_data is not a dict",
            )
            continue

        # Check if tasks field exists
        if "tasks" not in harness_data:
            logger.warning(
                "Skipping harness entry without tasks",
                harness_name=harness_name,
            )
            continue

        if not isinstance(harness_data["tasks"], dict):
            logger.warning(
                "Skipping invalid harness entry",
                harness_name=harness_name,
                reason="tasks is not a dict",
            )
            continue

        # Get container, which may be optional
        container = harness_data.get("container")
        if not container:
            logger.debug(
                "Harness entry without container",
                harness_name=harness_name,
            )

        for endpoint_type, harness_tasks in harness_data["tasks"].items():
            if not isinstance(harness_tasks, dict):
                logger.warning(
                    "Skipping invalid endpoint type",
                    harness_name=harness_name,
                    endpoint_type=endpoint_type,
                    reason="harness_tasks is not a dict",
                )
                continue

            for task_name, task_data in harness_tasks.items():
                if not isinstance(task_data, dict):
                    logger.warning(
                        "Skipping invalid task entry",
                        harness_name=harness_name,
                        task_name=task_name,
                        reason="task_data is not a dict",
                    )
                    continue

                key = (harness_name, task_name)
                if key in mapping:
                    raise KeyError(
                        f"(harness,task)-tuple key {repr(key)} already exists in the mapping"
                    )

                # Validate required fields exist in task_data
                # task_name and harness_name are already validated above
                # endpoint_type is validated as a key in harness_tasks
                # task_data must be a dict (validated above)

                mapping[key] = {
                    "task": task_name,
                    "harness": harness_name,
                    "endpoint_type": endpoint_type,
                }
                # Only add container if it exists
                if container:
                    mapping[key]["container"] = container

                # Validate task_data keys before updating
                for task_data_key in task_data.keys():
                    if task_data_key in mapping[key]:
                        raise KeyError(
                            f"{repr(task_data_key)} is not allowed as key under {repr(key)} in the mapping"
                        )
                    # Validate that task_data values are valid types (basic check)
                    if task_data_key not in ("description", "type") and not isinstance(
                        task_data[task_data_key],
                        (str, int, float, bool, dict, list, type(None)),
                    ):
                        logger.warning(
                            "Unexpected value type in task_data",
                            harness_name=harness_name,
                            task_name=task_name,
                            key=task_data_key,
                            value_type=type(task_data[task_data_key]).__name__,
                        )

                mapping[key].update(task_data)
    return mapping


def _extract_tasks_from_framework_yml(
    framework_yml_content: str, harness_name: str, container: str
) -> dict[tuple[str, str], dict]:
    """Extract tasks from framework.yml content and return as mapping entries.

    Args:
        framework_yml_content: YAML content from framework.yml file
        harness_name: Name of the harness
        container: Container image string

    Returns:
        Dictionary mapping (harness_name, task_name) to task configuration
    """
    tasks = {}
    try:
        framework_data = yaml.safe_load(framework_yml_content)
        if not framework_data or "evaluations" not in framework_data:
            logger.warning(
                "No evaluations found in framework.yml",
                harness=harness_name,
                container=container,
            )
            return tasks

        evaluations = framework_data.get("evaluations", [])
        for eval_config in evaluations:
            task_name = eval_config.get("name")
            description = eval_config.get("description", "")

            if not task_name:
                continue

            # Extract endpoint types from the evaluation config
            defaults = eval_config.get("defaults", {})
            config = defaults.get("config", {})
            supported_endpoint_types = config.get("supported_endpoint_types", ["chat"])
            task_type = config.get("type", "")  # Extract type from defaults.config.type

            # Use first endpoint type (mapping key is (harness, task), so one entry per task)
            endpoint_type = (
                supported_endpoint_types[0] if supported_endpoint_types else "chat"
            )

            key = (harness_name, task_name)
            # Only add if not already in mapping (don't override existing entries)
            if key not in tasks:
                tasks[key] = {
                    "task": task_name,
                    "harness": harness_name,
                    "container": container,
                    "endpoint_type": endpoint_type,
                    "description": description,
                    "type": task_type,  # Store type from defaults.config.type
                }
                # Merge any additional config from defaults
                if defaults:
                    tasks[key].update(defaults)

        logger.info(
            "Extracted tasks from framework.yml",
            harness=harness_name,
            container=container,
            num_tasks=len(tasks),
        )
    except yaml.YAMLError as e:
        logger.warning(
            "Failed to parse framework.yml",
            harness=harness_name,
            container=container,
            error=str(e),
        )
    except Exception as e:
        logger.warning(
            "Error extracting tasks from framework.yml",
            harness=harness_name,
            container=container,
            error=str(e),
        )

    return tasks


def _convert_irs_to_mapping_format(
    tasks: list[TaskIntermediateRepresentation],
) -> dict[tuple[str, str], dict]:
    """Convert list of TaskIntermediateRepresentation objects to mapping dict format.

    Args:
        tasks: List of TaskIntermediateRepresentation objects.
        harnesses_by_name: Optional mapping of harness name -> Harness IR. If provided,
            adds harness-level metadata (e.g., arch) to each task mapping entry.

    Returns:
        dict: Mapping of (harness_name, task_name) to dict holding their configuration.
    """
    mapping: dict[tuple[str, str], dict] = {}

    for task_ir in tasks:
        harness_name = task_ir.harness
        task_name = task_ir.name
        key = (harness_name, task_name)

        if key in mapping:
            logger.warning(
                "Duplicate task key found in IRs, keeping first occurrence",
                harness=harness_name,
                task=task_name,
            )
            continue

        # Extract endpoint_type from defaults.config.supported_endpoint_types
        defaults = task_ir.defaults or {}
        config = defaults.get("config", {})
        supported_endpoint_types = config.get("supported_endpoint_types", ["chat"])
        endpoint_type = (
            supported_endpoint_types[0] if supported_endpoint_types else "chat"
        )

        # Extract type from defaults.config.type
        task_type = config.get("type", "")

        # Build mapping entry
        mapping[key] = {
            "task": task_name,
            "harness": harness_name,
            "endpoint_type": endpoint_type,
            "container": task_ir.container,
        }

        if task_ir.container_arch:
            mapping[key]["arch"] = task_ir.container_arch

        # Backwards-compatible enhancement: keep full IR defaults available.
        # Existing code uses flattened defaults (excluding `config`) below; this adds a
        # new field without changing any existing keys.
        mapping[key]["defaults"] = defaults

        # Backwards-compatible enhancement: surface command explicitly if present.
        # Note: `command` is already included via flattened defaults merge, but
        # keeping it explicit makes downstream usage simpler.
        if "command" in defaults and "command" not in mapping[key]:
            mapping[key]["command"] = defaults["command"]

        # Add description if available
        if task_ir.description:
            mapping[key]["description"] = task_ir.description

        # Add type if available
        if task_type:
            mapping[key]["type"] = task_type

        # Add container_digest if available
        if task_ir.container_digest:
            mapping[key]["container_digest"] = task_ir.container_digest

        # Merge defaults (flattened, excluding config which is already processed)
        defaults_copy = {k: v for k, v in defaults.items() if k != "config"}
        mapping[key].update(defaults_copy)

    return mapping


def load_tasks_mapping(
    mapping_toml: pathlib.Path | str | None = None,
    *,
    from_container: str | None = None,
) -> dict[tuple[str, str], dict]:
    """Load tasks mapping.

    The function obeys the following priority rules:
    1. If from_container is not None -> extract framework.yml from that container and build mapping from the resulting IRs.
    2. Otherwise -> load packaged IRs (all_tasks_irs.yaml) and build mapping from those IRs.

    Args:
        mapping_toml: Deprecated. mapping.toml is no longer supported (IR-only mode).
        from_container: Optional container image identifier. If provided, tasks are loaded on-the-fly from that container.

    Returns:
        dict: Mapping of (harness_name, task_name) to dict holding their configuration.

    """
    if mapping_toml is not None:
        raise ValueError(
            "mapping_toml is no longer supported. This project has switched to packaged IRs (all_tasks_irs.yaml)."
        )

    # Explicit container path: extract tasks from container and return mapping built from IRs.
    # This bypasses packaged IRs.
    if from_container is not None:
        try:
            # Optional dependency path; importing may fail in "IR-only" environments.
            from nemo_evaluator_launcher.common.container_metadata import (
                load_tasks_from_container,
            )
        except ModuleNotFoundError as e:
            raise RuntimeError(
                "Loading tasks from a container requires optional dependencies. "
                "Install nemo-evaluator-launcher with the full runtime dependencies."
            ) from e

        tasks = load_tasks_from_container(from_container)
        if not tasks:
            logger.warning(
                "No tasks loaded from container via container-metadata",
                container=from_container,
            )
        else:
            logger.info(
                "Loaded tasks from container via container-metadata",
                container=from_container,
                num_tasks=len(tasks),
            )

        return _convert_irs_to_mapping_format(tasks)

    try:
        tasks, mapping_verified = load_tasks_from_tasks_file()
    except Exception as e:
        raise RuntimeError("Failed to load tasks from packaged IRs") from e

    if not tasks:
        raise RuntimeError("No tasks available in packaged IRs (all_tasks_irs.yaml)")

    logger.info(
        "Loaded tasks from packaged IRs",
        num_tasks=len(tasks),
        mapping_verified=mapping_verified,
    )
    return _convert_irs_to_mapping_format(tasks)


def get_task_from_mapping(query: str, mapping: dict[Any, Any]) -> dict[Any, Any]:
    """Unambiguously selects one task from the mapping based on the query.

    Args:
        query: Either `task_name` or `harness_name.task_name`.
        mapping: The object returned from `load_tasks_mapping` function.

    Returns:
        dict: Task data.

    """
    num_dots = query.count(".")

    # if there are no dots in query, treat it like a task name
    if num_dots == 0:
        matching_keys = [key for key in mapping.keys() if key[1] == query]
        # if exactly one task matching the query has been found:
        if len(matching_keys) == 1:
            key = matching_keys[0]
            return mapping[key]  # type: ignore[no-any-return]
        # if more than one task matching the query has been found:
        elif len(matching_keys) > 1:
            matching_queries = [
                f"{harness_name}.{task_name}"
                for harness_name, task_name in matching_keys
            ]
            raise ValueError(
                f"there are multiple tasks named {repr(query)} in the mapping,"
                f" please select one of {repr(matching_queries)}"
            )
        # no tasks have been found:
        else:
            raise ValueError(f"task {repr(query)} does not exist in the mapping")

    # if there is one dot in query, treat it like "{harness_name}.{task_name}"
    elif num_dots == 1:
        harness_name, task_name = query.split(".")
        matching_keys = [
            key for key in mapping.keys() if key == (harness_name, task_name)
        ]
        # if exactly one task matching the query has been found:
        if len(matching_keys) == 1:
            key = matching_keys[0]
            return mapping[key]  # type: ignore[no-any-return]
        # if more than one task matching the query has been found:
        elif len(matching_keys) >= 2:
            raise ValueError(
                f"there are multiple matches for {repr(query)} in the mapping,"
                " which means the mapping is not correct"
            )
        # no tasks have been found:
        else:
            raise ValueError(
                f"harness.task {repr(query)} does not exist in the mapping"
            )

    # invalid query
    else:
        raise ValueError(
            f"invalid query={repr(query)} for task mapping,"
            " it must contain exactly zero or one occurrence of '.' character"
        )


def _minimal_task_definition(task_query: str, *, container: str) -> dict[str, Any]:
    """Create a minimal task definition when task is not known in any mapping."""
    if task_query.count(".") == 1:
        harness, task = task_query.split(".")
    else:
        harness, task = "", task_query

    # Default to chat; most configs and endpoints use chat unless explicitly known.
    return {
        "task": task,
        "harness": harness,
        "endpoint_type": "chat",
        "container": container,
    }


def get_task_definition_for_job(
    *,
    task_query: str,
    base_mapping: dict[Any, Any],
    container: str | None = None,
) -> dict[str, Any]:
    """Resolve task definition for a job.

    If a container is provided, tasks are loaded from that container (using
    container-metadata) and we attempt to resolve the task from that mapping.
    If the task isn't found in the container, we warn and return a minimal
    task definition so submission can proceed.
    """
    if not container:
        return get_task_from_mapping(task_query, base_mapping)

    # `load_tasks_mapping(from_container=...)` uses container-metadata extraction,
    # which already has its own caching (e.g., caching extracted framework.yml).
    container_mapping = load_tasks_mapping(from_container=container)

    try:
        return get_task_from_mapping(task_query, container_mapping)
    except ValueError as e:
        logger.warning(
            "Task not found in provided container; proceeding with minimal task definition",
            task=task_query,
            container=container,
            error=str(e),
        )
        return _minimal_task_definition(task_query, container=container)
