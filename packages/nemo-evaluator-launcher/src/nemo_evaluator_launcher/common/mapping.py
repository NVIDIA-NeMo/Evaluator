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
import importlib
import pathlib
import sys
from importlib import resources
from typing import Any

import yaml

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from nemo_evaluator_launcher.common.container_metadata import (
    TaskIntermediateRepresentation,
    load_tasks_from_tasks_file,
)
from nemo_evaluator_launcher.common.logging_utils import logger

# Configuration constants
CACHE_FILENAME = "mapping.toml"
INTERNAL_RESOURCES_PKG = "nemo_evaluator_launcher_internal.resources"


def _load_packaged_resource(
    resource_name: str, pkg_name: str = "nemo_evaluator_launcher.resources"
) -> dict[str, Any]:
    """Load a resource from the packaged resources.

    Args:
        resource_name: The name of the resource to load.
    """
    try:
        resource_toml: dict[str, Any] = {}
        with resources.files(pkg_name).joinpath(resource_name).open("rb") as f:
            resource_toml = tomllib.load(f)
        logger.info(
            "Loaded resource from packaged file", resource=resource_name, pkg=pkg_name
        )
        return resource_toml
    except (OSError, tomllib.TOMLDecodeError) as e:
        logger.error(
            "Failed to load from packaged file",
            resource=resource_name,
            pkg=pkg_name,
            error=str(e),
        )
        raise RuntimeError(f"Failed to load {resource_name} from packaged file") from e


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
) -> dict[tuple[str, str], dict]:
    """Load tasks mapping.

    The function obeys the following priority rules:
    1. (Default) If mapping_toml is None -> try IRs first, then packaged mapping.
    2. If mapping_toml is not None -> load mapping from this path (uses old mapping.toml path).

    Args:
        mapping_toml: Optional path to mapping TOML file (uses old mapping.toml path).

    Returns:
        dict: Mapping of (harness_name, task_name) to dict holding their configuration.

    """
    # For explicit mapping_toml path, use old mapping.toml loading
    if mapping_toml is not None:
        with open(mapping_toml, "rb") as f:
            local_mapping = _process_mapping(tomllib.load(f))

        # Merge internal mapping if available
        try:
            importlib.import_module("nemo_evaluator_launcher_internal")
            logger.debug("Internal package available, loading internal mapping")
            internal_mapping = _process_mapping(
                _load_packaged_resource(CACHE_FILENAME, INTERNAL_RESOURCES_PKG)
            )
            local_mapping.update(internal_mapping)
            logger.info(
                "Successfully merged internal mapping",
                internal_tasks=len(internal_mapping),
            )
        except ImportError:
            logger.debug("Internal package not available, using external mapping only")
        except Exception as e:
            logger.warning("Failed to load internal mapping", error=str(e))

        return local_mapping

    # Default path: try IRs first, fall back to mapping.toml
    local_mapping: dict = {}
    irs_loaded = False

    # Try to load from IRs
    try:
        tasks, mapping_verified = load_tasks_from_tasks_file()
        if tasks:
            local_mapping = _convert_irs_to_mapping_format(tasks)
            irs_loaded = True
            if not mapping_verified:
                logger.warning(
                    "Loaded tasks from IRs but mapping.toml checksum mismatch detected",
                )
            else:
                logger.info(
                    "Loaded tasks from IRs",
                    num_tasks=len(tasks),
                    mapping_verified=mapping_verified,
                )
    except Exception as e:
        logger.debug(
            "Failed to load from IRs, falling back to mapping.toml",
            error=str(e),
        )

    # Fall back to old mapping.toml loading if IRs not available
    if not irs_loaded:
        logger.debug("Loading tasks from mapping.toml (IRs not available)")
        local_mapping = _process_mapping(_load_packaged_resource(CACHE_FILENAME))

    # Merge internal mapping if available (for both IR and mapping.toml paths)
    # Note: Internal IRs are already included when IRs are loaded, but we still merge
    # internal mapping.toml to get any additional harness metadata or entries
    try:
        importlib.import_module("nemo_evaluator_launcher_internal")
        logger.debug("Internal package available, loading internal mapping")
        internal_mapping = _process_mapping(
            _load_packaged_resource(CACHE_FILENAME, INTERNAL_RESOURCES_PKG)
        )
        local_mapping.update(internal_mapping)
        logger.info(
            "Successfully merged internal mapping",
            internal_tasks=len(internal_mapping),
        )
    except ImportError:
        logger.debug("Internal package not available, using external mapping only")
    except Exception as e:
        logger.warning("Failed to load internal mapping", error=str(e))

    return local_mapping


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
