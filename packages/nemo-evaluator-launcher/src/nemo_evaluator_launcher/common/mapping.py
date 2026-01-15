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

from nemo_evaluator_launcher.common.container_metadata import (
    TaskIntermediateRepresentation,
    load_tasks_from_tasks_file,
)
from nemo_evaluator_launcher.common.logging_utils import logger


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


def get_task_from_mapping(
    query: str, mapping: dict[Any, Any], *, allow_missing: bool = False
) -> dict[Any, Any] | None:
    """Unambiguously selects one task from the mapping based on the query.

    Args:
        query: Either `task_name` or `harness_name.task_name`.
        mapping: The object returned from `load_tasks_mapping` function.
        allow_missing: If True, return None when task is not found instead of raising ValueError.

    Returns:
        dict: Task data, or None if allow_missing=True and task not found.

    Raises:
        ValueError: If task not found (when allow_missing=False), multiple tasks match,
            or query format is invalid.
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
            if allow_missing:
                return None
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
            if allow_missing:
                return None
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
    """Create a minimal task definition when task is not known in any mapping.

    This is used for unlisted tasks - tasks not exposed in the FDF but available
    in the evaluation container. The returned dict includes `is_unlisted_task: True`
    to signal that safeguards should be applied.
    """
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
        "is_unlisted_task": True,
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
    If the task isn't found in the container or base mapping, we return a minimal
    task definition so submission can proceed (requires explicit container).

    For unlisted tasks (not in any mapping), the returned dict includes
    `is_unlisted_task: True` to signal that safeguards should be applied.

    Args:
        task_query: Either `task_name` or `harness_name.task_name`.
        base_mapping: The base tasks mapping from packaged IRs.
        container: Optional container image identifier. Required for unlisted tasks.

    Returns:
        dict: Task definition with all required fields.

    Raises:
        ValueError: If task not found and no container specified for unlisted task.
    """
    # First try to find task in base mapping
    task_def = get_task_from_mapping(task_query, base_mapping, allow_missing=True)
    if task_def is not None:
        # Task found in base mapping - use container override if provided
        if container:
            task_def = dict(task_def)  # Make a copy to avoid mutating the mapping
            task_def["container"] = container
        return task_def

    # Task not found in base mapping - try container-specific mapping if provided
    if container:
        # `load_tasks_mapping(from_container=...)` uses container-metadata extraction,
        # which already has its own caching (e.g., caching extracted framework.yml).
        container_mapping = load_tasks_mapping(from_container=container)
        task_def = get_task_from_mapping(
            task_query, container_mapping, allow_missing=True
        )
        if task_def is not None:
            return task_def

        # Task not found in container either - return minimal definition for unlisted task
        logger.warning(
            "Task not found in any mapping; proceeding with minimal task definition (unlisted task)",
            task=task_query,
            container=container,
        )
        return _minimal_task_definition(task_query, container=container)

    # No container provided and task not found - fail with helpful error
    raise ValueError(
        f"Task {repr(task_query)} does not exist in the mapping. "
        f"To run an unlisted task, specify the container explicitly in the task config "
        f"(e.g., 'container: nvcr.io/nvidia/eval-factory/your-container:tag')."
    )
