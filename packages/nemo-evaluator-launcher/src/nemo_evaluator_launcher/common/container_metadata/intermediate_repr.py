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
"""Intermediate representations for harnesses and tasks."""

import copy
import hashlib
import importlib
import importlib.resources
import pathlib
import sys
from dataclasses import dataclass
from typing import Any, Optional

import yaml

from nemo_evaluator_launcher.common.logging_utils import logger


def _deep_merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two dictionaries, with override taking precedence."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge_dict(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


@dataclass
class HarnessIntermediateRepresentation:
    """Intermediate representation of a harness with metadata."""

    name: str
    description: str
    full_name: Optional[str]
    url: Optional[str]
    source: Optional[str]
    container: str
    container_digest: Optional[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "description": self.description,
            "full_name": self.full_name,
            "url": self.url,
            "source": self.source,
            "container": self.container,
            "container_digest": self.container_digest,
        }


@dataclass
class TaskIntermediateRepresentation:
    """Intermediate representation of a task with merged defaults and metadata."""

    name: str
    description: str
    harness: str
    container: str
    container_digest: Optional[str]
    defaults: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "description": self.description,
            "harness": self.harness,
            "container": self.container,
            "container_digest": self.container_digest,
            "defaults": self.defaults,
        }


def _calculate_mapping_checksum(mapping_file: pathlib.Path) -> Optional[str]:
    """Calculate SHA256 checksum of mapping.toml file."""
    if not mapping_file.exists():
        return None

    try:
        with open(mapping_file, "rb") as f:
            file_content = f.read()

        checksum = hashlib.sha256(file_content).hexdigest()
        return f"sha256:{checksum}"
    except Exception as e:
        logger.debug(
            "Failed to calculate mapping.toml checksum",
            path=str(mapping_file),
            error=str(e),
        )
        return None


def _validate_checksum(
    stored_checksum: Optional[str],
    mapping_file: pathlib.Path,
    source: str,
) -> bool:
    """Validate checksum consistency between stored value and current mapping.toml.

    Args:
        stored_checksum: Checksum stored in metadata
        mapping_file: Path to mapping.toml file
        source: Source identifier for logging (e.g., "internal", "external")

    Returns:
        True if checksums match, False otherwise
    """
    if not stored_checksum:
        return False

    current_checksum = _calculate_mapping_checksum(mapping_file)
    if not current_checksum:
        return False

    verified = stored_checksum == current_checksum
    if not verified:
        logger.warning(
            f"{source.capitalize()} mapping.toml checksum mismatch detected",
            stored_checksum=stored_checksum,
            current_checksum=current_checksum,
        )
    else:
        logger.info(
            f"{source.capitalize()} mapping.toml checksum matches all_tasks_irs.yaml",
            checksum=stored_checksum,
        )

    return verified


def _parse_tasks_from_yaml_data(
    yaml_data: dict,
) -> list[TaskIntermediateRepresentation]:
    """Parse TaskIntermediateRepresentation objects from YAML data.

    Args:
        yaml_data: Parsed YAML data dictionary

    Returns:
        List of TaskIntermediateRepresentation objects
    """
    tasks: list[TaskIntermediateRepresentation] = []
    tasks_data = yaml_data.get("tasks", [])

    for task_dict in tasks_data:
        task_ir = TaskIntermediateRepresentation(
            name=task_dict["name"],
            description=task_dict.get("description", ""),
            harness=task_dict["harness"],
            container=task_dict["container"],
            container_digest=task_dict.get("container_digest"),
            defaults=task_dict.get("defaults", {}),
        )
        tasks.append(task_ir)

    return tasks


def _find_internal_mapping_file() -> Optional[pathlib.Path]:
    """Find internal mapping.toml file in package paths.

    Returns:
        Path to internal mapping.toml or None if not found
    """
    for pkg_path in sys.path:
        potential_path = (
            pathlib.Path(pkg_path)
            / "nemo_evaluator_launcher_internal"
            / "resources"
            / "mapping.toml"
        )
        if potential_path.exists():
            return potential_path
    return None


def _load_irs_from_package(
    package_name: str, resource_name: str, source: str
) -> tuple[list[TaskIntermediateRepresentation], bool]:
    """Load task IRs from a package resource.

    Args:
        package_name: Package name (e.g., "nemo_evaluator_launcher_internal.resources")
        resource_name: Resource file name (e.g., "all_tasks_irs.yaml")
        source: Source identifier for logging (e.g., "internal", "external")

    Returns:
        Tuple of (tasks list, checksum verified bool)
    """
    try:
        content = importlib.resources.read_text(
            package_name,
            resource_name,
            encoding="utf-8",
        )
        yaml_data = yaml.safe_load(content)

        logger.info(
            f"Loaded {source} task IRs from package resources",
            num_tasks=yaml_data.get("metadata", {}).get("num_tasks", 0),
        )

        metadata = yaml_data.get("metadata", {})
        stored_checksum = metadata.get("mapping_toml_checksum")

        # Find mapping file for checksum validation
        if source == "internal":
            mapping_file = _find_internal_mapping_file()
        else:
            mapping_file = (
                pathlib.Path(__file__).parent.parent.parent
                / "resources"
                / "mapping.toml"
            )

        verified = False
        if mapping_file:
            verified = _validate_checksum(stored_checksum, mapping_file, source)

        tasks = _parse_tasks_from_yaml_data(yaml_data)
        logger.info(f"Loaded {source} tasks from IRs", total_tasks=len(tasks))

        return tasks, verified

    except (ImportError, FileNotFoundError) as e:
        logger.debug(f"{source.capitalize()} IRs not available", error=str(e))
        return [], False
    except Exception as e:
        logger.debug(f"Failed to load {source} IRs", error=str(e))
        return [], False


def _load_tasks_from_file(
    tasks_file: pathlib.Path,
) -> tuple[list[TaskIntermediateRepresentation], bool]:
    """Load tasks from a file path.

    Args:
        tasks_file: Path to all_tasks_irs.yaml file

    Returns:
        Tuple of (tasks list, checksum verified bool)
    """
    if not tasks_file.exists():
        logger.warning("Tasks file not found", path=str(tasks_file))
        return [], False

    logger.info("Loading tasks from tasks file", path=str(tasks_file))

    try:
        with open(tasks_file, "r", encoding="utf-8") as f:
            content = f.read()

        yaml_data = yaml.safe_load(content)

        metadata = yaml_data.get("metadata", {})
        stored_checksum = metadata.get("mapping_toml_checksum")

        mapping_file = tasks_file.parent / "mapping.toml"
        verified = _validate_checksum(stored_checksum, mapping_file, "file")

        tasks = _parse_tasks_from_yaml_data(yaml_data)

        logger.info(
            "Loaded tasks from tasks file", total_tasks=len(tasks), path=str(tasks_file)
        )

        return tasks, verified

    except yaml.YAMLError as e:
        logger.error(
            "Failed to parse tasks YAML",
            error=str(e),
            path=str(tasks_file),
            exc_info=True,
        )
        return [], False
    except Exception as e:
        logger.error(
            "Error loading tasks from tasks file",
            error=str(e),
            path=str(tasks_file),
            exc_info=True,
        )
        return [], False


def load_tasks_from_tasks_file(
    tasks_file: Optional[pathlib.Path] = None,
) -> tuple[list[TaskIntermediateRepresentation], bool]:
    """Load tasks from all_tasks_irs.yaml file.

    Public API function for loading task Intermediate Representations (IRs).

    Uses a simple swap strategy:
    - If internal package is available, load internal IRs only
    - Otherwise, load external IRs only
    - No merging is performed

    Validates checksum consistency with current mapping.toml.

    Args:
        tasks_file: Path to all_tasks_irs.yaml file. If None, uses default path.

    Returns:
        Tuple of (list of TaskIntermediateRepresentation objects, mapping_verified: bool)
    """
    # If file path provided, load directly from file
    if tasks_file is not None:
        return _load_tasks_from_file(tasks_file)

    # Try to load internal IRs first
    try:
        importlib.import_module("nemo_evaluator_launcher_internal")
        logger.debug("Internal package available, loading internal IRs")
        tasks, verified = _load_irs_from_package(
            "nemo_evaluator_launcher_internal.resources",
            "all_tasks_irs.yaml",
            "internal",
        )
        if tasks:
            logger.info(
                "Using internal IRs",
                total_tasks=len(tasks),
                mapping_verified=verified,
            )
            return tasks, verified
    except ImportError:
        logger.debug("Internal package not available, will use external IRs")
    except Exception as e:
        logger.debug("Failed to load internal IRs, will use external IRs", error=str(e))

    # Fallback to external IRs
    logger.debug("Loading external IRs")
    tasks, verified = _load_irs_from_package(
        "nemo_evaluator_launcher.resources",
        "all_tasks_irs.yaml",
        "external",
    )

    if tasks:
        logger.info(
            "Using external IRs",
            total_tasks=len(tasks),
            mapping_verified=verified,
        )
        return tasks, verified

    # Final fallback: try default file path
    logger.debug("No IRs loaded from package resources, trying file path")
    default_tasks_file = (
        pathlib.Path(__file__).parent.parent.parent / "resources" / "all_tasks_irs.yaml"
    )
    return _load_tasks_from_file(default_tasks_file)
