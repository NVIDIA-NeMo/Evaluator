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
"""Task Intermediate Representation for consolidating task information from frameworks."""

import copy
import pathlib
from dataclasses import dataclass
from typing import Any, Optional

import yaml

from nemo_evaluator_launcher.common.logging_utils import logger


def _deep_merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two dictionaries, with override taking precedence.

    Args:
        base: Base dictionary
        override: Override dictionary (values take precedence)

    Returns:
        Merged dictionary
    """
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

    name: str  # Harness name
    description: str  # Harness description
    full_name: Optional[str]  # Full display name
    url: Optional[str]  # Harness URL
    source: Optional[str]  # Source URL
    container: str  # Container image identifier
    container_digest: Optional[str]  # Container manifest digest

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary representation of the harness IR
        """
        return {
            "name": self.name,
            "description": self.description,
            "full_name": self.full_name,
            "url": self.url,
            "source": self.source,
            "container": self.container,
            "container_digest": self.container_digest,
        }


def _load_harnesses_from_frameworks(
    frameworks_file: Optional[pathlib.Path] = None,
) -> dict[str, HarnessIntermediateRepresentation]:
    """Load all harnesses from all_frameworks.yaml and create intermediate representations.

    Args:
        frameworks_file: Path to all_frameworks.yaml file. If None, uses default path.

    Returns:
        Dictionary mapping harness name to HarnessIntermediateRepresentation
    """
    harnesses: dict[str, HarnessIntermediateRepresentation] = {}
    yaml_documents = []

    if frameworks_file is None:
        import importlib.resources

        try:
            content = importlib.resources.read_text(
                "nemo_evaluator_launcher.resources",
                "all_frameworks.yaml",
                encoding="utf-8",
            )
            for doc in yaml.safe_load_all(content):
                if doc:
                    yaml_documents.append(doc)
        except (ImportError, FileNotFoundError, Exception) as e:
            logger.debug(
                "Failed to load from package resources, trying file path",
                error=str(e),
            )
            # Fallback: construct path relative to this file
            frameworks_file = (
                pathlib.Path(__file__).parent.parent
                / "resources"
                / "all_frameworks.yaml"
            )

    # If we don't have documents yet and frameworks_file is set, try to load from file
    if not yaml_documents:
        if frameworks_file and frameworks_file.exists():
            with open(frameworks_file, "r", encoding="utf-8") as f:
                content = f.read()
            for doc in yaml.safe_load_all(content):
                if doc:
                    yaml_documents.append(doc)
        else:
            logger.warning("Frameworks file not found, cannot load harnesses")
            return harnesses

    for framework_doc in yaml_documents:
        container = framework_doc.get("__container", "")
        container_digest = framework_doc.get("__container_digest")

        if not container:
            continue

        framework_info = framework_doc.get("framework", {})
        harness_name = framework_info.get("name", "")
        if not harness_name:
            harness_name = _extract_harness_from_container(container)
        else:
            harness_name = harness_name.replace("_", "-")

        # Skip if we already have this harness
        if harness_name in harnesses:
            continue

        description = framework_info.get("description", "")
        full_name = framework_info.get("full_name")
        url = framework_info.get("url")
        source = framework_info.get("source")

        harness_ir = HarnessIntermediateRepresentation(
            name=harness_name,
            description=description,
            full_name=full_name,
            url=url,
            source=source,
            container=container,
            container_digest=container_digest,
        )

        harnesses[harness_name] = harness_ir

    return harnesses


def _extract_harness_from_container(container: str) -> str:
    """Extract harness name from container image identifier.

    Args:
        container: Container image identifier

    Returns:
        Harness name (normalized)
    """
    # Try to extract from GitLab path format
    # gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/{harness}:{tag}
    if "ci-llm/" in container:
        parts = container.split("ci-llm/")
        if len(parts) > 1:
            harness_part = parts[1].split(":")[0]
            return harness_part

    # Also check for ci-vlm/ format (for VLM containers)
    if "ci-vlm/" in container:
        parts = container.split("ci-vlm/")
        if len(parts) > 1:
            harness_part = parts[1].split(":")[0]
            return harness_part

    # Fallback: try to extract from nvcr.io format
    # nvcr.io/nvidia/eval-factory/{harness}:{tag}
    if "eval-factory/" in container:
        parts = container.split("eval-factory/")
        if len(parts) > 1:
            harness_part = parts[1].split(":")[0]
            return harness_part.replace("_", "-")

    # Last resort: use last part before tag
    # Use rsplit to split on the LAST colon (tag separator), not the first (port separator)
    if ":" in container:
        container = container.rsplit(":", 1)[0]
    parts = container.split("/")
    return parts[-1].replace("_", "-")


@dataclass
class TaskIntermediateRepresentation:
    """Intermediate representation of a task with merged defaults and metadata."""

    name: str  # Task name
    description: str  # Task description
    harness: str  # Harness name
    container: str  # Container image identifier
    container_digest: Optional[str]  # Container manifest digest
    defaults: dict[str, Any]  # Merged defaults (framework defaults + task overrides)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary representation of the task IR
        """
        return {
            "name": self.name,
            "description": self.description,
            "harness": self.harness,
            "container": self.container,
            "container_digest": self.container_digest,
            "defaults": self.defaults,
        }


def _load_tasks_from_frameworks(
    frameworks_file: Optional[pathlib.Path] = None,
) -> list[TaskIntermediateRepresentation]:
    """Load all tasks from all_frameworks.yaml and create intermediate representations.

    Args:
        frameworks_file: Path to all_frameworks.yaml file. If None, uses default path.

    Returns:
        List of TaskIntermediateRepresentation objects
    """
    if frameworks_file is None:
        # Default path relative to package resources
        import importlib.resources

        # Try to read directly from package resources
        try:
            content = importlib.resources.read_text(
                "nemo_evaluator_launcher.resources",
                "all_frameworks.yaml",
                encoding="utf-8",
            )
            # Parse content directly
            yaml_documents = []
            for doc in yaml.safe_load_all(content):
                if doc:
                    yaml_documents.append(doc)

            logger.info(
                "Loaded framework documents from package resources",
                num_documents=len(yaml_documents),
            )

            # Process documents (same logic as below)
            tasks: list[TaskIntermediateRepresentation] = []
            for framework_doc in yaml_documents:
                # Extract container metadata
                container = framework_doc.get("__container", "")
                container_digest = framework_doc.get("__container_digest")

                if not container:
                    logger.warning(
                        "Framework document missing __container field, skipping",
                    )
                    continue

                # Extract harness name from framework.name field
                framework_info = framework_doc.get("framework", {})
                harness = framework_info.get("name", "")
                if not harness:
                    # Fallback to extracting from container if framework.name is missing
                    logger.warning(
                        "Framework document missing framework.name field, extracting from container",
                        container=container,
                    )
                    harness = _extract_harness_from_container(container)
                else:
                    # Normalize the harness name (replace underscores with hyphens for consistency)
                    harness = harness.replace("_", "-")

                # Get framework-level defaults (if any)
                framework_defaults = framework_doc.get("defaults", {})

                # Process evaluations
                evaluations = framework_doc.get("evaluations", [])
                if not evaluations:
                    logger.debug(
                        "No evaluations found in framework",
                        container=container,
                        harness=harness,
                    )
                    continue

                for eval_config in evaluations:
                    task_name = eval_config.get("name")
                    if not task_name:
                        logger.debug(
                            "Evaluation missing name, skipping",
                            container=container,
                        )
                        continue

                    description = eval_config.get("description", "")

                    # Get task-specific defaults
                    task_defaults = eval_config.get("defaults", {})

                    # Merge defaults: framework defaults + task defaults (task overrides)
                    merged_defaults = _deep_merge_dict(
                        framework_defaults, task_defaults
                    )

                    # Create task IR
                    task_ir = TaskIntermediateRepresentation(
                        name=task_name,
                        description=description,
                        harness=harness,
                        container=container,
                        container_digest=container_digest,
                        defaults=merged_defaults,
                    )

                    tasks.append(task_ir)

                    logger.debug(
                        "Created task IR",
                        harness=harness,
                        task=task_name,
                        container=container,
                    )

            logger.info(
                "Loaded tasks from frameworks",
                total_tasks=len(tasks),
                num_frameworks=len(yaml_documents),
            )
            return tasks

        except (ImportError, FileNotFoundError, Exception) as e:
            logger.debug(
                "Failed to load from package resources, trying file path",
                error=str(e),
            )
            # Fallback: construct path relative to this file
            frameworks_file = (
                pathlib.Path(__file__).parent.parent
                / "resources"
                / "all_frameworks.yaml"
            )

    if not frameworks_file.exists():
        logger.warning(
            "Frameworks file not found",
            path=str(frameworks_file),
        )
        return []

    logger.info("Loading tasks from frameworks file", path=str(frameworks_file))

    tasks: list[TaskIntermediateRepresentation] = []

    try:
        with open(frameworks_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Parse multi-YAML (documents separated by ---)
        yaml_documents = []
        for doc in yaml.safe_load_all(content):
            if doc:
                yaml_documents.append(doc)

        logger.info(
            "Loaded framework documents",
            num_documents=len(yaml_documents),
        )

        for framework_doc in yaml_documents:
            # Extract container metadata
            container = framework_doc.get("__container", "")
            container_digest = framework_doc.get("__container_digest")

            if not container:
                logger.warning(
                    "Framework document missing __container field, skipping",
                )
                continue

            # Extract harness name from framework.name field
            framework_info = framework_doc.get("framework", {})
            harness = framework_info.get("name", "")
            if not harness:
                # Fallback to extracting from container if framework.name is missing
                logger.warning(
                    "Framework document missing framework.name field, extracting from container",
                    container=container,
                )
                harness = _extract_harness_from_container(container)
            else:
                # Normalize the harness name (replace underscores with hyphens for consistency)
                harness = harness.replace("_", "-")

            # Get framework-level defaults (if any)
            framework_defaults = framework_doc.get("defaults", {})

            # Process evaluations
            evaluations = framework_doc.get("evaluations", [])
            if not evaluations:
                logger.debug(
                    "No evaluations found in framework",
                    container=container,
                    harness=harness,
                )
                continue

            for eval_config in evaluations:
                task_name = eval_config.get("name")
                if not task_name:
                    logger.debug(
                        "Evaluation missing name, skipping",
                        container=container,
                    )
                    continue

                description = eval_config.get("description", "")

                # Get task-specific defaults
                task_defaults = eval_config.get("defaults", {})

                # Merge defaults: framework defaults + task defaults (task overrides)
                merged_defaults = _deep_merge_dict(framework_defaults, task_defaults)

                # Create task IR
                task_ir = TaskIntermediateRepresentation(
                    name=task_name,
                    description=description,
                    harness=harness,
                    container=container,
                    container_digest=container_digest,
                    defaults=merged_defaults,
                )

                tasks.append(task_ir)

                logger.debug(
                    "Created task IR",
                    harness=harness,
                    task=task_name,
                    container=container,
                )

        logger.info(
            "Loaded tasks from frameworks",
            total_tasks=len(tasks),
            num_frameworks=len(yaml_documents),
        )

    except yaml.YAMLError as e:
        logger.error(
            "Failed to parse frameworks YAML",
            error=str(e),
            path=str(frameworks_file),
            exc_info=True,
        )
    except Exception as e:
        logger.error(
            "Error loading tasks from frameworks",
            error=str(e),
            path=str(frameworks_file),
            exc_info=True,
        )

    return tasks
