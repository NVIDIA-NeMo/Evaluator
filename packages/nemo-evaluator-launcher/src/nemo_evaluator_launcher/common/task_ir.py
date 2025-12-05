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
import hashlib
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


def _calculate_mapping_checksum(mapping_file: pathlib.Path) -> Optional[str]:
    """Calculate SHA256 checksum of mapping.toml file.

    Args:
        mapping_file: Path to mapping.toml file

    Returns:
        SHA256 checksum as string (format: "sha256:...") or None if file doesn't exist
    """
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


def _validate_mapping_checksum(
    stored_checksum: Optional[str], mapping_file: Optional[pathlib.Path] = None
) -> None:
    """Validate checksum consistency between stored value and current mapping.toml.

    Logs INFO if checksums match, WARNING if they don't match.

    Args:
        stored_checksum: Checksum stored in all_tasks_irs.yaml metadata
        mapping_file: Path to mapping.toml file. If None, uses default path.
    """
    if not stored_checksum:
        logger.debug("No stored checksum found in all_tasks_irs.yaml metadata")
        return

    # Determine mapping.toml path
    if mapping_file is None:
        # Default path relative to package resources
        try:
            # Try to get the path to the packaged mapping.toml
            mapping_file = (
                pathlib.Path(__file__).parent.parent / "resources" / "mapping.toml"
            )
        except Exception:
            logger.debug(
                "Could not determine mapping.toml path for checksum validation"
            )
            return

    if not mapping_file.exists():
        logger.debug(
            "mapping.toml not found, skipping checksum validation",
            path=str(mapping_file),
        )
        return

    current_checksum = _calculate_mapping_checksum(mapping_file)
    if not current_checksum:
        logger.debug("Could not calculate current mapping.toml checksum")
        return

    if stored_checksum == current_checksum:
        logger.info(
            "mapping.toml checksum matches all_tasks_irs.yaml",
            checksum=stored_checksum,
        )
    else:
        logger.warning(
            "mapping.toml checksum mismatch detected",
            stored_checksum=stored_checksum,
            current_checksum=current_checksum,
            message=(
                "all_tasks_irs.yaml may be outdated. "
                "Consider regenerating it with: "
                "python packages/nemo-evaluator-launcher/scripts/load_framework_definitions.py"
            ),
        )


def load_tasks_from_tasks_file(
    tasks_file: Optional[pathlib.Path] = None,
) -> tuple[list[TaskIntermediateRepresentation], bool]:
    """Load tasks from all_tasks_irs.yaml file.

    Public API function for loading task Intermediate Representations (IRs).

    Validates checksum consistency with current mapping.toml:
    - If checksums match: Logs INFO and returns mapping_verified=True
    - If checksums don't match: Logs WARNING and returns mapping_verified=False

    Args:
        tasks_file: Path to all_tasks_irs.yaml file. If None, uses default path.

    Returns:
        Tuple of (list of TaskIntermediateRepresentation objects, mapping_verified: bool)
    """
    if tasks_file is None:
        # Default path relative to package resources
        import importlib.resources

        try:
            content = importlib.resources.read_text(
                "nemo_evaluator_launcher.resources",
                "all_tasks_irs.yaml",
                encoding="utf-8",
            )
            # Parse content directly
            yaml_data = yaml.safe_load(content)

            logger.info(
                "Loaded task IRs from package resources",
                num_tasks=yaml_data.get("metadata", {}).get("num_tasks", 0),
            )

            # Extract metadata and validate checksum
            metadata = yaml_data.get("metadata", {})
            stored_checksum = metadata.get("mapping_toml_checksum")

            # Determine mapping.toml path (derive from tasks_file location or use default)
            mapping_file = (
                pathlib.Path(__file__).parent.parent / "resources" / "mapping.toml"
            )

            # Calculate current checksum
            current_checksum = _calculate_mapping_checksum(mapping_file)

            # Determine if mapping is verified
            mapping_verified = (
                (stored_checksum == current_checksum)
                if stored_checksum and current_checksum
                else False
            )

            if not mapping_verified:
                logger.warning(
                    "mapping.toml checksum mismatch detected",
                    stored_checksum=stored_checksum,
                    current_checksum=current_checksum,
                    message=(
                        "all_tasks_irs.yaml may be outdated. "
                        "Consider regenerating it with: "
                        "python packages/nemo-evaluator-launcher/scripts/load_framework_definitions.py"
                    ),
                )
            else:
                logger.info(
                    "mapping.toml checksum matches all_tasks_irs.yaml",
                    checksum=stored_checksum,
                )

            # Parse tasks from tasks section
            tasks_data = yaml_data.get("tasks", [])
            tasks: list[TaskIntermediateRepresentation] = []
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

            logger.info(
                "Loaded tasks from tasks file",
                total_tasks=len(tasks),
            )
            return tasks, mapping_verified

        except (ImportError, FileNotFoundError, Exception) as e:
            logger.debug(
                "Failed to load from package resources, trying file path",
                error=str(e),
            )
            # Fallback: construct path relative to this file
            tasks_file = (
                pathlib.Path(__file__).parent.parent
                / "resources"
                / "all_tasks_irs.yaml"
            )

    if not tasks_file.exists():
        logger.warning(
            "Tasks file not found",
            path=str(tasks_file),
        )
        return [], False

    logger.info("Loading tasks from tasks file", path=str(tasks_file))

    tasks: list[TaskIntermediateRepresentation] = []

    try:
        with open(tasks_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Parse single YAML document
        yaml_data = yaml.safe_load(content)

        # Extract metadata and validate checksum
        metadata = yaml_data.get("metadata", {})
        stored_checksum = metadata.get("mapping_toml_checksum")

        # Derive mapping.toml path from tasks_file location
        mapping_file = tasks_file.parent / "mapping.toml"

        # Calculate current checksum
        current_checksum = _calculate_mapping_checksum(mapping_file)

        # Determine if mapping is verified
        mapping_verified = (
            (stored_checksum == current_checksum)
            if stored_checksum and current_checksum
            else False
        )

        if not mapping_verified:
            logger.warning(
                "mapping.toml checksum mismatch detected",
                stored_checksum=stored_checksum,
                current_checksum=current_checksum,
                message=(
                    "all_tasks_irs.yaml may be outdated. "
                    "Consider regenerating it with: "
                    "python packages/nemo-evaluator-launcher/scripts/load_framework_definitions.py"
                ),
            )
        else:
            logger.info(
                "mapping.toml checksum matches all_tasks_irs.yaml",
                checksum=stored_checksum,
            )

        # Parse tasks from tasks section
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

        logger.info(
            "Loaded tasks from tasks file",
            total_tasks=len(tasks),
            path=str(tasks_file),
        )

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

    return tasks, mapping_verified
