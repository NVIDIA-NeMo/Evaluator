#!/usr/bin/env python3
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
"""Script to load framework.yml files from all containers and convert to IRs.

This script reads all containers from mapping.toml, extracts framework.yml from each
container using OCI layer inspection, parses them using nemo_evaluator.core.input,
converts them to Intermediate Representations (IRs), and serializes them into
all_tasks_irs.yaml.
"""

import argparse
import hashlib
import json
import os
import pathlib
import sys
import tempfile
from typing import Any, Optional

import yaml

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from nemo_evaluator.core.input import get_framework_evaluations

from nemo_evaluator_launcher.common.logging_utils import logger
from nemo_evaluator_launcher.common.partial_pull import (
    GitlabRegistryAuthenticator,
    NvcrRegistryAuthenticator,
    RegistryAuthenticator,
    find_file_in_image_layers,
    find_file_matching_pattern_in_image_layers,
)
from nemo_evaluator_launcher.common.task_ir import (
    HarnessIntermediateRepresentation,
    TaskIntermediateRepresentation,
)


def parse_container_image(container_image: str) -> tuple[str, str, str, str]:
    """Parse a container image string into registry type, registry URL, repository, and tag.

    Args:
        container_image: Container image string (e.g., "nvcr.io/nvidia/eval-factory/simple-evals:25.10")

    Returns:
        Tuple of (registry_type, registry_url, repository, tag)
    """
    # Split tag from image
    if ":" in container_image:
        image_part, tag = container_image.rsplit(":", 1)
    else:
        image_part = container_image
        tag = "latest"

    # Parse registry and repository
    parts = image_part.split("/")
    if len(parts) < 2:
        raise ValueError(f"Invalid container image format: {container_image}")

    # Check if first part is a registry (contains '.' or is 'localhost')
    if "." in parts[0] or parts[0] == "localhost":
        registry_host = parts[0]
        # Determine registry type
        if "gitlab" in registry_host.lower():
            registry_type = "gitlab"
        elif "nvcr.io" in registry_host:
            registry_type = "nvcr"
        else:
            registry_type = "nvcr"  # Default to nvcr for other registries

        # Check if registry has a port
        if ":" in registry_host:
            registry_url = registry_host
        else:
            registry_url = registry_host
        repository = "/".join(parts[1:])
    else:
        # Default registry (Docker Hub)
        registry_type = "nvcr"
        registry_url = "registry-1.docker.io"
        repository = image_part

    return registry_type, registry_url, repository, tag


def create_authenticator(
    registry_type: str, registry_url: str, repository: str
) -> RegistryAuthenticator:
    """Create the appropriate authenticator based on registry type.

    Args:
        registry_type: Type of registry ('gitlab' or 'nvcr')
        registry_url: Registry URL
        repository: Repository name

    Returns:
        Registry authenticator instance
    """
    if registry_type == "gitlab":
        username = os.getenv("DOCKER_USERNAME", "gitlab-ci-token")
        password = os.getenv("GITLAB_TOKEN")
        if not password:
            raise ValueError(
                "GITLAB_TOKEN environment variable is required for GitLab registry"
            )
        return GitlabRegistryAuthenticator(
            registry_url=registry_url,
            username=username,
            password=password,
            repository=repository,
        )
    elif registry_type == "nvcr":
        username = os.getenv("NVCR_USERNAME") or os.getenv(
            "DOCKER_USERNAME", "$oauthtoken"
        )
        password = os.getenv("NVCR_PASSWORD") or os.getenv("NVCR_API_KEY")
        if not password:
            raise ValueError(
                "NVCR_PASSWORD or NVCR_API_KEY environment variable is required for nvcr.io registry"
            )
        return NvcrRegistryAuthenticator(
            registry_url=registry_url,
            username=username,
            password=password,
        )
    else:
        raise ValueError(f"Unknown registry type: {registry_type}")


def get_container_digest(
    authenticator: RegistryAuthenticator, repository: str, reference: str
) -> Optional[str]:
    """Get the manifest digest for a container image.

    Args:
        authenticator: Registry authenticator instance
        repository: Repository name
        reference: Tag or digest

    Returns:
        Container digest (sha256:...) or None if failed
    """
    try:
        manifest = authenticator.get_manifest(repository, reference)
        if not manifest:
            return None

        # Compute manifest digest (SHA256 of canonical JSON)
        manifest_json = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
        manifest_digest = (
            f"sha256:{hashlib.sha256(manifest_json.encode('utf-8')).hexdigest()}"
        )
        return manifest_digest
    except Exception as e:
        logger.warning(
            "Failed to get container digest",
            repository=repository,
            reference=reference,
            error=str(e),
        )
        return None


def calculate_mapping_checksum(mapping_file: pathlib.Path) -> str:
    """Calculate SHA256 checksum of mapping.toml file.

    Args:
        mapping_file: Path to mapping.toml file

    Returns:
        SHA256 checksum as string (format: "sha256:...")
    """
    try:
        with open(mapping_file, "rb") as f:
            file_content = f.read()

        checksum = hashlib.sha256(file_content).hexdigest()
        return f"sha256:{checksum}"
    except Exception as e:
        logger.error(
            "Failed to calculate mapping.toml checksum", error=str(e), exc_info=True
        )
        raise


def load_mapping_toml(mapping_file: pathlib.Path) -> tuple[dict[str, str], str]:
    """Load mapping.toml and extract unique containers per harness.

    Args:
        mapping_file: Path to mapping.toml file

    Returns:
        Tuple of:
        - Dictionary mapping harness_name to container image
        - SHA256 checksum of mapping.toml file (format: "sha256:...")
    """
    try:
        # Calculate checksum first (before parsing)
        checksum = calculate_mapping_checksum(mapping_file)

        with open(mapping_file, "rb") as f:
            mapping_toml = tomllib.load(f)

        harness_containers: dict[str, str] = {}
        for section_name, section_data in mapping_toml.items():
            # Extract harness name from section (e.g., "[lm-evaluation-harness]" -> "lm-evaluation-harness")
            if "." not in section_name and isinstance(section_data, dict):
                container = section_data.get("container")
                if container:
                    harness_containers[section_name] = container

        logger.info(
            "Loaded containers from mapping.toml",
            num_containers=len(harness_containers),
            checksum=checksum,
        )
        return harness_containers, checksum
    except Exception as e:
        logger.error("Failed to load mapping.toml", error=str(e), exc_info=True)
        raise


def extract_framework_yml(
    container: str,
    harness_name: str,
    max_layer_size: Optional[int] = None,
    use_cache: bool = True,
) -> tuple[Optional[str], Optional[str]]:
    """Extract framework.yml from a container using layer inspection.

    Args:
        container: Container image identifier
        harness_name: Name of the harness (for logging)
        max_layer_size: Optional maximum layer size in bytes
        use_cache: Whether to use caching

    Returns:
        Tuple of (framework_yml_content, container_digest) or (None, None) if failed
    """
    try:
        registry_type, registry_url, repository, tag = parse_container_image(container)

        logger.info(
            "Extracting framework.yml from container",
            container=container,
            harness=harness_name,
            registry_type=registry_type,
        )

        # Create authenticator
        authenticator = create_authenticator(registry_type, registry_url, repository)

        # Authenticate
        if not authenticator.authenticate(repository=repository):
            logger.error(
                "Authentication failed",
                container=container,
                harness=harness_name,
            )
            return None, None

        # Get container digest
        container_digest = get_container_digest(authenticator, repository, tag)
        if not container_digest:
            logger.warning(
                "Could not get container digest, continuing without it",
                container=container,
                harness=harness_name,
            )

        # Extract framework.yml
        # Use original container string as docker_id for cache consistency
        # This ensures cache hits work regardless of how the container string is formatted
        docker_id = container

        # First, try the standard location: /opt/metadata/framework.yml
        # Note: Digest is always validated for cache hits (prevents stale cache for 'latest' tags)
        framework_yml_content = find_file_in_image_layers(
            authenticator=authenticator,
            repository=repository,
            reference=tag,
            target_file="/opt/metadata/framework.yml",
            max_layer_size=max_layer_size,
            docker_id=docker_id,
            use_cache=use_cache,
            check_invalidated_digest=False,  # Deprecated parameter - digest always checked now
        )

        # If not found, try to find framework.yml in any subdirectory under /opt/metadata/
        # Note: find_file_matching_pattern_in_image_layers uses pattern-based caching
        # (cache key: /opt/metadata/framework.yml) so cache hits work regardless of subdirectory location
        if not framework_yml_content:
            logger.debug(
                "framework.yml not found at standard location, searching subdirectories",
                container=container,
                harness=harness_name,
            )
            result = find_file_matching_pattern_in_image_layers(
                authenticator=authenticator,
                repository=repository,
                reference=tag,
                prefix="/opt/metadata",
                filename="framework.yml",
                max_layer_size=max_layer_size,
                docker_id=docker_id,
                use_cache=use_cache,
            )
            if result:
                file_path, framework_yml_content = result
                logger.info(
                    "Found framework.yml in subdirectory",
                    container=container,
                    harness=harness_name,
                    file_path=file_path,
                )

        if not framework_yml_content:
            logger.warning(
                "framework.yml not found in container",
                container=container,
                harness=harness_name,
            )
            return None, None

        logger.info(
            "Successfully extracted framework.yml",
            container=container,
            harness=harness_name,
            content_size=len(framework_yml_content),
            digest=container_digest,
        )

        return framework_yml_content, container_digest

    except Exception as e:
        logger.warning(
            "Failed to extract framework.yml",
            container=container,
            harness=harness_name,
            error=str(e),
            exc_info=True,
        )
        return None, None


def parse_framework_to_irs(
    framework_content: str,
    container_id: str,
    container_digest: Optional[str],
    harness_name: str,
) -> tuple[HarnessIntermediateRepresentation, list[TaskIntermediateRepresentation]]:
    """Parse framework.yml content and convert to Intermediate Representations.

    Args:
        framework_content: Original framework.yml content as string
        container_id: Full container image identifier
        container_digest: Container manifest digest (optional)
        harness_name: Name of the harness (for logging and IR creation)

    Returns:
        Tuple of (HarnessIntermediateRepresentation, list[TaskIntermediateRepresentation])
    """
    try:
        # Parse framework.yml to get metadata (description, url, etc.)
        framework_data = yaml.safe_load(framework_content)
        if not framework_data:
            raise ValueError("Empty framework.yml content")

        framework_info = framework_data.get("framework", {})
        framework_name = framework_info.get("name", harness_name)
        description = framework_info.get("description", "")
        full_name = framework_info.get("full_name")
        url = framework_info.get("url")
        source = framework_info.get("source")

        # Normalize harness name (replace underscores with hyphens)
        normalized_harness_name = framework_name.replace("_", "-")

        # Write framework.yml to temporary file (required by get_framework_evaluations)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yml", delete=False
        ) as temp_file:
            temp_file.write(framework_content)
            temp_file_path = temp_file.name

        try:
            # Parse using get_framework_evaluations (handles merging internally)
            (
                parsed_framework_name,
                framework_defaults,
                evaluations,
            ) = get_framework_evaluations(temp_file_path)

            # Create HarnessIntermediateRepresentation
            harness_ir = HarnessIntermediateRepresentation(
                name=normalized_harness_name,
                description=description,
                full_name=full_name,
                url=url,
                source=source,
                container=container_id,
                container_digest=container_digest,
            )

            # Convert Evaluation objects to TaskIntermediateRepresentation
            task_irs = []
            for task_name, evaluation in evaluations.items():
                # Get description from original framework.yml (more complete)
                task_description = ""
                for eval_config in framework_data.get("evaluations", []):
                    eval_task_name = (
                        eval_config.get("defaults", {}).get("config", {}).get("type")
                    )
                    if eval_task_name == task_name:
                        task_description = eval_config.get("description", "")
                        break

                # Use evaluation.model_dump() for defaults (already merged)
                evaluation_dict = evaluation.model_dump(exclude_none=True)

                # Create TaskIntermediateRepresentation
                task_ir = TaskIntermediateRepresentation(
                    name=task_name,
                    description=task_description,
                    harness=normalized_harness_name,
                    container=container_id,
                    container_digest=container_digest,
                    defaults=evaluation_dict,
                )

                task_irs.append(task_ir)

                logger.debug(
                    "Created task IR",
                    harness=normalized_harness_name,
                    task=task_name,
                    container=container_id,
                )

            logger.info(
                "Parsed framework to IRs",
                harness=normalized_harness_name,
                num_tasks=len(task_irs),
                container=container_id,
            )

            return harness_ir, task_irs

        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file_path)
            except Exception:
                pass

    except Exception as e:
        logger.error(
            "Failed to parse framework.yml to IRs",
            error=str(e),
            container=container_id,
            harness=harness_name,
            exc_info=True,
        )
        raise


def enrich_framework_yml(
    framework_content: str, container_id: str, container_digest: Optional[str]
) -> dict[str, Any]:
    """Enrich framework.yml content with container metadata.

    DEPRECATED: This function is kept for backward compatibility during migration.
    Use parse_framework_to_irs() for new code.

    Args:
        framework_content: Original framework.yml content as string
        container_id: Full container image identifier
        container_digest: Container manifest digest (optional)

    Returns:
        Enriched framework data as dictionary
    """
    try:
        framework_data = yaml.safe_load(framework_content)
        if not framework_data:
            framework_data = {}

        # Add metadata fields at the top level
        framework_data["__container"] = container_id
        if container_digest:
            framework_data["__container_digest"] = container_digest

        return framework_data
    except yaml.YAMLError as e:
        logger.error("Failed to parse framework.yml", error=str(e), exc_info=True)
        raise


def serialize_tasks_irs(tasks: list[Any], metadata: dict[str, Any]) -> str:
    """Serialize task IRs and metadata into single YAML format.

    Args:
        tasks: List of TaskIntermediateRepresentation objects
        metadata: Metadata dictionary (must include 'mapping_toml_checksum')

    Returns:
        Single YAML string with metadata and tasks sections
    """
    # Combine metadata and tasks into single dict
    output_dict = {
        "metadata": metadata,
        "tasks": [task.to_dict() for task in tasks],
    }

    # Serialize to YAML
    return yaml.dump(output_dict, default_flow_style=False, sort_keys=False)


def combine_frameworks(frameworks: list[dict[str, Any]]) -> str:
    """Combine multiple framework definitions into multi-YAML format.

    DEPRECATED: This function is kept for backward compatibility during migration.
    Use serialize_tasks_irs() for new code.

    Args:
        frameworks: List of enriched framework dictionaries

    Returns:
        Multi-YAML string with documents separated by '---'
    """
    yaml_documents = []
    for framework in frameworks:
        yaml_doc = yaml.dump(framework, default_flow_style=False, sort_keys=False)
        yaml_documents.append(yaml_doc)

    return "---\n".join(yaml_documents) + "\n"


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Load framework.yml files from all containers and convert to Intermediate Representations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (default paths)
  %(prog)s

  # Custom mapping file and output
  %(prog)s --mapping-file custom_mapping.toml --output-file output.yaml

  # Disable caching
  %(prog)s --no-cache

  # Set max layer size (100MB)
  %(prog)s --max-layer-size 104857600

Environment Variables:
  DOCKER_USERNAME: GitLab registry username (default: gitlab-ci-token)
  GITLAB_TOKEN: GitLab registry token/password (required for GitLab)
  NVCR_USERNAME: nvcr.io username (default: $oauthtoken)
  NVCR_PASSWORD or NVCR_API_KEY: nvcr.io password/API key (required for nvcr.io)
        """,
    )
    parser.add_argument(
        "--mapping-file",
        type=pathlib.Path,
        default=pathlib.Path(__file__).parent.parent
        / "src"
        / "nemo_evaluator_launcher"
        / "resources"
        / "mapping.toml",
        help="Path to mapping.toml file",
    )
    parser.add_argument(
        "--output-file",
        type=pathlib.Path,
        default=pathlib.Path(__file__).parent.parent
        / "src"
        / "nemo_evaluator_launcher"
        / "resources"
        / "all_tasks_irs.yaml",
        help="Path to output YAML file with task IRs",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching of framework.yml files",
    )
    parser.add_argument(
        "--max-layer-size",
        type=int,
        default=100 * 1024,  # 100KB
        help="Maximum layer size in bytes. Only layers smaller than this will be checked. (default: 100KB)",
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.mapping_file.exists():
        logger.error("Mapping file not found", path=str(args.mapping_file))
        sys.exit(1)

    # Ensure output directory exists
    args.output_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Starting framework definitions extraction",
        mapping_file=str(args.mapping_file),
        output_file=str(args.output_file),
        use_cache=not args.no_cache,
        max_layer_size=args.max_layer_size,
    )

    # Load containers from mapping.toml
    try:
        harness_containers, mapping_checksum = load_mapping_toml(args.mapping_file)
    except Exception as e:
        logger.error("Failed to load mapping.toml", error=str(e))
        sys.exit(1)

    if not harness_containers:
        logger.warning("No containers found in mapping.toml")
        sys.exit(0)

    # Extract framework.yml from each container and convert to IRs
    all_task_irs: list[TaskIntermediateRepresentation] = []
    successful = 0
    failed = 0

    for harness_name, container in harness_containers.items():
        logger.info(
            "Processing container",
            harness=harness_name,
            container=container,
            progress=f"{successful + failed + 1}/{len(harness_containers)}",
        )

        framework_content, container_digest = extract_framework_yml(
            container=container,
            harness_name=harness_name,
            max_layer_size=args.max_layer_size,
            use_cache=not args.no_cache,
        )

        if framework_content:
            try:
                harness_ir, task_irs = parse_framework_to_irs(
                    framework_content=framework_content,
                    container_id=container,
                    container_digest=container_digest,
                    harness_name=harness_name,
                )
                all_task_irs.extend(task_irs)
                successful += 1
                logger.info(
                    "Successfully processed container",
                    harness=harness_name,
                    container=container,
                    num_tasks=len(task_irs),
                )
            except Exception as e:
                failed += 1
                logger.error(
                    "Failed to parse framework.yml to IRs",
                    harness=harness_name,
                    container=container,
                    error=str(e),
                    exc_info=True,
                )
        else:
            failed += 1
            logger.warning(
                "Skipping container (framework.yml not found)",
                harness=harness_name,
                container=container,
            )

    # Serialize all task IRs to YAML
    if all_task_irs:
        logger.info(
            "Serializing task IRs to YAML",
            num_tasks=len(all_task_irs),
        )
        try:
            # Create metadata with checksum
            metadata = {
                "mapping_toml_checksum": mapping_checksum,
                "num_tasks": len(all_task_irs),
            }

            # Serialize to YAML
            yaml_content = serialize_tasks_irs(all_task_irs, metadata)

            # Write to output file
            with open(args.output_file, "w", encoding="utf-8") as f:
                f.write(yaml_content)

            logger.info(
                "Successfully wrote task IRs YAML file",
                output_file=str(args.output_file),
                num_tasks=len(all_task_irs),
                file_size=len(yaml_content),
                mapping_checksum=mapping_checksum,
            )
        except Exception as e:
            logger.error("Failed to write output file", error=str(e), exc_info=True)
            sys.exit(1)
    else:
        logger.warning("No task IRs to write (all extractions failed)")
        sys.exit(1)

    # Summary
    logger.info(
        "Framework extraction complete",
        total=len(harness_containers),
        successful=successful,
        failed=failed,
        output_file=str(args.output_file),
        num_tasks=len(all_task_irs),
    )

    if failed > 0:
        logger.warning(
            "Some containers failed to process",
            failed=failed,
            total=len(harness_containers),
        )


if __name__ == "__main__":
    main()
