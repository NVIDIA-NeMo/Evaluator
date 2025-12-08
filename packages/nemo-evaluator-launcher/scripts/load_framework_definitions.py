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

Workflow diagram:
mapping.toml
(container = "..")
(# container-digest:sha256:....)              <---- CI: check digests are relevant

   |
   |
   v
scripts/load_framework_definitions.py
(updates the toml
 AND
creates the resources/all_tasks_irs,
records TOML checksum)                        <---- pre-commit guard: checks TOML checksum
   |          \
   |           \
   |            ------------------->    make docs-build
   |                                  (builds docs on the fly)
   v
scripts/update_readme.py
(updates README, records checksum)           <----- pre-commit guard: checks TOML checksum
"""

import argparse
import hashlib
import json
import os
import pathlib
import sys
from typing import Any, Optional

import yaml

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from nemo_evaluator_launcher.common.framework_loader import (
    extract_framework_yml,
    parse_framework_to_irs,
)
from nemo_evaluator_launcher.common.logging_utils import logger
from nemo_evaluator_launcher.common.partial_pull import (
    GitlabRegistryAuthenticator,
    NvcrRegistryAuthenticator,
    RegistryAuthenticator,
)
from nemo_evaluator_launcher.common.task_ir import (
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
        # Credentials are optional - try anonymous access first
        username = os.getenv("DOCKER_USERNAME")
        password = os.getenv("GITLAB_TOKEN")

        # If no password from environment, try Docker credentials file
        if not password:
            from nemo_evaluator_launcher.common.partial_pull import (
                _read_docker_credentials,
            )

            docker_creds = _read_docker_credentials(registry_url)
            if docker_creds:
                docker_username, docker_password = docker_creds
                # Use Docker credentials if username not set, or use Docker username if both available
                if not username:
                    username = docker_username
                password = docker_password
                logger.debug(
                    "Using credentials from Docker config file",
                    registry_url=registry_url,
                    username=username,
                )

        # If still no credentials provided, try anonymous access (for public registries)
        if not password:
            logger.debug(
                "No GITLAB_TOKEN or Docker credentials found, attempting anonymous access to GitLab registry",
                registry_url=registry_url,
                repository=repository,
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

        # If no password from environment, try Docker credentials file
        if not password:
            from nemo_evaluator_launcher.common.partial_pull import (
                _read_docker_credentials,
            )

            docker_creds = _read_docker_credentials(registry_url)
            if docker_creds:
                docker_username, docker_password = docker_creds
                # Use Docker credentials if username not set or is default, otherwise keep env username
                if not username or username == "$oauthtoken":
                    username = docker_username
                password = docker_password
                logger.debug(
                    "Using credentials from Docker config file",
                    registry_url=registry_url,
                    username=username,
                )

        if not password:
            raise ValueError(
                "NVCR_PASSWORD, NVCR_API_KEY environment variable, or Docker credentials "
                "are required for nvcr.io registry. Check ~/.docker/config.json for credentials."
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


def update_digest_comment_in_mapping_toml(
    mapping_file: pathlib.Path, container: str, digest: str
) -> None:
    """Update or add digest comment for container in mapping.toml.

    Args:
        mapping_file: Path to mapping.toml file
        container: Container image identifier
        digest: Container digest (sha256:...)
    """
    import re

    try:
        # Read file as text
        with open(mapping_file, "r", encoding="utf-8") as f:
            content = f.read()

        lines = content.split("\n")
        updated = False

        # Find the container line
        for i, line in enumerate(lines):
            # Match container = "..." lines (harness-level sections only)
            match = re.match(r'^(\s*)container\s*=\s*"([^"]+)"', line)
            if match and match.group(2) == container:
                indent = match.group(1)
                # Check if this is a harness-level section
                section_line_idx = i
                while section_line_idx >= 0 and not lines[
                    section_line_idx
                ].strip().startswith("["):
                    section_line_idx -= 1
                if section_line_idx >= 0:
                    section_line = lines[section_line_idx].strip()
                    # Harness-level sections don't have dots
                    if "." not in section_line.strip("[]"):
                        # Check if digest comment already exists
                        digest_comment_pattern = (
                            r"#\s*container-digest:\s*(sha256:[a-f0-9]+)"
                        )
                        existing_digest = None
                        comment_line_idx = None

                        # Look for existing digest comment in next few lines
                        for offset in range(1, 5):
                            if i + offset >= len(lines):
                                break
                            next_line = lines[i + offset]
                            match_comment = re.search(
                                digest_comment_pattern, next_line, re.IGNORECASE
                            )
                            if match_comment:
                                existing_digest = match_comment.group(1)
                                comment_line_idx = i + offset
                                break
                            # Stop if we hit another section or key
                            if next_line.strip().startswith("[") or (
                                next_line.strip()
                                and not next_line.strip().startswith("#")
                                and "=" in next_line
                            ):
                                break

                        # Update or insert digest comment
                        digest_comment = f"{indent}# container-digest:{digest}"

                        if existing_digest:
                            if existing_digest.lower() == digest.lower():
                                logger.debug(
                                    "Digest already correct",
                                    container=container,
                                    digest=digest,
                                )
                                continue
                            # Update existing comment
                            lines[comment_line_idx] = digest_comment
                            logger.debug(
                                "Updated digest comment",
                                container=container,
                                old_digest=existing_digest,
                                new_digest=digest,
                            )
                        else:
                            # Insert new comment after container line
                            lines.insert(i + 1, digest_comment)
                            logger.debug(
                                "Added digest comment",
                                container=container,
                                digest=digest,
                            )
                        updated = True
                        break

        # Write updated content if changed
        if updated:
            with open(mapping_file, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            logger.debug(
                "Updated mapping.toml with digest comment",
                container=container,
                digest=digest,
            )
    except Exception as e:
        logger.warning(
            "Failed to update digest comment in mapping.toml",
            container=container,
            digest=digest,
            error=str(e),
            exc_info=True,
        )
        # Don't fail the whole process if digest update fails


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

    # Set log level to INFO so extraction progress messages are visible
    # (default is WARNING which suppresses INFO messages)
    # This matches the behavior of CLI commands like 'ls task --from'
    if not os.getenv("NEMO_EVALUATOR_LOG_LEVEL") and not os.getenv("LOG_LEVEL"):
        os.environ["NEMO_EVALUATOR_LOG_LEVEL"] = "INFO"
        # Reconfigure logging with new level by re-importing and calling configure
        import importlib

        from nemo_evaluator_launcher.common import logging_utils

        importlib.reload(logging_utils)
        logging_utils._configure_structlog()

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

        # Update digest comment in mapping.toml if we got a digest
        if container_digest:
            update_digest_comment_in_mapping_toml(
                args.mapping_file, container, container_digest
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

    # Recalculate checksum after digest updates (digests may have been updated)
    try:
        mapping_checksum = calculate_mapping_checksum(args.mapping_file)
        logger.debug(
            "Recalculated mapping.toml checksum after digest updates",
            checksum=mapping_checksum,
        )
    except Exception as e:
        logger.warning(
            "Failed to recalculate mapping.toml checksum after digest updates",
            error=str(e),
        )
        # Continue with original checksum

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
