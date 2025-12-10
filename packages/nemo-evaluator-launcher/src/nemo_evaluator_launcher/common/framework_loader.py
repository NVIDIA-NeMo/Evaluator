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
"""Framework loading utilities for extracting and parsing framework.yml from containers."""

import os
import tempfile
from typing import Optional

import yaml
from nemo_evaluator.core.input import get_framework_evaluations

from nemo_evaluator_launcher.common.helpers import parse_container_image
from nemo_evaluator_launcher.common.logging_utils import logger
from nemo_evaluator_launcher.common.partial_pull import (
    GitlabRegistryAuthenticator,
    NvcrRegistryAuthenticator,
    RegistryAuthenticator,
    _read_docker_credentials,
    find_file_matching_pattern_in_image_layers,
)
from nemo_evaluator_launcher.common.task_ir import (
    HarnessIntermediateRepresentation,
    TaskIntermediateRepresentation,
    _extract_harness_from_container,
)


def _get_credentials_from_env_or_docker(
    registry_url: str,
    env_username_key: str,
    env_password_key: str,
    default_username: Optional[str] = None,
) -> tuple[Optional[str], Optional[str]]:
    """Get credentials from environment variables or Docker config file.

    Priority:
    1. Try token from env - if username in env, use it; if not, use standard username
    2. Only if no token in env, fallback to Docker config

    Args:
        registry_url: Registry URL to look up credentials for
        env_username_key: Environment variable name for username
        env_password_key: Environment variable name for password
        default_username: Default username to use if password is set but username is not

    Returns:
        Tuple of (username, password)
    """
    username = os.getenv(env_username_key)
    password = os.getenv(env_password_key)

    # Priority 1: If password from env, use it (with username from env or default)
    if password:
        if not username and default_username:
            username = default_username
        logger.info(
            "Using credentials from environment",
            registry_url=registry_url,
            username=username,
            has_password=True,
        )
        return username, password

    # Priority 2: No password in env - fallback to Docker config
    docker_creds = _read_docker_credentials(registry_url)
    if docker_creds:
        username, password = docker_creds
        logger.info(
            "Using credentials from Docker config file",
            registry_url=registry_url,
            username=username,
        )

    return username, password


def _create_authenticator(
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
        username, password = _get_credentials_from_env_or_docker(
            registry_url=registry_url,
            env_username_key="DOCKER_USERNAME",
            env_password_key="GITLAB_TOKEN",
            default_username="gitlab-ci-token",
        )

        if not password:
            logger.debug(
                "No credentials found, attempting anonymous access",
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

        # Try Docker config if no password
        if not password:
            docker_creds = _read_docker_credentials(registry_url)
            if docker_creds:
                username, password = docker_creds
                logger.info(
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


def _get_container_digest(
    authenticator: RegistryAuthenticator, repository: str, reference: str
) -> Optional[str]:
    """Get the manifest digest for a container image.

    Uses the Docker-Content-Digest header from the registry response if available,
    falling back to computing the digest from the manifest JSON if the header is absent.

    Args:
        authenticator: Registry authenticator instance
        repository: Repository name
        reference: Tag or digest

    Returns:
        Container digest (sha256:...) or None if failed
    """
    import hashlib
    import json

    try:
        manifest, headers = authenticator.get_manifest_with_headers(
            repository, reference
        )
        if not manifest:
            return None

        # Try to use Docker-Content-Digest header from registry response
        # This matches the registry's computed digest exactly
        if headers:
            # Look for Docker-Content-Digest header (case-insensitive)
            docker_content_digest = None
            for header_name, header_value in headers.items():
                if header_name.lower() == "docker-content-digest":
                    docker_content_digest = header_value
                    break
            if docker_content_digest:
                logger.debug(
                    "Using Docker-Content-Digest header from registry",
                    repository=repository,
                    reference=reference,
                    digest=docker_content_digest,
                )
                return docker_content_digest

        # Fallback: Compute manifest digest (SHA256 of canonical JSON)
        # This may not match the registry's digest if JSON serialization differs
        logger.debug(
            "Docker-Content-Digest header not available, computing digest from manifest",
            repository=repository,
            reference=reference,
        )
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


def extract_framework_yml(
    container: str,
    harness_name: Optional[str] = None,
    max_layer_size: Optional[int] = None,
    use_cache: bool = True,
) -> tuple[Optional[str], Optional[str]]:
    """Extract framework.yml from a container using layer inspection.

    Args:
        container: Container image identifier
        harness_name: Name of the harness (for logging). If None, extracted from container.
        max_layer_size: Optional maximum layer size in bytes
        use_cache: Whether to use caching

    Returns:
        Tuple of (framework_yml_content, container_digest) or (None, None) if failed
    """
    container_digest = None  # Initialize to track if we got digest before exception
    try:
        registry_type, registry_url, repository, tag = parse_container_image(container)

        # Extract harness name if not provided
        if not harness_name:
            harness_name = _extract_harness_from_container(container)

        logger.info(
            "Extracting framework.yml from container",
            container=container,
            harness=harness_name,
            registry_type=registry_type,
        )

        # Create authenticator
        authenticator = _create_authenticator(registry_type, registry_url, repository)

        # Authenticate
        if not authenticator.authenticate(repository=repository):
            logger.error(
                "Authentication failed",
                container=container,
                harness=harness_name,
            )
            return None, None

        # Get container digest (try to get it even if framework extraction might fail)
        container_digest = _get_container_digest(authenticator, repository, tag)
        if not container_digest:
            logger.warning(
                "Could not get container digest, continuing without it",
                container=container,
                harness=harness_name,
            )

        # Extract framework.yml using pattern-based search
        # Use original container string as docker_id for cache consistency
        docker_id = container

        # Use pattern-based search to find framework.yml in any subdirectory under /opt/metadata/
        logger.debug(
            "Searching for framework.yml using pattern-based search",
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
                "Found framework.yml",
                container=container,
                harness=harness_name,
                file_path=file_path,
            )
        else:
            framework_yml_content = None

        if not framework_yml_content:
            logger.warning(
                "framework.yml not found in container",
                container=container,
                harness=harness_name,
            )
            # Return digest even if framework extraction failed
            return None, container_digest

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
        # Preserve digest if we got it before the exception occurred
        return None, container_digest


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
