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
import os
import pathlib
import sys
from importlib import resources
from typing import Any, Optional

import requests
import yaml

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from nemo_evaluator_launcher.common.logging_utils import logger
from nemo_evaluator_launcher.common.partial_pull import (
    GitlabRegistryAuthenticator,
    NvcrRegistryAuthenticator,
    find_file_in_image_layers,
    find_file_matching_pattern_in_image_layers,
)

# Configuration constants
# For below, see docs: https://docs.github.com/en/rest/repos/contents
MAPPING_URL = "https://raw.githubusercontent.com/NVIDIA-NeMo/Eval/main/packages/nemo-evaluator-launcher/src/nemo_evaluator_launcher/resources/mapping.toml"
CACHE_DIR = pathlib.Path.home() / ".nemo-evaluator" / "cache"
CACHE_FILENAME = "mapping.toml"
INTERNAL_RESOURCES_PKG = "nemo_evaluator_launcher_internal.resources"


def _ensure_cache_dir() -> None:
    """Ensure the cache directory exists."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _get_cache_file() -> pathlib.Path:
    """Get the cache file path.

    Returns:
        pathlib.Path: Path to the cache file.
    """
    return CACHE_DIR / CACHE_FILENAME


def _download_latest_mapping() -> Optional[bytes]:
    """Download latest mapping from MAPPING_URL and return raw bytes.

    Returns:
        Optional[bytes]: Downloaded mapping bytes, or None if download fails.
    """
    try:
        response = requests.get(MAPPING_URL, timeout=10)
        response.raise_for_status()

        # For GitHub raw URLs, the response content is the file content directly
        mapping_bytes = response.content
        assert isinstance(mapping_bytes, bytes)

        logger.debug("Successfully downloaded mapping from remote URL")
        return mapping_bytes
    except (requests.RequestException, OSError) as e:
        logger.warning("Failed to download mapping from remote URL", error=str(e))
        return None


def _load_cached_mapping() -> Optional[dict[Any, Any]]:
    """Load mapping from cache file.

    Returns:
        Optional[dict]: Loaded mapping data, or None if loading fails.
    """
    cache_file = _get_cache_file()
    if not cache_file.exists():
        return None

    try:
        with open(cache_file, "rb") as f:
            mapping = tomllib.load(f)
        logger.debug("Loaded mapping from cache")
        return mapping  # type: ignore[no-any-return]
    except (OSError, tomllib.TOMLDecodeError) as e:
        logger.warning("Failed to load mapping from cache", error=str(e))
        return None


def _save_mapping_to_cache(mapping_bytes: bytes) -> None:
    """Save mapping to cache file.

    Args:
        mapping_bytes: Mapping data to save.
    """
    try:
        _ensure_cache_dir()
        cache_file = _get_cache_file()

        # Save the mapping data
        with open(cache_file, "wb") as f:
            f.write(mapping_bytes)

    except OSError as e:
        logger.warning("Failed to save mapping to cache", error=str(e))


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
                mapping[key] = {
                    "task": task_name,
                    "harness": harness_name,
                    "endpoint_type": endpoint_type,
                }
                # Only add container if it exists
                if container:
                    mapping[key]["container"] = container

                for task_data_key in task_data.keys():
                    if task_data_key in mapping[key]:
                        raise KeyError(
                            f"{repr(task_data_key)} is not allowed as key under {repr(key)} in the mapping"
                        )
                mapping[key].update(task_data)
    return mapping


def _parse_container_image(container_image: str) -> tuple[str, str, str, str]:
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


def _inspect_container_for_tasks(
    container: str, harness_name: str
) -> dict[tuple[str, str], dict]:
    """Inspect a container image to extract tasks from framework.yml.

    Args:
        container: Container image string (original, may be replaced by PoC stub)
        harness_name: Name of the harness

    Returns:
        Dictionary mapping (harness_name, task_name) to task configuration
    """
    try:
        # Parse container image (use the container from mapping.toml as-is)
        registry_type, registry_url, repository, tag = _parse_container_image(container)

        # Construct docker_id for caching
        docker_id = f"{registry_url}/{repository}:{tag}"
        target_file = "/opt/metadata/framework.yml"

        # Get credentials from environment
        if registry_type == "gitlab":
            username = os.getenv("DOCKER_USERNAME")
            password = os.getenv("GITLAB_TOKEN")
            if not username or not password:
                logger.debug(
                    "Skipping container inspection (missing GitLab credentials)",
                    container=container,
                )
                return {}
            authenticator = GitlabRegistryAuthenticator(
                registry_url=registry_url,
                username=username,
                password=password,
                repository=repository,
            )
        elif registry_type == "nvcr":
            username = os.getenv("NVCR_USERNAME") or os.getenv("DOCKER_USERNAME")
            password = os.getenv("NVCR_PASSWORD") or os.getenv("NVCR_API_KEY")
            if not username or not password:
                logger.debug(
                    "Skipping container inspection (missing nvcr credentials)",
                    container=container,
                )
                return {}
            authenticator = NvcrRegistryAuthenticator(
                registry_url=registry_url,
                username=username,
                password=password,
            )
        else:
            logger.warning(
                "Unknown registry type, skipping",
                container=container,
                registry_type=registry_type,
            )
            return {}

        # Get framework.yml from container (authentication handled inside)
        # Try the specified tag first, then fallback to "latest" if it fails
        framework_yml_content = None

        def _try_extract_framework_yml(
            ref_tag: str, ref_docker_id: str
        ) -> Optional[str]:
            """Helper function to try extracting framework.yml with fallback to subdirectories."""
            # First, try the standard location: /opt/metadata/framework.yml
            content = find_file_in_image_layers(
                authenticator=authenticator,
                repository=repository,
                reference=ref_tag,
                target_file=target_file,
                max_layer_size=100 * 1024,  # 100KB
                docker_id=ref_docker_id,
                use_cache=True,
                check_invalidated_digest=False,  # Use fast cache path
            )

            # If not found, try to find framework.yml in any subdirectory under /opt/metadata/
            if not content:
                logger.debug(
                    "framework.yml not found at standard location, searching subdirectories",
                    container=container,
                    tag=ref_tag,
                )
                result = find_file_matching_pattern_in_image_layers(
                    authenticator=authenticator,
                    repository=repository,
                    reference=ref_tag,
                    prefix="/opt/metadata",
                    filename="framework.yml",
                    max_layer_size=100 * 1024,  # 100KB
                    docker_id=ref_docker_id,
                    use_cache=True,
                )
                if result:
                    file_path, content = result
                    logger.info(
                        "Found framework.yml in subdirectory",
                        container=container,
                        tag=ref_tag,
                        file_path=file_path,
                    )

            return content

        try:
            framework_yml_content = _try_extract_framework_yml(tag, docker_id)
        except ValueError as e:
            # Tag might not exist, try "latest" as fallback
            if "Failed to get manifest" in str(e) and tag != "latest":
                logger.debug(
                    "Tag not found, trying 'latest' as fallback",
                    original_tag=tag,
                    container=container,
                )
                docker_id_latest = f"{registry_url}/{repository}:latest"
                try:
                    framework_yml_content = _try_extract_framework_yml(
                        "latest", docker_id_latest
                    )
                    # Update container to use latest tag for consistency
                    container = f"{registry_url}/{repository}:latest"
                except Exception as e2:
                    logger.debug(
                        "Failed to get framework.yml with 'latest' tag",
                        container=container,
                        error=str(e2),
                    )
            else:
                # Re-raise if it's a different error
                raise

        if not framework_yml_content:
            logger.debug(
                "framework.yml not found in container",
                container=container,
                tag=tag,
            )
            return {}

        # Extract tasks from framework.yml
        return _extract_tasks_from_framework_yml(
            framework_yml_content, harness_name, container
        )

    except Exception as e:
        logger.warning(
            "Failed to inspect container",
            container=container,
            harness=harness_name,
            error=str(e),
        )
        return {}


def load_tasks_mapping(
    latest: bool = False,
    mapping_toml: pathlib.Path | str | None = None,
) -> dict[tuple[str, str], dict]:
    """Load tasks mapping.

    The function obeys the following priority rules:
    1. (Default) If latest==False and mapping_toml is None -> load packaged mapping.
    2. If latest==True -> fetch MAPPING_URL, save to cache, load it.
    3. If mapping_toml is not None -> load mapping from this path.

    Docker container inspection can be enabled by setting the environment variable
    NE_USE_DOCKER_INSPECT=1. When enabled, the function will inspect Docker containers
    to extract additional tasks from framework.yml files.

    Args:
        latest: If True, fetch latest mapping from remote URL.
        mapping_toml: Optional path to mapping TOML file.

    Returns:
        dict: Mapping of (harness_name, task_name) to dict holding their configuration.

    """
    # Check environment variable for Docker inspection
    use_docker_inspect = os.getenv("NE_USE_DOCKER_INSPECT", "").strip() in (
        "1",
        "true",
        "yes",
    )
    if use_docker_inspect:
        logger.info(
            "Docker inspection enabled via NE_USE_DOCKER_INSPECT environment variable"
        )
    else:
        logger.debug(
            "Docker inspection disabled (set NE_USE_DOCKER_INSPECT=1 to enable)"
        )
    local_mapping: dict = {}
    if latest:
        mapping_bytes = _download_latest_mapping()
        if mapping_bytes:
            _save_mapping_to_cache(mapping_bytes)
            local_mapping = _process_mapping(
                tomllib.loads(mapping_bytes.decode("utf-8"))
            )
        else:
            # Fallback to cached mapping; raise only if cache is missing/invalid
            cached = _load_cached_mapping()
            if cached:
                local_mapping = _process_mapping(cached)
            else:
                raise RuntimeError("could not download latest mapping")

    elif mapping_toml is not None:
        with open(mapping_toml, "rb") as f:
            local_mapping = _process_mapping(tomllib.load(f))
    else:
        local_mapping = _process_mapping(_load_packaged_resource(CACHE_FILENAME))

    # TODO: make more elegant. We consider it ok to avoid a fully-blown plugin system.
    # Check if nemo_evaluator_launcher_internal is available and load its mapping.toml
    # CAVEAT: lazy-loading here, not somewhere top level, is important, to ensure
    # order of package initialization.
    try:
        importlib.import_module("nemo_evaluator_launcher_internal")
        logger.debug("Internal package available, loading internal mapping")
        internal_mapping = _process_mapping(
            _load_packaged_resource(CACHE_FILENAME, INTERNAL_RESOURCES_PKG)
        )

        # Merge internal mapping with local mapping (internal takes precedence)
        local_mapping.update(internal_mapping)
        logger.info(
            "Successfully merged internal mapping", internal_tasks=len(internal_mapping)
        )
    except ImportError:
        logger.debug("Internal package not available, using external mapping only")
    except Exception as e:
        logger.warning("Failed to load internal mapping", error=str(e))

    # Inspect Docker containers if requested
    if use_docker_inspect:
        logger.info("Inspecting Docker containers for additional tasks")
        # Collect unique containers per harness
        harness_containers: dict[str, set[str]] = {}
        for (harness_name, task_name), task_data in local_mapping.items():
            container = task_data.get("container")
            if container:
                if harness_name not in harness_containers:
                    harness_containers[harness_name] = set()
                harness_containers[harness_name].add(container)

        # Inspect each unique container
        inspected_tasks = {}
        for harness_name, containers in harness_containers.items():
            for container in containers:
                container_tasks = _inspect_container_for_tasks(container, harness_name)
                # Only add tasks that don't already exist in mapping
                for key, task_data in container_tasks.items():
                    inspected_tasks[key] = task_data

        if inspected_tasks:
            logger.info(
                "Added tasks from Docker inspection",
                num_new_tasks=len(inspected_tasks),
            )
            # Merge inspected tasks into local_mapping (don't replace - inspected tasks override existing ones)
            local_mapping.update(inspected_tasks)
        else:
            logger.debug("No new tasks found from Docker inspection")

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
