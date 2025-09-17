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
from typing import Any, Optional

import requests

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from nemo_evaluator_launcher.common.logging_utils import logger

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
        assert isinstance(harness_data["tasks"], dict)
        for endpoint_type, harness_tasks in harness_data["tasks"].items():
            assert isinstance(harness_tasks, dict)
            for task_name, task_data in harness_tasks.items():
                assert isinstance(task_data, dict)
                key = (harness_name, task_name)
                if key in mapping:
                    raise KeyError(
                        f"(harness,task)-tuple key {repr(key)} already exists in the mapping"
                    )
                mapping[key] = {
                    "task": task_name,
                    "harness": harness_name,
                    "container": harness_data["container"],
                    "endpoint_type": endpoint_type,
                }
                for task_data_key in task_data.keys():
                    if task_data_key in mapping[key]:
                        raise KeyError(
                            f"{repr(task_data_key)} is not allowed as key under {repr(key)} in the mapping"
                        )
                mapping[key].update(task_data)
    return mapping


def load_tasks_mapping(
    latest: bool = False,
    mapping_toml: pathlib.Path | str | None = None,
) -> dict[tuple[str, str], dict]:
    """Load tasks mapping.

    The function obeys the following priority rules:
    1. (Default) If latest==False and mapping_toml is None -> load packaged mapping.
    2. If latest==True -> fetch MAPPING_URL, save to cache, load it.
    3. If mapping_toml is not None -> load mapping from this path.

    Returns:
        dict: Mapping of (harness_name, task_name) to dict holding their configuration.

    """
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
