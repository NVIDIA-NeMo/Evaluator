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
"""Unified loading utilities for extracting and parsing framework.yml from containers."""

import hashlib
import json
import os
import pathlib
import tarfile
import tempfile
from typing import Optional

import yaml
from nemo_evaluator.core.input import get_framework_evaluations

from nemo_evaluator_launcher.common.container_metadata.intermediate_repr import (
    HarnessIntermediateRepresentation,
    TaskIntermediateRepresentation,
)
from nemo_evaluator_launcher.common.container_metadata.registries import (
    DockerRegistryHandler,
    create_authenticator,
)
from nemo_evaluator_launcher.common.container_metadata.utils import (
    parse_container_image,
)
from nemo_evaluator_launcher.common.logging_utils import logger

# Default max layer size for framework.yml extraction (100KB)
DEFAULT_MAX_LAYER_SIZE = 100 * 1024

# Framework.yml location in containers
FRAMEWORK_YML_PREFIX = "/opt/metadata"
FRAMEWORK_YML_FILENAME = "framework.yml"

# Cache directory for Docker metadata
CACHE_DIR = pathlib.Path.home() / ".nemo-evaluator" / "docker-meta"
MAX_CACHED_DATA = 200  # Maximum number of cache entries


# ============================================================================
# Cache Management Functions
# ============================================================================


def _ensure_cache_dir() -> pathlib.Path:
    """Ensure the cache directory exists and return its path.

    Returns:
        Path to the cache directory
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR


def _get_cache_key(docker_id: str, target_file: str) -> str:
    """Generate a cache key from docker_id and target_file.

    Args:
        docker_id: Docker image identifier (e.g., 'nvcr.io/nvidia/eval-factory/simple-evals:25.10')
        target_file: Target file path (e.g., '/opt/metadata/framework.yml')

    Returns:
        Cache key (hash-based filename)
    """
    # Create a unique key from docker_id and target_file
    key_string = f"{docker_id}|{target_file}"
    # Use SHA256 hash to create a filesystem-safe filename
    hash_obj = hashlib.sha256(key_string.encode("utf-8"))
    return hash_obj.hexdigest()


def _get_cache_path(docker_id: str, target_file: str) -> pathlib.Path:
    """Get the cache file path for a given docker_id and target_file.

    Args:
        docker_id: Docker image identifier
        target_file: Target file path

    Returns:
        Path to the cache file
    """
    cache_dir = _ensure_cache_dir()
    cache_key = _get_cache_key(docker_id, target_file)
    return cache_dir / f"{cache_key}.json"


def _evict_lru_cache_entries() -> None:
    """Evict least recently used cache entries if cache exceeds MAX_CACHED_DATA.

    Uses file modification time to determine least recently used entries.
    """
    cache_dir = _ensure_cache_dir()
    cache_files = list(cache_dir.glob("*.json"))

    if len(cache_files) < MAX_CACHED_DATA:
        return

    # Sort by modification time (oldest first)
    cache_files.sort(key=lambda p: p.stat().st_mtime)

    # Delete oldest entries until we're under the limit
    num_to_delete = (
        len(cache_files) - MAX_CACHED_DATA + 1
    )  # +1 to make room for new entry
    for cache_file in cache_files[:num_to_delete]:
        try:
            cache_file.unlink()
            logger.debug("Evicted cache entry", cache_path=str(cache_file))
        except OSError as e:
            logger.warning(
                "Failed to evict cache entry", cache_path=str(cache_file), error=str(e)
            )


def read_from_cache(
    docker_id: str, target_file: str, check_digest: str
) -> tuple[Optional[str], Optional[str]]:
    """Read metadata from cache, validating digest.

    Args:
        docker_id: Docker image identifier
        target_file: Target file path
        check_digest: Manifest digest to validate against stored digest.
            Must match stored digest for cache hit. If doesn't match, returns
            (None, stored_digest) to indicate cache is invalid.

    Returns:
        Tuple of (cached metadata string if found and valid, stored_digest).
        Returns (None, None) if cache miss, (None, stored_digest) if digest mismatch.
    """
    cache_path = _get_cache_path(docker_id, target_file)
    if not cache_path.exists():
        logger.debug(
            "Cache miss (file not found)",
            docker_id=docker_id,
            target_file=target_file,
            cache_path=str(cache_path),
        )
        return None, None

    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            cache_data = json.load(f)
            stored_digest = cache_data.get("digest")
            metadata_str = cache_data.get("metadata")

            # Always check digest - required for cache validity
            if stored_digest is None:
                logger.info(
                    "Cache invalidated (no stored digest - old cache entry)",
                    docker_id=docker_id,
                    target_file=target_file,
                    cache_path=str(cache_path),
                )
                return None, None

            if stored_digest != check_digest:
                logger.info(
                    "Cache invalidated (digest mismatch)",
                    docker_id=docker_id,
                    target_file=target_file,
                    stored_digest=stored_digest,
                    current_digest=check_digest,
                )
                return None, stored_digest

            # Digest matches - cache hit!
            # Update file modification time for LRU tracking
            try:
                cache_path.touch()
            except OSError:
                pass  # Ignore errors updating mtime

            logger.info(
                "Cache hit (digest validated)",
                docker_id=docker_id,
                target_file=target_file,
                digest=stored_digest,
                cache_path=str(cache_path),
            )
            return metadata_str, stored_digest
    except (OSError, json.JSONDecodeError, KeyError) as e:
        logger.warning(
            "Failed to read from cache",
            docker_id=docker_id,
            target_file=target_file,
            cache_path=str(cache_path),
            error=str(e),
        )
        return None, None


def write_to_cache(
    docker_id: str,
    target_file: str,
    metadata_str: str,
    digest: str,
    cached_file_path: Optional[str] = None,
) -> None:
    """Write metadata to cache with digest.

    Args:
        docker_id: Docker image identifier
        target_file: Target file path (or pattern for pattern-based searches)
        metadata_str: Metadata content to cache
        digest: Manifest digest of the container image. Required and stored in
            the cache entry for validation on subsequent reads.
        cached_file_path: Optional resolved file path (for pattern-based searches).
            If provided, stored in cache for retrieval on cache hits.
    """
    # Evict old entries if cache is full
    _evict_lru_cache_entries()

    cache_path = _get_cache_path(docker_id, target_file)
    try:
        cache_data = {
            "docker_id": docker_id,
            "target_file": target_file,
            "metadata": metadata_str,
            "digest": digest,  # Always store digest - required for validation
        }
        # Always include cached_file_path if provided (standardized cache structure)
        if cached_file_path is not None:
            cache_data["cached_file_path"] = cached_file_path

        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, indent=2)

        # Update file modification time for LRU tracking
        try:
            cache_path.touch()
        except OSError:
            pass  # Ignore errors updating mtime

        logger.info(
            "Cached metadata",
            docker_id=docker_id,
            target_file=target_file,
            digest=digest,
            cache_path=str(cache_path),
            cached_file_path=cached_file_path,
        )
    except OSError as e:
        logger.warning(
            "Failed to write to cache",
            docker_id=docker_id,
            target_file=target_file,
            digest=digest,
            cache_path=str(cache_path),
            error=str(e),
        )


# ============================================================================
# Layer Inspection Functions
# ============================================================================


class LayerInspector:
    """Utility class for inspecting Docker layers."""

    @staticmethod
    def extract_file_from_layer(
        layer_content: bytes, target_file: str
    ) -> Optional[str]:
        """Extract a specific file from a layer tar archive.

        Args:
            layer_content: The layer content as bytes (tar.gz format)
            target_file: The file path to extract

        Returns:
            The file content as string if found, None otherwise
        """
        try:
            with tempfile.NamedTemporaryFile() as temp_file:
                temp_file.write(layer_content)
                temp_file.flush()

                with tarfile.open(temp_file.name, "r:gz") as tar:
                    logger.debug(
                        "Searching for file in layer",
                        target_file=target_file,
                        files_in_layer=len(tar.getmembers()),
                    )

                    # Look for the file in the tar archive
                    for member in tar.getmembers():
                        if member.name.endswith(
                            target_file
                        ) or member.name == target_file.lstrip("/"):
                            logger.debug("Found file in layer", file_path=member.name)
                            file_obj = tar.extractfile(member)
                            if file_obj:
                                content = file_obj.read().decode("utf-8")
                                logger.debug(
                                    "Extracted file content",
                                    file_path=member.name,
                                    content_size_chars=len(content),
                                )
                                return content

                    logger.debug(
                        "File not found in layer",
                        target_file=target_file,
                        sample_files=[m.name for m in tar.getmembers()[:10]],
                    )

        except Exception as e:
            logger.error(
                "Error extracting file from layer",
                error=str(e),
                target_file=target_file,
                exc_info=True,
            )

        return None

    @staticmethod
    def extract_file_matching_pattern(
        layer_content: bytes, prefix: str, filename: str
    ) -> Optional[tuple[str, str]]:
        """Extract a file matching a pattern from a layer tar archive.

        Searches for files that start with the given prefix and end with the given filename.
        For example, prefix="/opt/metadata/" and filename="framework.yml" will match
        "/opt/metadata/framework.yml" or "/opt/metadata/some_folder/framework.yml".

        Args:
            layer_content: The layer content as bytes (tar.gz format)
            prefix: The path prefix to match (e.g., "/opt/metadata/")
            filename: The filename to match (e.g., "framework.yml")

        Returns:
            Tuple of (file_path, file_content) if found, None otherwise
        """
        try:
            with tempfile.NamedTemporaryFile() as temp_file:
                temp_file.write(layer_content)
                temp_file.flush()

                with tarfile.open(temp_file.name, "r:gz") as tar:
                    logger.debug(
                        "Searching for file matching pattern in layer",
                        prefix=prefix,
                        filename=filename,
                        files_in_layer=len(tar.getmembers()),
                    )

                    # Normalize prefix to ensure it ends with /
                    normalized_prefix = prefix.rstrip("/") + "/"
                    # Also check without leading /
                    normalized_prefix_no_leading = normalized_prefix.lstrip("/")

                    # Look for files matching the pattern
                    for member in tar.getmembers():
                        # Check if file matches the pattern
                        # Match files that start with prefix and end with filename
                        member_name = member.name
                        member_name_no_leading = member_name.lstrip("/")

                        # Check if file starts with the prefix (with or without leading slash)
                        matches_prefix = member_name.startswith(
                            normalized_prefix
                        ) or member_name_no_leading.startswith(
                            normalized_prefix_no_leading
                        )

                        if not matches_prefix:
                            continue

                        # Check if file ends with the filename (exact match or with path separator)
                        # This ensures we match:
                        # - /opt/metadata/framework.yml
                        # - /opt/metadata/some_folder/framework.yml
                        # But not:
                        # - /opt/metadata/framework.yml.backup
                        # - /opt/metadata/framework_yml
                        matches_filename = (
                            member_name == normalized_prefix + filename
                            or member_name == normalized_prefix_no_leading + filename
                            or member_name.endswith(f"/{filename}")
                            or member_name_no_leading.endswith(f"/{filename}")
                        )

                        if matches_filename:
                            logger.debug(
                                "Found file matching pattern in layer",
                                file_path=member_name,
                                prefix=prefix,
                                filename=filename,
                            )
                            file_obj = tar.extractfile(member)
                            if file_obj:
                                content = file_obj.read().decode("utf-8")
                                logger.debug(
                                    "Extracted file content",
                                    file_path=member_name,
                                    content_size_chars=len(content),
                                )
                                return member_name, content

                    logger.debug(
                        "File matching pattern not found in layer",
                        prefix=prefix,
                        filename=filename,
                        sample_files=[m.name for m in tar.getmembers()[:10]],
                    )

        except Exception as e:
            logger.error(
                "Error extracting file matching pattern from layer",
                error=str(e),
                prefix=prefix,
                filename=filename,
                exc_info=True,
            )

        return None


# ============================================================================
# File Finding Functions
# ============================================================================


def find_file_matching_pattern_in_image_layers(
    authenticator: DockerRegistryHandler,
    repository: str,
    reference: str,
    prefix: str,
    filename: str,
    max_layer_size: Optional[int] = None,
    docker_id: Optional[str] = None,
    use_cache: bool = True,
) -> Optional[tuple[str, str]]:
    """Find a file matching a pattern in Docker image layers without pulling the entire image.

    This function searches through image layers (optionally filtered by size)
    to find a file matching the pattern (prefix + filename). Layers are checked
    in reverse order (last to first) to find the most recent version of the file.

    Args:
        authenticator: Registry authenticator instance (will be authenticated if needed)
        repository: The repository name (e.g., 'agronskiy/idea/poc-for-partial-pull')
        reference: The tag or digest (e.g., 'latest')
        prefix: The path prefix to match (e.g., '/opt/metadata/')
        filename: The filename to match (e.g., 'framework.yml')
        max_layer_size: Optional maximum layer size in bytes. Only layers smaller
            than this size will be checked. If None, all layers are checked.
        docker_id: Optional Docker image identifier for caching. If provided and
            use_cache is True, will check cache before searching and write to cache
            after finding the file.
        use_cache: Whether to use caching. Defaults to True.

    Returns:
        Tuple of (file_path, file_content) if found, None otherwise

    Raises:
        ValueError: If authentication fails or manifest cannot be retrieved
    """
    # Authenticate if needed (but don't fail if it returns False - may work for public containers)
    if not getattr(authenticator, "bearer_token", None):
        authenticator.authenticate(repository=repository)
        # Don't fail here - authentication may fail but public containers can still be accessed

    # Get top-level manifest and digest (tag may resolve to multi-arch index).
    top_manifest, top_digest = authenticator.get_manifest_and_digest(
        repository, reference
    )
    if not top_manifest:
        raise ValueError(f"Failed to get manifest for {repository}:{reference}")
    if not top_digest:
        raise ValueError(f"Failed to get digest for {repository}:{reference}")

    # Keep top-level digest for caching/validation, but resolve to a platform-specific
    # manifest for layer inspection when the top-level is an index/manifest list.
    manifest = top_manifest
    manifest_digest = top_digest
    if isinstance(top_manifest, dict) and isinstance(
        top_manifest.get("manifests"), list
    ):
        # Prefer registry's default platform resolver by requesting a single-image manifest.
        single_accept = (
            "application/vnd.oci.image.manifest.v1+json, "
            "application/vnd.docker.distribution.manifest.v2+json"
        )
        resolved, _ = authenticator.get_manifest_and_digest(
            repository, reference, accept=single_accept
        )
        if resolved:
            manifest = resolved

        # If a registry still returns an index (ignoring Accept), fall back to the
        # first digest entry for layer inspection. This does NOT affect recorded digests.
        if isinstance(manifest, dict) and isinstance(manifest.get("manifests"), list):
            for m in manifest.get("manifests") or []:
                if isinstance(m, dict):
                    d = m.get("digest")
                    if isinstance(d, str) and d.startswith("sha256:"):
                        resolved2, _ = authenticator.get_manifest_and_digest(
                            repository, d, accept=single_accept
                        )
                        if resolved2:
                            manifest = resolved2
                        break

    # Check cache with digest validation (always validates digest)
    # For pattern searches, use pattern-based cache key (not resolved path)
    # This allows cache hits regardless of where the file is found
    if docker_id and use_cache:
        # Create pattern-based cache key: prefix + filename
        # This ensures same cache key regardless of subdirectory location
        pattern_key = f"{prefix.rstrip('/')}/{filename}"
        logger.debug(
            "Checking cache for pattern",
            docker_id=docker_id,
            pattern=pattern_key,
            current_digest=manifest_digest,
        )
        cached_result, stored_digest = read_from_cache(
            docker_id, pattern_key, check_digest=manifest_digest
        )
        if cached_result is not None:
            # Parse the cached result to extract file path and content
            # The cached metadata should be the file content
            # We need to return (file_path, file_content) but we don't know the path
            # So we'll need to search for it or store the path in cache
            # For now, let's store the path in the cache entry
            logger.info(
                "Using cached metadata (pattern-based, digest validated)",
                docker_id=docker_id,
                pattern=pattern_key,
                digest=manifest_digest,
            )
            # Try to get the file path from cache entry
            cache_path = _get_cache_path(docker_id, pattern_key)
            if cache_path.exists():
                try:
                    with open(cache_path, "r", encoding="utf-8") as f:
                        cache_data = json.load(f)
                        cached_file_path = cache_data.get("cached_file_path")
                        if cached_file_path:
                            return (cached_file_path, cached_result)
                except (OSError, json.JSONDecodeError, KeyError) as e:
                    # Log specific exception types for better debugging
                    logger.debug(
                        "Failed to read cached_file_path from cache",
                        docker_id=docker_id,
                        pattern=pattern_key,
                        error=str(e),
                        error_type=type(e).__name__,
                    )
                except Exception as e:
                    # Log unexpected exceptions at warning level
                    logger.warning(
                        "Unexpected error reading cached_file_path from cache",
                        docker_id=docker_id,
                        pattern=pattern_key,
                        error=str(e),
                        error_type=type(e).__name__,
                        exc_info=True,
                    )
            # Fallback: try to infer path from pattern
            # Most common case: file is at prefix/filename
            inferred_path = f"{prefix.rstrip('/')}/{filename}"
            logger.debug(
                "Using inferred file path from pattern (cached_file_path not in cache)",
                docker_id=docker_id,
                pattern=pattern_key,
                inferred_path=inferred_path,
            )
            return (inferred_path, cached_result)
        elif stored_digest is not None:
            # Digest mismatch - cache invalidated
            logger.info(
                "Cache invalidated (digest changed), re-searching",
                docker_id=docker_id,
                pattern=pattern_key,
                stored_digest=stored_digest,
                current_digest=manifest_digest,
            )
        else:
            logger.debug(
                "Cache miss - no cached entry found for pattern",
                docker_id=docker_id,
                pattern=pattern_key,
            )

    # Get layers from manifest (single-arch image manifest).
    layers = manifest.get("layers", []) if isinstance(manifest, dict) else []
    logger.info(
        "Searching for file matching pattern in image layers",
        repository=repository,
        reference=reference,
        prefix=prefix,
        filename=filename,
        total_layers=len(layers),
        max_layer_size=max_layer_size,
    )

    # Initialize layer inspector
    inspector = LayerInspector()

    # Check each layer for files matching the pattern (in reverse order)
    # Reverse order ensures we get the most recent version of the file
    for i, layer in enumerate(reversed(layers)):
        original_index = len(layers) - 1 - i
        layer_digest = layer.get("digest")
        layer_size = layer.get("size", 0)

        if not layer_digest:
            logger.warning(
                "Layer has no digest, skipping",
                layer_index=original_index,
            )
            continue

        # Filter by size if max_layer_size is specified
        if max_layer_size is not None and layer_size >= max_layer_size:
            logger.debug(
                "Skipping layer (too large)",
                layer_index=original_index,
                layer_size=layer_size,
                max_layer_size=max_layer_size,
            )
            continue

        logger.debug(
            "Checking layer for pattern match",
            layer_index=original_index,
            reverse_index=i,
            digest_preview=layer_digest[:20],
            size=layer_size,
            media_type=layer.get("mediaType", "unknown"),
        )

        # Download the layer
        layer_content = authenticator.get_blob(repository, layer_digest)
        if not layer_content:
            logger.warning(
                "Failed to download layer",
                layer_index=original_index,
                digest_preview=layer_digest[:20],
            )
            continue

        # Extract files matching the pattern from this layer
        result = inspector.extract_file_matching_pattern(
            layer_content, prefix, filename
        )
        if result:
            file_path, file_content = result
            logger.info(
                "Found file matching pattern in layer",
                file_path=file_path,
                layer_index=original_index,
                digest_preview=layer_digest[:20],
            )
            # Cache the result if docker_id is provided and caching is enabled
            # Always store digest for validation on subsequent reads
            # Use pattern-based cache key (not resolved file path) for consistency
            if docker_id and use_cache:
                pattern_key = f"{prefix.rstrip('/')}/{filename}"
                # Store both the content and the resolved file path in cache
                # Standardized cache structure always includes cached_file_path
                write_to_cache(
                    docker_id=docker_id,
                    target_file=pattern_key,
                    metadata_str=file_content,
                    digest=manifest_digest,
                    cached_file_path=file_path,  # Store resolved path
                )
                logger.info(
                    "Cached metadata (pattern-based)",
                    docker_id=docker_id,
                    pattern=pattern_key,
                    resolved_path=file_path,
                    digest=manifest_digest,
                )
            return result
        else:
            logger.debug(
                "File matching pattern not found in layer",
                prefix=prefix,
                filename=filename,
                layer_index=original_index,
            )

    logger.warning(
        "File matching pattern not found in any layer",
        prefix=prefix,
        filename=filename,
        repository=repository,
        reference=reference,
    )
    return None


def get_container_digest(
    authenticator: DockerRegistryHandler, repository: str, reference: str
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
    try:
        _, digest = authenticator.get_manifest_and_digest(repository, reference)
        return digest

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
    max_layer_size: Optional[int] = None,
    use_cache: bool = True,
) -> tuple[Optional[str], Optional[str]]:
    """Extract framework.yml from a container using layer inspection.

    Args:
        container: Container image identifier
        max_layer_size: Optional maximum layer size in bytes
        use_cache: Whether to use caching

    Returns:
        Tuple of (framework_yml_content, container_digest) or (None, None) if failed
    """
    container_digest = None
    try:
        registry_type, registry_url, repository, tag = parse_container_image(container)

        logger.info(
            "Extracting frame definition file from the container",
            container=container,
            registry_type=registry_type,
            filename=FRAMEWORK_YML_FILENAME,
        )

        # Create authenticator and authenticate
        authenticator = create_authenticator(registry_type, registry_url, repository)
        authenticator.authenticate(repository=repository)

        # Get container digest
        container_digest = get_container_digest(authenticator, repository, tag)
        if not container_digest:
            logger.warning(
                "Could not get container digest, continuing without it",
                container=container,
            )

        # Search for framework.yml in container layers
        logger.debug(
            "Searching for frame definition file using pattern-based search",
            filename=FRAMEWORK_YML_FILENAME,
            container=container,
        )
        result = find_file_matching_pattern_in_image_layers(
            authenticator=authenticator,
            repository=repository,
            reference=tag,
            prefix=FRAMEWORK_YML_PREFIX,
            filename=FRAMEWORK_YML_FILENAME,
            max_layer_size=max_layer_size,
            docker_id=container,
            use_cache=use_cache,
        )

        if not result:
            logger.warning(
                "Frame definition file not found in container",
                filename=FRAMEWORK_YML_FILENAME,
                container=container,
            )
            return None, container_digest

        file_path, framework_yml_content = result
        logger.info(
            "Successfully extracted frame definition file",
            filename=FRAMEWORK_YML_FILENAME,
            container=container,
            file_path=file_path,
            content_size=len(framework_yml_content),
            digest=container_digest,
        )

        return framework_yml_content, container_digest

    except Exception as e:
        logger.warning(
            "Failed to extract frame definition file",
            filename=FRAMEWORK_YML_FILENAME,
            container=container,
            error=str(e),
            exc_info=True,
        )
        return None, container_digest


def _extract_task_description(framework_data: dict, task_name: str) -> str:
    """Extract task description from framework.yml data.

    Args:
        framework_data: Parsed framework.yml dictionary
        task_name: Name of the task

    Returns:
        Task description string
    """
    for eval_config in framework_data.get("evaluations", []):
        eval_task_name = eval_config.get("defaults", {}).get("config", {}).get("type")
        if eval_task_name == task_name:
            return eval_config.get("description", "")
    return ""


def _create_task_irs(
    evaluations: dict,
    framework_data: dict,
    harness_name: str,
    container_id: str,
    container_digest: Optional[str],
    container_arch: Optional[str] = None,
) -> list[TaskIntermediateRepresentation]:
    """Create TaskIntermediateRepresentation objects from evaluations.

    Args:
        evaluations: Dictionary of evaluation objects from get_framework_evaluations
        framework_data: Parsed framework.yml dictionary
        harness_name: Harness name (original, not normalized)
        container_id: Container image identifier
        container_digest: Container manifest digest

    Returns:
        List of TaskIntermediateRepresentation objects
    """
    task_irs = []
    for task_name, evaluation in evaluations.items():
        task_description = _extract_task_description(framework_data, task_name)
        evaluation_dict = evaluation.model_dump(exclude_none=True)

        task_ir = TaskIntermediateRepresentation(
            name=task_name,
            description=task_description,
            harness=harness_name,
            container=container_id,
            container_digest=container_digest,
            container_arch=container_arch,
            defaults=evaluation_dict,
        )

        task_irs.append(task_ir)

        logger.debug(
            "Created task IR",
            harness=harness_name,
            task=task_name,
            container=container_id,
        )

    return task_irs


def parse_framework_to_irs(
    framework_content: str,
    container_id: str,
    container_digest: Optional[str],
    container_arch: Optional[str] = None,
) -> tuple[HarnessIntermediateRepresentation, list[TaskIntermediateRepresentation]]:
    """Parse framework.yml content and convert to Intermediate Representations.

    Args:
        framework_content: Original framework.yml content as string
        container_id: Full container image identifier
        container_digest: Container manifest digest (optional)

    Returns:
        Tuple of (HarnessIntermediateRepresentation, list[TaskIntermediateRepresentation])

    Raises:
        ValueError: If framework.yml is empty or missing framework.name
    """
    try:
        framework_data = yaml.safe_load(framework_content)
        if not framework_data:
            raise ValueError("Empty framework.yml content")

        # Extract harness metadata from framework.yml
        framework_info = framework_data.get("framework", {})
        harness_name = framework_info.get("name")
        if not harness_name:
            raise ValueError(
                "framework.yml missing required 'framework.name' field. "
                "The harness name must be specified in the framework.yml file."
            )

        if not isinstance(harness_name, str):
            raise ValueError(
                f"framework.name must be a string, got {type(harness_name).__name__}"
            )

        # Write to temporary file for get_framework_evaluations
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yml", delete=False
        ) as temp_file:
            temp_file.write(framework_content)
            temp_file_path = temp_file.name

        try:
            # Parse evaluations using nemo_evaluator
            (
                parsed_framework_name,
                framework_defaults,
                evaluations,
            ) = get_framework_evaluations(temp_file_path)

            # Create harness IR
            harness_ir = HarnessIntermediateRepresentation(
                name=harness_name,
                description=framework_info.get("description", ""),
                full_name=framework_info.get("full_name"),
                url=framework_info.get("url"),
                container=container_id,
                container_digest=container_digest,
            )

            # Create task IRs
            task_irs = _create_task_irs(
                evaluations,
                framework_data,
                harness_name,
                container_id,
                container_digest,
                container_arch,
            )

            logger.info(
                "Parsed framework to IRs",
                harness=harness_name,
                num_tasks=len(task_irs),
                container=container_id,
            )

            return harness_ir, task_irs

        finally:
            try:
                os.unlink(temp_file_path)
            except Exception:
                pass

    except Exception as e:
        logger.error(
            "Failed to parse frame definition file to IRs",
            filename=FRAMEWORK_YML_FILENAME,
            error=str(e),
            container=container_id,
            exc_info=True,
        )
        raise


def load_tasks_from_container(
    container: str,
    max_layer_size: Optional[int] = None,
    use_cache: bool = True,
) -> list[TaskIntermediateRepresentation]:
    """Load tasks from container by extracting and parsing framework.yml.

    Args:
        container: Container image identifier
        max_layer_size: Optional maximum layer size in bytes for layer inspection
        use_cache: Whether to use caching for framework.yml extraction

    Returns:
        List of TaskIntermediateRepresentation objects

    Raises:
        ValueError: If container filtering results in no tasks
    """
    logger.debug("Loading tasks from container", container=container)

    def _normalize_platform_arch(arch: object) -> Optional[str]:
        if not arch:
            return None
        arch_l = str(arch).lower()
        if arch_l in {"amd64", "x86_64"}:
            return "amd"
        if arch_l in {"arm64", "aarch64"}:
            return "arm"
        return None

    def _arch_label_from_arch_set(archs: set[str]) -> Optional[str]:
        if not archs:
            return None
        if "amd" in archs and "arm" in archs:
            return "multiarch"
        if archs == {"amd"}:
            return "amd"
        if archs == {"arm"}:
            return "arm"
        return None

    def _get_container_arch(container_ref: str) -> Optional[str]:
        """Best-effort: derive 'amd'|'arm'|'multiarch' from registry manifest APIs."""
        try:
            registry_type, registry_url, repository, tag = parse_container_image(
                container_ref
            )
            authenticator = create_authenticator(
                registry_type, registry_url, repository
            )
            authenticator.authenticate(repository=repository)

            manifest, _ = authenticator.get_manifest_and_digest(repository, tag)
            if not isinstance(manifest, dict):
                return None

            manifests = manifest.get("manifests")
            if isinstance(manifests, list):
                archs = {
                    _normalize_platform_arch(
                        (m.get("platform") or {}).get("architecture")
                    )
                    for m in manifests
                    if isinstance(m, dict)
                }
                archs.discard(None)  # type: ignore[arg-type]
                return _arch_label_from_arch_set(set(archs))  # type: ignore[arg-type]

            cfg = manifest.get("config") or {}
            cfg_digest = cfg.get("digest") if isinstance(cfg, dict) else None
            if not (isinstance(cfg_digest, str) and cfg_digest.startswith("sha256:")):
                return None
            blob = authenticator.get_blob(repository, cfg_digest)
            if not blob:
                return None
            try:
                cfg_json = json.loads(blob.decode("utf-8"))
            except Exception:
                return None
            if not isinstance(cfg_json, dict):
                return None
            return _normalize_platform_arch(cfg_json.get("architecture"))
        except Exception:
            return None

    container_arch = _get_container_arch(container)

    # Extract framework.yml from container
    framework_content, container_digest = extract_framework_yml(
        container=container,
        max_layer_size=max_layer_size or DEFAULT_MAX_LAYER_SIZE,
        use_cache=use_cache,
    )

    if not framework_content:
        logger.error(
            "Could not extract frame definition file from container",
            container=container,
            filename=FRAMEWORK_YML_FILENAME,
        )
        return []

    try:
        # Parse framework.yml to IRs (harness name will be extracted from framework.yml)
        harness_ir, task_irs = parse_framework_to_irs(
            framework_content=framework_content,
            container_id=container,
            container_digest=container_digest,
            container_arch=container_arch,
        )

        logger.info(
            "Loaded tasks from container",
            container=container,
            num_tasks=len(task_irs),
        )

        if len(task_irs) == 0:
            logger.warning(
                "No tasks found in the specified container",
                container=container,
            )

        return task_irs

    except Exception as e:
        logger.error(
            "Failed to parse frame definition file from container",
            filename=FRAMEWORK_YML_FILENAME,
            container=container,
            error=str(e),
            exc_info=True,
        )
        return []
