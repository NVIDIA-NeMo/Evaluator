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
"""Utilities for partial Docker image pulls to find specific files in layers.

This module provides functionality to search for files in Docker image layers
without pulling the entire image. It supports searching through layers filtered
by size and extracting specific files from layer tar archives.
"""

import base64
import hashlib
import json
import os
import pathlib
import tarfile
import tempfile
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import requests

from nemo_evaluator_launcher.common.logging_utils import logger

# Cache directory for Docker metadata
CACHE_DIR = pathlib.Path.home() / ".nemo-evaluator" / "docker-meta"
MAX_CACHED_DATA = 200  # Maximum number of cache entries

# Docker credentials file location
DOCKER_CONFIG_PATH = pathlib.Path.home() / ".docker" / "config.json"


def _read_docker_credentials(registry_url: str) -> Optional[Tuple[str, str]]:
    """Read Docker credentials from Docker config file.

    Docker stores credentials in ~/.docker/config.json with format:
    {
      "auths": {
        "registry-url": {
          "auth": "base64(username:password)"
        }
      }
    }

    Args:
        registry_url: Registry URL to look up credentials for (e.g., 'nvcr.io', 'gitlab-master.nvidia.com:5005')

    Returns:
        Tuple of (username, password) if found, None otherwise
    """
    if not DOCKER_CONFIG_PATH.exists():
        logger.debug(
            "Docker config file not found",
            config_path=str(DOCKER_CONFIG_PATH),
        )
        return None

    try:
        with open(DOCKER_CONFIG_PATH, "r", encoding="utf-8") as f:
            config = json.load(f)

        auths = config.get("auths", {})
        if not auths:
            logger.debug("No auths section in Docker config file")
            return None

        # Try exact match first
        registry_auth = auths.get(registry_url)
        if not registry_auth:
            # Try matching by hostname (without port)
            registry_host = registry_url.split(":")[0]
            registry_auth = auths.get(registry_host)
            # Also try with https:// prefix
            if not registry_auth:
                registry_auth = auths.get(f"https://{registry_url}")
            if not registry_auth:
                registry_auth = auths.get(f"https://{registry_host}")

        if not registry_auth:
            logger.debug(
                "No credentials found for registry in Docker config",
                registry_url=registry_url,
                available_registries=list(auths.keys()),
            )
            return None

        # Decode base64 auth string
        auth_string = registry_auth.get("auth")
        if not auth_string:
            logger.debug(
                "No auth field in Docker config for registry",
                registry_url=registry_url,
            )
            return None

        try:
            decoded = base64.b64decode(auth_string).decode("utf-8")
            if ":" not in decoded:
                logger.warning(
                    "Invalid auth format in Docker config (expected username:password)",
                    registry_url=registry_url,
                )
                return None
            username, password = decoded.split(":", 1)
            logger.debug(
                "Found credentials in Docker config",
                registry_url=registry_url,
                username=username,
            )
            return username, password
        except Exception as e:
            logger.warning(
                "Failed to decode auth from Docker config",
                registry_url=registry_url,
                error=str(e),
            )
            return None

    except json.JSONDecodeError as e:
        logger.warning(
            "Failed to parse Docker config file",
            config_path=str(DOCKER_CONFIG_PATH),
            error=str(e),
        )
        return None
    except Exception as e:
        logger.warning(
            "Error reading Docker config file",
            config_path=str(DOCKER_CONFIG_PATH),
            error=str(e),
        )
        return None


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
    docker_id: str, target_file: str, metadata_str: str, digest: str
) -> None:
    """Write metadata to cache with digest.

    Args:
        docker_id: Docker image identifier
        target_file: Target file path (or pattern for pattern-based searches)
        metadata_str: Metadata content to cache
        digest: Manifest digest of the container image. Required and stored in
            the cache entry for validation on subsequent reads.
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


class RegistryAuthenticator(ABC):
    """Abstract base class for Docker registry authentication and operations."""

    @abstractmethod
    def authenticate(self, repository: Optional[str] = None) -> bool:
        """Authenticate with the registry to obtain JWT token.

        Args:
            repository: Optional repository name for authentication scope.
                Implementation-specific.

        Returns:
            True if authentication successful, False otherwise
        """
        pass

    @abstractmethod
    def get_manifest(self, repository: str, reference: str) -> Optional[Dict]:
        """Get the manifest for a specific image reference.

        Args:
            repository: The repository name
            reference: The tag or digest

        Returns:
            The manifest as a dictionary, or None if failed
        """
        pass

    @abstractmethod
    def get_blob(self, repository: str, digest: str) -> Optional[bytes]:
        """Download a blob (layer) by its digest.

        Args:
            repository: The repository name
            digest: The blob digest

        Returns:
            The blob content as bytes, or None if failed
        """
        pass


class GitlabRegistryAuthenticator(RegistryAuthenticator):
    """GitLab-specific implementation of Docker registry authentication flow."""

    def __init__(
        self,
        registry_url: str,
        username: Optional[str] = None,
        password: Optional[str] = None,
        repository: Optional[str] = None,
    ):
        """Initialize the authenticator.

        Args:
            registry_url: The registry URL (e.g., 'gitlab-master.nvidia.com:5005')
            username: Registry username (optional, for authenticated access)
            password: Registry password or token (optional, for authenticated access)
            repository: Optional repository name for JWT scope. If not provided,
                a default scope will be used. The repository should be in the format
                'namespace/project' (e.g., 'agronskiy/idea/poc-for-partial-pull').
        """
        self.registry_url = registry_url.rstrip("/")
        self.username = username
        self.password = password
        self.repository = repository
        self.bearer_token: Optional[str] = None
        self.session = requests.Session()

    def authenticate(self, repository: Optional[str] = None) -> bool:
        """Authenticate with GitLab registry using Bearer Token flow.

        First attempts anonymous access (for public registries), then falls back to
        JWT authentication if credentials are available and anonymous access fails.

        Args:
            repository: Optional repository name for JWT scope. If provided, overrides
                the repository set during initialization. The repository should be in
                the format 'namespace/project' (e.g., 'agronskiy/idea/poc-for-partial-pull').

        Returns:
            True if authentication successful (anonymous or authenticated), False otherwise
        """
        try:
            logger.debug(
                "Authenticating with GitLab registry",
                registry_url=self.registry_url,
                repository=repository or self.repository,
                has_credentials=bool(self.username and self.password),
            )

            # Step 1: Try anonymous access first (for public registries)
            # Docker Registry API v2: check if /v2/ endpoint is accessible without auth
            v2_url = f"https://{self.registry_url}/v2/"
            logger.debug("Checking for public registry access", url=v2_url)

            response = self.session.get(v2_url)

            # If we get 200, registry is public - no authentication needed
            if response.status_code == 200:
                logger.debug("Registry is public, no authentication needed")
                # Set Accept header for Docker Registry API v2
                self.session.headers.update(
                    {
                        "Accept": "application/vnd.docker.distribution.manifest.v2+json",
                    }
                )
                return True

            # If we get 401, registry requires authentication
            if response.status_code != 401:
                logger.error(
                    "Unexpected response from registry",
                    status_code=response.status_code,
                    response_preview=response.text[:200],
                )
                # If we don't have credentials, try to proceed anyway
                # (some registries allow anonymous access to manifests even if /v2/ requires auth)
                if not (self.username and self.password):
                    logger.debug(
                        "No credentials available, attempting to proceed without authentication"
                    )
                    self.session.headers.update(
                        {
                            "Accept": "application/vnd.docker.distribution.manifest.v2+json",
                        }
                    )
                    return True
                return False

            # Step 2: Registry requires authentication - check if credentials are available
            if not (self.username and self.password):
                logger.debug(
                    "Registry requires authentication but no credentials provided, "
                    "attempting to proceed without authentication (may work for public repos)"
                )
                # Try to proceed anyway - some GitLab registries allow anonymous access
                # to manifests/blobs even if /v2/ endpoint requires auth
                self.session.headers.update(
                    {
                        "Accept": "application/vnd.docker.distribution.manifest.v2+json",
                    }
                )
                return True

            # Step 3: Use credentials for JWT authentication
            # Use provided repository or fall back to instance repository or default
            repo_for_scope = (
                repository
                or self.repository
                or f"{self.username}/idea/poc-for-partial-pull"
            )

            # GitLab-specific authentication flow
            # Get Bearer Token from GitLab JWT endpoint
            # Extract hostname from registry URL (remove port if present)
            gitlab_host = self.registry_url.split(":")[0]
            jwt_url = f"https://{gitlab_host}/jwt/auth?service=container_registry&scope=repository:{repo_for_scope}:pull"
            logger.debug("Requesting Bearer token", jwt_url=jwt_url)

            token_response = self.session.get(
                jwt_url, auth=(self.username, self.password)
            )

            if token_response.status_code != 200:
                logger.error(
                    "Token request failed",
                    status_code=token_response.status_code,
                    response_preview=token_response.text[:200],
                )
                return False

            token_data = token_response.json()
            self.bearer_token = token_data.get("token")

            if not self.bearer_token:
                logger.error(
                    "No token in response",
                    response_keys=list(token_data.keys()),
                )
                return False

            logger.debug(
                "Authentication successful",
                token_length=len(self.bearer_token) if self.bearer_token else 0,
            )

            # Set up session with bearer token
            self.session.headers.update(
                {
                    "Authorization": f"Bearer {self.bearer_token}",
                    "Accept": "application/vnd.docker.distribution.manifest.v2+json",
                }
            )

            return True

        except Exception as e:
            logger.error("Authentication error", error=str(e), exc_info=True)
            # On error, try to proceed without authentication (may work for public repos)
            if not (self.username and self.password):
                logger.debug(
                    "Authentication error but no credentials, attempting to proceed without authentication"
                )
                self.session.headers.update(
                    {
                        "Accept": "application/vnd.docker.distribution.manifest.v2+json",
                    }
                )
                return True
            return False

    def get_manifest(self, repository: str, reference: str) -> Optional[Dict]:
        """Get the manifest for a specific image reference.

        Args:
            repository: The repository name
            reference: The tag or digest

        Returns:
            The manifest as a dictionary, or None if failed
        """
        try:
            url = f"https://{self.registry_url}/v2/{repository}/manifests/{reference}"
            logger.debug("Fetching manifest", url=url)

            response = self.session.get(url)

            if response.status_code == 200:
                manifest = response.json()
                logger.debug(
                    "Successfully retrieved manifest",
                    schema_version=manifest.get("schemaVersion", "unknown"),
                    media_type=manifest.get("mediaType", "unknown"),
                    layers_count=len(manifest.get("layers", [])),
                )
                return manifest
            else:
                logger.error(
                    "Failed to get manifest",
                    status_code=response.status_code,
                    response_preview=response.text[:200],
                )
                return None

        except Exception as e:
            logger.error("Error fetching manifest", error=str(e), exc_info=True)
            return None

    def get_blob(self, repository: str, digest: str) -> Optional[bytes]:
        """Download a blob (layer) by its digest.

        Args:
            repository: The repository name
            digest: The blob digest

        Returns:
            The blob content as bytes, or None if failed
        """
        try:
            url = f"https://{self.registry_url}/v2/{repository}/blobs/{digest}"
            logger.debug("Downloading blob", digest_preview=digest[:20])

            response = self.session.get(url, stream=True)

            if response.status_code == 200:
                content = response.content
                logger.debug("Downloaded blob", size_bytes=len(content))
                return content
            else:
                logger.error(
                    "Failed to download blob",
                    status_code=response.status_code,
                    digest_preview=digest[:20],
                )
                return None

        except Exception as e:
            logger.error(
                "Error downloading blob",
                error=str(e),
                digest_preview=digest[:20],
                exc_info=True,
            )
            return None


class NvcrRegistryAuthenticator(RegistryAuthenticator):
    """NVIDIA Container Registry (nvcr.io) implementation using Docker Registry API v2."""

    def __init__(self, registry_url: str, username: str, password: str):
        """Initialize the authenticator.

        Args:
            registry_url: The registry URL (e.g., 'nvcr.io')
            username: Registry username (typically '$oauthtoken' for API key auth)
            password: Registry password or API key
        """
        self.registry_url = registry_url.rstrip("/")
        self.username = username
        self.password = password
        self.bearer_token: Optional[str] = None
        self.session = requests.Session()

    def authenticate(self, repository: Optional[str] = None) -> bool:
        """Authenticate with nvcr.io using Docker Registry API v2 Bearer token flow.

        Args:
            repository: Optional repository name for authentication scope.
                If provided, will be used in the token request scope.

        Returns:
            True if authentication successful, False otherwise
        """
        try:
            logger.debug(
                "Authenticating with nvcr.io registry",
                registry_url=self.registry_url,
            )

            # Docker Registry API v2 authentication flow
            # Step 1: Try to access /v2/ endpoint to get authentication challenge
            v2_url = f"https://{self.registry_url}/v2/"
            logger.debug("Requesting authentication challenge", url=v2_url)

            response = self.session.get(v2_url)

            # If we get 200, no auth needed (public registry)
            if response.status_code == 200:
                logger.debug("Registry is public, no authentication needed")
                return True

            # If we get 401, parse WWW-Authenticate header
            if response.status_code != 401:
                logger.error(
                    "Unexpected response from registry",
                    status_code=response.status_code,
                    response_preview=response.text[:200],
                )
                return False

            www_authenticate = response.headers.get("WWW-Authenticate", "")
            if not www_authenticate:
                logger.error("No WWW-Authenticate header in 401 response")
                return False

            # Parse WWW-Authenticate header
            # Format: Bearer realm="https://auth.example.com/token",service="registry.example.com",scope="repository:name:pull"
            auth_params = {}
            for part in www_authenticate.replace("Bearer ", "").split(","):
                if "=" in part:
                    key, value = part.split("=", 1)
                    auth_params[key.strip()] = value.strip('"')

            realm = auth_params.get("realm")
            service = auth_params.get("service", "")
            scope = auth_params.get("scope", "")

            if not realm:
                logger.error("No realm in WWW-Authenticate header")
                return False

            # Step 2: Request token from realm
            token_url = realm
            params = {"service": service}
            if scope:
                params["scope"] = scope
            elif repository:
                # Construct scope if not provided
                params["scope"] = f"repository:{repository}:pull"

            logger.debug("Requesting Bearer token", token_url=token_url, params=params)

            token_response = self.session.get(
                token_url,
                params=params,
                auth=(self.username, self.password),
            )

            if token_response.status_code != 200:
                logger.error(
                    "Token request failed",
                    status_code=token_response.status_code,
                    response_preview=token_response.text[:200],
                )
                return False

            token_data = token_response.json()
            self.bearer_token = token_data.get("token") or token_data.get(
                "access_token"
            )

            if not self.bearer_token:
                logger.error(
                    "No token in response",
                    response_keys=list(token_data.keys()),
                )
                return False

            logger.debug(
                "Authentication successful",
                token_length=len(self.bearer_token) if self.bearer_token else 0,
            )

            # Set up session with bearer token
            self.session.headers.update(
                {
                    "Authorization": f"Bearer {self.bearer_token}",
                    "Accept": "application/vnd.docker.distribution.manifest.v2+json",
                }
            )

            return True

        except Exception as e:
            logger.error("Authentication error", error=str(e), exc_info=True)
            return False

    def get_manifest(self, repository: str, reference: str) -> Optional[Dict]:
        """Get the manifest for a specific image reference.

        Args:
            repository: The repository name
            reference: The tag or digest

        Returns:
            The manifest as a dictionary, or None if failed
        """
        try:
            url = f"https://{self.registry_url}/v2/{repository}/manifests/{reference}"
            logger.debug("Fetching manifest", url=url)

            response = self.session.get(url)

            if response.status_code == 200:
                manifest = response.json()
                logger.debug(
                    "Successfully retrieved manifest",
                    schema_version=manifest.get("schemaVersion", "unknown"),
                    media_type=manifest.get("mediaType", "unknown"),
                    layers_count=len(manifest.get("layers", [])),
                )
                return manifest
            else:
                logger.error(
                    "Failed to get manifest",
                    status_code=response.status_code,
                    response_preview=response.text[:200],
                )
                return None

        except Exception as e:
            logger.error("Error fetching manifest", error=str(e), exc_info=True)
            return None

    def get_blob(self, repository: str, digest: str) -> Optional[bytes]:
        """Download a blob (layer) by its digest.

        Args:
            repository: The repository name
            digest: The blob digest

        Returns:
            The blob content as bytes, or None if failed
        """
        try:
            url = f"https://{self.registry_url}/v2/{repository}/blobs/{digest}"
            logger.debug("Downloading blob", digest_preview=digest[:20])

            response = self.session.get(url, stream=True)

            if response.status_code == 200:
                content = response.content
                logger.debug("Downloaded blob", size_bytes=len(content))
                return content
            else:
                logger.error(
                    "Failed to download blob",
                    status_code=response.status_code,
                    digest_preview=digest[:20],
                )
                return None

        except Exception as e:
            logger.error(
                "Error downloading blob",
                error=str(e),
                digest_preview=digest[:20],
                exc_info=True,
            )
            return None


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


def find_file_in_image_layers(
    authenticator: RegistryAuthenticator,
    repository: str,
    reference: str,
    target_file: str,
    max_layer_size: Optional[int] = None,
    docker_id: Optional[str] = None,
    use_cache: bool = True,
    check_invalidated_digest: bool = False,  # Deprecated - always checks digest now
) -> Optional[str]:
    """DEPRECATED: Use find_file_matching_pattern_in_image_layers() instead.

    This function is deprecated and will be removed in a future version.
    Use find_file_matching_pattern_in_image_layers() for pattern-based search
    which handles both exact paths and subdirectories.

    Find a file in Docker image layers without pulling the entire image.

    .. deprecated:: 4.2.0
        Use :func:`find_file_matching_pattern_in_image_layers` instead.

    This function searches through image layers (optionally filtered by size)
    to find a specific file. Layers are checked in reverse order (last to first)
    to find the most recent version of the file. Results are cached with digest
    validation to ensure cache hits only occur when the image digest matches.

    Cache always validates digest to prevent stale cache hits (e.g., for 'latest' tags).
    This requires fetching the manifest to get the current digest, but ensures
    cache correctness.

    Args:
        authenticator: Registry authenticator instance (will be authenticated if needed)
        repository: The repository name (e.g., 'agronskiy/idea/poc-for-partial-pull')
        reference: The tag or digest (e.g., 'latest')
        target_file: The file path to find (e.g., '/app/metadata.json')
        max_layer_size: Optional maximum layer size in bytes. Only layers smaller
            than this size will be checked. If None, all layers are checked.
        docker_id: Optional Docker image identifier for caching (e.g.,
            'nvcr.io/nvidia/eval-factory/simple-evals:25.10'). If provided and
            use_cache is True, will check cache before searching and write to cache
            after finding the file.
        use_cache: Whether to use caching. Defaults to True.
        check_invalidated_digest: Deprecated - digest is always checked now.
            Kept for backward compatibility but ignored.

    Returns:
        The file content as string if found, None otherwise

    Raises:
        ValueError: If authentication fails or manifest cannot be retrieved
    """
    import warnings

    warnings.warn(
        "find_file_in_image_layers() is deprecated. "
        "Use find_file_matching_pattern_in_image_layers() instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    # Convert exact file path to pattern-based search
    # Extract directory prefix and filename
    target_path = os.path.normpath(target_file).lstrip("/")
    path_parts = target_path.split("/")
    if len(path_parts) < 2:
        # Single filename, use root as prefix
        prefix = "/"
        filename = path_parts[0] if path_parts else target_file.lstrip("/")
    else:
        filename = path_parts[-1]
        prefix = "/" + "/".join(path_parts[:-1])

    # Use pattern-based search instead
    result = find_file_matching_pattern_in_image_layers(
        authenticator=authenticator,
        repository=repository,
        reference=reference,
        prefix=prefix,
        filename=filename,
        max_layer_size=max_layer_size,
        docker_id=docker_id,
        use_cache=use_cache,
    )

    if result:
        file_path, file_content = result
        return file_content
    return None


def find_file_matching_pattern_in_image_layers(
    authenticator: RegistryAuthenticator,
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
    # Authenticate if needed
    if not getattr(authenticator, "bearer_token", None):
        if not authenticator.authenticate(repository=repository):
            raise ValueError(f"Failed to authenticate for {repository}:{reference}")

    # Get manifest (required for digest validation)
    manifest = authenticator.get_manifest(repository, reference)
    if not manifest:
        raise ValueError(f"Failed to get manifest for {repository}:{reference}")

    # Compute manifest digest for cache validation
    manifest_json = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    manifest_digest = (
        f"sha256:{hashlib.sha256(manifest_json.encode('utf-8')).hexdigest()}"
    )

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
                        # Fallback: try to infer path from pattern
                        # Most common case: file is at prefix/filename
                        inferred_path = f"{prefix.rstrip('/')}/{filename}"
                        return (inferred_path, cached_result)
                except Exception:
                    pass
            # If we can't get the path, return None to trigger search
            # But this shouldn't happen if cache is properly structured
            logger.warning(
                "Cache hit but couldn't determine file path, re-searching",
                docker_id=docker_id,
                pattern=pattern_key,
            )
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

    # Get layers from manifest
    layers = manifest.get("layers", [])
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
                cache_path = _get_cache_path(docker_id, pattern_key)
                _evict_lru_cache_entries()
                try:
                    cache_data = {
                        "docker_id": docker_id,
                        "pattern": pattern_key,
                        "cached_file_path": file_path,  # Store resolved path
                        "metadata": file_content,
                        "digest": manifest_digest,
                    }
                    with open(cache_path, "w", encoding="utf-8") as f:
                        json.dump(cache_data, f, indent=2)
                    cache_path.touch()
                    logger.info(
                        "Cached metadata (pattern-based)",
                        docker_id=docker_id,
                        pattern=pattern_key,
                        resolved_path=file_path,
                        digest=manifest_digest,
                        cache_path=str(cache_path),
                    )
                except OSError as e:
                    logger.warning(
                        "Failed to write to cache",
                        docker_id=docker_id,
                        pattern=pattern_key,
                        error=str(e),
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
