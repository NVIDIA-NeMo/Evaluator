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
"""Registry authentication and credential management for container registries."""

import base64
import hashlib
import json
import os
import pathlib
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import requests

from nemo_evaluator_launcher.common.logging_utils import logger

# Docker credentials file location for falling back if public and PAT auth failed
_DOCKER_CONFIG_PATH = pathlib.Path.home() / ".docker" / "config.json"

# Docker Registry API v2 manifest Accept header.
# IMPORTANT: include *manifest list* / *OCI index* types so multi-arch tags return
# the index by default (otherwise registries may negotiate down to a single-arch
# manifest, typically amd64).
_DOCKER_MANIFEST_MEDIA_TYPE = (
    "application/vnd.oci.image.index.v1+json, "
    "application/vnd.docker.distribution.manifest.list.v2+json, "
    "application/vnd.oci.image.manifest.v1+json, "
    "application/vnd.docker.distribution.manifest.v2+json"
)


def _build_key_variants(registry_url: str) -> list[str]:
    """Build list of key variants to try when looking up Docker credentials.

    Args:
        registry_url: Registry URL

    Returns:
        List of key variants to try
    """
    registry_host = registry_url.split(":")[0]
    variants = [
        registry_url,
        registry_host,
        f"https://{registry_url}",
        f"https://{registry_host}",
    ]

    # For GitLab, also try common ports
    if "gitlab" in registry_host.lower():
        for port in ["5005", "5050"]:
            variants.extend(
                [f"{registry_host}:{port}", f"https://{registry_host}:{port}"]
            )

    return variants


def _find_auth_in_config(
    auths: dict, registry_url: str
) -> tuple[Optional[dict], Optional[str]]:
    """Find authentication entry in Docker config auths section.

    Args:
        auths: Auths dictionary from Docker config
        registry_url: Registry URL to look up

    Returns:
        Tuple of (auth dict, matched key) or (None, None) if not found
    """
    registry_host = registry_url.split(":")[0]
    key_variants = _build_key_variants(registry_url)

    # Try exact matches first
    for key in key_variants:
        if key in auths:
            return auths[key], key

    # Fallback: match by hostname
    for key in auths.keys():
        key_host = key.split("://")[-1].split(":")[0].split("/")[0]
        if key_host == registry_host:
            logger.info(
                "Found credentials using hostname match",
                registry_url=registry_url,
                matched_key=key,
            )
            return auths[key], key

    return None, None


def _decode_auth_string(
    auth_string: str, registry_url: str
) -> Optional[Tuple[str, str]]:
    """Decode base64 auth string from Docker config.

    Args:
        auth_string: Base64 encoded auth string
        registry_url: Registry URL (for logging)

    Returns:
        Tuple of (username, password) or None if decoding fails
    """
    try:
        decoded = base64.b64decode(auth_string).decode("utf-8")
        if ":" not in decoded:
            logger.warning(
                "Invalid auth format in Docker config (expected username:password)",
                registry_url=registry_url,
            )
            return None
        return decoded.split(":", 1)
    except Exception as e:
        logger.warning(
            "Failed to decode auth from Docker config",
            registry_url=registry_url,
            error=str(e),
        )
        return None


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
        registry_url: Registry URL to look up credentials for

    Returns:
        Tuple of (username, password) if found, None otherwise
    """
    if not _DOCKER_CONFIG_PATH.exists():
        logger.debug(
            "Docker config file not found", config_path=str(_DOCKER_CONFIG_PATH)
        )
        return None

    try:
        with open(_DOCKER_CONFIG_PATH, "r", encoding="utf-8") as f:
            config = json.load(f)

        auths = config.get("auths", {})
        if not auths:
            logger.debug("No auths section in Docker config file")
            return None

        logger.debug(
            "Looking up Docker credentials",
            registry_url=registry_url,
            available_keys=list(auths.keys()),
        )

        registry_auth, matched_key = _find_auth_in_config(auths, registry_url)
        if not registry_auth:
            registry_host = registry_url.split(":")[0]
            logger.debug(
                "No credentials found for registry in Docker config",
                registry_url=registry_url,
                registry_host=registry_host,
                available_registries=list(auths.keys()),
            )
            return None

        auth_string = registry_auth.get("auth")
        if not auth_string:
            logger.debug(
                "No auth field in Docker config for registry", registry_url=registry_url
            )
            return None

        result = _decode_auth_string(auth_string, registry_url)
        if result:
            username, password = result
            logger.info(
                "Found credentials in Docker config",
                registry_url=registry_url,
                username=username,
                matched_key=matched_key or registry_url,
            )
        return result

    except json.JSONDecodeError as e:
        logger.warning(
            "Failed to parse Docker config file",
            config_path=str(_DOCKER_CONFIG_PATH),
            error=str(e),
        )
        return None
    except Exception as e:
        logger.warning(
            "Error reading Docker config file",
            config_path=str(_DOCKER_CONFIG_PATH),
            error=str(e),
        )
        return None


def _retry_without_auth(
    url: str, stream: bool = False, accept: Optional[str] = None
) -> Optional[requests.Response]:
    """Retry HTTP request without authentication headers.

    Used for accessing public containers that may return 401/403 even for
    anonymous access.

    Args:
        url: URL to request
        stream: Whether to stream the response

    Returns:
        Response object if successful, None otherwise
    """
    temp_session = requests.Session()
    temp_session.headers.update({"Accept": accept or _DOCKER_MANIFEST_MEDIA_TYPE})
    response = temp_session.get(url, stream=stream)
    if response.status_code == 200:
        return response
    return None


class DockerRegistryHandler(ABC):
    """Abstract base class for Docker registry authentication and operations."""

    @abstractmethod
    def authenticate(self, repository: Optional[str] = None) -> bool:
        """Authenticate with the registry to obtain JWT token.

        Args:
            repository: Optional repository name for authentication scope.

        Returns:
            True if authentication successful, False otherwise
        """
        pass

    def get_manifest_and_digest(
        self, repository: str, reference: str, accept: Optional[str] = None
    ) -> Tuple[Optional[Dict], Optional[str]]:
        """Get the manifest and digest for a specific image reference.

        Default implementation that handles common retry logic for public containers.
        Subclasses can override if they need custom behavior.

        Args:
            repository: The repository name
            reference: The tag or digest

        Returns:
            Tuple of (manifest dictionary, digest string).
            Returns (None, None) if failed.
            Digest is extracted from Docker-Content-Digest header if available,
            otherwise computed from manifest JSON.
        """
        try:
            # Build URL - subclasses should set self.registry_url
            url = f"https://{self.registry_url}/v2/{repository}/manifests/{reference}"
            logger.debug("Fetching manifest", url=url)

            accept_header = (
                accept
                or self.session.headers.get("Accept")
                or _DOCKER_MANIFEST_MEDIA_TYPE
            )
            response = self.session.get(url, headers={"Accept": accept_header})

            if response.status_code == 200:
                manifest = response.json()
                headers_dict = dict(response.headers)

                # Extract digest from Docker-Content-Digest header
                digest = None
                for header_name, header_value in headers_dict.items():
                    if header_name.lower() == "docker-content-digest":
                        digest = header_value
                        break

                # Fallback: compute digest from manifest JSON
                if not digest:
                    manifest_json = json.dumps(
                        manifest, sort_keys=True, separators=(",", ":")
                    )
                    digest = f"sha256:{hashlib.sha256(manifest_json.encode('utf-8')).hexdigest()}"
                    logger.debug(
                        "Computed digest from manifest JSON",
                        repository=repository,
                        reference=reference,
                    )
                else:
                    logger.debug(
                        "Using Docker-Content-Digest header from registry",
                        repository=repository,
                        reference=reference,
                    )

                logger.debug(
                    "Successfully retrieved manifest",
                    schema_version=manifest.get("schemaVersion", "unknown"),
                    media_type=manifest.get("mediaType", "unknown"),
                    layers_count=len(manifest.get("layers", [])),
                    digest=digest,
                )
                return manifest, digest

            # Retry without authentication for public containers
            if response.status_code in (401, 403):
                logger.debug(
                    "Got 401/403, retrying without authentication",
                    status_code=response.status_code,
                )
                retry_response = _retry_without_auth(url, accept=accept_header)
                if retry_response:
                    manifest = retry_response.json()
                    headers_dict = dict(retry_response.headers)

                    # Extract digest from Docker-Content-Digest header
                    digest = None
                    for header_name, header_value in headers_dict.items():
                        if header_name.lower() == "docker-content-digest":
                            digest = header_value
                            break

                    # Fallback: compute digest from manifest JSON
                    if not digest:
                        manifest_json = json.dumps(
                            manifest, sort_keys=True, separators=(",", ":")
                        )
                        digest = f"sha256:{hashlib.sha256(manifest_json.encode('utf-8')).hexdigest()}"

                    return manifest, digest
                logger.error("Failed to get manifest", status_code=response.status_code)
                return None, None

            logger.error(
                "Failed to get manifest",
                status_code=response.status_code,
                response_preview=response.text[:200],
            )
            return None, None

        except Exception as e:
            logger.error("Error fetching manifest", error=str(e), exc_info=True)
            return None, None

    def get_blob(self, repository: str, digest: str) -> Optional[bytes]:
        """Download a blob (layer) by its digest.

        Default implementation that handles common retry logic for public containers.
        Subclasses can override if they need custom behavior.

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

            # Retry without authentication for public containers
            if response.status_code in (401, 403):
                logger.debug(
                    "Got 401/403, retrying without authentication",
                    status_code=response.status_code,
                )
                retry_response = _retry_without_auth(url, stream=True)
                if retry_response:
                    return retry_response.content
                logger.error(
                    "Failed to download blob", status_code=response.status_code
                )
                return None

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


class GitlabDockerRegistryHandler(DockerRegistryHandler):
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
            repository: Optional repository name for JWT scope.
        """
        self.registry_url = registry_url.rstrip("/")
        self.username = username
        self.password = password
        self.repository = repository
        self.bearer_token: Optional[str] = None
        self.session = requests.Session()

    def _check_public_access(self) -> bool:
        """Check if registry allows public access without authentication.

        Returns:
            True if public access is available, False otherwise
        """
        v2_url = f"https://{self.registry_url}/v2/"
        logger.debug("Checking for public registry access", url=v2_url)
        response = self.session.get(v2_url)

        if response.status_code == 200:
            logger.debug("Registry is public, no authentication needed")
            self.session.headers.update({"Accept": _DOCKER_MANIFEST_MEDIA_TYPE})
            return True
        return False

    def _request_gitlab_token(self, repository: str) -> Optional[str]:
        """Request Bearer token from GitLab JWT endpoint.

        Args:
            repository: Repository name for JWT scope

        Returns:
            Bearer token string or None if failed
        """
        gitlab_host = self.registry_url.split(":")[0]
        jwt_url = (
            f"https://{gitlab_host}/jwt/auth?"
            f"service=container_registry&scope=repository:{repository}:pull"
        )

        if self.username and self.password:
            logger.debug("Requesting Bearer token with credentials", jwt_url=jwt_url)
            token_response = self.session.get(
                jwt_url, auth=(self.username, self.password)
            )
        else:
            logger.debug(
                "Requesting anonymous token for public container", jwt_url=jwt_url
            )
            token_response = self.session.get(jwt_url)

        if token_response.status_code != 200:
            logger.error(
                "Token request failed",
                status_code=token_response.status_code,
                has_credentials=bool(self.username and self.password),
            )
            return None

        token_data = token_response.json()
        bearer_token = token_data.get("token")
        if not bearer_token:
            logger.error("No token in response", response_keys=list(token_data.keys()))
            return None

        return bearer_token

    def authenticate(self, repository: Optional[str] = None) -> bool:
        """Authenticate with GitLab registry using Bearer Token flow.

        Supports both authenticated and anonymous token requests for public containers.

        Args:
            repository: Optional repository name for JWT scope.

        Returns:
            True if authentication successful, False otherwise
        """
        try:
            repo_for_scope = repository or self.repository
            logger.debug(
                "Authenticating with GitLab registry",
                registry_url=self.registry_url,
                repository=repo_for_scope,
                has_credentials=bool(self.username and self.password),
            )

            # Check for public access first
            if self._check_public_access():
                return True

            # Handle unexpected responses
            v2_url = f"https://{self.registry_url}/v2/"
            response = self.session.get(v2_url)
            if response.status_code not in (200, 401):
                logger.error(
                    "Unexpected response from registry",
                    status_code=response.status_code,
                    response_preview=response.text[:200],
                )
                if not (self.username and self.password):
                    logger.debug(
                        "No credentials available, attempting to proceed without authentication"
                    )
                    self.session.headers.update({"Accept": _DOCKER_MANIFEST_MEDIA_TYPE})
                    return True
                return False

            # Request token
            if not repo_for_scope:
                logger.error(
                    "Repository name required for GitLab authentication",
                    registry_url=self.registry_url,
                )
                return False

            self.bearer_token = self._request_gitlab_token(repo_for_scope)
            if not self.bearer_token:
                return False

            # Set up session with bearer token
            self.session.headers.update(
                {
                    "Authorization": f"Bearer {self.bearer_token}",
                    "Accept": _DOCKER_MANIFEST_MEDIA_TYPE,
                }
            )

            return True

        except Exception as e:
            logger.error("Authentication error", error=str(e))
            return False

    # get_manifest_and_digest and get_blob are inherited from base class


class NvcrDockerRegistryHandler(DockerRegistryHandler):
    """NVIDIA Container Registry (nvcr.io) implementation using Docker Registry API v2."""

    def __init__(
        self,
        registry_url: str,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ):
        """Initialize the authenticator.

        Args:
            registry_url: The registry URL (e.g., 'nvcr.io')
            username: Registry username. Optional for anonymous access to public containers.
            password: Registry password or API key. Optional for anonymous access to public containers.
        """
        self.registry_url = registry_url.rstrip("/")
        self.username = username
        self.password = password
        self.bearer_token: Optional[str] = None
        self.session = requests.Session()

    def _parse_www_authenticate(self, www_authenticate: str) -> Optional[dict]:
        """Parse WWW-Authenticate header to extract auth parameters.

        Args:
            www_authenticate: WWW-Authenticate header value

        Returns:
            Dictionary with realm, service, scope or None if parsing fails
        """
        auth_params = {}
        for part in www_authenticate.replace("Bearer ", "").split(","):
            if "=" in part:
                key, value = part.split("=", 1)
                auth_params[key.strip()] = value.strip('"')

        realm = auth_params.get("realm")
        if not realm:
            logger.error("No realm in WWW-Authenticate header")
            return None

        return {
            "realm": realm,
            "service": auth_params.get("service", ""),
            "scope": auth_params.get("scope", ""),
        }

    def _request_token(
        self, auth_params: dict, repository: Optional[str]
    ) -> Optional[str]:
        """Request Bearer token from authentication realm.

        Args:
            auth_params: Authentication parameters (realm, service, scope)
            repository: Optional repository name for scope

        Returns:
            Bearer token string or None if failed
        """
        token_url = auth_params["realm"]
        params = {"service": auth_params["service"]}
        if auth_params["scope"]:
            params["scope"] = auth_params["scope"]
        elif repository:
            params["scope"] = f"repository:{repository}:pull"

        if self.username and self.password:
            token_response = self.session.get(
                token_url, params=params, auth=(self.username, self.password)
            )
        else:
            logger.debug("Requesting anonymous token for public container")
            token_response = self.session.get(token_url, params=params)

        if token_response.status_code != 200:
            logger.error(
                "Token request failed",
                status_code=token_response.status_code,
                has_credentials=bool(self.username and self.password),
            )
            return None

        token_data = token_response.json()
        bearer_token = token_data.get("token") or token_data.get("access_token")
        if not bearer_token:
            logger.error("No token in response")
            return None

        return bearer_token

    def authenticate(self, repository: Optional[str] = None) -> bool:
        """Authenticate with nvcr.io using Docker Registry API v2 Bearer token flow.

        Supports both authenticated and anonymous token requests for public containers.

        Args:
            repository: Optional repository name for authentication scope.

        Returns:
            True if authentication successful, False otherwise
        """
        try:
            v2_url = f"https://{self.registry_url}/v2/"
            response = self.session.get(v2_url)

            if response.status_code == 200:
                return True

            if response.status_code != 401:
                logger.error(
                    "Unexpected response from registry",
                    status_code=response.status_code,
                )
                return False

            www_authenticate = response.headers.get("WWW-Authenticate", "")
            if not www_authenticate:
                logger.error("No WWW-Authenticate header in 401 response")
                return False

            auth_params = self._parse_www_authenticate(www_authenticate)
            if not auth_params:
                return False

            self.bearer_token = self._request_token(auth_params, repository)
            if not self.bearer_token:
                return False

            self.session.headers.update(
                {
                    "Authorization": f"Bearer {self.bearer_token}",
                    "Accept": _DOCKER_MANIFEST_MEDIA_TYPE,
                }
            )

            return True

        except Exception as e:
            logger.error("Authentication error", error=str(e))
            return False

    # get_manifest_and_digest and get_blob are inherited from base class


def _resolve_gitlab_credentials(
    registry_url: str,
) -> tuple[Optional[str], Optional[str]]:
    """Resolve GitLab credentials from environment variables and Docker config.

    Args:
        registry_url: Registry URL

    Returns:
        Tuple of (username, password)
    """
    username = os.getenv("DOCKER_USERNAME")
    password = os.getenv("GITLAB_TOKEN")

    # If password from env but no username, try Docker config for username
    if password and not username:
        docker_creds = _read_docker_credentials(registry_url)
        if docker_creds:
            username, _ = docker_creds
        else:
            username = "gitlab-ci-token"

    # If no password from env, try Docker config
    if not password:
        docker_creds = _read_docker_credentials(registry_url)
        if docker_creds:
            username, password = docker_creds

    return username, password


def _resolve_nvcr_credentials(registry_url: str) -> tuple[Optional[str], Optional[str]]:
    """Resolve NVCR credentials from environment variables and Docker config.

    Args:
        registry_url: Registry URL

    Returns:
        Tuple of (username, password)
    """
    username = os.getenv("NVCR_USERNAME") or os.getenv("DOCKER_USERNAME", "$oauthtoken")
    password = os.getenv("NVCR_PASSWORD") or os.getenv("NVCR_API_KEY")

    # If no password from env, try Docker config
    if not password:
        docker_creds = _read_docker_credentials(registry_url)
        if docker_creds:
            username, password = docker_creds

    # Allow None credentials for anonymous access
    if not password:
        username = None
        password = None

    return username, password


def create_authenticator(
    registry_type: str, registry_url: str, repository: Optional[str] = None
) -> DockerRegistryHandler:
    """Create the appropriate authenticator based on registry type.

    Unified authenticator creation that supports:
    1. Public containers (anonymous token access)
    2. Environment variable credentials
    3. Docker config file credentials
    4. No credentials (falls back to anonymous access)

    Args:
        registry_type: Type of registry ('gitlab' or 'nvcr')
        registry_url: Registry URL
        repository: Optional repository name (required for GitLab)

    Returns:
        Registry authenticator instance
    """
    if registry_type == "gitlab":
        username, password = _resolve_gitlab_credentials(registry_url)
        logger.debug(
            "Creating GitLab authenticator",
            registry_url=registry_url,
            repository=repository,
            has_credentials=bool(username and password),
        )
        return GitlabDockerRegistryHandler(
            registry_url=registry_url,
            username=username,
            password=password,
            repository=repository,
        )

    elif registry_type == "nvcr":
        username, password = _resolve_nvcr_credentials(registry_url)
        logger.debug(
            "Creating NVCR authenticator",
            registry_url=registry_url,
            has_credentials=bool(username and password),
        )
        return NvcrDockerRegistryHandler(
            registry_url=registry_url,
            username=username,
            password=password,
        )

    else:
        raise ValueError(f"Unknown registry type: {registry_type}")
