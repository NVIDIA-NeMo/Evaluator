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
"""Tests for registry authentication and credential management."""

import base64
import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from nemo_evaluator_launcher.common.container_metadata.registries import (
    GitlabDockerRegistryHandler,
    NvcrDockerRegistryHandler,
    _build_key_variants,
    _decode_auth_string,
    _find_auth_in_config,
    _read_docker_credentials,
    _resolve_gitlab_credentials,
    _resolve_nvcr_credentials,
    create_authenticator,
)


class TestBuildKeyVariants:
    """Test key variant building for Docker credential lookup."""

    def test_build_key_variants_basic(self):
        """Test basic key variant building."""
        variants = _build_key_variants("registry.example.com")
        assert "registry.example.com" in variants
        assert "https://registry.example.com" in variants

    def test_build_key_variants_gitlab(self):
        """Test GitLab-specific key variants."""
        variants = _build_key_variants("gitlab.example.com:5005")
        assert "gitlab.example.com:5005" in variants
        assert "gitlab.example.com:5005" in variants
        assert "gitlab.example.com:5005" in variants
        assert any("5005" in v for v in variants)


class TestDecodeAuthString:
    """Test base64 auth string decoding."""

    def test_decode_auth_string_valid(self):
        """Test decoding valid auth string."""
        username = "testuser"
        password = "testpass"
        auth_string = base64.b64encode(f"{username}:{password}".encode()).decode()
        result = _decode_auth_string(auth_string, "test-registry")
        assert result is not None
        assert result[0] == username
        assert result[1] == password

    def test_decode_auth_string_invalid_format(self):
        """Test decoding invalid auth format."""
        auth_string = base64.b64encode(b"no-colon").decode()
        result = _decode_auth_string(auth_string, "test-registry")
        assert result is None


class TestFindAuthInConfig:
    """Test finding auth in Docker config."""

    def test_find_auth_exact_match(self):
        """Test exact match lookup."""
        auths = {
            "registry.example.com": {"auth": "dGVzdHVzZXI6dGVzdHBhc3M="},
            "other-registry.com": {"auth": "b3RoZXI6cGFzcw=="},
        }
        auth_dict, matched_key = _find_auth_in_config(auths, "registry.example.com")
        assert auth_dict is not None
        assert matched_key == "registry.example.com"

    def test_find_auth_hostname_match(self):
        """Test hostname-based match."""
        auths = {
            "https://registry.example.com:5000": {"auth": "dGVzdHVzZXI6dGVzdHBhc3M="},
        }
        auth_dict, matched_key = _find_auth_in_config(auths, "registry.example.com")
        assert auth_dict is not None


class TestReadDockerCredentials:
    """Test reading Docker credentials from config file."""

    def test_read_docker_credentials_not_found(self, tmp_path, monkeypatch):
        """Test when Docker config file doesn't exist."""
        # Point DOCKER_CONFIG at an empty temp dir (no config.json).
        monkeypatch.setenv("DOCKER_CONFIG", str(tmp_path))
        result = _read_docker_credentials("test-registry")
        assert result is None

    def test_read_docker_credentials_found(self, tmp_path, monkeypatch):
        """Test reading credentials from Docker config."""
        # Docker looks for ${DOCKER_CONFIG}/config.json when DOCKER_CONFIG is set.
        monkeypatch.setenv("DOCKER_CONFIG", str(tmp_path))
        config_file = tmp_path / "config.json"
        username = "testuser"
        password = "testpass"
        auth_string = base64.b64encode(f"{username}:{password}".encode()).decode()

        config_data = {
            "auths": {
                "test-registry": {"auth": auth_string},
            }
        }
        config_file.write_text(json.dumps(config_data))

        result = _read_docker_credentials("test-registry")
        assert result is not None
        assert result[0] == username
        assert result[1] == password

    @patch(
        "nemo_evaluator_launcher.common.container_metadata.registries.subprocess.run"
    )
    def test_read_docker_credentials_creds_store(self, mock_run, tmp_path, monkeypatch):
        """Test reading credentials via credsStore helper when auths entry is empty."""
        monkeypatch.setenv("DOCKER_CONFIG", str(tmp_path))
        config_file = tmp_path / "config.json"
        config_data = {
            "auths": {
                "gitlab-master.nvidia.com": {},
                "gitlab-master.nvidia.com:5005": {},
            },
            "credsStore": "osxkeychain",
        }
        config_file.write_text(json.dumps(config_data))

        mock_run.return_value = SimpleNamespace(
            returncode=0,
            stdout='{"Username":"u","Secret":"s"}',
            stderr="",
        )

        result = _read_docker_credentials("gitlab-master.nvidia.com:5005")
        assert result == ("u", "s")

        assert mock_run.call_count == 1
        args, kwargs = mock_run.call_args
        assert args[0] == ["docker-credential-osxkeychain", "get"]
        assert kwargs["text"] is True
        assert kwargs["capture_output"] is True
        assert kwargs["check"] is False
        # First tried server should be the exact matched key.
        assert kwargs["input"] == "gitlab-master.nvidia.com:5005\n"

    @patch(
        "nemo_evaluator_launcher.common.container_metadata.registries.subprocess.run"
    )
    def test_read_docker_credentials_creds_store_helper_missing(
        self, mock_run, tmp_path, monkeypatch
    ):
        """Test credsStore helper missing returns None."""
        monkeypatch.setenv("DOCKER_CONFIG", str(tmp_path))
        config_file = tmp_path / "config.json"
        config_data = {
            "auths": {"gitlab-master.nvidia.com:5005": {}},
            "credsStore": "osxkeychain",
        }
        config_file.write_text(json.dumps(config_data))

        mock_run.side_effect = FileNotFoundError()

        result = _read_docker_credentials("gitlab-master.nvidia.com:5005")
        assert result is None


class TestResolveCredentials:
    """Test credential resolution from environment and Docker config."""

    @patch.dict("os.environ", {}, clear=True)
    @patch(
        "nemo_evaluator_launcher.common.container_metadata.registries._read_docker_credentials"
    )
    def test_resolve_gitlab_credentials_from_env(self, mock_read):
        """Test GitLab credentials from environment."""
        with patch.dict(
            "os.environ", {"DOCKER_USERNAME": "user", "GITLAB_TOKEN": "token"}
        ):
            username, password = _resolve_gitlab_credentials("gitlab.example.com")
            assert username == "user"
            assert password == "token"

    @patch.dict("os.environ", {}, clear=True)
    @patch(
        "nemo_evaluator_launcher.common.container_metadata.registries._read_docker_credentials"
    )
    def test_resolve_nvcr_credentials_from_env(self, mock_read):
        """Test NVCR credentials from environment."""
        with patch.dict(
            "os.environ", {"NVCR_USERNAME": "user", "NVCR_PASSWORD": "pass"}
        ):
            username, password = _resolve_nvcr_credentials("nvcr.io")
            assert username == "user"
            assert password == "pass"

    @patch.dict("os.environ", {}, clear=True)
    @patch(
        "nemo_evaluator_launcher.common.container_metadata.registries._read_docker_credentials"
    )
    def test_resolve_nvcr_credentials_anonymous(self, mock_read):
        """Test NVCR credentials allow anonymous access."""
        mock_read.return_value = None
        username, password = _resolve_nvcr_credentials("nvcr.io")
        assert username is None
        assert password is None


class TestGitlabDockerRegistryHandler:
    """Test GitLab registry authenticator."""

    def test_init(self):
        """Test authenticator initialization."""
        auth = GitlabDockerRegistryHandler(
            "gitlab.example.com:5005", username="user", password="pass"
        )
        assert auth.registry_url == "gitlab.example.com:5005"
        assert auth.username == "user"
        assert auth.password == "pass"

    @patch("requests.Session")
    def test_authenticate_public_access(self, mock_session_class):
        """Test authentication with public access."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        mock_response = Mock()
        mock_response.status_code = 200
        mock_session.get.return_value = mock_response

        auth = GitlabDockerRegistryHandler("gitlab.example.com:5005")
        result = auth.authenticate(repository="test/repo")

        assert result is True
        assert mock_session.headers.get("Accept") is not None

    @patch("requests.Session")
    def test_get_manifest_and_digest_success(self, mock_session_class):
        """Test getting manifest and digest successfully."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session

        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "schemaVersion": 2,
            "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
            "layers": [],
        }
        mock_response.headers = {"Docker-Content-Digest": "sha256:abc123"}
        mock_session.get.return_value = mock_response

        auth = GitlabDockerRegistryHandler("gitlab.example.com:5005")
        auth.session = mock_session
        manifest, digest = auth.get_manifest_and_digest("test/repo", "latest")

        assert manifest is not None
        assert digest == "sha256:abc123"

    @patch("requests.Session")
    def test_get_manifest_and_digest_computed(self, mock_session_class):
        """Test computing digest when header not available."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session

        # Mock response without Docker-Content-Digest header
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "schemaVersion": 2,
            "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
            "layers": [],
        }
        mock_response.headers = {}
        mock_session.get.return_value = mock_response

        auth = GitlabDockerRegistryHandler("gitlab.example.com:5005")
        auth.session = mock_session
        manifest, digest = auth.get_manifest_and_digest("test/repo", "latest")

        assert manifest is not None
        assert digest is not None
        assert digest.startswith("sha256:")


class TestNvcrDockerRegistryHandler:
    """Test NVCR registry authenticator."""

    def test_init(self):
        """Test authenticator initialization."""
        auth = NvcrDockerRegistryHandler("nvcr.io", username="user", password="pass")
        assert auth.registry_url == "nvcr.io"
        assert auth.username == "user"
        assert auth.password == "pass"

    @patch("requests.Session")
    def test_authenticate_public_access(self, mock_session_class):
        """Test authentication with public access."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        mock_response = Mock()
        mock_response.status_code = 200
        mock_session.get.return_value = mock_response

        auth = NvcrDockerRegistryHandler("nvcr.io")
        result = auth.authenticate()

        assert result is True

    @patch("requests.Session")
    def test_get_manifest_and_digest_success(self, mock_session_class):
        """Test getting manifest and digest successfully."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session

        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "schemaVersion": 2,
            "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
            "layers": [],
        }
        mock_response.headers = {"Docker-Content-Digest": "sha256:def456"}
        mock_session.get.return_value = mock_response

        auth = NvcrDockerRegistryHandler("nvcr.io")
        auth.session = mock_session
        manifest, digest = auth.get_manifest_and_digest("test/repo", "latest")

        assert manifest is not None
        assert digest == "sha256:def456"


class TestCreateAuthenticator:
    """Test authenticator factory function."""

    @patch(
        "nemo_evaluator_launcher.common.container_metadata.registries._resolve_gitlab_credentials_with_sources"
    )
    def test_create_authenticator_gitlab(self, mock_resolve):
        """Test creating GitLab authenticator."""
        mock_resolve.return_value = (
            "user",
            "pass",
            {"username_source": "env:GITLAB_TOKEN"},
        )
        auth = create_authenticator("gitlab", "gitlab.example.com:5005", "test/repo")
        assert isinstance(auth, GitlabDockerRegistryHandler)

    @patch(
        "nemo_evaluator_launcher.common.container_metadata.registries._resolve_nvcr_credentials"
    )
    def test_create_authenticator_nvcr(self, mock_resolve):
        """Test creating NVCR authenticator."""
        mock_resolve.return_value = ("user", "pass")
        auth = create_authenticator("nvcr", "nvcr.io")
        assert isinstance(auth, NvcrDockerRegistryHandler)

    def test_create_authenticator_unknown(self):
        """Test creating authenticator with unknown registry type."""
        with pytest.raises(ValueError, match="Unknown registry type"):
            create_authenticator("unknown", "registry.example.com")
