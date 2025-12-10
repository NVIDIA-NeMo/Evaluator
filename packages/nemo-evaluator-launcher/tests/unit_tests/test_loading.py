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
"""Tests for container loading and framework.yml extraction."""

import io
import tarfile
import tempfile
from unittest.mock import Mock, patch

import pytest

from nemo_evaluator_launcher.common.container_metadata.intermediate_repr import (
    HarnessIntermediateRepresentation,
    TaskIntermediateRepresentation,
)
from nemo_evaluator_launcher.common.container_metadata.loading import (
    LayerInspector,
    _create_task_irs,
    _extract_task_description,
    extract_framework_yml,
    find_file_matching_pattern_in_image_layers,
    get_container_digest,
    parse_framework_to_irs,
    read_from_cache,
    write_to_cache,
)
from nemo_evaluator_launcher.common.container_metadata.registries import (
    RegistryAuthenticator,
)


class TestLayerInspector:
    """Test layer inspection utilities."""

    def test_extract_file_from_layer_not_found(self):
        """Test extracting file that doesn't exist."""
        # Create empty tar.gz
        layer_content = self._create_tar_gz({})
        inspector = LayerInspector()
        result = inspector.extract_file_from_layer(
            layer_content, "/opt/metadata/framework.yml"
        )
        assert result is None

    def test_extract_file_from_layer_found(self):
        """Test extracting file that exists."""
        content = "test: content"
        layer_content = self._create_tar_gz({"/opt/metadata/framework.yml": content})
        inspector = LayerInspector()
        result = inspector.extract_file_from_layer(
            layer_content, "/opt/metadata/framework.yml"
        )
        assert result == content

    def test_extract_file_matching_pattern_found(self):
        """Test extracting file matching pattern."""
        content = "framework: test"
        layer_content = self._create_tar_gz(
            {"/opt/metadata/subdir/framework.yml": content}
        )
        inspector = LayerInspector()
        result = inspector.extract_file_matching_pattern(
            layer_content, "/opt/metadata", "framework.yml"
        )
        assert result is not None
        file_path, file_content = result
        assert file_content == content
        assert "framework.yml" in file_path

    def test_extract_file_matching_pattern_not_found(self):
        """Test pattern matching when file doesn't exist."""
        layer_content = self._create_tar_gz({"/other/path/file.txt": "content"})
        inspector = LayerInspector()
        result = inspector.extract_file_matching_pattern(
            layer_content, "/opt/metadata", "framework.yml"
        )
        assert result is None

    @staticmethod
    def _create_tar_gz(files: dict[str, str]) -> bytes:
        """Create a tar.gz archive with given files."""
        with tempfile.NamedTemporaryFile() as tmp:
            with tarfile.open(tmp.name, "w:gz") as tar:
                for filepath, content in files.items():
                    info = tarfile.TarInfo(name=filepath)
                    info.size = len(content.encode())
                    tar.addfile(info, io.BytesIO(content.encode()))
            tmp.seek(0)
            return tmp.read()


class TestCacheFunctions:
    """Test cache management functions."""

    def test_write_and_read_cache(self, tmp_path, monkeypatch):
        """Test writing and reading from cache."""
        # Mock cache directory
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        monkeypatch.setattr(
            "nemo_evaluator_launcher.common.container_metadata.loading.CACHE_DIR",
            cache_dir,
        )

        docker_id = "test:latest"
        target_file = "/opt/metadata/framework.yml"
        metadata = "test content"
        digest = "sha256:abc123"

        write_to_cache(docker_id, target_file, metadata, digest)
        result, stored_digest = read_from_cache(docker_id, target_file, digest)

        assert result == metadata
        assert stored_digest == digest

    def test_read_cache_mismatched_digest(self, tmp_path, monkeypatch):
        """Test reading cache with mismatched digest."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        monkeypatch.setattr(
            "nemo_evaluator_launcher.common.container_metadata.loading.CACHE_DIR",
            cache_dir,
        )

        docker_id = "test:latest"
        target_file = "/opt/metadata/framework.yml"
        metadata = "test content"
        digest = "sha256:abc123"

        write_to_cache(docker_id, target_file, metadata, digest)
        result, stored_digest = read_from_cache(
            docker_id, target_file, "sha256:different"
        )

        assert result is None
        assert stored_digest == digest


class TestExtractTaskDescription:
    """Test task description extraction."""

    def test_extract_task_description_found(self):
        """Test extracting task description when found."""
        framework_data = {
            "evaluations": [
                {
                    "defaults": {"config": {"type": "task1"}},
                    "description": "Task 1 description",
                },
                {
                    "defaults": {"config": {"type": "task2"}},
                    "description": "Task 2 description",
                },
            ]
        }
        result = _extract_task_description(framework_data, "task1")
        assert result == "Task 1 description"

    def test_extract_task_description_not_found(self):
        """Test extracting task description when not found."""
        framework_data = {"evaluations": []}
        result = _extract_task_description(framework_data, "task1")
        assert result == ""


class TestCreateTaskIrs:
    """Test task IR creation."""

    @patch(
        "nemo_evaluator_launcher.common.container_metadata.loading._extract_task_description"
    )
    def test_create_task_irs(self, mock_extract):
        """Test creating task IRs from evaluations."""
        mock_extract.return_value = "Test description"

        evaluations = {
            "task1": Mock(model_dump=Mock(return_value={"type": "task1"})),
            "task2": Mock(model_dump=Mock(return_value={"type": "task2"})),
        }
        framework_data = {}
        harness_name = "test-harness"
        container_id = "test:latest"
        container_digest = "sha256:abc123"

        task_irs = _create_task_irs(
            evaluations, framework_data, harness_name, container_id, container_digest
        )

        assert len(task_irs) == 2
        assert all(isinstance(ir, TaskIntermediateRepresentation) for ir in task_irs)
        assert all(ir.harness == harness_name for ir in task_irs)
        assert all(ir.container == container_id for ir in task_irs)


class TestParseFrameworkToIrs:
    """Test parsing framework.yml to IRs."""

    @patch(
        "nemo_evaluator_launcher.common.container_metadata.loading.get_framework_evaluations"
    )
    def test_parse_framework_to_irs_success(self, mock_get_evaluations):
        """Test successful parsing of framework.yml."""
        framework_content = """
framework:
  name: test-harness
  description: Test harness description
evaluations:
  - defaults:
      config:
        type: task1
    description: Task 1 description
"""
        mock_get_evaluations.return_value = (
            "test-harness",
            {},
            {"task1": Mock(model_dump=Mock(return_value={"type": "task1"}))},
        )

        harness_ir, task_irs = parse_framework_to_irs(
            framework_content, "test:latest", "sha256:abc123"
        )

        assert isinstance(harness_ir, HarnessIntermediateRepresentation)
        assert harness_ir.name == "test-harness"
        assert len(task_irs) > 0

    def test_parse_framework_to_irs_missing_name(self):
        """Test parsing framework.yml without framework.name."""
        framework_content = """
framework:
  description: Test harness description
"""
        with pytest.raises(ValueError, match="framework.name"):
            parse_framework_to_irs(framework_content, "test:latest", None)

    def test_parse_framework_to_irs_empty(self):
        """Test parsing empty framework.yml."""
        with pytest.raises(ValueError, match="Empty"):
            parse_framework_to_irs("", "test:latest", None)


class TestGetContainerDigest:
    """Test container digest retrieval."""

    def test_get_container_digest_success(self):
        """Test getting container digest successfully."""
        mock_authenticator = Mock(spec=RegistryAuthenticator)
        mock_authenticator.get_manifest_and_digest.return_value = (
            {"schemaVersion": 2},
            "sha256:abc123",
        )

        result = get_container_digest(mock_authenticator, "test/repo", "latest")
        assert result == "sha256:abc123"

    def test_get_container_digest_failure(self):
        """Test getting container digest when it fails."""
        mock_authenticator = Mock(spec=RegistryAuthenticator)
        mock_authenticator.get_manifest_and_digest.side_effect = Exception(
            "Network error"
        )

        result = get_container_digest(mock_authenticator, "test/repo", "latest")
        assert result is None


class TestFindFileMatchingPatternInImageLayers:
    """Test finding files matching pattern in image layers."""

    @staticmethod
    def _create_tar_gz(files: dict[str, str]) -> bytes:
        """Create a tar.gz archive with given files."""
        with tempfile.NamedTemporaryFile() as tmp:
            with tarfile.open(tmp.name, "w:gz") as tar:
                for filepath, content in files.items():
                    info = tarfile.TarInfo(name=filepath)
                    info.size = len(content.encode())
                    tar.addfile(info, io.BytesIO(content.encode()))
            tmp.seek(0)
            return tmp.read()

    def test_find_file_matching_pattern_cache_hit(self, tmp_path, monkeypatch):
        """Test finding file with cache hit."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        monkeypatch.setattr(
            "nemo_evaluator_launcher.common.container_metadata.loading.CACHE_DIR",
            cache_dir,
        )

        mock_authenticator = Mock(spec=RegistryAuthenticator)
        mock_authenticator.get_manifest_and_digest.return_value = (
            {"layers": []},
            "sha256:abc123",
        )

        # Pre-populate cache
        write_to_cache(
            "test:latest",
            "/opt/metadata/framework.yml",
            "cached content",
            "sha256:abc123",
            cached_file_path="/opt/metadata/framework.yml",
        )

        result = find_file_matching_pattern_in_image_layers(
            mock_authenticator,
            "test/repo",
            "latest",
            "/opt/metadata",
            "framework.yml",
            docker_id="test:latest",
            use_cache=True,
        )

        assert result is not None
        file_path, content = result
        assert content == "cached content"

    def test_find_file_matching_pattern_found_in_layer(self):
        """Test finding file in layer."""
        mock_authenticator = Mock(spec=RegistryAuthenticator)
        mock_authenticator.get_manifest_and_digest.return_value = (
            {
                "layers": [
                    {
                        "digest": "sha256:layer1",
                        "size": 1000,
                        "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
                    }
                ]
            },
            "sha256:abc123",
        )

        # Create layer content with framework.yml
        layer_content = self._create_tar_gz(
            {"/opt/metadata/framework.yml": "framework: test"}
        )
        mock_authenticator.get_blob.return_value = layer_content

        result = find_file_matching_pattern_in_image_layers(
            mock_authenticator,
            "test/repo",
            "latest",
            "/opt/metadata",
            "framework.yml",
            docker_id="test:latest",
            use_cache=False,
        )

        assert result is not None
        file_path, content = result
        assert "framework.yml" in file_path
        assert content == "framework: test"

    def test_find_file_matching_pattern_not_found(self):
        """Test when file is not found in any layer."""
        mock_authenticator = Mock(spec=RegistryAuthenticator)
        mock_authenticator.get_manifest_and_digest.return_value = (
            {
                "layers": [
                    {
                        "digest": "sha256:layer1",
                        "size": 1000,
                        "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
                    }
                ]
            },
            "sha256:abc123",
        )

        # Create layer content without framework.yml
        layer_content = self._create_tar_gz({"/other/file.txt": "content"})
        mock_authenticator.get_blob.return_value = layer_content

        result = find_file_matching_pattern_in_image_layers(
            mock_authenticator,
            "test/repo",
            "latest",
            "/opt/metadata",
            "framework.yml",
            docker_id="test:latest",
            use_cache=False,
        )

        assert result is None


class TestExtractFrameworkYml:
    """Test framework.yml extraction from containers."""

    @patch(
        "nemo_evaluator_launcher.common.container_metadata.loading.create_authenticator"
    )
    @patch(
        "nemo_evaluator_launcher.common.container_metadata.loading.find_file_matching_pattern_in_image_layers"
    )
    @patch(
        "nemo_evaluator_launcher.common.container_metadata.loading.get_container_digest"
    )
    @patch(
        "nemo_evaluator_launcher.common.container_metadata.loading.parse_container_image"
    )
    def test_extract_framework_yml_success(
        self, mock_parse, mock_get_digest, mock_find_file, mock_create_auth
    ):
        """Test successful framework.yml extraction."""
        mock_parse.return_value = ("nvcr", "nvcr.io", "test/repo", "latest")
        mock_authenticator = Mock()
        mock_create_auth.return_value = mock_authenticator
        mock_get_digest.return_value = "sha256:abc123"
        mock_find_file.return_value = (
            "/opt/metadata/framework.yml",
            "framework:\n  name: test",
        )

        content, digest = extract_framework_yml("nvcr.io/test/repo:latest")

        assert content == "framework:\n  name: test"
        assert digest == "sha256:abc123"

    @patch(
        "nemo_evaluator_launcher.common.container_metadata.loading.create_authenticator"
    )
    @patch(
        "nemo_evaluator_launcher.common.container_metadata.loading.find_file_matching_pattern_in_image_layers"
    )
    def test_extract_framework_yml_not_found(self, mock_find_file, mock_create_auth):
        """Test when framework.yml is not found."""
        mock_authenticator = Mock()
        mock_create_auth.return_value = mock_authenticator
        mock_find_file.return_value = None

        content, digest = extract_framework_yml("test:latest")

        assert content is None
        assert digest is None
