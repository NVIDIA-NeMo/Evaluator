# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the interceptor registry and metadata functionality."""

import tempfile
from pathlib import Path
from typing import Union
from unittest.mock import patch

from nvidia_eval_commons.adapters.registry import (
    InterceptorMetadata,
    InterceptorRegistry,
)
from nvidia_eval_commons.adapters.types import (
    AdapterGlobalContext,
    AdapterRequest,
    AdapterResponse,
    RequestInterceptor,
    ResponseInterceptor,
)


class MockRequestInterceptor(RequestInterceptor):
    """Mock request interceptor for testing."""

    def intercept_request(
        self, req: AdapterRequest, context: AdapterGlobalContext
    ) -> AdapterRequest:
        return req


class MockRequestResponseInterceptor(RequestInterceptor):
    """Mock request interceptor that can return both request and response."""

    def intercept_request(
        self, req: AdapterRequest, context: AdapterGlobalContext
    ) -> Union[AdapterRequest, AdapterResponse]:
        return req


class MockResponseInterceptor(ResponseInterceptor):
    """Mock response interceptor for testing."""

    def intercept_response(
        self, resp: AdapterResponse, context: AdapterGlobalContext
    ) -> AdapterResponse:
        return resp


class TestInterceptorMetadata:
    """Test the InterceptorMetadata class with stage validation."""

    def test_metadata_creation_with_valid_stage(self):
        """Test creating metadata with valid stage."""
        metadata = InterceptorMetadata(
            name="test_interceptor",
            description="Test interceptor",
            interceptor_class=MockRequestInterceptor,
        )
        assert metadata.name == "test_interceptor"

    def test_metadata_creation_without_stage(self):
        """Test creating metadata without stage (backward compatibility)."""
        metadata = InterceptorMetadata(
            name="test_interceptor",
            description="Test interceptor",
            interceptor_class=MockRequestInterceptor,
        )
        assert hasattr(metadata, "name")

    def test_supports_request_interception(self):
        """Test supports_request_interception method."""
        metadata = InterceptorMetadata(
            name="test_interceptor",
            description="Test interceptor",
            interceptor_class=MockRequestInterceptor,
        )
        assert metadata.supports_request_interception() is True
        assert metadata.supports_response_interception() is False

    def test_supports_response_interception(self):
        """Test supports_response_interception method."""
        metadata = InterceptorMetadata(
            name="test_interceptor",
            description="Test interceptor",
            interceptor_class=MockResponseInterceptor,
        )
        assert metadata.supports_request_interception() is False
        assert metadata.supports_response_interception() is True


class TestInterceptorRegistry:
    """Test the InterceptorRegistry with stage validation."""

    def setup_method(self):
        """Reset registry before each test."""
        registry = InterceptorRegistry.get_instance()
        registry.reset()

    def test_register(self):
        """Test registering interceptor with valid stage."""
        registry = InterceptorRegistry.get_instance()
        metadata = InterceptorMetadata(
            name="test_interceptor",
            description="Test interceptor",
            interceptor_class=MockRequestInterceptor,
        )
        registry.register("test_interceptor", MockRequestInterceptor, metadata)
        assert "test_interceptor" in registry.get_all_components()

    def test_register_duplicate(self):
        """Test registering interceptor with duplicate."""
        registry = InterceptorRegistry.get_instance()
        metadata = InterceptorMetadata(
            name="test_interceptor",
            description="Test interceptor",
            interceptor_class=MockRequestInterceptor,
        )
        registry.register("test_interceptor", MockRequestInterceptor, metadata)
        registry.register("test_interceptor", MockRequestInterceptor, metadata)
        assert "test_interceptor" in registry.get_all_components()

    def test_get_metadata(self):
        """Test getting metadata with stage information."""
        registry = InterceptorRegistry.get_instance()
        metadata = InterceptorMetadata(
            name="test_interceptor",
            description="Test interceptor",
            interceptor_class=MockRequestInterceptor,
        )
        registry.register("test_interceptor", MockRequestInterceptor, metadata)

        retrieved_metadata = registry.get_metadata("test_interceptor")
        assert retrieved_metadata is not None
        assert retrieved_metadata.name == "test_interceptor"

    def test_discover_components_with_modules(self):
        """Test discovering interceptors from modules."""
        registry = InterceptorRegistry.get_instance()

        with patch("importlib.import_module") as mock_import:
            registry.discover_components(modules=["test.module", "another.module"])

            assert (
                mock_import.call_count == 5
            )  # 3 Default modules + 2 specified modules
            mock_import.assert_any_call("nvidia_eval_commons.adapters.interceptors")
            mock_import.assert_any_call("nvidia_eval_commons.adapters.reports")
            mock_import.assert_any_call("test.module")
            mock_import.assert_any_call("another.module")
            # Note: internal module is imported but not asserted to avoid test dependencies

    def test_discover_components_with_dirs(self):
        """Test discovering interceptors from directories."""
        registry = InterceptorRegistry.get_instance()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            test_dir = temp_path / "test_plugins"
            test_dir.mkdir()

            # Create a test Python file with @register_for_adapter
            test_file = test_dir / "test_interceptor.py"
            test_file.write_text("@register_for_adapter\ndef test_func():\n    pass")

            with patch("importlib.import_module") as mock_import:
                registry.discover_components(dirs=[str(test_dir)])

                assert (
                    mock_import.call_count == 4
                )  # 3 Default modules + discovered module
                mock_import.assert_any_call("nvidia_eval_commons.adapters.interceptors")
                mock_import.assert_any_call("nvidia_eval_commons.adapters.reports")
                mock_import.assert_any_call("test_plugins.test_interceptor")
                # Note: internal module is imported but not asserted to avoid test dependencies

    def test_discover_components_with_both_modules_and_dirs(self):
        """Test discovering interceptors from both modules and directories."""
        registry = InterceptorRegistry.get_instance()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            test_dir = temp_path / "test_plugins"
            test_dir.mkdir()

            test_file = test_dir / "test_interceptor.py"
            test_file.write_text("@register_for_adapter\ndef test_func():\n    pass")

            with patch("importlib.import_module") as mock_import:
                registry.discover_components(
                    modules=["test.module"], dirs=[str(test_dir)]
                )

                assert (
                    mock_import.call_count == 5
                )  # 3 Default modules + specified module + discovered module
                mock_import.assert_any_call("nvidia_eval_commons.adapters.interceptors")
                mock_import.assert_any_call("nvidia_eval_commons.adapters.reports")
                mock_import.assert_any_call("test.module")
                mock_import.assert_any_call("test_plugins.test_interceptor")
                # Note: internal module is imported but not asserted to avoid test dependencies

    def test_discover_components_with_none_values(self):
        """Test discovering interceptors with None values."""
        registry = InterceptorRegistry.get_instance()

        # Should not raise any exceptions
        registry.discover_components(modules=None, dirs=None)
        registry.discover_components(modules=[], dirs=[])

    def test_discover_from_modules_with_import_error(self):
        """Test module discovery handles import errors gracefully."""
        registry = InterceptorRegistry.get_instance()

        with patch(
            "importlib.import_module", side_effect=ImportError("Module not found")
        ):
            # Should not raise exception, just log warning
            registry.discover_components(modules=["nonexistent.module"])

    def test_discover_from_directories_with_nonexistent_dir(self):
        """Test directory discovery handles nonexistent directories gracefully."""
        registry = InterceptorRegistry.get_instance()

        # Should not raise exception, just log warning
        registry.discover_components(dirs=["/nonexistent/directory"])

    def test_discover_from_directories_with_file_path(self):
        """Test directory discovery handles file paths gracefully."""
        registry = InterceptorRegistry.get_instance()

        with tempfile.NamedTemporaryFile(suffix=".py") as temp_file:
            # Should not raise exception, just log warning
            registry.discover_components(dirs=[temp_file.name])

    def test_should_process_file_with_valid_content(self):
        """Test _should_process_file with valid content."""
        registry = InterceptorRegistry.get_instance()

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as temp_file:
            temp_file.write("@register_for_adapter\ndef test_func():\n    pass")
            temp_file.flush()

            assert registry._should_process_file(Path(temp_file.name)) is True

            # Clean up
            Path(temp_file.name).unlink()

    def test_should_process_file_without_decorator(self):
        """Test _should_process_file without @register_for_adapter decorator."""
        registry = InterceptorRegistry.get_instance()

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as temp_file:
            temp_file.write("def test_func():\n    pass")
            temp_file.flush()

            assert registry._should_process_file(Path(temp_file.name)) is False

            # Clean up
            Path(temp_file.name).unlink()

    def test_should_process_file_with_read_error(self):
        """Test _should_process_file handles read errors gracefully."""
        registry = InterceptorRegistry.get_instance()

        # Create a directory with the same name as a file to cause read error
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            test_file = temp_path / "test.py"
            test_file.mkdir()  # Make it a directory instead of a file

            assert registry._should_process_file(test_file) is False
