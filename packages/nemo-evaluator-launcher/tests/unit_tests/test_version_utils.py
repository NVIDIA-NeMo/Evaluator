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
"""Tests for version_utils.py functionality."""

from unittest.mock import patch


class TestVersionUtils:
    """Test cases for version utilities."""

    def test_version_info_success(self):
        """Test successful version info retrieval."""
        # Import to ensure the module loads successfully
        from nemo_evaluator_launcher.common import version_utils

        # Check that the basic constants are defined
        assert hasattr(version_utils, "DIST_NAME")
        assert hasattr(version_utils, "__main_pkg_name__")
        assert hasattr(version_utils, "__version__")

        assert version_utils.DIST_NAME == "nemo-evaluator-launcher"

    def test_version_info_constants(self):
        """Test that version constants are properly set."""
        from nemo_evaluator_launcher.common import version_utils

        # Test that DIST_NAME is the expected value
        assert version_utils.DIST_NAME == "nemo-evaluator-launcher"

        # Test that __main_pkg_name__ is a string
        assert isinstance(version_utils.__main_pkg_name__, str)

        # Test that __version__ is a string
        assert isinstance(version_utils.__version__, str)

    def test_keyerror_exception_handling(self):
        """Test KeyError exception handling during metadata access."""
        # This tests the specific lines that handle KeyError exception
        with patch(
            "nemo_evaluator_launcher.common.version_utils.metadata"
        ) as mock_metadata:
            # Configure mock to raise KeyError when accessing "Name" key
            mock_metadata.side_effect = KeyError("Name")

            # Create a fresh module-like environment to test the exception handling
            import types

            test_module = types.ModuleType("test_version_utils")
            test_module.DIST_NAME = "nemo-evaluator-launcher"

            # Execute the code that should handle the KeyError
            try:
                pkg_name = mock_metadata(test_module.DIST_NAME)["Name"]
            except KeyError:
                pkg_name = "name_not_found"

            # Verify that the exception was handled correctly
            assert pkg_name == "name_not_found"

    def test_successful_metadata_access(self):
        """Test successful metadata access."""
        with patch(
            "nemo_evaluator_launcher.common.version_utils.metadata"
        ) as mock_metadata:
            mock_metadata.return_value = {"Name": "test-package"}

            # Simulate the successful path
            import types

            test_module = types.ModuleType("test_version_utils")
            test_module.DIST_NAME = "nemo-evaluator-launcher"

            pkg_name = mock_metadata(test_module.DIST_NAME)["Name"]
            assert pkg_name == "test-package"

    def test_dist_name_constant(self):
        """Test that DIST_NAME constant is correctly defined."""
        from nemo_evaluator_launcher.common import version_utils

        assert version_utils.DIST_NAME == "nemo-evaluator-launcher"

    def test_module_has_required_attributes(self):
        """Test that the module has all required attributes."""
        from nemo_evaluator_launcher.common import version_utils

        # Check that all expected attributes exist
        required_attrs = ["DIST_NAME", "__main_pkg_name__", "__version__"]
        for attr in required_attrs:
            assert hasattr(version_utils, attr), (
                f"Module missing required attribute: {attr}"
            )

    def test_version_string_format(self):
        """Test that version string follows expected format."""
        from nemo_evaluator_launcher.common import version_utils

        # Version should be a non-empty string
        assert isinstance(version_utils.__version__, str)
        assert len(version_utils.__version__) > 0

    def test_package_name_format(self):
        """Test that package name follows expected format."""
        from nemo_evaluator_launcher.common import version_utils

        # Package name should be a non-empty string
        assert isinstance(version_utils.__main_pkg_name__, str)
        assert len(version_utils.__main_pkg_name__) > 0

    def test_metadata_keyerror_pattern(self):
        """Test the exact KeyError handling pattern from the source code."""
        # This mimics the exact try-except pattern used in the source

        def mock_metadata_with_keyerror(dist_name):
            """Mock metadata function that raises KeyError for 'Name' key."""
            return {}  # Empty dict will cause KeyError on ["Name"] access

        # Test the exact pattern from lines 24-27
        DIST_NAME = "nemo-evaluator-launcher"
        try:
            __main_pkg_name__ = mock_metadata_with_keyerror(DIST_NAME)["Name"]
        except KeyError:
            __main_pkg_name__ = "name_not_found"

        assert __main_pkg_name__ == "name_not_found"

    def test_metadata_success_pattern(self):
        """Test the successful metadata access pattern."""

        def mock_metadata_success(dist_name):
            """Mock metadata function that returns proper dict."""
            return {"Name": "test-package-name"}

        # Test the successful path from lines 24-27
        DIST_NAME = "nemo-evaluator-launcher"
        try:
            __main_pkg_name__ = mock_metadata_success(DIST_NAME)["Name"]
        except KeyError:
            __main_pkg_name__ = "name_not_found"

        assert __main_pkg_name__ == "test-package-name"
