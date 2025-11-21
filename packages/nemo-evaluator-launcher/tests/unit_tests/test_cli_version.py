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
"""Tests for the CLI version command and --version flag."""

from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest

from nemo_evaluator_launcher import __package_name__, __version__
from nemo_evaluator_launcher.cli.main import create_parser, main
from nemo_evaluator_launcher.cli.version import Cmd as VersionCmd


class TestVersionCommand:
    """Test the version CLI command."""

    def test_version_cmd_execute_basic(self):
        """Test basic version command execution."""
        cmd = VersionCmd()

        # Capture output
        output = StringIO()
        with redirect_stdout(output):
            cmd.execute()

        result = output.getvalue().strip()

        # Should contain the main package version
        assert f"{__package_name__}: {__version__}" in result

    @patch("importlib.import_module")
    def test_version_cmd_with_internal_package_available(self, mock_import_module):
        """Test version command when internal package is available with version."""
        # Mock internal module with version
        mock_internal = MagicMock()
        mock_internal.__version__ = "1.2.3-internal"
        mock_import_module.return_value = mock_internal

        cmd = VersionCmd()

        # Capture output
        output = StringIO()
        with redirect_stdout(output):
            cmd.execute()

        result = output.getvalue().strip()
        lines = result.split("\n")

        # Should contain main package version
        assert f"{__package_name__}: {__version__}" in lines[0]

        # Should contain internal package version
        assert "nemo_evaluator_launcher_internal: 1.2.3-internal" in lines[1]

        # Verify import was called
        mock_import_module.assert_called_with("nemo_evaluator_launcher_internal")

    @patch("importlib.import_module")
    def test_version_cmd_with_internal_package_no_version(self, mock_import_module):
        """Test version command when internal package is available but has no version."""
        # Mock internal module without version
        mock_internal = MagicMock()
        del mock_internal.__version__  # Ensure no version attribute
        mock_import_module.return_value = mock_internal

        cmd = VersionCmd()

        # Capture output
        output = StringIO()
        with redirect_stdout(output):
            cmd.execute()

        result = output.getvalue().strip()
        lines = result.split("\n")

        # Should contain main package version
        assert f"{__package_name__}: {__version__}" in lines[0]

        # Should indicate internal package is available but version unknown
        assert (
            "nemo_evaluator_launcher_internal: available (version unknown)" in lines[1]
        )

    @patch("importlib.import_module")
    def test_version_cmd_with_internal_package_import_error(self, mock_import_module):
        """Test version command when internal package import fails."""
        # Mock ImportError
        mock_import_module.side_effect = ImportError(
            "No module named 'nemo_evaluator_launcher_internal'"
        )

        cmd = VersionCmd()

        # Capture output
        output = StringIO()
        with redirect_stdout(output):
            cmd.execute()

        result = output.getvalue().strip()

        # Should only contain main package version (no internal package line)
        assert f"{__package_name__}: {__version__}" in result
        assert "nemo_evaluator_launcher_internal" not in result

    @patch("importlib.import_module")
    def test_version_cmd_with_internal_package_other_error(self, mock_import_module):
        """Test version command when internal package has other loading errors."""
        # Mock other error
        mock_import_module.side_effect = Exception("Some other error")

        cmd = VersionCmd()

        with pytest.raises(Exception, match=".*Some other error.*"):
            cmd.execute()


class TestVersionCLIIntegration:
    """Test version functionality integration with CLI parser."""

    def test_version_subcommand_in_parser(self):
        """Test that version subcommand is properly registered in parser."""
        parser = create_parser()

        # Test parsing version subcommand
        args = parser.parse_args(["version"])
        assert args.command == "version"
        assert hasattr(args, "version")
        assert isinstance(args.version, VersionCmd)

    def test_version_flag_in_parser(self):
        """Test that --version flag is properly registered in parser."""
        parser = create_parser()

        # Test parsing --version flag
        args = parser.parse_args(["--version"])
        assert hasattr(args, "version")
        assert args.version is True

    @patch("sys.argv", ["nemo-evaluator-launcher", "--version"])
    @patch("nemo_evaluator_launcher.cli.version.Cmd.execute")
    def test_main_with_version_flag(self, mock_execute):
        """Test main function with --version flag."""
        main()

        # Should call version command execute
        mock_execute.assert_called_once()

    @patch("sys.argv", ["nemo-evaluator-launcher", "version"])
    def test_main_with_version_subcommand(self):
        """Test main function with version subcommand."""
        # Capture output
        output = StringIO()
        with redirect_stdout(output):
            main()

        result = output.getvalue().strip()

        # Should contain version information
        assert f"{__package_name__}: {__version__}" in result

    @patch("sys.argv", ["nemo-evaluator-launcher"])
    def test_main_without_command_shows_help(self):
        """Test that main shows help when no command is provided."""
        # Capture output
        output = StringIO()
        with redirect_stdout(output):
            main()

        result = output.getvalue()

        # Should show help (contains usage information)
        assert "usage:" in result.lower() or "help" in result.lower()

    def test_version_flag_and_subcommand_compatibility(self):
        """Test that both --version flag and version subcommand work similarly."""
        # Test subcommand output
        cmd = VersionCmd()
        subcommand_output = StringIO()
        with redirect_stdout(subcommand_output):
            cmd.execute()
        subcommand_result = subcommand_output.getvalue().strip()

        # Test flag output via main (using argv patching)
        with patch("sys.argv", ["nemo-evaluator-launcher", "--version"]):
            flag_output = StringIO()
            with redirect_stdout(flag_output):
                main()
            flag_result = flag_output.getvalue().strip()

        # Both should produce the same output
        assert subcommand_result == flag_result
