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

"""Unit tests for BYOB requirements validation."""

from unittest.mock import MagicMock, patch

from nemo_evaluator.contrib.byob.runner import check_requirements, ensure_requirements


class TestCheckRequirements:
    """Tests for check_requirements function."""

    @patch("importlib.metadata.version")
    def test_all_present(self, mock_version):
        """Test that no warnings are emitted when all packages are installed."""
        mock_version.side_effect = lambda pkg: {"numpy": "1.24.0", "pandas": "2.0.0"}[
            pkg
        ]
        warnings = check_requirements(["numpy", "pandas"])
        assert warnings == []

    @patch("importlib.metadata.version")
    def test_missing_package(self, mock_version):
        """Test that a missing package produces a warning."""
        import importlib.metadata

        mock_version.side_effect = importlib.metadata.PackageNotFoundError(
            "nonexistent"
        )
        warnings = check_requirements(["nonexistent-pkg"])
        assert len(warnings) == 1
        assert "Missing package" in warnings[0]
        assert "nonexistent-pkg" in warnings[0]

    @patch("importlib.metadata.version")
    def test_version_satisfied(self, mock_version):
        """Test that version specifiers are checked correctly."""
        mock_version.return_value = "1.24.0"
        warnings = check_requirements(["numpy>=1.20"])
        assert warnings == []

    @patch("importlib.metadata.version")
    def test_version_mismatch(self, mock_version):
        """Test that version mismatch produces a warning."""
        mock_version.return_value = "1.18.0"
        warnings = check_requirements(["numpy>=1.20"])
        assert len(warnings) == 1
        assert "Version mismatch" in warnings[0]
        assert "1.18.0" in warnings[0]

    @patch("importlib.metadata.version")
    def test_empty_requirements(self, mock_version):
        """Test that empty list returns no warnings."""
        warnings = check_requirements([])
        assert warnings == []
        mock_version.assert_not_called()

    @patch("importlib.metadata.version")
    def test_whitespace_requirements(self, mock_version):
        """Test that whitespace-only requirements are skipped."""
        warnings = check_requirements(["", "  "])
        assert warnings == []

    @patch("importlib.metadata.version")
    def test_mixed_results(self, mock_version):
        """Test mix of present, missing, and mismatched packages."""
        import importlib.metadata

        def version_lookup(pkg):
            if pkg == "numpy":
                return "1.24.0"
            if pkg == "pandas":
                raise importlib.metadata.PackageNotFoundError(pkg)
            if pkg == "scipy":
                return "1.5.0"
            raise importlib.metadata.PackageNotFoundError(pkg)

        mock_version.side_effect = version_lookup
        warnings = check_requirements(["numpy>=1.20", "pandas", "scipy>=1.10"])
        # pandas is missing, scipy version is too old
        assert len(warnings) == 2
        assert any("pandas" in w for w in warnings)
        assert any("scipy" in w for w in warnings)

    @patch("importlib.metadata.version")
    def test_exact_version(self, mock_version):
        """Test exact version match (==)."""
        mock_version.return_value = "1.24.0"
        warnings = check_requirements(["numpy==1.24.0"])
        assert warnings == []

    @patch("importlib.metadata.version")
    def test_exact_version_mismatch(self, mock_version):
        """Test exact version mismatch."""
        mock_version.return_value = "1.23.0"
        warnings = check_requirements(["numpy==1.24.0"])
        assert len(warnings) == 1
        assert "Version mismatch" in warnings[0]


class TestEnsureRequirements:
    """Tests for ensure_requirements function."""

    @patch("nemo_evaluator.contrib.byob.runner.check_requirements")
    def test_no_install_when_satisfied(self, mock_check):
        """Test that subprocess is NOT called when all requirements are met."""
        mock_check.return_value = []
        with patch("subprocess.run") as mock_run:
            ensure_requirements(["numpy", "pandas"])
            mock_run.assert_not_called()

    @patch("nemo_evaluator.contrib.byob.runner.check_requirements")
    def test_installs_missing_packages(self, mock_check):
        """Test that pip/uv install IS called for missing packages."""
        mock_check.return_value = ["Missing package: datasets (required: datasets)"]
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            ensure_requirements(["datasets"])
            mock_run.assert_called_once()
            cmd = mock_run.call_args[0][0]
            assert "pip" in cmd
            assert "install" in cmd
            assert "datasets" in cmd

    @patch("nemo_evaluator.contrib.byob.runner.check_requirements")
    def test_installs_only_missing(self, mock_check):
        """Test that only missing packages are installed, not satisfied ones."""
        mock_check.return_value = ["Missing package: datasets (required: datasets)"]
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            ensure_requirements(["numpy", "datasets"])
            cmd = mock_run.call_args[0][0]
            assert "datasets" in cmd
            assert "numpy" not in cmd
