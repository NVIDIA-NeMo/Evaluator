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

"""Tests for custom harness discovery in nemo_evaluator namespace."""

from unittest.mock import MagicMock, patch

import pytest


class TestCustomHarnessDiscovery:
    """Test harness discovery from both nemo_evaluator and core_evals namespaces."""

    def test_get_harness_packages_empty(self):
        """Test _get_harness_packages when no harnesses are installed."""
        from nemo_evaluator.core.input import _get_harness_packages

        # Mock both imports to fail
        with patch("nemo_evaluator.core.input.pkgutil.iter_modules") as mock_iter:
            mock_iter.return_value = []
            packages = _get_harness_packages()
            assert isinstance(packages, list)

    def test_get_harness_packages_core_evals_only(self):
        """Test _get_harness_packages when only core_evals harnesses exist."""
        from nemo_evaluator.core.input import _get_harness_packages

        # Create mock package info
        mock_pkg = MagicMock()
        mock_pkg.name = "test_harness"
        mock_pkg.module_finder.path = "/fake/path"

        with patch("nemo_evaluator.core.input.pkgutil.iter_modules") as mock_iter:
            # First call for nemo_evaluator (empty), second for core_evals
            mock_iter.side_effect = [
                [],  # nemo_evaluator namespace (empty/not found)
                [mock_pkg],  # core_evals namespace
            ]

            packages = _get_harness_packages()
            assert len(packages) == 1
            assert packages[0].name == "test_harness"

    def test_get_harness_packages_both_namespaces(self, tmp_path, monkeypatch):
        """Test _get_harness_packages when harnesses exist in both nemo_evaluator and core_evals namespaces."""
        from nemo_evaluator.core.input import _get_harness_packages

        # Mock nemo_evaluator namespace
        mock_nemo_pkg = MagicMock()
        mock_nemo_pkg.name = "custom_harness"
        mock_nemo_pkg.module_finder.path = str(tmp_path)

        # Mock core_evals namespace
        mock_core_pkg = MagicMock()
        mock_core_pkg.name = "core_harness"
        mock_core_pkg.module_finder.path = "/fake/path"

        with patch("nemo_evaluator.core.input.pkgutil.iter_modules") as mock_iter:
            # First call for nemo_evaluator namespace, second for core_evals
            mock_iter.side_effect = [
                [mock_nemo_pkg],  # nemo_evaluator namespace
                [mock_core_pkg],  # core_evals namespace
            ]

            with patch("nemo_evaluator.core.input.os.path.exists") as mock_exists:
                # Only custom_harness has framework.yml
                mock_exists.side_effect = lambda p: "custom_harness" in p

                packages = _get_harness_packages()

                # Should find both packages
                assert len(packages) == 2
                assert any(p.name == "custom_harness" for p in packages)
                assert any(p.name == "core_harness" for p in packages)

    def test_parse_output_nemo_evaluator_namespace(self, tmp_path):
        """Test parse_output tries nemo_evaluator namespace first."""
        from nemo_evaluator.api.api_dataclasses import (
            Evaluation,
            EvaluationConfig,
            EvaluationTarget,
        )
        from nemo_evaluator.core.evaluate import parse_output

        # Create a mock evaluation
        evaluation = Evaluation(
            command="test",
            framework_name="test",
            pkg_name="test_harness",
            config=EvaluationConfig(output_dir=str(tmp_path)),
            target=EvaluationTarget(),
        )

        # Mock the import to simulate nemo_evaluator.test_harness.output exists
        mock_module = MagicMock()
        mock_result = MagicMock()
        mock_module.parse_output.return_value = mock_result

        with patch(
            "nemo_evaluator.core.evaluate.importlib.import_module"
        ) as mock_import:
            mock_import.return_value = mock_module
            result = parse_output(evaluation)

            # Verify it tried to import from nemo_evaluator namespace first
            mock_import.assert_called_with("nemo_evaluator.test_harness.output")
            assert result == mock_result

    def test_parse_output_fallback_to_core_evals(self, tmp_path):
        """Test parse_output falls back to core_evals if nemo_evaluator import fails."""
        from nemo_evaluator.api.api_dataclasses import (
            Evaluation,
            EvaluationConfig,
            EvaluationTarget,
        )
        from nemo_evaluator.core.evaluate import parse_output

        # Create a mock evaluation
        evaluation = Evaluation(
            command="test",
            framework_name="test",
            pkg_name="test_harness",
            config=EvaluationConfig(output_dir=str(tmp_path)),
            target=EvaluationTarget(),
        )

        # Mock the import to simulate nemo_evaluator import fails, core_evals succeeds
        mock_module = MagicMock()
        mock_result = MagicMock()
        mock_module.parse_output.return_value = mock_result

        with patch(
            "nemo_evaluator.core.evaluate.importlib.import_module"
        ) as mock_import:

            def import_side_effect(module_name):
                if "nemo_evaluator.test_harness" in module_name:
                    raise ImportError("Not found")
                return mock_module

            mock_import.side_effect = import_side_effect
            result = parse_output(evaluation)

            # Verify it tried both imports
            assert mock_import.call_count == 2
            assert result == mock_result

    def test_parse_output_both_imports_fail(self, tmp_path):
        """Test parse_output raises clear error when both imports fail."""
        from nemo_evaluator.api.api_dataclasses import (
            Evaluation,
            EvaluationConfig,
            EvaluationTarget,
        )
        from nemo_evaluator.core.evaluate import parse_output

        # Create a mock evaluation
        evaluation = Evaluation(
            command="test",
            framework_name="test",
            pkg_name="nonexistent_harness",
            config=EvaluationConfig(output_dir=str(tmp_path)),
            target=EvaluationTarget(),
        )

        with patch(
            "nemo_evaluator.core.evaluate.importlib.import_module"
        ) as mock_import:
            mock_import.side_effect = ImportError("Not found")

            with pytest.raises(ImportError) as exc_info:
                parse_output(evaluation)

            # Verify error message mentions the harness and relevant paths
            assert "nonexistent_harness" in str(exc_info.value)
            assert ".nemo_evaluator" in str(exc_info.value) or "core_evals" in str(
                exc_info.value
            )

    def test_get_available_evaluations_uses_new_discovery(self):
        """Test get_available_evaluations uses _get_harness_packages."""
        from nemo_evaluator.core.input import get_available_evaluations

        with patch(
            "nemo_evaluator.core.input._get_harness_packages"
        ) as mock_get_packages:
            mock_get_packages.return_value = []
            result = get_available_evaluations()

            # Verify _get_harness_packages was called
            mock_get_packages.assert_called_once()
            # Result should be tuple of 3 dicts
            assert isinstance(result, tuple)
            assert len(result) == 3
