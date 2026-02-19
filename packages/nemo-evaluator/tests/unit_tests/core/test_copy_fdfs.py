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

import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from nemo_evaluator.core.input import _copy_fdfs


def _make_entry_point(pkg_dir):
    """Create a mock entry point that loads a module with __file__ in pkg_dir."""
    ep = MagicMock()
    mod = MagicMock()
    mod.__file__ = str(pkg_dir / "__init__.py")
    ep.load.return_value = mod
    return ep


def _make_eps_result(entry_points):
    """Create a mock entry_points() return value with .select() support."""
    mock_eps = MagicMock()
    mock_eps.select.return_value = entry_points
    return mock_eps


def test_copy_fdfs_creates_target_dir(tmp_path):
    """Test that _copy_fdfs creates the target directory if it doesn't exist."""
    target_dir = tmp_path / "new_target_dir"
    assert not target_dir.exists()

    # Mock both core_evals (ImportError) and entry_points (no results)
    with patch.dict(sys.modules, {"core_evals": None}):
        with patch("importlib.metadata.entry_points", return_value=_make_eps_result([])):
            _copy_fdfs(str(target_dir))

    assert target_dir.exists()
    assert target_dir.is_dir()


def test_copy_fdfs_no_packages(tmp_path):
    """Test _copy_fdfs with no core_evals and no entry points - target_dir created but empty."""
    target_dir = tmp_path / "empty_target"

    with patch.dict(sys.modules, {"core_evals": None}):
        with patch("importlib.metadata.entry_points", return_value=_make_eps_result([])):
            _copy_fdfs(str(target_dir))

    assert target_dir.exists()
    assert target_dir.is_dir()
    assert len(list(target_dir.iterdir())) == 0


def test_copy_fdfs_entry_point_discovery(tmp_path):
    """Test entry point discovery with a .nemo_evaluator package containing framework.yml."""
    target_dir = tmp_path / "target"

    # Create a mock package structure with .nemo_evaluator
    mock_pkg_dir = tmp_path / "mock_package"
    mock_pkg_dir.mkdir()
    nemo_eval_dir = mock_pkg_dir / ".nemo_evaluator"
    nemo_eval_dir.mkdir()
    harness_dir = nemo_eval_dir / "test_harness"
    harness_dir.mkdir()
    (harness_dir / "framework.yml").write_text("framework:\n  name: test_harness\n  pkg_name: test_pkg\n")

    ep = _make_entry_point(mock_pkg_dir)

    with patch.dict(sys.modules, {"core_evals": None}):
        with patch("importlib.metadata.entry_points", return_value=_make_eps_result([ep])):
            _copy_fdfs(str(target_dir))

    copied_file = target_dir / "test_harness" / "framework.yml"
    assert copied_file.exists()
    assert "test_harness" in copied_file.read_text()


def test_copy_fdfs_multiple_harnesses(tmp_path):
    """Test multiple harnesses in one .nemo_evaluator directory."""
    target_dir = tmp_path / "target"

    mock_pkg_dir = tmp_path / "mock_package"
    mock_pkg_dir.mkdir()
    nemo_eval_dir = mock_pkg_dir / ".nemo_evaluator"
    nemo_eval_dir.mkdir()

    (nemo_eval_dir / "harness1").mkdir()
    (nemo_eval_dir / "harness1" / "framework.yml").write_text("framework:\n  name: harness1\n")

    (nemo_eval_dir / "harness2").mkdir()
    (nemo_eval_dir / "harness2" / "framework.yml").write_text("framework:\n  name: harness2\n")

    ep = _make_entry_point(mock_pkg_dir)

    with patch.dict(sys.modules, {"core_evals": None}):
        with patch("importlib.metadata.entry_points", return_value=_make_eps_result([ep])):
            _copy_fdfs(str(target_dir))

    assert (target_dir / "harness1" / "framework.yml").exists()
    assert (target_dir / "harness2" / "framework.yml").exists()
    assert "harness1" in (target_dir / "harness1" / "framework.yml").read_text()
    assert "harness2" in (target_dir / "harness2" / "framework.yml").read_text()


def test_copy_fdfs_entry_point_load_failure(tmp_path):
    """Test that one entry point failing to load doesn't prevent others from working."""
    target_dir = tmp_path / "target"

    # Create a working package
    mock_pkg_dir = tmp_path / "mock_package"
    mock_pkg_dir.mkdir()
    nemo_eval_dir = mock_pkg_dir / ".nemo_evaluator"
    nemo_eval_dir.mkdir()
    harness_dir = nemo_eval_dir / "working_harness"
    harness_dir.mkdir()
    (harness_dir / "framework.yml").write_text("framework:\n  name: working_harness\n")

    # One that fails, one that works
    failing_ep = MagicMock()
    failing_ep.load.side_effect = Exception("Failed to load")

    working_ep = _make_entry_point(mock_pkg_dir)

    with patch.dict(sys.modules, {"core_evals": None}):
        with patch("importlib.metadata.entry_points", return_value=_make_eps_result([failing_ep, working_ep])):
            _copy_fdfs(str(target_dir))

    copied_file = target_dir / "working_harness" / "framework.yml"
    assert copied_file.exists()
    assert "working_harness" in copied_file.read_text()


def test_copy_fdfs_missing_framework_yml(tmp_path):
    """Test that harness directories without framework.yml are skipped."""
    target_dir = tmp_path / "target"

    mock_pkg_dir = tmp_path / "mock_package"
    mock_pkg_dir.mkdir()
    nemo_eval_dir = mock_pkg_dir / ".nemo_evaluator"
    nemo_eval_dir.mkdir()

    # Harness without framework.yml
    (nemo_eval_dir / "incomplete_harness").mkdir()

    # Harness with framework.yml
    (nemo_eval_dir / "complete_harness").mkdir()
    (nemo_eval_dir / "complete_harness" / "framework.yml").write_text("framework:\n  name: complete_harness\n")

    ep = _make_entry_point(mock_pkg_dir)

    with patch.dict(sys.modules, {"core_evals": None}):
        with patch("importlib.metadata.entry_points", return_value=_make_eps_result([ep])):
            _copy_fdfs(str(target_dir))

    assert not (target_dir / "incomplete_harness").exists()
    assert (target_dir / "complete_harness" / "framework.yml").exists()


def test_copy_fdfs_both_patterns(tmp_path):
    """Test that both core_evals (legacy) and entry_point patterns work together."""
    target_dir = tmp_path / "target"

    # Legacy core_evals structure
    legacy_pkg_dir = tmp_path / "legacy_package"
    legacy_pkg_dir.mkdir()
    legacy_harness_dir = legacy_pkg_dir / "legacy_harness"
    legacy_harness_dir.mkdir()
    (legacy_harness_dir / "framework.yml").write_text("framework:\n  name: legacy_harness\n")

    mock_module_info = SimpleNamespace()
    mock_module_info.name = "legacy_harness"
    mock_module_info.module_finder = SimpleNamespace()
    mock_module_info.module_finder.path = str(legacy_pkg_dir)

    # OSS .nemo_evaluator structure
    oss_pkg_dir = tmp_path / "oss_package"
    oss_pkg_dir.mkdir()
    nemo_eval_dir = oss_pkg_dir / ".nemo_evaluator"
    nemo_eval_dir.mkdir()
    oss_harness_dir = nemo_eval_dir / "oss_harness"
    oss_harness_dir.mkdir()
    (oss_harness_dir / "framework.yml").write_text("framework:\n  name: oss_harness\n")

    oss_ep = _make_entry_point(oss_pkg_dir)

    # Mock core_evals as importable with iter_modules returning our mock
    mock_core_evals = MagicMock()
    mock_core_evals.__path__ = [str(legacy_pkg_dir)]

    with patch.dict(sys.modules, {"core_evals": mock_core_evals}):
        with patch("nemo_evaluator.core.input.pkgutil.iter_modules", return_value=[mock_module_info]):
            with patch("importlib.metadata.entry_points", return_value=_make_eps_result([oss_ep])):
                _copy_fdfs(str(target_dir))

    assert (target_dir / "legacy_harness" / "framework.yml").exists()
    assert (target_dir / "oss_harness" / "framework.yml").exists()
    assert "legacy_harness" in (target_dir / "legacy_harness" / "framework.yml").read_text()
    assert "oss_harness" in (target_dir / "oss_harness" / "framework.yml").read_text()
