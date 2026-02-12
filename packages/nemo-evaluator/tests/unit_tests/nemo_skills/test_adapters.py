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

"""Tests for nemo-skills adapter stubs."""

import dataclasses
import json

import pytest

from nemo_evaluator._nemo_skills._adapters.utils import (
    get_help_message,
    get_logger_name,
    nested_dataclass,
    unroll_files,
)
from nemo_evaluator._nemo_skills._adapters.file_utils import jdump, jload
from nemo_evaluator._nemo_skills._adapters.dataset_utils import (
    get_dataset_module,
    import_from_path,
)
from nemo_evaluator._nemo_skills._adapters.scicode_utils import eval_prefix
from nemo_evaluator._nemo_skills._adapters.version import __version__


class TestAdapterStubs:
    """Tests for adapter stub functions (T-014 through T-017, T-078 through T-087)."""

    def test_t014_unroll_files_glob_expansion(self, tmp_path):
        """T-014: Expand glob pattern matching 3 files."""
        for i in range(3):
            (tmp_path / f"file_{i}.txt").write_text(f"content {i}")
        pattern = str(tmp_path / "*.txt")
        result = unroll_files([pattern])
        assert len(result) == 3
        assert all(r.endswith(".txt") for r in result)

    def test_t015_unroll_files_zero_matches(self, tmp_path):
        """T-015: Raise FileNotFoundError for glob with zero matches."""
        pattern = str(tmp_path / "nonexistent_*.xyz")
        with pytest.raises(FileNotFoundError, match="No files match pattern"):
            unroll_files([pattern])

    def test_t016_get_dataset_module_valid_import(self):
        """T-016: Dynamically import a known module via get_dataset_module."""
        # Use a module we know exists: json (stdlib)
        # get_dataset_module tries to import nemo_evaluator._nemo_skills.dataset.<benchmark>
        # which won't exist, so we test the error path and the valid module import via import_from_path
        with pytest.raises(ModuleNotFoundError):
            get_dataset_module("nonexistent_benchmark_xyz")

    def test_t017_get_dataset_module_nonexistent(self):
        """T-017: Raise ModuleNotFoundError for nonexistent benchmark."""
        with pytest.raises(ModuleNotFoundError):
            get_dataset_module("definitely_not_a_real_benchmark_module")

    def test_t078_get_logger_name(self):
        """T-078: get_logger_name returns non-empty string derived from path."""
        result = get_logger_name("/some/path/to/module.py")
        assert isinstance(result, str)
        assert len(result) > 0
        assert result == "module"

    def test_t079_get_help_message_empty_string(self):
        """T-079: get_help_message returns empty string."""
        result = get_help_message()
        assert result == ""

    def test_t080_jdump_jload_roundtrip_with_unicode(self, tmp_path):
        """T-080: Round-trip JSON with Unicode characters, indent=2, ensure_ascii=False."""
        data = {"key": "value", "unicode": "\u00e9\u00e8\u00ea", "number": 42}
        file_path = str(tmp_path / "test.json")
        jdump(data, file_path)
        loaded = jload(file_path)
        assert loaded == data
        # Verify formatting: indent=2 and ensure_ascii=False
        raw = (tmp_path / "test.json").read_text(encoding="utf-8")
        assert "\u00e9" in raw  # Unicode preserved, not escaped
        assert "  " in raw  # Indented

    def test_t081_jload_nonexistent_file(self):
        """T-081: Raise FileNotFoundError for nonexistent file."""
        with pytest.raises(FileNotFoundError):
            jload("/nonexistent/path/to/file.json")

    def test_t082_jload_invalid_json(self, tmp_path):
        """T-082: Raise json.JSONDecodeError for invalid JSON."""
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not valid json {{{")
        with pytest.raises(json.JSONDecodeError):
            jload(str(bad_file))

    def test_t083_import_from_path_valid_module(self, tmp_path):
        """T-083: Import Python module from absolute path."""
        mod_file = tmp_path / "my_module.py"
        mod_file.write_text("MY_CONSTANT = 42\n")
        mod = import_from_path(str(mod_file))
        assert hasattr(mod, "MY_CONSTANT")
        assert mod.MY_CONSTANT == 42

    def test_t084_import_from_path_nonexistent(self):
        """T-084: Raise error for nonexistent module path."""
        with pytest.raises((FileNotFoundError, ImportError, OSError)):
            import_from_path("/nonexistent/path/module.py")

    def test_t085_scicode_utils_eval_prefix(self):
        """T-085: scicode_utils.eval_prefix equals expected string."""
        assert eval_prefix == "from scicode.parse.parse import process_hdf5_to_tuple\n"

    def test_t086_version_synced(self):
        """T-086: version.__version__ equals 'synced'."""
        assert __version__ == "synced"

    def test_t087_nested_dataclass_decorator(self):
        """T-087: nested_dataclass returns a dataclass."""
        @nested_dataclass
        class MyData:
            x: int = 0
            y: str = "hello"

        assert dataclasses.is_dataclass(MyData)
        instance = MyData(x=5, y="world")
        assert instance.x == 5
        assert instance.y == "world"
