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

"""Unit tests for BYOB dataset fetching and loading."""

import json
from pathlib import Path
from typing import Optional

import pytest
from unittest.mock import patch

from nemo_evaluator.contrib.byob.dataset import (
    FetchResult,
    HuggingFaceFetcher,
    LocalFetcher,
    _FETCHER_REGISTRY,
    _detect_format,
    _remap_fields,
    get_fetcher_for_uri,
    load_csv,
    load_dataset,
    load_jsonl,
    register_fetcher,
)


# ---------------------------------------------------------------------------
# TestDetectFormat
# ---------------------------------------------------------------------------


class TestDetectFormat:
    """Tests for _detect_format extension-based format detection."""

    @pytest.mark.parametrize(
        "filename,expected_format",
        [
            ("data.jsonl", "jsonl"),
            ("data.json", "jsonl"),
            ("data.csv", "csv"),
            ("data.tsv", "tsv"),
        ],
    )
    def test_known_extensions(self, filename: str, expected_format: str) -> None:
        """Validate known extensions map to the correct format string."""
        result = _detect_format(Path(filename))
        assert result == expected_format, (
            f"_detect_format({filename!r}) returned {result!r}, "
            f"expected {expected_format!r}"
        )

    @pytest.mark.parametrize(
        "filename",
        [
            "data.parquet",
            "data.txt",
            "data.xml",
            "data",
        ],
    )
    def test_unknown_extension_defaults_to_jsonl(self, filename: str) -> None:
        """Validate unknown/missing extensions fall back to 'jsonl'."""
        result = _detect_format(Path(filename))
        assert result == "jsonl", (
            f"_detect_format({filename!r}) returned {result!r}, "
            f"expected default 'jsonl'"
        )

    def test_case_insensitive(self) -> None:
        """Validate extension detection is case-insensitive."""
        for ext, expected in [(".JSONL", "jsonl"), (".CSV", "csv"), (".TSV", "tsv")]:
            result = _detect_format(Path(f"data{ext}"))
            assert result == expected, (
                f"_detect_format('data{ext}') returned {result!r}, "
                f"expected {expected!r} (case-insensitive)"
            )


# ---------------------------------------------------------------------------
# TestLoadJsonl
# ---------------------------------------------------------------------------


class TestLoadJsonl:
    """Tests for load_jsonl JSONL parser."""

    def test_basic_load(self, tmp_path: Path) -> None:
        """Validate loading a well-formed JSONL file returns all records."""
        path = tmp_path / "data.jsonl"
        records = [
            {"question": "Q1", "answer": "A1"},
            {"question": "Q2", "answer": "A2"},
            {"question": "Q3", "answer": "A3"},
        ]
        path.write_text(
            "\n".join(json.dumps(r) for r in records),
            encoding="utf-8",
        )

        data = load_jsonl(path)
        assert len(data) == 3, f"Expected 3 records, got {len(data)}"
        assert data == records, "Loaded data does not match written records"

    def test_empty_lines_skipped(self, tmp_path: Path) -> None:
        """Validate blank lines interspersed in the file are silently skipped."""
        path = tmp_path / "data.jsonl"
        content = '{"a": 1}\n\n   \n{"b": 2}\n\n'
        path.write_text(content, encoding="utf-8")

        data = load_jsonl(path)
        assert len(data) == 2, (
            f"Expected 2 records (empty lines skipped), got {len(data)}"
        )
        assert data[0] == {"a": 1}
        assert data[1] == {"b": 2}

    def test_limit(self, tmp_path: Path) -> None:
        """Validate limit parameter stops reading after N records."""
        path = tmp_path / "data.jsonl"
        records = [{"i": i} for i in range(10)]
        path.write_text(
            "\n".join(json.dumps(r) for r in records),
            encoding="utf-8",
        )

        data = load_jsonl(path, limit=3)
        assert len(data) == 3, f"Expected 3 records with limit=3, got {len(data)}"
        assert data == records[:3], f"Expected first 3 records, got {data}"

    def test_limit_greater_than_file_returns_all(self, tmp_path: Path) -> None:
        """Validate limit larger than file size returns all records."""
        path = tmp_path / "data.jsonl"
        records = [{"i": i} for i in range(3)]
        path.write_text(
            "\n".join(json.dumps(r) for r in records),
            encoding="utf-8",
        )

        data = load_jsonl(path, limit=100)
        assert len(data) == 3, f"Expected all 3 records with limit=100, got {len(data)}"

    def test_non_dict_line_raises_value_error(self, tmp_path: Path) -> None:
        """Validate that a JSON line that is not a dict raises ValueError."""
        path = tmp_path / "data.jsonl"
        content = '{"valid": true}\n[1, 2, 3]\n'
        path.write_text(content, encoding="utf-8")

        with pytest.raises(ValueError, match="not a JSON object"):
            load_jsonl(path)

    def test_malformed_json_raises(self, tmp_path: Path) -> None:
        """Validate malformed JSON raises json.JSONDecodeError."""
        path = tmp_path / "data.jsonl"
        content = '{"valid": true}\n{invalid json}\n'
        path.write_text(content, encoding="utf-8")

        with pytest.raises(json.JSONDecodeError):
            load_jsonl(path)

    def test_empty_file_returns_empty_list(self, tmp_path: Path) -> None:
        """Validate an empty file returns an empty list."""
        path = tmp_path / "empty.jsonl"
        path.write_text("", encoding="utf-8")

        data = load_jsonl(path)
        assert data == [], f"Expected empty list for empty file, got {data}"


# ---------------------------------------------------------------------------
# TestLoadCsv
# ---------------------------------------------------------------------------


class TestLoadCsv:
    """Tests for load_csv CSV/TSV parser."""

    def test_basic_csv(self, tmp_path: Path) -> None:
        """Validate loading a standard CSV file with headers."""
        path = tmp_path / "data.csv"
        content = "question,answer\nWhat is 1+1?,2\nWhat is 2+2?,4\n"
        path.write_text(content, encoding="utf-8")

        data = load_csv(path, delimiter=",")
        assert len(data) == 2, f"Expected 2 rows, got {len(data)}"
        assert data[0] == {"question": "What is 1+1?", "answer": "2"}, (
            f"Row 0 mismatch: {data[0]}"
        )
        assert data[1] == {"question": "What is 2+2?", "answer": "4"}, (
            f"Row 1 mismatch: {data[1]}"
        )

    def test_tsv(self, tmp_path: Path) -> None:
        """Validate loading a TSV file with tab delimiter."""
        path = tmp_path / "data.tsv"
        content = "name\tscore\nAlice\t95\nBob\t87\n"
        path.write_text(content, encoding="utf-8")

        data = load_csv(path, delimiter="\t")
        assert len(data) == 2, f"Expected 2 rows, got {len(data)}"
        assert data[0] == {"name": "Alice", "score": "95"}, f"Row 0 mismatch: {data[0]}"
        assert data[1] == {"name": "Bob", "score": "87"}, f"Row 1 mismatch: {data[1]}"

    def test_limit(self, tmp_path: Path) -> None:
        """Validate limit parameter stops reading after N rows."""
        path = tmp_path / "data.csv"
        lines = ["id,value"] + [f"{i},{i * 10}" for i in range(10)]
        path.write_text("\n".join(lines), encoding="utf-8")

        data = load_csv(path, delimiter=",", limit=3)
        assert len(data) == 3, f"Expected 3 rows with limit=3, got {len(data)}"
        assert data[0] == {"id": "0", "value": "0"}, f"Row 0 mismatch: {data[0]}"

    def test_empty_csv_header_only(self, tmp_path: Path) -> None:
        """Validate a CSV with only a header row returns an empty list."""
        path = tmp_path / "empty.csv"
        path.write_text("col_a,col_b\n", encoding="utf-8")

        data = load_csv(path, delimiter=",")
        assert data == [], f"Expected empty list for header-only CSV, got {data}"

    def test_completely_empty_csv(self, tmp_path: Path) -> None:
        """Validate a completely empty CSV returns an empty list."""
        path = tmp_path / "empty.csv"
        path.write_text("", encoding="utf-8")

        data = load_csv(path, delimiter=",")
        assert data == [], f"Expected empty list for completely empty CSV, got {data}"


# ---------------------------------------------------------------------------
# TestLocalFetcher
# ---------------------------------------------------------------------------


class TestLocalFetcher:
    """Tests for LocalFetcher filesystem fetcher."""

    def test_supports_existing_file(self, tmp_path: Path) -> None:
        """Validate supports() returns True for an existing file."""
        path = tmp_path / "exists.jsonl"
        path.write_text('{"a": 1}\n', encoding="utf-8")

        fetcher = LocalFetcher()
        assert fetcher.supports(str(path)) is True, (
            f"Expected supports('{path}') to be True for existing file"
        )

    def test_not_supports_nonexistent(self) -> None:
        """Validate supports() returns False for a nonexistent path."""
        fetcher = LocalFetcher()
        assert fetcher.supports("/nonexistent/path/data.jsonl") is False, (
            "Expected supports() to be False for nonexistent path"
        )

    def test_fetch_returns_correct_fetch_result(self, tmp_path: Path) -> None:
        """Validate fetch() returns a FetchResult with correct path, format, and metadata."""
        path = tmp_path / "test.csv"
        path.write_text("col\nval\n", encoding="utf-8")

        fetcher = LocalFetcher()
        result = fetcher.fetch(str(path))

        assert isinstance(result, FetchResult), (
            f"Expected FetchResult, got {type(result).__name__}"
        )
        assert result.local_path == path, (
            f"Expected local_path={path}, got {result.local_path}"
        )
        assert result.format == "csv", f"Expected format='csv', got {result.format!r}"
        assert result.metadata == {"source": "local"}, (
            f"Expected metadata={{'source': 'local'}}, got {result.metadata}"
        )

    def test_fetch_nonexistent_raises_file_not_found(self) -> None:
        """Validate fetch() raises FileNotFoundError for missing files."""
        fetcher = LocalFetcher()
        with pytest.raises(FileNotFoundError, match="Dataset not found"):
            fetcher.fetch("/nonexistent/path/data.jsonl")

    def test_fetch_directory_raises_value_error(self, tmp_path: Path) -> None:
        """Validate fetch() raises ValueError when path is a directory, not a file."""
        fetcher = LocalFetcher()
        with pytest.raises(ValueError, match="not a file"):
            fetcher.fetch(str(tmp_path))

    def test_fetch_detects_jsonl_format(self, tmp_path: Path) -> None:
        """Validate fetch correctly detects .jsonl format."""
        path = tmp_path / "data.jsonl"
        path.write_text('{"x": 1}\n', encoding="utf-8")

        result = LocalFetcher().fetch(str(path))
        assert result.format == "jsonl", (
            f"Expected format='jsonl', got {result.format!r}"
        )

    def test_fetch_detects_tsv_format(self, tmp_path: Path) -> None:
        """Validate fetch correctly detects .tsv format."""
        path = tmp_path / "data.tsv"
        path.write_text("a\tb\n1\t2\n", encoding="utf-8")

        result = LocalFetcher().fetch(str(path))
        assert result.format == "tsv", f"Expected format='tsv', got {result.format!r}"


# ---------------------------------------------------------------------------
# TestFetcherRegistry
# ---------------------------------------------------------------------------


class TestFetcherRegistry:
    """Tests for the fetcher registry (register_fetcher, get_fetcher_for_uri)."""

    def test_default_registry_contains_local_fetcher(self) -> None:
        """Validate the default registry contains exactly one LocalFetcher."""
        local_fetchers = [f for f in _FETCHER_REGISTRY if isinstance(f, LocalFetcher)]
        assert len(local_fetchers) >= 1, (
            f"Expected at least one LocalFetcher in registry, "
            f"found: {[type(f).__name__ for f in _FETCHER_REGISTRY]}"
        )

    def test_register_fetcher_adds_to_front(self) -> None:
        """Validate register_fetcher inserts at position 0 (LIFO order)."""

        class DummyFetcher:
            def supports(self, uri: str) -> bool:
                return False

            def fetch(self, uri: str, cache_dir: Optional[Path] = None) -> FetchResult:
                raise NotImplementedError

        original_length = len(_FETCHER_REGISTRY)
        dummy = DummyFetcher()
        register_fetcher(dummy)

        assert len(_FETCHER_REGISTRY) == original_length + 1, (
            f"Expected registry length {original_length + 1}, "
            f"got {len(_FETCHER_REGISTRY)}"
        )
        assert _FETCHER_REGISTRY[0] is dummy, (
            f"Expected newly registered fetcher at position 0, "
            f"got {type(_FETCHER_REGISTRY[0]).__name__}"
        )

    def test_custom_fetcher_overrides_local(self, tmp_path: Path) -> None:
        """Validate a custom fetcher that supports a URI takes priority over LocalFetcher."""
        sentinel_path = tmp_path / "sentinel.jsonl"
        sentinel_path.write_text('{"from": "custom"}\n', encoding="utf-8")

        class CustomFetcher:
            def supports(self, uri: str) -> bool:
                return uri.startswith("custom://")

            def fetch(self, uri: str, cache_dir: Optional[Path] = None) -> FetchResult:
                return FetchResult(
                    local_path=sentinel_path,
                    format="jsonl",
                    metadata={"source": "custom"},
                )

        register_fetcher(CustomFetcher())

        fetcher = get_fetcher_for_uri("custom://my-dataset")
        assert isinstance(fetcher, CustomFetcher), (
            f"Expected CustomFetcher for 'custom://' URI, got {type(fetcher).__name__}"
        )

    def test_get_fetcher_for_uri_raises_for_unsupported(self) -> None:
        """Validate get_fetcher_for_uri raises ValueError for unsupported URIs."""
        with pytest.raises(ValueError, match="No fetcher supports URI"):
            get_fetcher_for_uri("s3://bucket/data.jsonl")

    def test_get_fetcher_for_uri_error_lists_registered_fetchers(self) -> None:
        """Validate the ValueError message lists registered fetcher class names."""
        try:
            get_fetcher_for_uri("unknown://nowhere")
            pytest.fail("Expected ValueError")
        except ValueError as e:
            error_msg = str(e)
            assert "LocalFetcher" in error_msg, (
                f"Expected 'LocalFetcher' in error message, got: {error_msg}"
            )

    def test_register_multiple_fetchers_lifo_order(self) -> None:
        """Validate multiple registered fetchers are checked in LIFO order."""

        class FetcherA:
            def supports(self, uri: str) -> bool:
                return uri.startswith("proto://")

            def fetch(self, uri: str, cache_dir: Optional[Path] = None) -> FetchResult:
                return FetchResult(
                    local_path=Path("/tmp/a"),
                    format="jsonl",
                    metadata={"source": "A"},
                )

        class FetcherB:
            def supports(self, uri: str) -> bool:
                return uri.startswith("proto://")

            def fetch(self, uri: str, cache_dir: Optional[Path] = None) -> FetchResult:
                return FetchResult(
                    local_path=Path("/tmp/b"),
                    format="jsonl",
                    metadata={"source": "B"},
                )

        register_fetcher(FetcherA())
        register_fetcher(FetcherB())  # B registered last, so checked first

        fetcher = get_fetcher_for_uri("proto://data")
        assert isinstance(fetcher, FetcherB), (
            f"Expected FetcherB (registered last = checked first), "
            f"got {type(fetcher).__name__}"
        )


# ---------------------------------------------------------------------------
# TestLoadDataset
# ---------------------------------------------------------------------------


class TestLoadDataset:
    """Tests for the high-level load_dataset entry point."""

    def test_load_jsonl_dataset(self, tmp_path: Path) -> None:
        """Validate load_dataset with a JSONL file returns parsed records."""
        path = tmp_path / "data.jsonl"
        records = [{"q": "Q1", "a": "A1"}, {"q": "Q2", "a": "A2"}]
        path.write_text(
            "\n".join(json.dumps(r) for r in records),
            encoding="utf-8",
        )

        data = load_dataset(str(path))
        assert len(data) == 2, f"Expected 2 records, got {len(data)}"
        assert data == records, f"Loaded data does not match: {data}"

    def test_load_csv_dataset(self, tmp_path: Path) -> None:
        """Validate load_dataset with a CSV file returns parsed rows."""
        path = tmp_path / "data.csv"
        path.write_text("name,age\nAlice,30\nBob,25\n", encoding="utf-8")

        data = load_dataset(str(path))
        assert len(data) == 2, f"Expected 2 rows, got {len(data)}"
        assert data[0] == {"name": "Alice", "age": "30"}, f"Row 0 mismatch: {data[0]}"

    def test_load_tsv_dataset(self, tmp_path: Path) -> None:
        """Validate load_dataset with a TSV file returns parsed rows."""
        path = tmp_path / "data.tsv"
        path.write_text("col1\tcol2\nval1\tval2\n", encoding="utf-8")

        data = load_dataset(str(path))
        assert len(data) == 1, f"Expected 1 row, got {len(data)}"
        assert data[0] == {"col1": "val1", "col2": "val2"}, f"Row 0 mismatch: {data[0]}"

    def test_load_dataset_with_limit(self, tmp_path: Path) -> None:
        """Validate load_dataset respects the limit parameter end-to-end."""
        path = tmp_path / "data.jsonl"
        records = [{"i": i} for i in range(20)]
        path.write_text(
            "\n".join(json.dumps(r) for r in records),
            encoding="utf-8",
        )

        data = load_dataset(str(path), limit=5)
        assert len(data) == 5, f"Expected 5 records with limit=5, got {len(data)}"
        assert data == records[:5], f"Expected first 5 records, got {data}"

    def test_load_dataset_nonexistent_file_raises(self) -> None:
        """Validate load_dataset raises ValueError for nonexistent files.

        The LocalFetcher.supports() returns False for missing files,
        so get_fetcher_for_uri raises ValueError (no fetcher found).
        """
        with pytest.raises(ValueError, match="No fetcher supports URI"):
            load_dataset("/nonexistent/path/to/data.jsonl")

    def test_load_dataset_unsupported_uri_raises(self) -> None:
        """Validate load_dataset raises ValueError for unsupported URI schemes."""
        with pytest.raises(ValueError, match="No fetcher supports URI"):
            load_dataset("s3://my-bucket/data.jsonl")

    def test_load_dataset_with_custom_fetcher(self, tmp_path: Path) -> None:
        """Validate load_dataset works end-to-end with a custom fetcher."""
        # Create a real backing file
        backing_file = tmp_path / "backing.jsonl"
        records = [{"source": "custom", "id": 1}, {"source": "custom", "id": 2}]
        backing_file.write_text(
            "\n".join(json.dumps(r) for r in records),
            encoding="utf-8",
        )

        class MockRemoteFetcher:
            def supports(self, uri: str) -> bool:
                return uri.startswith("mock://")

            def fetch(self, uri: str, cache_dir: Optional[Path] = None) -> FetchResult:
                return FetchResult(
                    local_path=backing_file,
                    format="jsonl",
                    metadata={"source": "mock-remote"},
                )

        register_fetcher(MockRemoteFetcher())

        data = load_dataset("mock://my-dataset")
        assert len(data) == 2, f"Expected 2 records, got {len(data)}"
        assert data == records, f"Custom fetcher data mismatch: {data}"

    def test_load_dataset_json_extension_treated_as_jsonl(self, tmp_path: Path) -> None:
        """Validate .json files are treated as JSONL (line-delimited JSON)."""
        path = tmp_path / "data.json"
        records = [{"k": "v1"}, {"k": "v2"}]
        path.write_text(
            "\n".join(json.dumps(r) for r in records),
            encoding="utf-8",
        )

        data = load_dataset(str(path))
        assert len(data) == 2, f"Expected 2 records from .json file, got {len(data)}"
        assert data == records, f"Data mismatch: {data}"


# ---------------------------------------------------------------------------
# TestHuggingFaceFetcher
# ---------------------------------------------------------------------------


class TestHuggingFaceFetcher:
    """Tests for the HuggingFaceFetcher class."""

    def test_supports_hf_prefix(self) -> None:
        """Validate supports() returns True for URIs with hf:// prefix."""
        fetcher = HuggingFaceFetcher()
        assert fetcher.supports("hf://my_dataset") is True, (
            "Expected supports('hf://my_dataset') to be True"
        )

    def test_not_supports_other_prefix(self) -> None:
        """Validate supports() returns False for non-hf:// URIs."""
        fetcher = HuggingFaceFetcher()
        assert fetcher.supports("/local/path") is False, (
            "Expected supports('/local/path') to be False"
        )
        assert fetcher.supports("s3://bucket/data.jsonl") is False, (
            "Expected supports('s3://...') to be False"
        )
        assert fetcher.supports("https://example.com/data") is False, (
            "Expected supports('https://...') to be False"
        )

    def test_import_error_when_datasets_missing(self) -> None:
        """Validate clear ImportError when 'datasets' package is not installed."""
        fetcher = HuggingFaceFetcher()
        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "datasets":
                raise ImportError("No module named 'datasets'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(ImportError, match="datasets"):
                fetcher.fetch("hf://test/dataset")

    def test_uri_parsing(self) -> None:
        """Validate _parse_uri with various URI patterns returns correct tuples."""
        # hf://org/dataset/config/split -> (org/dataset, config, split)
        result = HuggingFaceFetcher._parse_uri("hf://org/dataset/config/split")
        assert result == ("org/dataset", "config", "split"), (
            f"Expected ('org/dataset', 'config', 'split'), got {result}"
        )

        # hf://org/dataset/config -> (org/dataset, config, None)
        result = HuggingFaceFetcher._parse_uri("hf://org/dataset/config")
        assert result == ("org/dataset", "config", None), (
            f"Expected ('org/dataset', 'config', None), got {result}"
        )

        # hf://org/dataset -> (org/dataset, None, None)
        result = HuggingFaceFetcher._parse_uri("hf://org/dataset")
        assert result == ("org/dataset", None, None), (
            f"Expected ('org/dataset', None, None), got {result}"
        )

        # hf://single_name -> (single_name, None, None)
        result = HuggingFaceFetcher._parse_uri("hf://single_name")
        assert result == ("single_name", None, None), (
            f"Expected ('single_name', None, None), got {result}"
        )


# ---------------------------------------------------------------------------
# TestFieldRemapping
# ---------------------------------------------------------------------------


class TestFieldRemapping:
    """Tests for field remapping in dataset loading."""

    def test_basic_remap(self) -> None:
        """Validate basic field remapping with {old_key: new_key}."""
        records = [
            {"old_key": "value1", "other": "keep"},
            {"old_key": "value2", "other": "keep2"},
        ]
        result = _remap_fields(records, {"old_key": "new_key"})

        assert len(result) == 2, f"Expected 2 records, got {len(result)}"
        assert "new_key" in result[0], (
            f"Expected 'new_key' in remapped record, got keys {list(result[0].keys())}"
        )
        assert "old_key" not in result[0], (
            "Expected 'old_key' to be renamed in remapped record"
        )
        assert result[0]["new_key"] == "value1", (
            f"Expected new_key='value1', got {result[0]['new_key']!r}"
        )
        assert result[0]["other"] == "keep", (
            f"Expected unmapped key 'other' to be passed through, got {result[0]['other']!r}"
        )

    def test_no_mapping_passthrough(self, tmp_path: Path) -> None:
        """Validate field_mapping=None returns data unchanged via load_dataset."""
        path = tmp_path / "data.jsonl"
        records = [{"question": "Q1", "answer": "A1"}]
        path.write_text(json.dumps(records[0]), encoding="utf-8")

        data = load_dataset(str(path), field_mapping=None)
        assert data == records, (
            f"Expected data unchanged when field_mapping=None, got {data}"
        )

    def test_missing_source_key_ignored(self) -> None:
        """Validate mapping referencing a key not in data is silently ignored."""
        records = [{"existing": "value"}]
        result = _remap_fields(records, {"nonexistent": "new_name"})

        assert len(result) == 1, f"Expected 1 record, got {len(result)}"
        assert "existing" in result[0], "Expected 'existing' key to be preserved"
        assert "new_name" not in result[0], (
            "Expected 'new_name' not in result when source key is missing"
        )
        assert "nonexistent" not in result[0], "Expected 'nonexistent' not in result"

    def test_remap_integration(self, tmp_path: Path) -> None:
        """Validate load_dataset with field_mapping parameter end-to-end."""
        path = tmp_path / "data.jsonl"
        records = [
            {"src_question": "What?", "src_answer": "Yes"},
            {"src_question": "How?", "src_answer": "No"},
        ]
        path.write_text(
            "\n".join(json.dumps(r) for r in records),
            encoding="utf-8",
        )

        data = load_dataset(
            str(path),
            field_mapping={"src_question": "question", "src_answer": "answer"},
        )

        assert len(data) == 2, f"Expected 2 records, got {len(data)}"
        assert "question" in data[0], (
            f"Expected 'question' key after remapping, got keys {list(data[0].keys())}"
        )
        assert "answer" in data[0], (
            f"Expected 'answer' key after remapping, got keys {list(data[0].keys())}"
        )
        assert data[0]["question"] == "What?", (
            f"Expected question='What?', got {data[0]['question']!r}"
        )
        assert data[1]["answer"] == "No", (
            f"Expected answer='No', got {data[1]['answer']!r}"
        )
        # Original keys should be gone
        assert "src_question" not in data[0], (
            "Expected 'src_question' to be renamed, not present"
        )
        assert "src_answer" not in data[0], (
            "Expected 'src_answer' to be renamed, not present"
        )
