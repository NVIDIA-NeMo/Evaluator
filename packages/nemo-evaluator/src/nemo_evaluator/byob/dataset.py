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

"""Dataset fetching and loading for BYOB benchmarks.

Two-layer architecture:
  Fetcher: URI -> local file path (handles downloading, caching)
  Loader:  local file path -> List[Dict] (handles parsing)

The fetcher registry is extensible -- users can register custom fetchers
for S3, GCS, HuggingFace, or any other data source.

Example::

    from nemo_evaluator.byob.dataset import load_dataset, register_fetcher

    # Load from local JSONL (default)
    data = load_dataset("data.jsonl")

    # Load from CSV
    data = load_dataset("data.csv")

    # Register a custom fetcher
    register_fetcher(MyS3Fetcher())
    data = load_dataset("s3://bucket/data.jsonl")
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@dataclass
class FetchResult:
    """Result of fetching a dataset."""

    local_path: Path
    format: str  # "jsonl", "csv", "tsv"
    metadata: Dict = field(default_factory=dict)


@runtime_checkable
class DatasetFetcher(Protocol):
    """Protocol for dataset fetchers.

    A fetcher resolves a URI to a local file path. It does NOT parse
    the file -- that's the loader's job.

    Implement this protocol to add support for new data sources
    (S3, GCS, HuggingFace, HTTP URLs, etc.).
    """

    def supports(self, uri: str) -> bool:
        """Return True if this fetcher can handle the given URI."""
        ...

    def fetch(self, uri: str, cache_dir: Optional[Path] = None) -> FetchResult:
        """Fetch the dataset and return a local path + detected format."""
        ...


class LocalFetcher:
    """Fetcher for local filesystem paths.

    This is a no-op fetcher -- it validates that the file exists and
    detects its format from the file extension.
    """

    def supports(self, uri: str) -> bool:
        """Support any URI that points to an existing local file."""
        return Path(uri).exists()

    def fetch(self, uri: str, cache_dir: Optional[Path] = None) -> FetchResult:
        """Validate file exists and detect format."""
        path = Path(uri)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {uri}")
        if not path.is_file():
            raise ValueError(f"Dataset path is not a file: {uri}")
        fmt = _detect_format(path)
        return FetchResult(local_path=path, format=fmt, metadata={"source": "local"})


# --- Format detection ---


def _detect_format(path: Path) -> str:
    """Detect dataset format from file extension."""
    suffix = path.suffix.lower()
    format_map = {
        ".jsonl": "jsonl",
        ".json": "jsonl",  # treat .json as JSONL (line-delimited)
        ".csv": "csv",
        ".tsv": "tsv",
    }
    return format_map.get(suffix, "jsonl")  # default to JSONL


# --- Format-specific loaders ---


def load_jsonl(path: Path, limit: Optional[int] = None) -> List[Dict]:
    """Load a JSONL file into a list of dicts.

    Args:
        path: Path to JSONL file.
        limit: Optional limit on number of records to load.

    Returns:
        List of parsed dictionaries.

    Raises:
        json.JSONDecodeError: On malformed JSON lines.
        ValueError: If a parsed line is not a dict.
    """
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parsed = json.loads(line)
            if not isinstance(parsed, dict):
                raise ValueError(
                    f"Line {line_num} in {path} is not a JSON object: {type(parsed).__name__}"
                )
            data.append(parsed)
            if limit and len(data) >= limit:
                break
    return data


def load_csv(path: Path, delimiter: str = ",", limit: Optional[int] = None) -> List[Dict]:
    """Load a CSV/TSV file into a list of dicts.

    Uses csv.DictReader so the first row is treated as column headers.

    Args:
        path: Path to CSV/TSV file.
        delimiter: Column delimiter ("," for CSV, "\t" for TSV).
        limit: Optional limit on number of records to load.

    Returns:
        List of dictionaries (one per row, keyed by header names).
    """
    data = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            data.append(dict(row))
            if limit and len(data) >= limit:
                break
    return data


# Loader dispatch table: format string -> loader function
_LOADERS: Dict[str, Callable[..., List[Dict]]] = {
    "jsonl": load_jsonl,
    "csv": lambda path, limit=None: load_csv(path, delimiter=",", limit=limit),
    "tsv": lambda path, limit=None: load_csv(path, delimiter="\t", limit=limit),
}


# --- Fetcher registry ---

_FETCHER_REGISTRY: List[DatasetFetcher] = [LocalFetcher()]


def register_fetcher(fetcher: DatasetFetcher) -> None:
    """Register a custom dataset fetcher.

    Newly registered fetchers are checked first (LIFO order).
    This allows users to override default fetchers or add support
    for new data sources.

    Args:
        fetcher: A DatasetFetcher implementation.
    """
    _FETCHER_REGISTRY.insert(0, fetcher)


def get_fetcher_for_uri(uri: str) -> DatasetFetcher:
    """Find a fetcher that supports the given URI.

    Args:
        uri: Dataset URI to resolve.

    Returns:
        The first fetcher that supports this URI.

    Raises:
        ValueError: If no fetcher supports the URI.
    """
    for fetcher in _FETCHER_REGISTRY:
        if fetcher.supports(uri):
            return fetcher
    raise ValueError(
        f"No fetcher supports URI: {uri}. "
        f"Registered fetchers: {[type(f).__name__ for f in _FETCHER_REGISTRY]}"
    )


# --- High-level API ---


def load_dataset(uri: str, limit: Optional[int] = None) -> List[Dict]:
    """Load a dataset from any supported source.

    This is the main entry point for dataset loading. It:
    1. Finds a fetcher that supports the URI
    2. Fetches the data (downloads if remote, validates if local)
    3. Detects the format
    4. Loads and parses the data

    Args:
        uri: Dataset URI. Can be a local file path, or any URI
             supported by registered fetchers.
        limit: Optional limit on number of records to load.
              Applied during loading (not after), so large files
              don't need to be fully read into memory.

    Returns:
        List of sample dictionaries.

    Raises:
        ValueError: If no fetcher supports the URI or no loader for the format.
        FileNotFoundError: If the dataset file doesn't exist.
    """
    fetcher = get_fetcher_for_uri(uri)
    result = fetcher.fetch(uri)
    logger.info(
        "Fetched dataset: %s (format=%s, source=%s)",
        result.local_path,
        result.format,
        result.metadata.get("source", "unknown"),
    )

    loader = _LOADERS.get(result.format)
    if not loader:
        raise ValueError(
            f"No loader for format '{result.format}'. "
            f"Supported formats: {list(_LOADERS.keys())}"
        )

    data = loader(result.local_path, limit=limit)
    logger.info("Loaded %d records from %s", len(data), result.local_path)
    return data
