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

    from nemo_evaluator.contrib.byob.dataset import load_dataset, register_fetcher

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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Protocol, runtime_checkable

from nemo_evaluator.logging import get_logger

logger = get_logger(__name__)


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


class HuggingFaceFetcher:
    """Fetcher for HuggingFace datasets using ``hf://`` URIs.

    URI format::

        hf://dataset_name
        hf://dataset_name/config
        hf://dataset_name/config/split

    When *config* or *split* are omitted the HuggingFace ``datasets`` library
    defaults are used (typically ``"default"`` config and ``"test"`` split).

    The ``datasets`` package is imported lazily so it remains an optional
    dependency -- only users who actually reference ``hf://`` URIs need it
    installed.

    This fetcher is **not** auto-registered.  To enable it call::

        from nemo_evaluator.contrib.byob.dataset import register_fetcher
        register_fetcher(HuggingFaceFetcher())

    Args:
        default_cache_dir: Directory for downloaded/converted files.
            Defaults to ``~/.cache/nemo_evaluator/hf_datasets/``.
    """

    _HF_PREFIX = "hf://"

    def __init__(
        self,
        default_cache_dir: Optional[Path] = None,
    ) -> None:
        self._default_cache_dir = default_cache_dir or Path.home() / ".cache" / "nemo_evaluator" / "hf_datasets"

    # -- protocol methods --------------------------------------------------

    def supports(self, uri: str) -> bool:
        """Return True for URIs starting with ``hf://``."""
        return uri.startswith(self._HF_PREFIX)

    def fetch(self, uri: str, cache_dir: Optional[Path] = None) -> FetchResult:
        """Download the HuggingFace dataset and convert it to JSONL.

        Args:
            uri: A ``hf://dataset_name[/config[/split]]`` URI.
            cache_dir: Override for the download / conversion cache directory.

        Returns:
            FetchResult pointing to the generated JSONL file.

        Raises:
            ImportError: If the ``datasets`` library is not installed.
            ValueError: If the URI cannot be parsed.
        """
        try:
            import datasets as hf_datasets  # noqa: F811 – lazy import
        except ImportError as exc:
            raise ImportError(
                "The 'datasets' package is required to fetch HuggingFace datasets. "
                "Install it with:  pip install datasets"
            ) from exc

        dataset_name, config, split = self._parse_uri(uri)
        cache = cache_dir or self._default_cache_dir
        cache.mkdir(parents=True, exist_ok=True)

        # Build a deterministic filename so repeated fetches are cached.
        safe_name = dataset_name.replace("/", "__")
        parts = [safe_name]
        if config is not None:
            parts.append(config)
        if split is not None:
            parts.append(split)
        out_path = cache / ("_".join(parts) + ".jsonl")

        if out_path.exists():
            logger.info("HuggingFace dataset already cached", path=str(out_path))
            return FetchResult(
                local_path=out_path,
                format="jsonl",
                metadata={
                    "source": "huggingface",
                    "dataset": dataset_name,
                    "config": config,
                    "split": split,
                },
            )

        logger.info(
            "Downloading HuggingFace dataset",
            dataset=dataset_name, config=config, split=split,
        )

        load_kwargs: Dict[str, str] = {}
        if config is not None:
            load_kwargs["name"] = config
        if split is not None:
            load_kwargs["split"] = split

        ds = hf_datasets.load_dataset(dataset_name, **load_kwargs)

        # If no split was requested, ``load_dataset`` returns a DatasetDict.
        # In that case take the first available split.
        if isinstance(ds, hf_datasets.DatasetDict):
            first_split = next(iter(ds))
            logger.info("No split specified; using first available split", split=first_split)
            ds = ds[first_split]

        # Write out as JSONL.
        with open(out_path, "w", encoding="utf-8") as fout:
            for record in ds:
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info("Converted HuggingFace dataset to JSONL", path=str(out_path), records=len(ds))

        return FetchResult(
            local_path=out_path,
            format="jsonl",
            metadata={
                "source": "huggingface",
                "dataset": dataset_name,
                "config": config,
                "split": split,
            },
        )

    # -- internal helpers --------------------------------------------------

    @staticmethod
    def _parse_uri(uri: str) -> tuple:
        """Parse ``hf://dataset_name[/config[/split]][?split=X]``.

        Supports two ways to specify the split:

        - **Query parameter** (recommended for namespaced datasets)::

              hf://google/boolq?split=validation
              hf://CohereForAI/Global-MMLU-Lite/en?split=test

        - **Path segments** (positional)::

              hf://dataset_name                         -> (name, None, None)
              hf://namespace/dataset                    -> (ns/name, None, None)
              hf://namespace/dataset/config             -> (ns/name, config, None)
              hf://namespace/dataset/config/split       -> (ns/name, config, split)

        Returns:
            A 3-tuple ``(dataset_name, config_or_None, split_or_None)``.

        Raises:
            ValueError: If the URI is empty after the ``hf://`` prefix.
        """
        body = uri[len(HuggingFaceFetcher._HF_PREFIX) :]
        if not body:
            raise ValueError(f"Invalid HuggingFace URI (empty body): {uri}")

        # Extract ?split=X query parameter if present
        query_split = None
        if "?" in body:
            body, query = body.split("?", 1)
            for param in query.split("&"):
                if param.startswith("split="):
                    query_split = param[len("split="):]

        segments = [s for s in body.split("/") if s]
        if not segments:
            raise ValueError(f"Invalid HuggingFace URI (empty body): {uri}")

        if len(segments) == 1:
            return (segments[0], None, query_split)
        elif len(segments) == 2:
            return ("/".join(segments[:2]), None, query_split)
        elif len(segments) == 3:
            return ("/".join(segments[:2]), segments[2], query_split)
        elif len(segments) >= 4:
            return ("/".join(segments[:2]), segments[2], query_split or segments[3])

        return (body, None, query_split)  # pragma: no cover


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

_FETCHER_REGISTRY: List[DatasetFetcher] = [LocalFetcher(), HuggingFaceFetcher()]


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


# --- Field remapping ---


def _remap_fields(records: List[Dict], field_mapping: Dict[str, str]) -> List[Dict]:
    """Rename keys in each record according to *field_mapping*.

    Args:
        records: List of dictionaries (dataset rows).
        field_mapping: Mapping of ``{source_key: target_key}``.
            Keys present in a record are renamed; keys absent from a
            record are silently ignored.  Keys not listed in the mapping
            are passed through unchanged.

    Returns:
        A new list of dictionaries with renamed keys.
    """
    remapped: List[Dict] = []
    for record in records:
        new_record: Dict = {}
        for key, value in record.items():
            new_key = field_mapping.get(key, key)
            new_record[new_key] = value
        remapped.append(new_record)
    return remapped


# --- High-level API ---


def load_dataset(
    uri: str,
    limit: Optional[int] = None,
    field_mapping: Optional[Dict[str, str]] = None,
) -> List[Dict]:
    """Load a dataset from any supported source.

    This is the main entry point for dataset loading. It:
    1. Finds a fetcher that supports the URI
    2. Fetches the data (downloads if remote, validates if local)
    3. Detects the format
    4. Loads and parses the data
    5. Optionally remaps field names

    Args:
        uri: Dataset URI. Can be a local file path, or any URI
             supported by registered fetchers.
        limit: Optional limit on number of records to load.
              Applied during loading (not after), so large files
              don't need to be fully read into memory.
        field_mapping: Optional mapping of ``{source_key: target_key}``
              used to rename fields in every loaded record.  Keys not
              present in the mapping are preserved unchanged.

    Returns:
        List of sample dictionaries.

    Raises:
        ValueError: If no fetcher supports the URI or no loader for the format.
        FileNotFoundError: If the dataset file doesn't exist.
    """
    fetcher = get_fetcher_for_uri(uri)
    result = fetcher.fetch(uri)
    logger.info(
        "Fetched dataset",
        path=str(result.local_path), format=result.format,
        source=result.metadata.get("source", "unknown"),
    )

    loader = _LOADERS.get(result.format)
    if not loader:
        raise ValueError(
            f"No loader for format '{result.format}'. "
            f"Supported formats: {list(_LOADERS.keys())}"
        )

    data = loader(result.local_path, limit=limit)
    logger.info("Loaded dataset records", records=len(data), path=str(result.local_path))

    if field_mapping is not None:
        data = _remap_fields(data, field_mapping)
        logger.info("Applied field remapping", field_mapping=field_mapping)

    return data
