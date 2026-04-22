"""Dataset loader module for evaluator SDK runtime."""

# Migrated from: services/evaluator/src/nmp/evaluator/app/datasets/loader.py

import gzip
import io
import json
import logging
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.csv as pa_csv
import pyarrow.feather as pa_feather
import pyarrow.json as pa_json
import pyarrow.parquet as pa_parquet

from nemo_evaluator.sdk.values.datasets import DatasetInput, DatasetRows

_log = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {
    ".json",
    ".jsonl",
    ".csv",
    ".parquet",
    ".feather",
    ".arrow",
    ".orc",
}

COMPRESSION_EXTENSIONS = {".gz", ".gzip"}


class DatasetLoadError(Exception):
    """Raised when a dataset cannot be loaded."""


def rows_from_dataset(dataset: DatasetInput) -> list[dict[str, Any]]:
    """Convert supported dataset containers into a list of row dictionaries.

    Args:
        dataset: Materialized dataset payload or container to normalize.

    Returns:
        The dataset rows as plain dictionaries.

    Raises:
        TypeError: If the dataset container type is unsupported.
    """
    if isinstance(dataset, DatasetRows):
        return dataset.rows
    if isinstance(dataset, list):
        return dataset
    if isinstance(dataset, pa.Table):
        return dataset.to_pylist()
    raise TypeError(f"Unsupported dataset type: {type(dataset).__name__}")


def normalize_dataset(
    dataset: DatasetInput | str | Path,
    pattern: str | None,
) -> DatasetInput:
    """Normalize dataset inputs into an in-memory evaluation payload.

    Args:
        dataset: Dataset input accepted by evaluator execution.
        pattern: Optional file selector used when ``dataset`` points to a directory.

    Returns:
        A materialized dataset payload suitable for row extraction.

    Raises:
        TypeError: If the dataset input type is unsupported.
        FileNotFoundError: If the provided dataset path does not exist.
        ValueError: If ``pattern`` is used with a file path instead of a directory.
    """
    if isinstance(dataset, (DatasetRows, pa.Table, list)):
        return dataset
    if not isinstance(dataset, (str, Path)):
        raise TypeError(f"Unsupported dataset type: {type(dataset).__name__}")

    path = Path(dataset)
    if not path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {path}")

    if path.is_dir():
        return load_dataset_as_dicts(path, pattern)

    if pattern is not None:
        raise ValueError("pattern can only be used when dataset points to a directory")

    return load_dataset_as_dicts(path.parent, path.name)


def prepare_dataset_rows(
    dataset: DatasetInput | str | Path,
    pattern: str | None,
    max_size: int | None,
) -> list[dict[str, Any]]:
    """Materialize one dataset input into row dictionaries with optional truncation.

    Args:
        dataset: Dataset input accepted by evaluator execution.
        pattern: Optional file selector used when ``dataset`` points to a directory.
        max_size: Optional maximum number of rows to retain from the materialized dataset.

    Returns:
        The prepared row dictionaries used by local metric execution.
    """
    rows = rows_from_dataset(normalize_dataset(dataset, pattern))
    if max_size is not None:
        rows = rows[:max_size]
    return rows


def is_glob_pattern(pattern: str) -> bool:
    """Check whether a file pattern contains glob metacharacters.

    Args:
        pattern: User-provided file selector.

    Returns:
        ``True`` when ``pattern`` should be interpreted as a glob.
    """
    glob_chars = {"*", "?", "[", "]"}
    return any(c in pattern for c in glob_chars)


def _is_glob_pattern(pattern: str) -> bool:
    """Backward-compatible alias for :func:`is_glob_pattern`."""
    return is_glob_pattern(pattern)


def discover_files(base_path: Path, pattern: str | None) -> list[Path]:
    """Resolve dataset files under a base path.

    Args:
        base_path: Directory that contains dataset files.
        pattern: Optional explicit file name or glob pattern.

    Returns:
        List of discovered files.

    Raises:
        DatasetLoadError: If files cannot be found or selected paths are invalid.
    """
    if not base_path.exists():
        raise DatasetLoadError(f"Dataset directory not found: {base_path}")

    if pattern is None:
        files = [f for f in base_path.rglob("*") if f.is_file()]
        if not files:
            raise DatasetLoadError(f"No files found in {base_path}")
        return files

    if is_glob_pattern(pattern):
        files = list(base_path.glob(pattern))
        if not files:
            raise DatasetLoadError(f"No files found matching pattern '{pattern}' in {base_path}")
        return [f for f in files if f.is_file()]

    file_path = base_path / pattern
    if not file_path.exists():
        raise DatasetLoadError(f"File not found: {file_path}")
    if not file_path.is_file():
        raise DatasetLoadError(f"Path is not a file: {file_path}")
    return [file_path]


def _discover_files(base_path: Path, pattern: str | None) -> list[Path]:
    """Backward-compatible alias for :func:`discover_files`."""
    return discover_files(base_path, pattern)


def _get_file_format(path: Path) -> str | None:
    """Infer logical dataset format from a file path.

    For compressed paths such as ``data.jsonl.gz``, this function inspects the
    extension before the compression suffix.

    Args:
        path: Candidate dataset file path.

    Returns:
        Supported data extension (for example ``.jsonl``), or ``None``.
    """
    suffixes = path.suffixes
    if suffixes and suffixes[-1].lower() in COMPRESSION_EXTENSIONS:
        if len(suffixes) < 2:
            return None
        data_ext = suffixes[-2].lower()
    else:
        data_ext = path.suffix.lower() if path.suffix else None

    if data_ext in SUPPORTED_EXTENSIONS:
        return data_ext
    return None


def _is_compressed(path: Path) -> bool:
    """Check whether a file path uses a supported gzip extension.

    Args:
        path: Candidate dataset file path.

    Returns:
        ``True`` when the file suffix indicates gzip compression.
    """
    return path.suffix.lower() in COMPRESSION_EXTENSIONS


def _load_json_file(source: Path | io.BytesIO) -> pa.Table:
    """Load JSON content as a PyArrow table.

    The loader supports both JSON arrays and JSONL streams. Arrays are parsed
    via ``json.loads`` and converted with ``Table.from_pylist``; line-delimited
    JSON is delegated to ``pyarrow.json.read_json``.

    Args:
        source: JSON source as filesystem path or in-memory bytes buffer.

    Returns:
        Parsed rows as a ``pyarrow.Table``.

    Raises:
        ValueError: If JSON array parsing produces a non-list payload.
        UnicodeDecodeError: If bytes cannot be decoded as UTF-8 text.
        json.JSONDecodeError: If JSON content is invalid.
    """
    if isinstance(source, Path):
        content = source.read_bytes()
    else:
        content = source.read()
        source.seek(0)

    text = content.decode("utf-8").strip()
    if text.startswith("["):
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("Expected JSON array")
        return pa.Table.from_pylist(data)

    if isinstance(source, io.BytesIO):
        source.seek(0)
        return pa_json.read_json(source)
    return pa_json.read_json(source)


def _load_content(source: Path | io.BytesIO, file_format: str) -> pa.Table:
    """Load tabular content for a known file format.

    Args:
        source: File path or in-memory bytes buffer.
        file_format: Normalized format extension returned by ``_get_file_format``.

    Returns:
        Parsed table for the given input format.

    Raises:
        ValueError: If the format is not supported.
    """
    if file_format in (".json", ".jsonl"):
        return _load_json_file(source)
    if file_format == ".csv":
        return pa_csv.read_csv(source)
    if file_format == ".parquet":
        return pa_parquet.read_table(source)
    if file_format in (".feather", ".arrow"):
        return pa_feather.read_table(source)
    if file_format == ".orc":
        import pyarrow.orc as pa_orc

        return pa_orc.read_table(source)
    raise ValueError(f"Unsupported format: {file_format}")


def load_file(path: Path) -> pa.Table | None:
    """Load one file into a table, returning ``None`` for skipped/failed files.

    Args:
        path: Dataset file path.

    Returns:
        Parsed table, or ``None`` when format is unsupported or parsing fails.
    """
    file_format = _get_file_format(path)
    if file_format is None:
        _log.debug("Skipping unsupported file format", extra={"path": str(path)})
        return None

    try:
        if _is_compressed(path):
            with gzip.open(path, "rb") as gz_file:
                content = gz_file.read()
            return _load_content(io.BytesIO(content), file_format)
        return _load_content(path, file_format)
    except Exception as e:  # pragma: no cover - defensive logging path
        _log.warning("Failed to load file", extra={"path": str(path), "error": str(e)})
        return None


def _load_file(path: Path) -> pa.Table | None:
    """Backward-compatible alias for :func:`load_file`."""
    return load_file(path)


def load_dataset(base_path: Path, pattern: str | None) -> pa.Table:
    """Load matching dataset files and concatenate them into one table.

    Args:
        base_path: Directory containing dataset files.
        pattern: Optional file name or glob pattern.

    Returns:
        Single table containing rows from all successfully loaded files.

    Raises:
        DatasetLoadError: If no files are discovered or no data can be loaded.
    """
    files = discover_files(base_path, pattern)

    tables: list[pa.Table] = []
    for file_path in files:
        table = load_file(file_path)
        if table is not None:
            tables.append(table)
            _log.debug("Loaded file", extra={"file": file_path.name, "rows": table.num_rows})

    if not tables:
        raise DatasetLoadError(f"No data could be loaded from {base_path} (pattern: {pattern})")

    if len(tables) == 1:
        return tables[0]
    return pa.concat_tables(tables, promote_options="default")


def load_dataset_as_dicts(base_path: Path, pattern: str | None) -> list[dict]:
    """Load a dataset and return rows as dictionaries.

    Args:
        base_path: Directory containing dataset files.
        pattern: Optional file name or glob pattern.

    Returns:
        List of row dictionaries ready for metric execution.
    """
    try:
        table = load_dataset(base_path, pattern)
        return table.to_pylist()
    except DatasetLoadError as e:
        rows = _load_json_from_fileset_path_as_dicts_fallback(Path(base_path), pattern)
        if rows:
            _log.warning(
                "Falling back to Python JSON loader for dataset path due to pyarrow parsing failure",
                extra={"base_path": str(base_path), "pattern": pattern, "rows": len(rows), "error": str(e)},
            )
            return rows
        raise


def _load_json_file_as_dicts(path: Path) -> list[dict]:
    """Load JSON/JSONL files as list[dict] without pyarrow schema inference."""
    if _is_compressed(path):
        with gzip.open(path, "rb") as gz_file:
            text = gz_file.read().decode("utf-8")
    else:
        text = path.read_text(encoding="utf-8")

    stripped = text.lstrip()
    if stripped.startswith("["):
        # startswith("[") means content is JSON array (not JSONL).
        data = json.loads(stripped)
        if isinstance(data, list):
            return [row for row in data if isinstance(row, dict)]
        if isinstance(data, dict):
            return [data]
        return []

    rows: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _load_json_from_fileset_path_as_dicts_fallback(
    fileset_path: Path,
    pattern: str | None = None,
) -> list[dict]:
    """Fallback loader for JSON/JSONL when pyarrow table loading fails.

    Supports .json, .jsonl, and .jsonl.gz (detected as .jsonl by _get_file_format).
    """

    if not fileset_path.exists():
        raise DatasetLoadError(f"Fileset directory not found: {fileset_path}")

    files = _discover_files(fileset_path, pattern)
    rows: list[dict] = []
    for file_path in files:
        file_format = _get_file_format(file_path)

        if file_format not in {".json", ".jsonl"}:
            continue
        try:
            rows.extend(_load_json_file_as_dicts(file_path))
        except Exception as e:
            _log.warning(
                "Fallback JSON loader failed for file",
                extra={"path": str(file_path), "error": str(e)},
            )
    return rows
