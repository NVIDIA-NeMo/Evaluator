"""Dataset loading and template utilities for evaluator SDK runtime."""

from nemo_evaluator.sdk.datasets.loader import (
    DatasetLoadError,
    discover_files,
    is_glob_pattern,
    load_dataset,
    load_dataset_as_dicts,
    load_file,
)
from nemo_evaluator.sdk.templates import (
    render_request,
    render_template,
)

__all__ = [
    "DatasetLoadError",
    "is_glob_pattern",
    "discover_files",
    "load_file",
    "load_dataset",
    "load_dataset_as_dicts",
    "render_request",
    "render_template",
]
