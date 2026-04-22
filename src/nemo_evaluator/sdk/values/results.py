"""Result types for evaluator SDK runtime."""

# Migrated from: services/evaluator/src/nmp/evaluator/app/values/results.py
# Migrated from: services/evaluator/src/nmp/evaluator/app/values/scores.py

from __future__ import annotations

import json
import math
from typing import Any, Literal, Self

import pyarrow as pa
from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator, model_serializer

ResultView = Literal["rows", "aggregate"]
AggregateFieldName = Literal[
    # Base statistics
    "nan_count",
    "sum",
    "mean",
    "min",
    "max",
    "std_dev",
    "variance",
    # Range-specific fields
    "score_type",
    "percentiles",
    "histogram",
    # Rubric-specific fields
    "rubric_distribution",
    "mode_category",
]
DefaultAggregateFieldName = Literal["nan_count", "sum", "mean", "min", "max"]


def flatten_dict(prefix: str, value: Any, output: dict[str, Any]) -> None:
    """Flatten nested dictionaries into dot-delimited key/value pairs.

    Args:
        prefix: Current key path prefix.
        value: Value to flatten (possibly nested dictionary).
        output: Mutable destination mapping populated in place.

    Returns:
        ``None``. ``output`` is mutated with flattened entries.
    """
    # Flatten nested payloads into dot-separated columns so row views round-trip
    # cleanly into tables and data frames.
    if isinstance(value, dict):
        for key, nested_value in value.items():
            nested_prefix = f"{prefix}.{key}" if prefix else str(key)
            flatten_dict(nested_prefix, nested_value, output)
        return

    output[prefix] = value


def serialize_value(value: Any) -> Any:
    """Recursively convert SDK values into JSON-serializable primitives.

    Args:
        value: Value to serialize.

    Returns:
        JSON-compatible representation of ``value``.
    """
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, list):
        return [serialize_value(item) for item in value]
    if isinstance(value, dict):
        return {key: serialize_value(item) for key, item in value.items()}
    return value


def _truncate(value: Any, max_length: int = 40) -> str:
    """Render values as bounded-width strings for text-table output.

    Args:
        value: Value to render as text.
        max_length: Maximum output length before ellipsis truncation.

    Returns:
        Truncated or original string representation.
    """
    text = str(value)
    if len(text) <= max_length:
        return text
    return f"{text[: max_length - 3]}..."


def format_table(records: list[dict[str, Any]]) -> str:
    """Render flat records as an aligned ASCII table.

    The formatter derives a stable column order from first appearance, computes
    per-column widths, and truncates long cell values for compact output.

    Args:
        records: Flat dictionaries to print.

    Returns:
        Multi-line table string.
    """
    if not records:
        return "(no rows)"

    seen: set[str] = set()
    columns: list[str] = []
    for record in records:
        for key in record:
            if key not in seen:
                seen.add(key)
                columns.append(key)

    widths: dict[str, int] = {}
    for column in columns:
        widths[column] = len(column)
        for record in records:
            widths[column] = max(widths[column], len(_truncate(record.get(column, ""))))

    header = " | ".join(column.ljust(widths[column]) for column in columns)
    divider = "-+-".join("-" * widths[column] for column in columns)
    body = [
        " | ".join(_truncate(record.get(column, "")).ljust(widths[column]) for column in columns) for record in records
    ]
    return "\n".join([header, divider, *body])


class RubricScoreValue(BaseModel):
    """Rubric-based score definition for grading criteria."""

    label: str = Field(description="The label to use for the level of the rubric grading criteria.")
    description: str | None = Field(
        default=None,
        description="Describe the semantic meaning of each criteria for the given rubric.",
    )
    value: float | int = Field(description="The score value to assign for the criteria.")


class RubricScoreStat(RubricScoreValue):
    """Rubric score with count statistics."""

    count: int = Field(default=0, description="The number of samples evaluated with the rubric level.")


class ScoreStats(BaseModel):
    """Stats for a score. Fields that are NaN are serialized as the string "NaN" in the API response."""

    count: int | None = Field(
        default=None,
        description="The number of values used for computing the score.",
    )
    sum: float | None = Field(
        default=None,
        description="The sum of all values used for computing the score.",
    )
    sum_squared: float | None = Field(
        default=None,
        description="The sum of the square of all values used for computing the score.",
    )
    min: float | None = Field(
        default=None,
        description="The minimum of all values used for computing the score.",
    )
    max: float | None = Field(
        default=None,
        description="The maximum of all values used for computing the score.",
    )
    mean: float | None = Field(
        default=None,
        description="The mean of all values used for computing the score.",
    )
    variance: float | None = Field(
        default=None,
        description="""The population variance, (note: not the sample variance).""",
    )
    stddev: float | None = Field(
        default=None,
        description="""The population standard deviation, (note: not the sample standard deviation).""",
    )
    stderr: float | None = Field(default=None, description="The standard error.")
    nan_count: int | None = Field(
        default=None,
        description="The number of values that are not a number (NaN) and are excluded from the score stats calculations.",
    )
    rubric_distribution: list[RubricScoreStat] | None = Field(
        default=None, description="The distribution of the rubric grading criteria for the score."
    )

    @field_serializer("sum", "sum_squared", "min", "max", "mean", "variance", "stddev", "stderr")
    def serialize_nan(self, v: float | None) -> float | str | None:
        """Serialize NaN stats as string values for JSON compatibility.

        Args:
            v: Float statistic value or ``None``.

        Returns:
            ``"NaN"`` for NaN floats, otherwise the original value.
        """
        if isinstance(v, float) and math.isnan(v):
            return "NaN"
        return v


class MetricScore(BaseModel):
    """
    A computed score for the metric
    """

    name: str
    value: float
    stats: ScoreStats | None = Field(
        default=None,
        description="Computed score statistics for the score.",
    )

    @field_validator("value", mode="before")
    @classmethod
    def convert_value(cls, v):
        """
        If incoming object is string with value "nan", it is converted to float nan.
        """
        if isinstance(v, str):
            if v.strip().lower() == "nan":
                return float("nan")
            raise ValueError("The only string value allowed for value is NaN")
        return v

    @field_serializer("value")
    def serialize_nan(self, v):
        """
        JSON serializers do not consistently support float NaN values.
        Serialize them as the string "NaN" so results remain portable.
        """
        if isinstance(v, float) and math.isnan(v):
            return "NaN"
        return v


class MetricResult(BaseModel):
    """Evaluation results for the metric"""

    scores: list[MetricScore]


class Percentiles(BaseModel):
    """Percentile distribution of scores."""

    model_config = ConfigDict(extra="forbid")
    p10: float | int = Field(description="10th percentile.")
    p20: float | int = Field(description="20th percentile.")
    p30: float | int = Field(description="30th percentile.")
    p40: float | int = Field(description="40th percentile.")
    p50: float | int = Field(description="50th percentile (median).")
    p60: float | int = Field(description="60th percentile.")
    p70: float | int = Field(description="70th percentile.")
    p80: float | int = Field(description="80th percentile.")
    p90: float | int = Field(description="90th percentile.")
    p100: float | int = Field(description="100th percentile.")


class HistogramBin(BaseModel):
    """A single bin in a histogram."""

    model_config = ConfigDict(extra="forbid")
    lower_bound: float | int = Field(description="Lower bound of the bin (inclusive).")
    upper_bound: float | int = Field(description="Upper bound of the bin (exclusive for all but last bin).")
    count: int = Field(description="Number of values in this bin.")


class Histogram(BaseModel):
    """Histogram of score distribution."""

    model_config = ConfigDict(extra="forbid")
    bins: list[HistogramBin] = Field(description="Histogram bins.")


class AggregateScoreBase(BaseModel):
    """Base statistics shared by all aggregated score types.

    This base class is used by both the app layer aggregation and API response schemas.
    """

    model_config = ConfigDict(extra="forbid")
    name: str = Field(description="Name of the score.")
    count: int = Field(description="Number of samples evaluated (excluding NaN).")
    nan_count: int = Field(description="Number of samples that produced NaN scores.")
    sum: float | None = Field(default=None, description="Sum of all score values.")
    mean: float | None = Field(default=None, description="Mean score value.")
    min: float | None = Field(default=None, description="Minimum score value.")
    max: float | None = Field(default=None, description="Maximum score value.")
    std_dev: float | None = Field(default=None, description="Standard deviation of the scores.")
    variance: float | None = Field(default=None, description="Variance of the scores.")


class AggregateRangeScore(AggregateScoreBase):
    """Aggregated statistics for a range-type score with percentiles and histogram."""

    score_type: Literal["range"] = Field(default="range", description="Type of score.")
    percentiles: Percentiles | None = Field(default=None, description="Percentile distribution of scores.")
    histogram: Histogram | None = Field(default=None, description="Histogram of score distribution.")

    _include_fields: frozenset[str] | None = None

    def with_fields(self, fields: frozenset[AggregateFieldName]) -> Self:
        """Return a copy configured to serialize only the specified fields."""
        copy = self.model_copy()
        object.__setattr__(copy, "_include_fields", {*fields, "name", "count"})
        return copy

    @model_serializer(mode="wrap")
    def _serialize(self, handler):
        data = handler(self)
        if self._include_fields is not None:
            # Always include required fields (name, count), plus requested fields
            fields_to_include = self._include_fields | {"name", "count"}
            return {k: v for k, v in data.items() if k in fields_to_include}
        return data


class AggregateRubricScore(AggregateScoreBase):
    """Aggregated statistics for a rubric-type score with category distribution."""

    score_type: Literal["rubric"] = Field(default="rubric", description="Type of score.")
    rubric_distribution: list[RubricScoreStat] = Field(description="Distribution of rubric categories.")
    mode_category: str | None = Field(default=None, description="Most frequent rubric category.")

    _include_fields: frozenset[str] | None = None

    def with_fields(self, fields: frozenset[AggregateFieldName]) -> Self:
        """Return a copy configured to serialize only the specified fields."""
        copy = self.model_copy()
        object.__setattr__(copy, "_include_fields", {*fields, "name", "count"})
        return copy

    @model_serializer(mode="wrap")
    def _serialize(self, handler):
        data = handler(self)
        if self._include_fields is not None:
            # Always include required fields (name, count), plus requested fields
            fields_to_include = self._include_fields | {"name", "count"}
            return {k: v for k, v in data.items() if k in fields_to_include}
        return data


AggregateScore = AggregateRangeScore | AggregateRubricScore


class AggregatedMetricResult(BaseModel):
    """Result of aggregating metric scores with full statistics."""

    model_config = ConfigDict(extra="forbid")
    scores: list[AggregateScore] = Field(description="The list of aggregated scores.")


class RowScore(BaseModel):
    """Normalized row-level score payload for metric/benchmark job results."""

    model_config = ConfigDict(extra="allow")

    row_index: int | None = Field(default=None, description="Stable row position used for result alignment.", ge=0)
    item: dict[str, Any] = Field(description="Input item metadata for the evaluated row.")
    sample: dict[str, Any] = Field(description="Sample output payload for the evaluated row.")
    metrics: dict[str, list[MetricScore]] = Field(description="Metric-level row scores by metric key.")
    requests: list[dict[str, Any]] = Field(description="Request details captured during evaluation.")
    metric_errors: dict[str, str] | None = Field(
        default=None,
        description="Full row-level error text keyed by metric for summary rendering.",
    )

    @property
    def error(self) -> str | None:
        """Derived row-level summary error text."""
        if self.metric_errors:
            return "; ".join(f"{metric_key}: {message}" for metric_key, message in self.metric_errors.items())
        return None


class SampleResult(BaseModel):
    """Result of evaluating a single sample."""

    model_config = ConfigDict(extra="forbid")

    index: int = Field(description="Index of the sample in the input dataset.")
    result: MetricResult | None = Field(default=None, description="Metric result if evaluation succeeded.")
    error: str | None = Field(default=None, description="Error message if evaluation failed.")

    @property
    def is_success(self) -> bool:
        """Return whether this sample completed without an error payload.

        Returns:
            ``True`` when ``result`` is present.
        """
        return self.result is not None

    @classmethod
    def success(cls, index: int, result: MetricResult) -> "SampleResult":
        """Create a successful ``SampleResult``.

        Args:
            index: Sample index in the source dataset.
            result: Metric result for that sample.

        Returns:
            Successful sample result object.
        """
        return cls(index=index, result=result)

    @classmethod
    def failure(cls, index: int, exc: Exception) -> "SampleResult":
        """Create a failed ``SampleResult`` from an exception.

        Args:
            index: Sample index in the source dataset.
            exc: Exception raised during sample evaluation.

        Returns:
            Failed sample result object with normalized error text.
        """
        return cls(index=index, error=str(exc) or exc.__class__.__name__)


def row_error_text(row_score: RowScore) -> str | None:
    """Return the human-facing row error text."""
    return row_score.error


def _row_has_scores(row_score: RowScore) -> bool:
    """Return whether the row contains any metric score values."""
    return any(metric_scores for metric_scores in row_score.metrics.values())


def _row_has_errors(row_score: RowScore) -> bool:
    """Return whether the row contains any error payload."""
    return bool(row_error_text(row_score))


def row_status(row_score: RowScore) -> str:
    """Return a compact row status used by summary and export views."""
    if _row_has_errors(row_score):
        return "error"
    return "ok"


def _summary_status_counts(row_scores: list[RowScore]) -> dict[str, int]:
    """Count summary statuses for inclusion in summary headers."""
    counts = {"ok": 0, "error": 0}
    for row_score in row_scores:
        counts[row_status(row_score)] += 1
    return counts


def summary_header(name: str, row_scores: list[RowScore], aggregate_count: int) -> str:
    """Build the compact summary header line for result objects."""
    counts = _summary_status_counts(row_scores)
    parts = [f"{name}(rows={len(row_scores)}, aggregate_scores={aggregate_count}"]
    if counts["ok"]:
        parts.append(f", ok={counts['ok']}")
    if counts["error"]:
        parts.append(f", error={counts['error']}")
    parts.append(")")
    return "".join(parts)


def _row_display_index(row_score: RowScore, index: int) -> int:
    """Resolve the row index used in preview tables and error sections."""
    return row_score.row_index if row_score.row_index is not None else index


def _flatten_summary_dict(prefix: str, value: Any, output: dict[str, Any]) -> None:
    """Flatten only summary-friendly scalar values into a preview record."""
    if isinstance(value, dict):
        for key, nested_value in value.items():
            nested_prefix = f"{prefix}.{key}" if prefix else str(key)
            _flatten_summary_dict(nested_prefix, nested_value, output)
        return
    if isinstance(value, (str, int, float, bool)) or value is None:
        output[prefix] = value


def summary_aggregate_record(score: AggregateScore) -> dict[str, Any]:
    """Project an aggregate score into a concise summary row."""
    record: dict[str, Any] = {
        "name": score.name,
        "count": score.count,
        "nan_count": score.nan_count,
        "mean": score.mean,
        "min": score.min,
        "max": score.max,
    }
    score_type = getattr(score, "score_type", None)
    if score_type is not None:
        record["score_type"] = score_type
    percentiles = getattr(score, "percentiles", None)
    if percentiles is not None and getattr(percentiles, "p50", None) is not None:
        record["p50"] = percentiles.p50
    mode_category = getattr(score, "mode_category", None)
    if mode_category is not None:
        record["mode_category"] = mode_category
    return record


def summary_row_base_record(row_score: RowScore, index: int) -> dict[str, Any]:
    """Build the shared non-score columns for summary row previews."""
    record: dict[str, Any] = {
        "row_index": _row_display_index(row_score, index),
        "status": row_status(row_score),
    }
    _flatten_summary_dict("item", serialize_value(row_score.item), record)
    sample = serialize_value(row_score.sample)
    if isinstance(sample, dict):
        sample = {key: value for key, value in sample.items() if key != "response"}
    _flatten_summary_dict("sample", sample, record)
    if error_text := row_error_text(row_score):
        record["error"] = error_text
    return record


def format_error_details(
    row_scores: list[RowScore],
    *,
    max_error_rows: int | None,
    label_metric_errors: bool,
) -> list[str]:
    """Render a detailed full-error section for failed rows."""
    failed_rows = [(index, row_score) for index, row_score in enumerate(row_scores) if _row_has_errors(row_score)]
    if not failed_rows:
        return []

    shown_limit = len(failed_rows) if max_error_rows is None else max(0, max_error_rows)
    shown_rows = failed_rows[:shown_limit]
    parts = [
        "",
        f"Error details ({len(shown_rows)} of {len(failed_rows)} failed rows)",
    ]

    for index, row_score in shown_rows:
        display_index = _row_display_index(row_score, index)
        parts.extend(["", f"[row {display_index}]"])

        if row_score.metric_errors:
            for metric_key, message in row_score.metric_errors.items():
                if label_metric_errors:
                    parts.append(f"{metric_key}: {message}")
                else:
                    parts.append(message)
        else:
            if error_text := row_error_text(row_score):
                parts.append(error_text)

    if len(shown_rows) < len(failed_rows):
        parts.extend(["", f"... {len(failed_rows) - len(shown_rows)} more failed rows omitted"])

    return parts


class EvaluationResult(BaseModel):
    """Result object returned by SDK offline evaluation."""

    row_scores: list[RowScore] = Field(description="Row-level scores.")
    aggregate_scores: AggregatedMetricResult = Field(description="Aggregate score statistics.")

    def to_records(self, view: ResultView = "rows") -> list[dict[str, Any]]:
        """Convert evaluation output into flat dictionaries.

        For ``view="rows"``, nested ``item`` and ``sample`` payloads are
        flattened with dotted keys. For ``view="aggregate"``, percentile fields
        are flattened while histograms are kept as JSON strings to preserve
        tabular shape.

        Args:
            view: Output projection, either ``"rows"`` or ``"aggregate"``.

        Returns:
            Flat record dictionaries for downstream table/dataframe conversion.

        Raises:
            ValueError: If ``view`` is unsupported.
        """
        if view == "rows":
            records: list[dict[str, Any]] = []
            for index, row_score in enumerate(self.row_scores):
                record: dict[str, Any] = {
                    "row_index": row_score.row_index if row_score.row_index is not None else index,
                    "status": row_status(row_score),
                }
                flatten_dict("item", serialize_value(row_score.item), record)
                flatten_dict("sample", serialize_value(row_score.sample), record)
                if error_text := row_error_text(row_score):
                    record["error"] = error_text
                for metric_scores in row_score.metrics.values():
                    for score in metric_scores:
                        record[f"score.{score.name}"] = serialize_value(score.value)
                records.append(record)
            return records

        if view == "aggregate":
            records = []
            for score in self.aggregate_scores.scores:
                record = {}
                for key, value in score.model_dump(mode="json").items():
                    if key == "percentiles" and isinstance(value, dict):
                        flatten_dict("percentiles", value, record)
                    elif key == "histogram" and value is not None:
                        # Histograms stay as JSON strings so aggregate views remain
                        # tabular instead of expanding variable-width nested columns.
                        record[key] = json.dumps(value, sort_keys=True)
                    else:
                        record[key] = value
                records.append(record)
            return records

        raise ValueError(f"Unsupported view {view!r}. Expected 'rows' or 'aggregate'.")

    def to_table(self, view: ResultView = "rows") -> pa.Table:
        """Convert records into a ``pyarrow.Table``.

        Args:
            view: Output projection, either ``"rows"`` or ``"aggregate"``.

        Returns:
            Table built from ``to_records(view=view)``.
        """
        return pa.Table.from_pylist(self.to_records(view=view))

    def to_pandas(self, view: ResultView = "rows"):
        """Convert records into a pandas ``DataFrame``.

        Args:
            view: Output projection, either ``"rows"`` or ``"aggregate"``.

        Returns:
            DataFrame built from ``to_records(view=view)``.
        """
        import pandas as pd

        return pd.DataFrame.from_records(self.to_records(view=view))

    def format_summary(self, max_rows: int = 10, *, max_error_rows: int | None = None) -> str:
        """Render a human-readable summary with aggregates and row preview.

        Args:
            max_rows: Maximum number of row-level records included in preview.
            max_error_rows: Maximum number of failed rows included in the full
                error-details section. Defaults to ``max_rows``.

        Returns:
            Multi-line summary string suitable for terminal/notebook display.
        """
        if max_error_rows is None:
            max_error_rows = max_rows
        aggregate_records = [summary_aggregate_record(score) for score in self.aggregate_scores.scores]
        preview_records = []
        for index, row_score in enumerate(self.row_scores[:max_rows]):
            record = summary_row_base_record(row_score, index)
            for metric_scores in row_score.metrics.values():
                for score in metric_scores:
                    record[f"score.{score.name}"] = serialize_value(score.value)
            preview_records.append(record)
        parts = [
            summary_header("EvaluationResult", self.row_scores, len(self.aggregate_scores.scores)),
            "",
            "Aggregate scores",
            format_table(aggregate_records),
        ]
        if preview_records:
            parts.extend(
                [
                    "",
                    f"Row preview (first {len(preview_records)} of {len(self.row_scores)})",
                    format_table(preview_records),
                ]
            )
        parts.extend(
            format_error_details(
                self.row_scores,
                max_error_rows=max_error_rows,
                label_metric_errors=False,
            )
        )
        return "\n".join(parts)

    def print_summary(self, max_rows: int = 10, *, max_error_rows: int | None = None) -> None:
        """Print ``format_summary`` output.

        Args:
            max_rows: Maximum number of row-level records included in preview.
            max_error_rows: Maximum number of failed rows included in the full
                error-details section. Defaults to ``max_rows``.
        """
        print(self.format_summary(max_rows=max_rows, max_error_rows=max_error_rows))

    def __str__(self) -> str:
        """Return the default compact summary representation.

        Returns:
            Summary string with up to five preview rows.
        """
        return self.format_summary(max_rows=5)
