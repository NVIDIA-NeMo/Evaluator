"""Helpers for assembling multi-metric evaluator results."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

from nemo_evaluator.sdk.values.results import (
    AggregatedMetricResult,
    AggregateFieldName,
    EvaluationResult,
    MetricScore,
    ResultView,
    RowScore,
    flatten_dict,
    format_error_details,
    format_table,
    row_error_text,
    row_status,
    serialize_value,
    summary_aggregate_record,
    summary_header,
    summary_row_base_record,
)


def _filter_aggregate_fields(
    aggregate_scores: AggregatedMetricResult, aggregate_fields: tuple[AggregateFieldName, ...] | None
) -> AggregatedMetricResult:
    """Trim aggregate score payloads to the requested field subset.

    Args:
        aggregate_scores: Full aggregate metric output for one or more metrics.
        aggregate_fields: Optional field names to retain on each score object.

    Returns:
        The original aggregate scores when no filtering is requested, otherwise
        a copy that keeps only the selected fields.
    """
    if not aggregate_fields:
        return aggregate_scores

    filtered_scores = [score.with_fields(frozenset(aggregate_fields)) for score in aggregate_scores.scores]
    return AggregatedMetricResult(scores=filtered_scores)


def _extract_metric_scores(row_score: RowScore, expected_key: str) -> list[MetricScore]:
    """Resolve the score list for one metric from a row result.

    This exists because local and remote execution paths may return row scores
    either already keyed by the final metric key or as a single unnamed metric
    payload that still needs to be attached to that key.

    Args:
        row_score: Row-level result payload to inspect.
        expected_key: Metric key the caller expects to find on the row.

    Returns:
        The score list for the requested metric key, or an empty list when the
        row has no metric output because evaluation failed.

    Raises:
        ValueError: If the row contains multiple metric entries and none match
            the requested key.
    """
    if expected_key in row_score.metrics:
        return row_score.metrics[expected_key]
    if not row_score.metrics:
        return []
    if len(row_score.metrics) == 1:
        return next(iter(row_score.metrics.values()))
    raise ValueError(f"Unable to resolve row metric scores for key {expected_key!r}")


def _extract_metric_error(row_score: RowScore, expected_key: str) -> str | None:
    """Resolve one metric error message from a row result."""
    metric_errors = row_score.metric_errors
    if metric_errors and expected_key in metric_errors:
        return metric_errors[expected_key]
    if metric_errors:
        if len(metric_errors) == 1:
            return next(iter(metric_errors.values()))
        raise ValueError(f"Unable to resolve row metric error for key {expected_key!r}")
    return row_score.error


def _row_identity(row_score: RowScore, fallback_index: int) -> int:
    """Resolve the stable row identity used when combining metric results.

    Args:
        row_score: Row-level result payload.
        fallback_index: Positional fallback when no explicit row identity is present.

    Returns:
        Stable row identity for result alignment.
    """
    return row_score.row_index if row_score.row_index is not None else fallback_index


def namespace_result(
    metric_key: str,
    result: EvaluationResult,
    aggregate_fields: tuple[AggregateFieldName, ...] | None,
) -> EvaluationResult:
    """Rewrite a single-metric result to use a stable `v4` metric namespace.

    Args:
        metric_key: Public metric key assigned by the evaluator.
        result: Raw single-metric evaluation result from one backend.
        aggregate_fields: Optional aggregate field subset to keep.

    Returns:
        A result whose row metrics and aggregate score names are prefixed with
        the evaluator-assigned metric key.
    """
    row_scores = [
        RowScore(
            row_index=row_score.row_index,
            item=row_score.item,
            sample=row_score.sample,
            metrics={metric_key: _extract_metric_scores(row_score, metric_key)},
            requests=row_score.requests,
            metric_errors={metric_key: error} if (error := _extract_metric_error(row_score, metric_key)) else None,
        )
        for row_score in result.row_scores
    ]
    aggregate_scores = AggregatedMetricResult(
        scores=[
            score.model_copy(update={"name": f"{metric_key}.{score.name}"}) for score in result.aggregate_scores.scores
        ]
    )
    aggregate_scores = _filter_aggregate_fields(aggregate_scores, aggregate_fields)
    return EvaluationResult(row_scores=row_scores, aggregate_scores=aggregate_scores)


def collapse_results(
    results_by_key: dict[str, EvaluationResult],
    aggregate_fields: tuple[AggregateFieldName, ...] | None,
) -> BenchmarkEvaluationResult:
    """Merge multiple single-metric results into one multi-metric view.

    Args:
        results_by_key: Namespaced single-metric results keyed by metric name.
        aggregate_fields: Optional aggregate field subset to keep.

    Returns:
        A combined multi-metric result with merged row scores, merged aggregate
        scores, and the original per-metric mapping.

    Raises:
        ValueError: If the provided metric results do not all contain the same
            number of rows.
    """
    if not results_by_key:
        empty = AggregatedMetricResult(scores=[])
        return BenchmarkEvaluationResult(row_scores=[], aggregate_scores=empty, per_metric={})

    ordered_keys = list(results_by_key.keys())
    row_count = len(results_by_key[ordered_keys[0]].row_scores)
    baseline_identities = [
        _row_identity(row_score, index) for index, row_score in enumerate(results_by_key[ordered_keys[0]].row_scores)
    ]
    for metric_key in ordered_keys[1:]:
        result = results_by_key[metric_key]
        if len(result.row_scores) != row_count:
            raise ValueError(f"Cannot combine metric results with different row counts: {metric_key}")
        current_identities = [_row_identity(row_score, index) for index, row_score in enumerate(result.row_scores)]
        if current_identities != baseline_identities:
            raise ValueError(f"Cannot combine metric results with different row identities: {metric_key}")

    combined_rows: list[RowScore] = []
    for index in range(row_count):
        first_row = results_by_key[ordered_keys[0]].row_scores[index]
        metrics: dict[str, list[MetricScore]] = {}
        requests: list[dict[str, Any]] = []
        metric_errors: dict[str, str] = {}
        for metric_key in ordered_keys:
            row_score = results_by_key[metric_key].row_scores[index]
            metrics[metric_key] = _extract_metric_scores(row_score, metric_key)
            requests.extend(row_score.requests)
            if row_score.metric_errors:
                metric_errors.update(row_score.metric_errors)
            elif row_score.error:
                metric_errors[metric_key] = row_score.error
        combined_rows.append(
            RowScore(
                row_index=_row_identity(first_row, index),
                item=first_row.item,
                sample=first_row.sample,
                metrics=metrics,
                requests=requests,
                metric_errors=metric_errors or None,
            )
        )

    aggregate_scores = AggregatedMetricResult(
        scores=[score for result in results_by_key.values() for score in result.aggregate_scores.scores]
    )
    aggregate_scores = _filter_aggregate_fields(aggregate_scores, aggregate_fields)
    return BenchmarkEvaluationResult(
        row_scores=combined_rows,
        aggregate_scores=aggregate_scores,
        per_metric=results_by_key,
    )


class BenchmarkEvaluationResult(BaseModel):
    """Unified benchmark evaluation result."""

    row_scores: list[RowScore]
    aggregate_scores: AggregatedMetricResult
    per_metric: dict[str, EvaluationResult]

    def metric_result(self, metric_key: str) -> EvaluationResult:
        """Return the original single-metric result for one metric key.

        Args:
            metric_key: Metric key to retrieve from the combined result.

        Returns:
            The single-metric result stored for that key.
        """
        return self.per_metric[metric_key]

    def to_records(self, view: ResultView = "rows") -> list[dict[str, Any]]:
        """Convert the result into flat dictionaries for export or inspection.

        Args:
            view: Which logical result view to flatten, either `"rows"` or
                `"aggregate"`.

        Returns:
            A list of flat dictionaries suitable for tabular rendering.

        Raises:
            ValueError: If `view` is not one of the supported options.
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
                for metric_key, metric_scores in row_score.metrics.items():
                    for score in metric_scores:
                        record[f"score.{metric_key}.{score.name}"] = serialize_value(score.value)
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
                        record[key] = json.dumps(value, sort_keys=True)
                    else:
                        record[key] = value
                records.append(record)
            return records

        raise ValueError(f"Unsupported view {view!r}. Expected 'rows' or 'aggregate'.")

    def to_table(self, view: ResultView = "rows"):
        """Convert the result into a `pyarrow.Table`.

        Args:
            view: Which logical result view to materialize.

        Returns:
            A PyArrow table built from the flattened result records.
        """
        import pyarrow as pa

        return pa.Table.from_pylist(self.to_records(view=view))

    def to_pandas(self, view: ResultView = "rows"):
        """Convert the result into a pandas `DataFrame`.

        Args:
            view: Which logical result view to materialize.

        Returns:
            A pandas dataframe built from the flattened result records.
        """
        import pandas as pd

        return pd.DataFrame.from_records(self.to_records(view=view))

    def format_summary(self, max_rows: int = 10, *, max_error_rows: int | None = None) -> str:
        """Build a human-readable text summary of the result.

        Args:
            max_rows: Maximum number of row records to include in the preview.
            max_error_rows: Maximum number of failed rows included in the full
                error-details section. Defaults to ``max_rows``.

        Returns:
            A formatted multiline string summary.
        """
        if max_error_rows is None:
            max_error_rows = max_rows
        aggregate_records = [summary_aggregate_record(score) for score in self.aggregate_scores.scores]
        preview_records = []
        for index, row_score in enumerate(self.row_scores[:max_rows]):
            record = summary_row_base_record(row_score, index)
            for metric_key, metric_scores in row_score.metrics.items():
                for score in metric_scores:
                    record[f"score.{metric_key}.{score.name}"] = serialize_value(score.value)
            preview_records.append(record)
        parts = [
            summary_header("BenchmarkEvaluationResult", self.row_scores, len(self.aggregate_scores.scores)),
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
                label_metric_errors=True,
            )
        )
        return "\n".join(parts)

    def print_summary(self, max_rows: int = 10, *, max_error_rows: int | None = None) -> None:
        """Print the formatted summary to standard output.

        Args:
            max_rows: Maximum number of row records to include in the preview.
            max_error_rows: Maximum number of failed rows included in the full
                error-details section. Defaults to ``max_rows``.

        Returns:
            None.
        """
        print(self.format_summary(max_rows=max_rows, max_error_rows=max_error_rows))
