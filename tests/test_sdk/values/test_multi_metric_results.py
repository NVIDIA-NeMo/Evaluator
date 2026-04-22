import pytest
from nemo_evaluator.sdk.values.multi_metric_results import (
    BenchmarkEvaluationResult,
    collapse_results,
    namespace_result,
)
from nemo_evaluator.sdk.values.results import (
    AggregatedMetricResult,
    AggregateRangeScore,
    EvaluationResult,
    MetricScore,
    Percentiles,
    RowScore,
)


def _make_aggregate_score(name: str, mean: float) -> AggregateRangeScore:
    return AggregateRangeScore(
        name=name,
        count=2,
        nan_count=0,
        sum=mean * 2,
        mean=mean,
        min=0.0,
        max=1.0,
        std_dev=0.25,
        variance=0.0625,
        percentiles=Percentiles(
            p10=0.1,
            p20=0.2,
            p30=0.3,
            p40=0.4,
            p50=mean,
            p60=0.6,
            p70=0.7,
            p80=0.8,
            p90=0.9,
            p100=1.0,
        ),
        histogram=None,
    )


def _make_evaluation_result(*, rows: list[RowScore], aggregate_name: str, mean: float) -> EvaluationResult:
    return EvaluationResult(
        row_scores=rows,
        aggregate_scores=AggregatedMetricResult(scores=[_make_aggregate_score(aggregate_name, mean)]),
    )


class TestNamespaceResult:
    def test_prefixes_metric_names_and_aggregate_scores(self):
        result = _make_evaluation_result(
            rows=[
                RowScore(
                    row_index=7,
                    item={"prompt": "q"},
                    sample={"output_text": "a"},
                    metrics={"score": [MetricScore(name="score", value=1.0)]},
                    requests=[{"id": "req-1"}],
                    metric_errors=None,
                )
            ],
            aggregate_name="score",
            mean=1.0,
        )

        namespaced = namespace_result("exact-match", result, None)

        assert namespaced == EvaluationResult(
            row_scores=[
                RowScore(
                    row_index=7,
                    item={"prompt": "q"},
                    sample={"output_text": "a"},
                    metrics={"exact-match": [MetricScore(name="score", value=1.0)]},
                    requests=[{"id": "req-1"}],
                    metric_errors=None,
                )
            ],
            aggregate_scores=AggregatedMetricResult(scores=[_make_aggregate_score("exact-match.score", 1.0)]),
        )

    def test_filters_aggregate_fields(self):
        result = _make_evaluation_result(
            rows=[
                RowScore(
                    row_index=0,
                    item={},
                    sample={},
                    metrics={"score": [MetricScore(name="score", value=1.0)]},
                    requests=[],
                    metric_errors=None,
                )
            ],
            aggregate_name="score",
            mean=1.0,
        )

        namespaced = namespace_result("metric-a", result, ("mean", "percentiles"))

        assert namespaced.aggregate_scores.model_dump(mode="json") == {
            "scores": [
                {
                    "name": "metric-a.score",
                    "count": 2,
                    "mean": 1.0,
                    "percentiles": {
                        "p10": 0.1,
                        "p20": 0.2,
                        "p30": 0.3,
                        "p40": 0.4,
                        "p50": 1.0,
                        "p60": 0.6,
                        "p70": 0.7,
                        "p80": 0.8,
                        "p90": 0.9,
                        "p100": 1.0,
                    },
                }
            ]
        }

    def test_uses_single_metric_fallback_and_preserves_metric_error(self):
        result = _make_evaluation_result(
            rows=[
                RowScore(
                    row_index=1,
                    item={"id": "row-1"},
                    sample={},
                    metrics={"raw": [MetricScore(name="score", value=0.0)]},
                    requests=[],
                    metric_errors={"raw": "boom"},
                )
            ],
            aggregate_name="score",
            mean=0.0,
        )

        namespaced = namespace_result("metric-a", result, None)

        assert namespaced.row_scores == [
            RowScore(
                row_index=1,
                item={"id": "row-1"},
                sample={},
                metrics={"metric-a": [MetricScore(name="score", value=0.0)]},
                requests=[],
                metric_errors={"metric-a": "boom"},
            )
        ]

    def test_returns_empty_metric_scores_for_failed_row_without_metrics(self):
        result = _make_evaluation_result(
            rows=[
                RowScore(
                    row_index=2,
                    item={"id": "row-2"},
                    sample={},
                    metrics={},
                    requests=[],
                    metric_errors={"metric-a": "failed"},
                )
            ],
            aggregate_name="score",
            mean=0.0,
        )

        namespaced = namespace_result("metric-a", result, None)

        assert namespaced.row_scores == [
            RowScore(
                row_index=2,
                item={"id": "row-2"},
                sample={},
                metrics={"metric-a": []},
                requests=[],
                metric_errors={"metric-a": "failed"},
            )
        ]


class TestCollapseResults:
    def test_merges_rows_aggregate_scores_and_errors(self):
        metric_a = _make_evaluation_result(
            rows=[
                RowScore(
                    row_index=10,
                    item={"id": "row-1"},
                    sample={"output_text": "first"},
                    metrics={"metric-a": [MetricScore(name="score", value=1.0)]},
                    requests=[{"request_id": "a-1"}],
                    metric_errors=None,
                ),
                RowScore(
                    row_index=11,
                    item={"id": "row-2"},
                    sample={"output_text": "second"},
                    metrics={"metric-a": []},
                    requests=[{"request_id": "a-2"}],
                    metric_errors={"metric-a": "failed-a"},
                ),
            ],
            aggregate_name="metric-a.score",
            mean=0.5,
        )
        metric_b = _make_evaluation_result(
            rows=[
                RowScore(
                    row_index=10,
                    item={"id": "row-1"},
                    sample={"output_text": "first"},
                    metrics={"metric-b": [MetricScore(name="score", value=0.5)]},
                    requests=[{"request_id": "b-1"}],
                    metric_errors=None,
                ),
                RowScore(
                    row_index=11,
                    item={"id": "row-2"},
                    sample={"output_text": "second"},
                    metrics={"metric-b": []},
                    requests=[{"request_id": "b-2"}],
                    metric_errors=None,
                ),
            ],
            aggregate_name="metric-b.score",
            mean=0.25,
        )

        combined = collapse_results({"metric-a": metric_a, "metric-b": metric_b}, None)

        assert combined == BenchmarkEvaluationResult(
            row_scores=[
                RowScore(
                    row_index=10,
                    item={"id": "row-1"},
                    sample={"output_text": "first"},
                    metrics={
                        "metric-a": [MetricScore(name="score", value=1.0)],
                        "metric-b": [MetricScore(name="score", value=0.5)],
                    },
                    requests=[{"request_id": "a-1"}, {"request_id": "b-1"}],
                    metric_errors=None,
                ),
                RowScore(
                    row_index=11,
                    item={"id": "row-2"},
                    sample={"output_text": "second"},
                    metrics={"metric-a": [], "metric-b": []},
                    requests=[{"request_id": "a-2"}, {"request_id": "b-2"}],
                    metric_errors={"metric-a": "failed-a"},
                ),
            ],
            aggregate_scores=AggregatedMetricResult(
                scores=[
                    _make_aggregate_score("metric-a.score", 0.5),
                    _make_aggregate_score("metric-b.score", 0.25),
                ]
            ),
            per_metric={"metric-a": metric_a, "metric-b": metric_b},
        )

    def test_to_records_marks_rows_with_metric_errors_as_error(self):
        metric_a = _make_evaluation_result(
            rows=[
                RowScore(
                    row_index=0,
                    item={"id": "row-1"},
                    sample={"output_text": "ok"},
                    metrics={"metric-a": [MetricScore(name="score", value=1.0)]},
                    requests=[],
                    metric_errors=None,
                ),
                RowScore(
                    row_index=1,
                    item={"id": "row-2"},
                    sample={"output_text": "bad"},
                    metrics={"metric-a": []},
                    requests=[],
                    metric_errors={"metric-a": "failed-a"},
                ),
            ],
            aggregate_name="metric-a.score",
            mean=0.5,
        )
        metric_b = _make_evaluation_result(
            rows=[
                RowScore(
                    row_index=0,
                    item={"id": "row-1"},
                    sample={"output_text": "ok"},
                    metrics={"metric-b": [MetricScore(name="score", value=0.5)]},
                    requests=[],
                    metric_errors=None,
                ),
                RowScore(
                    row_index=1,
                    item={"id": "row-2"},
                    sample={"output_text": "bad"},
                    metrics={"metric-b": [MetricScore(name="score", value=0.0)]},
                    requests=[],
                    metric_errors=None,
                ),
            ],
            aggregate_name="metric-b.score",
            mean=0.25,
        )

        combined = collapse_results({"metric-a": metric_a, "metric-b": metric_b}, None)
        records = combined.to_records(view="rows")

        assert records[0]["status"] == "ok"
        assert records[1]["status"] == "error"
        assert records[1]["error"] == "metric-a: failed-a"

    def test_format_summary_includes_multi_metric_row_errors(self):
        metric_a = _make_evaluation_result(
            rows=[
                RowScore(
                    row_index=0,
                    item={"id": "row-1"},
                    sample={"output_text": "bad"},
                    metrics={"metric-a": []},
                    requests=[],
                    metric_errors={"metric-a": "failed-a"},
                )
            ],
            aggregate_name="metric-a.score",
            mean=0.0,
        )
        metric_b = _make_evaluation_result(
            rows=[
                RowScore(
                    row_index=0,
                    item={"id": "row-1"},
                    sample={"output_text": "bad"},
                    metrics={"metric-b": [MetricScore(name="score", value=0.0)]},
                    requests=[],
                    metric_errors=None,
                )
            ],
            aggregate_name="metric-b.score",
            mean=0.0,
        )

        combined = collapse_results({"metric-a": metric_a, "metric-b": metric_b}, None)
        formatted = combined.format_summary(max_rows=1, max_error_rows=1)

        assert "BenchmarkEvaluationResult(rows=1, aggregate_scores=2, error=1)" in formatted
        assert "metric-a: failed-a" in formatted

    def test_imports_from_multi_metric_results_module(self):
        result = _make_evaluation_result(
            rows=[
                RowScore(
                    row_index=0,
                    item={},
                    sample={},
                    metrics={"score": [MetricScore(name="score", value=1.0)]},
                    requests=[],
                    metric_errors=None,
                )
            ],
            aggregate_name="score",
            mean=1.0,
        )

        namespaced = namespace_result("metric-a", result, None)
        collapsed = collapse_results({"metric-a": namespaced}, None)

        assert collapsed == BenchmarkEvaluationResult(
            row_scores=namespaced.row_scores,
            aggregate_scores=namespaced.aggregate_scores,
            per_metric={"metric-a": namespaced},
        )

    def test_applies_aggregate_field_filter_to_combined_result(self):
        metric_a = _make_evaluation_result(
            rows=[
                RowScore(
                    row_index=0,
                    item={},
                    sample={},
                    metrics={"metric-a": [MetricScore(name="score", value=1.0)]},
                    requests=[],
                    metric_errors=None,
                )
            ],
            aggregate_name="metric-a.score",
            mean=1.0,
        )

        combined = collapse_results({"metric-a": metric_a}, ("mean",))

        assert combined.aggregate_scores.model_dump(mode="json") == {
            "scores": [{"name": "metric-a.score", "count": 2, "mean": 1.0}]
        }

    @pytest.mark.parametrize(
        ("results_by_key", "match"),
        [
            (
                {
                    "metric-a": _make_evaluation_result(
                        rows=[
                            RowScore(
                                row_index=0,
                                item={},
                                sample={},
                                metrics={"metric-a": [MetricScore(name="score", value=1.0)]},
                                requests=[],
                                metric_errors=None,
                            )
                        ],
                        aggregate_name="metric-a.score",
                        mean=1.0,
                    ),
                    "metric-b": _make_evaluation_result(
                        rows=[],
                        aggregate_name="metric-b.score",
                        mean=0.0,
                    ),
                },
                "different row counts",
            ),
            (
                {
                    "metric-a": _make_evaluation_result(
                        rows=[
                            RowScore(
                                row_index=0,
                                item={},
                                sample={},
                                metrics={"metric-a": [MetricScore(name="score", value=1.0)]},
                                requests=[],
                                metric_errors=None,
                            )
                        ],
                        aggregate_name="metric-a.score",
                        mean=1.0,
                    ),
                    "metric-b": _make_evaluation_result(
                        rows=[
                            RowScore(
                                row_index=1,
                                item={},
                                sample={},
                                metrics={"metric-b": [MetricScore(name="score", value=0.0)]},
                                requests=[],
                                metric_errors=None,
                            )
                        ],
                        aggregate_name="metric-b.score",
                        mean=0.0,
                    ),
                },
                "different row identities",
            ),
        ],
    )
    def test_raises_for_incompatible_results(self, results_by_key: dict[str, EvaluationResult], match: str):
        with pytest.raises(ValueError, match=match):
            collapse_results(results_by_key, None)
