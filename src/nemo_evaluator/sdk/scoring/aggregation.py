"""Aggregation data structures and computations for metric results."""

# Migrated from: services/evaluator/src/nmp/evaluator/app/metrics/aggregation.py

import math
from collections import OrderedDict, defaultdict

from nemo_evaluator.sdk.values.results import (
    AggregatedMetricResult,
    AggregateRangeScore,
    AggregateRubricScore,
    AggregateScore,
    Histogram,
    HistogramBin,
    MetricResult,
    MetricScore,
    Percentiles,
    RubricScoreStat,
    ScoreStats,
)


def add_corpus_scores(aggregated_result: AggregatedMetricResult, corpus_result: MetricResult) -> None:
    """Append corpus-level scores using aggregate-score schema fields.

    Args:
        aggregated_result: Aggregate result object to mutate.
        corpus_result: Corpus-level metric output with one or more scores.

    Returns:
        ``None``. The ``aggregated_result`` object is updated in place.
    """
    for score in corpus_result.scores:
        value = score.value
        # Corpus-level metrics contribute one already-aggregated value, so
        # expose them through the same aggregate schema with count=1.
        corpus_score = AggregateRangeScore(
            name=score.name,
            count=1,
            nan_count=0,
            sum=value,
            mean=value,
            min=value,
            max=value,
            std_dev=0.0,
            variance=0.0,
            percentiles=Percentiles(
                p10=value,
                p20=value,
                p30=value,
                p40=value,
                p50=value,
                p60=value,
                p70=value,
                p80=value,
                p90=value,
                p100=value,
            ),
            histogram=Histogram(bins=[HistogramBin(lower_bound=value, upper_bound=value, count=1)]),
        )
        aggregated_result.scores.append(corpus_score)


def _compute_percentile(sorted_values: list[float], percentile: float) -> float:
    """Compute a percentile using linearly interpolated rank position.

    The rank position is computed as ``(p/100) * (n + 1) - 1`` and then
    interpolated between neighboring points when the position is fractional.

    Args:
        sorted_values: Score values sorted in ascending order.
        percentile: Percentile in the inclusive range ``[0, 100]``.

    Returns:
        The interpolated percentile value, or ``0.0`` when no values exist.
    """
    if not sorted_values:
        return 0.0
    n = len(sorted_values)
    pos = (percentile / 100.0) * (n + 1) - 1
    if pos <= 0:
        return sorted_values[0]
    if pos >= n - 1:
        return sorted_values[-1]
    lower_idx = int(pos)
    frac = pos - lower_idx
    return sorted_values[lower_idx] + frac * (sorted_values[lower_idx + 1] - sorted_values[lower_idx])


def _compute_percentiles(sorted_values: list[float]) -> Percentiles:
    """Compute the fixed percentile set used by SDK aggregate output.

    Args:
        sorted_values: Score values sorted in ascending order.

    Returns:
        ``Percentiles`` containing p10 through p100.
    """
    return Percentiles(
        p10=_compute_percentile(sorted_values, 10),
        p20=_compute_percentile(sorted_values, 20),
        p30=_compute_percentile(sorted_values, 30),
        p40=_compute_percentile(sorted_values, 40),
        p50=_compute_percentile(sorted_values, 50),
        p60=_compute_percentile(sorted_values, 60),
        p70=_compute_percentile(sorted_values, 70),
        p80=_compute_percentile(sorted_values, 80),
        p90=_compute_percentile(sorted_values, 90),
        p100=_compute_percentile(sorted_values, 100),
    )


def _compute_histogram(values: list[float], num_bins: int = 10) -> Histogram:
    """Build a fixed-width histogram for numeric score distribution.

    The algorithm uses equal-width bins between the global minimum and maximum.
    All bins except the last are half-open ``[lower, upper)``, while the final
    bin is closed ``[lower, upper]`` so the maximum value is always counted.

    Args:
        values: Numeric score values to bucket.
        num_bins: Number of equally sized bins.

    Returns:
        Histogram object containing ordered bin counts.
    """
    if not values:
        return Histogram(bins=[])

    min_val = min(values)
    max_val = max(values)
    if min_val == max_val:
        return Histogram(bins=[HistogramBin(lower_bound=min_val, upper_bound=max_val, count=len(values))])

    bin_width = (max_val - min_val) / num_bins
    bins: list[HistogramBin] = []
    for i in range(num_bins):
        lower = min_val + i * bin_width
        upper = min_val + (i + 1) * bin_width
        if i == num_bins - 1:
            count = sum(1 for v in values if lower <= v <= upper)
        else:
            count = sum(1 for v in values if lower <= v < upper)
        bins.append(HistogramBin(lower_bound=lower, upper_bound=upper, count=count))
    return Histogram(bins=bins)


def aggregate_metrics(items: list[MetricResult]) -> AggregatedMetricResult:
    """Aggregate row-level metric results into range or rubric summaries.

    This function performs two logical passes:
    1. Incremental accumulation of count/sum/min/max and rubric occurrences.
    2. Finalization of derived statistics (variance, percentiles, histogram,
       and rubric mode category) once complete value sets are known.

    Args:
        items: Row-level metric results to aggregate.

    Returns:
        Aggregate metric result with one aggregate score per score name.
    """
    score_values: dict[str, list[float]] = defaultdict(list)

    aggregated_results: dict[str, MetricScore] = {}
    rubric_distribution: dict[str, dict[str, dict]] = defaultdict(lambda: defaultdict(OrderedDict))
    has_rubric: dict[str, bool] = {}

    for item in items:
        for score in item.scores:
            if score.name not in aggregated_results:
                # Keep one running accumulator per score name; distribution
                # details are materialized in a second pass once all values exist.
                aggregated_results[score.name] = MetricScore(
                    name=score.name,
                    value=0.0,
                    stats=ScoreStats(
                        count=0,
                        sum=0,
                        sum_squared=0,
                        min=None,
                        max=None,
                        mean=0,
                        variance=None,
                        stddev=None,
                        nan_count=0,
                    ),
                )
                has_rubric[score.name] = bool(score.stats and score.stats.rubric_distribution)
                if score.stats and score.stats.rubric_distribution:
                    for rubric_stat in score.stats.rubric_distribution:
                        rubric_distribution[score.name][rubric_stat.label] = {
                            "label": rubric_stat.label,
                            "value": rubric_stat.value,
                            "count": 0,
                        }

            if score.stats and score.stats.rubric_distribution:
                for rubric_stat in score.stats.rubric_distribution:
                    if rubric_stat.count:
                        rubric_distribution[score.name][rubric_stat.label]["count"] += 1

            results = aggregated_results[score.name]
            assert results.stats is not None

            # int values can never be NaN in Python, so skipping int and
            # float checks is sufficient.
            # float('nan') is the only way to get a NaN in Python.
            if isinstance(score.value, float) and math.isnan(score.value):
                results.stats.nan_count = (results.stats.nan_count or 0) + 1
                continue

            score_values[score.name].append(score.value)
            results.stats.count = (results.stats.count or 0) + 1
            results.stats.sum = (results.stats.sum or 0) + score.value
            results.stats.sum_squared = (results.stats.sum_squared or 0) + score.value**2
            results.stats.mean = results.stats.sum / results.stats.count

            if results.stats.min is None or score.value < results.stats.min:
                results.stats.min = score.value
            if results.stats.max is None or score.value > results.stats.max:
                results.stats.max = score.value

            results.value = results.stats.mean

    for score_name, values in score_values.items():
        if not values:
            continue
        results = aggregated_results[score_name]
        assert results.stats is not None

        n = len(values)
        mean = results.stats.mean or 0
        # Use population variance because these rows are the full evaluation set,
        # not a sample intended to estimate a larger population.
        variance = sum((v - mean) ** 2 for v in values) / n if n > 0 else 0
        results.stats.variance = variance
        results.stats.stddev = math.sqrt(variance)

    aggregated_scores: list[AggregateScore] = []
    for score_name, metric_score in aggregated_results.items():
        stats = metric_score.stats
        assert stats is not None

        values = score_values[score_name]
        base_name = metric_score.name
        base_count = stats.count or 0
        base_nan_count = stats.nan_count or 0
        base_mean = stats.mean
        base_sum = stats.sum if stats.sum is not None else (None if base_mean is None else (base_mean * base_count))
        base_min = stats.min if stats.min is not None else base_mean
        base_max = stats.max if stats.max is not None else base_mean
        base_variance = stats.variance if stats.variance is not None else (None if base_mean is None else 0.0)
        base_std_dev = stats.stddev if stats.stddev is not None else (None if base_mean is None else 0.0)

        if base_count == 0:
            base_sum = None
            base_mean = None
            base_min = None
            base_max = None
            base_variance = None
            base_std_dev = None

        if has_rubric.get(score_name):
            rubric_dist = [
                RubricScoreStat(label=r["label"], value=r["value"], count=r["count"])
                for r in rubric_distribution[score_name].values()
            ]
            mode_category = None
            if rubric_dist:
                # Break ties deterministically so aggregate output is stable.
                max_count = max(r.count for r in rubric_dist)
                tied = sorted(r.label for r in rubric_dist if r.count == max_count)
                mode_category = tied[0]

            aggregated_scores.append(
                AggregateRubricScore(
                    name=base_name,
                    count=base_count,
                    nan_count=base_nan_count,
                    sum=base_sum,
                    mean=base_mean,
                    min=base_min,
                    max=base_max,
                    variance=base_variance,
                    std_dev=base_std_dev,
                    rubric_distribution=rubric_dist,
                    mode_category=mode_category,
                )
            )
        else:
            if values:
                # Range scores get richer distribution metadata than rubric scores.
                sorted_values = sorted(values)
                percentiles = _compute_percentiles(sorted_values)
                histogram = _compute_histogram(values)
            else:
                percentiles = None
                histogram = Histogram(bins=[])

            aggregated_scores.append(
                AggregateRangeScore(
                    name=base_name,
                    count=base_count,
                    nan_count=base_nan_count,
                    sum=base_sum,
                    mean=base_mean,
                    min=base_min,
                    max=base_max,
                    variance=base_variance,
                    std_dev=base_std_dev,
                    percentiles=percentiles,
                    histogram=histogram,
                )
            )

    return AggregatedMetricResult(scores=aggregated_scores)
