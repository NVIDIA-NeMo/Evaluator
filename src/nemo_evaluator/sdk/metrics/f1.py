"""F1 metric runtime implementation."""

import collections

from nemo_evaluator.sdk.metrics.base import normalize_text
from nemo_evaluator.sdk.metrics.template_rendering import render_reference_and_candidate, template_metric_repr
from nemo_evaluator.sdk.values.metrics import F1
from nemo_evaluator.sdk.values.results import MetricResult, MetricScore

__all__ = ["F1Metric"]


class F1Metric(F1):
    """F1 metric for token-overlap similarity scoring."""

    metric_threshold: float | None = None

    def score_names(self) -> list[str]:
        """Return score keys emitted by this metric."""
        return [self.type.value]

    def metric(self, item: dict, sample: dict, trace=None) -> float | bool:
        """Compute a single raw score for the metric."""
        ground_truth, prediction = render_reference_and_candidate(
            metric_repr=template_metric_repr(self),
            metric_name=self.__class__.__name__,
            reference_template=self.reference,
            candidate_template=self.candidate,
            item=item,
            sample=sample,
        )

        prediction_tokens = normalize_text(prediction).split()
        ground_truth_tokens = normalize_text(ground_truth).split()

        # If either token list is empty, the F1 is 1.0 if they agree, 0.0 otherwise.
        if len(ground_truth_tokens) == 0 or len(prediction_tokens) == 0:
            return float(ground_truth_tokens == prediction_tokens)

        common = collections.Counter(prediction_tokens) & collections.Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0.0

        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        score = (2 * precision * recall) / (precision + recall)

        if trace:
            if self.metric_threshold is None:
                raise ValueError(f"metric_threshold is required to compute metric for {self.type.value}")
            return score >= self.metric_threshold

        return score

    async def compute_scores(self, item: dict, sample: dict) -> MetricResult:
        """Compute the scores for the metric."""
        score = self.metric(item, sample)
        return MetricResult(scores=[MetricScore(name=self.type.value, value=score)])
