"""ROUGE metric runtime implementation."""

from functools import cached_property
from typing import ClassVar, Literal

from nemo_evaluator.sdk.scoring.base import NELScorerMixin
from nemo_evaluator.sdk.scoring.template_rendering import render_reference_and_candidate, template_metric_repr
from nemo_evaluator.sdk.values.metrics import ROUGE
from nemo_evaluator.sdk.values.results import MetricResult, MetricScore

__all__ = ["ROUGEMetric", "RougeScoreName"]

RougeScoreName = Literal["rouge_1_score", "rouge_2_score", "rouge_3_score", "rouge_L_score"]


class ROUGEMetric(NELScorerMixin, ROUGE):
    """ROUGE metric for overlap-based summarization quality scoring.

    Evaluator-driven runs expose dataset fields through ``item`` and generated
    model outputs through ``sample.output_text`` for online execution.
    """

    metric_threshold: float | None = None
    metric_threshold_score: RougeScoreName | None = None

    scores_mapping: ClassVar[dict[RougeScoreName, str]] = {
        # Maps the public MetricResult score name to the underlying rouge_scorer key.
        "rouge_1_score": "rouge1",
        "rouge_2_score": "rouge2",
        "rouge_3_score": "rouge3",
        "rouge_L_score": "rougeL",
    }

    @cached_property
    def _scorer(self):
        """Lazily initialize the ROUGE scorer to avoid expensive import-time setup."""
        # The RougeScorer loads NLTK's stemmer/tokenizer machinery, so keeping
        # this lazy avoids unnecessary startup cost for callers that never use
        # ROUGE in a given process.
        from rouge_score import rouge_scorer

        return rouge_scorer.RougeScorer(["rouge1", "rouge2", "rouge3", "rougeL"], use_stemmer=True)

    def score_names(self) -> list[str]:
        """Return score keys emitted by this metric."""
        return list(self.scores_mapping.keys())

    def _metric(self, item: dict, sample: dict) -> dict:
        """Compute raw ROUGE scores for one item/sample pair."""
        ground_truth, prediction = render_reference_and_candidate(
            metric_repr=template_metric_repr(self),
            metric_name=self.__class__.__name__,
            reference_template=self.reference,
            candidate_template=self.candidate,
            item=item,
            sample=sample,
        )
        return self._scorer.score(ground_truth, prediction)

    def metric(self, item: dict, sample: dict, trace=None) -> float | bool:
        """Compute a single raw score for the metric."""
        if not (self.metric_threshold and self.metric_threshold_score):
            raise ValueError(
                f"metric_threshold and metric_threshold_score are required to compute metric for {self.type.value}"
            )

        scores = self._metric(item, sample)
        score = scores[self.scores_mapping[self.metric_threshold_score]].fmeasure

        if trace:
            return score >= self.metric_threshold

        return score

    async def compute_scores(self, item: dict, sample: dict) -> MetricResult:
        """Compute structured score output for one item/sample pair."""
        scores = self._metric(item, sample)
        return MetricResult(
            scores=[
                MetricScore(name=score_name, value=scores[score_key].fmeasure)
                for score_name, score_key in self.scores_mapping.items()
            ]
        )
