"""BLEU metric runtime implementation."""

import sacrebleu
from nemo_evaluator.sdk.metrics.template_rendering import (
    build_template_context,
    render_default_output_text_candidate_or_raise,
    render_template_or_raise,
    template_metric_repr,
)
from nemo_evaluator.sdk.values.metrics import BLEU
from nemo_evaluator.sdk.values.results import MetricResult, MetricScore

__all__ = ["BLEUMetric"]


class BLEUMetric(BLEU):
    """BLEU metric for sentence- and corpus-level n-gram overlap.

    Evaluator-driven runs render references from dataset fields exposed through
    ``item``. Candidate text can come from explicit dataset fields or from
    ``sample.output_text`` when the evaluator generates model outputs online.
    """

    metric_threshold: float | None = None

    def score_names(self) -> list[str]:
        """Return score keys emitted by this metric."""
        return ["sentence"]

    def _render_references(self, item: dict, sample: dict) -> list[str]:
        """Render all reference templates for one row."""
        context = build_template_context(item, sample)
        metric_repr = template_metric_repr(self)
        references: list[str] = []
        for index, reference in enumerate(self.references):
            rendered_reference = render_template_or_raise(
                template_name=f"references[{index}]",
                template=reference,
                context=context,
                item=item,
                sample=sample,
                metric_repr=metric_repr,
                item_keys_label="reference",
                sample_keys_label="candidate",
            )
            if not isinstance(rendered_reference, str):
                raise TypeError("The reference must be a string.")
            references.append(rendered_reference)
        return references

    def _render_candidate(self, item: dict, sample: dict) -> str:
        """Render the candidate text for one row."""
        if self.candidate:
            context = build_template_context(item, sample)
            prediction = render_template_or_raise(
                template_name="candidate",
                template=self.candidate,
                context=context,
                item=item,
                sample=sample,
                metric_repr=template_metric_repr(self),
                item_keys_label="reference",
                sample_keys_label="candidate",
            )
        else:
            prediction = render_default_output_text_candidate_or_raise(
                sample=sample,
                metric_name=self.__class__.__name__,
            )

        if not isinstance(prediction, str):
            raise TypeError("The candidate must be a string.")
        return prediction

    def metric(self, item: dict, sample: dict, trace: bool | None = None) -> float | bool:
        """Compute a single BLEU score."""
        references = self._render_references(item, sample)
        candidate = self._render_candidate(item, sample)
        score = sacrebleu.sentence_bleu(candidate, references).score.real

        if trace:
            if self.metric_threshold is None:
                raise ValueError(f"metric_threshold is required to compute metric for {self.type.value}")
            return score >= self.metric_threshold

        return score

    async def compute_scores(self, item: dict, sample: dict) -> MetricResult:
        """Compute the scores for the metric."""
        score = self.metric(item, sample)
        return MetricResult(scores=[MetricScore(name="sentence", value=score)])

    async def compute_corpus_scores(self, items: list[dict], samples: list[dict]) -> MetricResult | None:
        """Compute the corpus-level BLEU metric."""
        references_raw = [self._render_references(item, sample) for item, sample in zip(items, samples)]
        # NOTE: because of the bug in sacrebleu, we need to flatten the references
        references = [[reference_set[0] for reference_set in references_raw]]
        candidates = [self._render_candidate(item, sample) for item, sample in zip(items, samples)]
        score = sacrebleu.corpus_bleu(candidates, references)
        return MetricResult(scores=[MetricScore(name="corpus", value=score.score.real)])
