from __future__ import annotations

import logging
from typing import TYPE_CHECKING

# Type-only imports for static analysis (not imported at runtime)
if TYPE_CHECKING:
    from ragas import EvaluationDataset
    from ragas.llms import LangchainLLMWrapper

from nemo_evaluator.sdk.enums import MetricType
from nemo_evaluator.sdk.scoring.ragas.base import BaseRAGASMetric

# Lazy imports for RAGAS metric classes - these are getter functions that defer
# the expensive RAGAS/langchain imports (~20-30s) until first use.
from nemo_evaluator.sdk.scoring.ragas.imports import (
    get_agent_goal_accuracy_with_reference_class,
    get_agent_goal_accuracy_without_reference_class,
    get_answer_accuracy_class,
    get_context_entity_recall_class,
    get_context_precision_class,
    get_context_recall_class,
    get_context_relevance_class,
    get_faithfulness_class,
    get_noise_sensitivity_class,
    get_response_groundedness_class,
    get_response_relevancy_class,
    get_tool_call_accuracy_class,
    get_topic_adherence_score_class,
)
from nemo_evaluator.sdk.values import metrics

log = logging.getLogger(__name__)


# Agentic metrics


class TopicAdherenceMetric(metrics.TopicAdherence, BaseRAGASMetric):
    """Metric for measuring topic adherence."""

    def _metric(self, data: EvaluationDataset, llm_judge: LangchainLLMWrapper | None) -> dict[str, float]:
        metric_mode = self.metric_mode
        TopicAdherenceScore = get_topic_adherence_score_class()
        metric = TopicAdherenceScore(llm=llm_judge, mode=metric_mode)
        return self._run_evaluate(data, [metric])


class ToolCallAccuracyMetric(metrics.ToolCallAccuracy, BaseRAGASMetric):
    """Metric for measuring tool call accuracy."""

    def _metric(self, data: EvaluationDataset, llm_judge: LangchainLLMWrapper | None) -> dict[str, float]:
        ToolCallAccuracy = get_tool_call_accuracy_class()
        metric = ToolCallAccuracy()
        return self._run_evaluate(data, [metric])


class AgentGoalAccuracyMetric(metrics.AgentGoalAccuracy, BaseRAGASMetric):
    """Metric for measuring agent goal accuracy."""

    def _metric(self, data: EvaluationDataset, llm_judge: LangchainLLMWrapper | None) -> dict[str, float]:
        # Choose between with/without reference based on params
        if self.use_reference:
            metric_class = get_agent_goal_accuracy_with_reference_class()
        else:
            metric_class = get_agent_goal_accuracy_without_reference_class()
        metric = metric_class(llm=llm_judge)
        return self._run_evaluate(data, [metric])


# Nvidia metrics


class AnswerAccuracyMetric(metrics.AnswerAccuracy, BaseRAGASMetric):
    """Metric for measuring answer accuracy."""

    def _metric(self, data: EvaluationDataset, llm_judge: LangchainLLMWrapper | None) -> dict[str, float]:
        AnswerAccuracy = get_answer_accuracy_class()
        metric = AnswerAccuracy(llm=llm_judge)
        return self._run_evaluate(data, [metric])


class ContextRelevanceMetric(metrics.ContextRelevance, BaseRAGASMetric):
    """Metric for measuring context relevance."""

    def _metric(self, data: EvaluationDataset, llm_judge: LangchainLLMWrapper | None) -> dict[str, float]:
        ContextRelevance = get_context_relevance_class()
        metric = ContextRelevance(llm=llm_judge)
        return self._run_evaluate(data, [metric])


class ResponseGroundednessMetric(metrics.ResponseGroundedness, BaseRAGASMetric):
    """Metric for measuring response groundedness."""

    def _metric(self, data: EvaluationDataset, llm_judge: LangchainLLMWrapper | None) -> dict[str, float]:
        ResponseGroundedness = get_response_groundedness_class()
        metric = ResponseGroundedness(llm=llm_judge)
        return self._run_evaluate(data, [metric])


# RAG Metrics


class ContextRecallMetric(metrics.ContextRecall, BaseRAGASMetric):
    """Metric for measuring context recall."""

    def _metric(self, data: EvaluationDataset, llm_judge: LangchainLLMWrapper | None) -> dict[str, float]:
        ContextRecall = get_context_recall_class()
        metric = ContextRecall(llm=llm_judge)
        return self._run_evaluate(data, [metric])


class ContextPrecisionMetric(metrics.ContextPrecision, BaseRAGASMetric):
    """Metric for measuring context precision."""

    def _metric(self, data: EvaluationDataset, llm_judge: LangchainLLMWrapper | None) -> dict[str, float]:
        ContextPrecision = get_context_precision_class()
        metric = ContextPrecision(llm=llm_judge)
        return self._run_evaluate(data, [metric])


class ContextEntityRecallMetric(metrics.ContextEntityRecall, BaseRAGASMetric):
    """Metric for measuring context entity recall."""

    def _metric(self, data: EvaluationDataset, llm_judge: LangchainLLMWrapper | None) -> dict[str, float]:
        ContextEntityRecall = get_context_entity_recall_class()
        metric = ContextEntityRecall(llm=llm_judge)
        return self._run_evaluate(data, [metric])


class ResponseRelevancyMetric(metrics.ResponseRelevancy, BaseRAGASMetric):
    """Metric for measuring response relevancy."""

    def _metric(self, data: EvaluationDataset, llm_judge: LangchainLLMWrapper | None) -> dict[str, float]:
        embeddings_client = self._get_embeddings_client()
        # Strictness defines number of parallel questions generated. NIM can only generate 1.
        # Having a configurable parameter allows computation with non-NIM judges.
        ResponseRelevancy = get_response_relevancy_class()
        metric = ResponseRelevancy(llm=llm_judge, embeddings=embeddings_client, strictness=self.strictness)
        return self._run_evaluate(data, [metric])


class FaithfulnessMetric(metrics.Faithfulness, BaseRAGASMetric):
    """Metric for measuring faithfulness."""

    def _metric(self, data: EvaluationDataset, llm_judge: LangchainLLMWrapper | None) -> dict[str, float]:
        Faithfulness = get_faithfulness_class()
        metric = Faithfulness(llm=llm_judge)
        return self._run_evaluate(data, [metric])


class NoiseSensitivityMetric(metrics.NoiseSensitivity, BaseRAGASMetric):
    """Metric for measuring noise sensitivity."""

    def _metric(self, data: EvaluationDataset, llm_judge: LangchainLLMWrapper | None) -> dict[str, float]:
        NoiseSensitivity = get_noise_sensitivity_class()
        metric = NoiseSensitivity(llm=llm_judge)
        return self._run_evaluate(data, [metric])


# List of all RAGAS metrics
ragas_metrics = [
    AgentGoalAccuracyMetric,
    AnswerAccuracyMetric,
    ContextEntityRecallMetric,
    ContextPrecisionMetric,
    ContextRecallMetric,
    ContextRelevanceMetric,
    FaithfulnessMetric,
    NoiseSensitivityMetric,
    ResponseGroundednessMetric,
    ResponseRelevancyMetric,
    ToolCallAccuracyMetric,
    TopicAdherenceMetric,
]

# Map of Metric enum values to class for all RAGAS metrics
RAGAS_METRIC_CLASSES: dict[MetricType, type[BaseRAGASMetric]] = {
    metric.model_fields["type"].default: metric for metric in ragas_metrics
}
