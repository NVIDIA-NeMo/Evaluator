# Apply Git patch before any other imports that might use GitPython (like ragas)
from nemo_evaluator.sdk.scoring.ragas.git_patch import apply_git_patch

apply_git_patch()

# ruff: noqa: E402
"""Lazy imports for RAGAS library to avoid slow startup times.

RAGAS and its dependencies (langchain, etc.) take 20-30 seconds to import.
This module provides lazy loading so that the import cost is only paid when
RAGAS metrics are actually used, not when the evaluator service starts.

Usage:
    # Instead of: from ragas import EvaluationDataset
    # Use: EvaluationDataset = get_evaluation_dataset_class()
"""
import os
from functools import cache

# Disable RAGAS telemetry/tracking before importing ragas
os.environ["RAGAS_DO_NOT_TRACK"] = "true"


@cache
def _load_ragas():
    """Load all RAGAS modules. Cached so imports only happen once."""
    import ragas
    from ragas import EvaluationDataset, RunConfig, evaluate
    from ragas.embeddings.base import LangchainEmbeddingsWrapper
    from ragas.llms.base import LangchainLLMWrapper
    from ragas.metrics import (
        AgentGoalAccuracyWithoutReference,
        AgentGoalAccuracyWithReference,
        AnswerAccuracy,
        AnswerCorrectness,
        AnswerSimilarity,
        ContextEntityRecall,
        ContextPrecision,
        ContextRecall,
        ContextRelevance,
        Faithfulness,
        NoiseSensitivity,
        ResponseGroundedness,
        ResponseRelevancy,
        ToolCallAccuracy,
        TopicAdherenceScore,
    )

    # Return as a dict for easy access
    return {
        "ragas": ragas,
        "EvaluationDataset": EvaluationDataset,
        "RunConfig": RunConfig,
        "evaluate": evaluate,
        "LangchainEmbeddingsWrapper": LangchainEmbeddingsWrapper,
        "LangchainLLMWrapper": LangchainLLMWrapper,
        "AgentGoalAccuracyWithoutReference": AgentGoalAccuracyWithoutReference,
        "AgentGoalAccuracyWithReference": AgentGoalAccuracyWithReference,
        "AnswerAccuracy": AnswerAccuracy,
        "AnswerCorrectness": AnswerCorrectness,
        "AnswerSimilarity": AnswerSimilarity,
        "ContextEntityRecall": ContextEntityRecall,
        "ContextPrecision": ContextPrecision,
        "ContextRecall": ContextRecall,
        "ContextRelevance": ContextRelevance,
        "Faithfulness": Faithfulness,
        "NoiseSensitivity": NoiseSensitivity,
        "ResponseGroundedness": ResponseGroundedness,
        "ResponseRelevancy": ResponseRelevancy,
        "ToolCallAccuracy": ToolCallAccuracy,
        "TopicAdherenceScore": TopicAdherenceScore,
    }


# =============================================================================
# Lazy accessor functions
# =============================================================================


def get_evaluation_dataset_class() -> type:
    """Get the EvaluationDataset class."""
    return _load_ragas()["EvaluationDataset"]


def get_run_config_class() -> type:
    """Get the RunConfig class."""
    return _load_ragas()["RunConfig"]


def get_evaluate_function():
    """Get the evaluate function."""
    return _load_ragas()["evaluate"]


def get_langchain_embeddings_wrapper_class() -> type:
    """Get the LangchainEmbeddingsWrapper class."""
    return _load_ragas()["LangchainEmbeddingsWrapper"]


def get_langchain_llm_wrapper_class() -> type:
    """Get the LangchainLLMWrapper class."""
    return _load_ragas()["LangchainLLMWrapper"]


def get_topic_adherence_score_class() -> type:
    """Get the TopicAdherenceScore metric class."""
    return _load_ragas()["TopicAdherenceScore"]


def get_tool_call_accuracy_class() -> type:
    """Get the ToolCallAccuracy metric class."""
    return _load_ragas()["ToolCallAccuracy"]


def get_agent_goal_accuracy_with_reference_class() -> type:
    """Get the AgentGoalAccuracyWithReference metric class."""
    return _load_ragas()["AgentGoalAccuracyWithReference"]


def get_agent_goal_accuracy_without_reference_class() -> type:
    """Get the AgentGoalAccuracyWithoutReference metric class."""
    return _load_ragas()["AgentGoalAccuracyWithoutReference"]


def get_answer_accuracy_class() -> type:
    """Get the AnswerAccuracy metric class."""
    return _load_ragas()["AnswerAccuracy"]


def get_context_relevance_class() -> type:
    """Get the ContextRelevance metric class."""
    return _load_ragas()["ContextRelevance"]


def get_response_groundedness_class() -> type:
    """Get the ResponseGroundedness metric class."""
    return _load_ragas()["ResponseGroundedness"]


def get_context_recall_class() -> type:
    """Get the ContextRecall metric class."""
    return _load_ragas()["ContextRecall"]


def get_context_precision_class() -> type:
    """Get the ContextPrecision metric class."""
    return _load_ragas()["ContextPrecision"]


def get_context_entity_recall_class() -> type:
    """Get the ContextEntityRecall metric class."""
    return _load_ragas()["ContextEntityRecall"]


def get_response_relevancy_class() -> type:
    """Get the ResponseRelevancy metric class."""
    return _load_ragas()["ResponseRelevancy"]


def get_faithfulness_class() -> type:
    """Get the Faithfulness metric class."""
    return _load_ragas()["Faithfulness"]


def get_noise_sensitivity_class() -> type:
    """Get the NoiseSensitivity metric class."""
    return _load_ragas()["NoiseSensitivity"]


__all__ = [
    "get_evaluation_dataset_class",
    "get_run_config_class",
    "get_evaluate_function",
    "get_langchain_embeddings_wrapper_class",
    "get_langchain_llm_wrapper_class",
    "get_topic_adherence_score_class",
    "get_tool_call_accuracy_class",
    "get_agent_goal_accuracy_with_reference_class",
    "get_agent_goal_accuracy_without_reference_class",
    "get_answer_accuracy_class",
    "get_context_relevance_class",
    "get_response_groundedness_class",
    "get_context_recall_class",
    "get_context_precision_class",
    "get_context_entity_recall_class",
    "get_response_relevancy_class",
    "get_faithfulness_class",
    "get_noise_sensitivity_class",
]
