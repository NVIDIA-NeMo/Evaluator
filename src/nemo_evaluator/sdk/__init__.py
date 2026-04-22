"""NeMo Evaluator SDK."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _package_version

from nemo_evaluator.sdk.datasets import DatasetLoadError, load_dataset, load_dataset_as_dicts
from nemo_evaluator.sdk.scoring.bleu import BLEUMetric
from nemo_evaluator.sdk.scoring.exact_match import ExactMatchMetric
from nemo_evaluator.sdk.scoring.f1 import F1Metric
from nemo_evaluator.sdk.scoring.llm_judge import LLMJudgeMetric
from nemo_evaluator.sdk.scoring.number_check import NumberCheckMetric
from nemo_evaluator.sdk.scoring.remote import NemoAgentToolkitRemoteMetric, RemoteMetric
from nemo_evaluator.sdk.scoring.rouge import ROUGEMetric
from nemo_evaluator.sdk.scoring.string_check import StringCheckMetric
from nemo_evaluator.sdk.scoring.tool_calling import ToolCallingMetric
from nemo_evaluator.sdk.structured_output import (
    InferenceFn,
    InferenceStructuredOutput,
    StructuredOutput,
    StructuredOutputMode,
    default_structured_output_mode,
    detect_structured_output_mode,
)
from nemo_evaluator.sdk.values import DatasetRows, EvaluationResult

try:
    version = _package_version("nemo-evaluator-sdk")
except PackageNotFoundError:
    version = "0.0.1"

__all__ = [
    "BLEUMetric",
    "DatasetLoadError",
    "DatasetRows",
    "ExactMatchMetric",
    "ExactMatchMetricConfig",
    "F1Metric",
    "F1MetricParams",
    "InferenceFn",
    "InferenceStructuredOutput",
    "LLMJudgeMetric",
    "LLMJudgeMetricConfig",
    "NemoAgentToolkitRemoteMetric",
    "NemoAgentToolkitRemoteMetricConfig",
    "NumberCheckMetric",
    "NumberCheckMetricConfig",
    "RemoteMetric",
    "RemoteMetricConfig",
    "EvaluationResult",
    "ROUGEMetric",
    "StringCheckMetric",
    "StringCheckMetricConfig",
    "StructuredOutput",
    "StructuredOutputMode",
    "ToolCallingMetric",
    "ToolCallingMetricConfig",
    "default_structured_output_mode",
    "detect_structured_output_mode",
    "load_dataset",
    "load_dataset_as_dicts",
    "version",
]
