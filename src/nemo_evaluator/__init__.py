"""NeMo Evaluator – benchmark environments, evaluation runner, and decision-grade scoring."""

__version__ = "0.3.0"

from nemo_evaluator.environments.base import (
    EvalEnvironment, SeedResult, VerifyResult,
    StepEnvironment, Observation,
)
from nemo_evaluator.environments.registry import get_environment, list_environments, register
from nemo_evaluator.runner.eval_loop import run_evaluation
from nemo_evaluator.runner.model_client import ModelClient

__all__ = [
    "EvalEnvironment",
    "SeedResult",
    "VerifyResult",
    "StepEnvironment",
    "Observation",
    "register",
    "get_environment",
    "list_environments",
    "run_evaluation",
    "ModelClient",
]
