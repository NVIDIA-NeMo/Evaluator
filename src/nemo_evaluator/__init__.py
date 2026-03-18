"""NeMo Evaluator -- environments, solvers, evaluation orchestration."""

__version__ = "0.9.0"

from nemo_evaluator.environments.base import EvalEnvironment, SeedResult, VerifyResult
from nemo_evaluator.environments.byob import benchmark, scorer
from nemo_evaluator.environments.registry import get_environment, list_environments, load_benchmark_file, register
from nemo_evaluator.eval.config import EndpointType
from nemo_evaluator.runner.eval_loop import run_evaluation
from nemo_evaluator.runner.model_client import ModelClient
from nemo_evaluator.solvers import (
    ChatSolver,
    CompletionSolver,
    NatSolver,
    OpenClawSolver,
    Solver,
    SolveResult,
    VLMSolver,
)
from nemo_evaluator.scoring import (
    ScorerInput,
    answer_line,
    code_sandbox,
    code_sandbox_async,
    exact_match,
    fuzzy_match,
    multichoice_regex,
    needs_judge,
    numeric_match,
)

__all__ = [
    # Core
    "EvalEnvironment", "SeedResult", "VerifyResult",
    "register", "get_environment", "list_environments", "load_benchmark_file",
    "run_evaluation", "ModelClient", "EndpointType",
    # Solver
    "Solver", "ChatSolver", "CompletionSolver",
    "NatSolver", "OpenClawSolver", "VLMSolver",
    "SolveResult",
    # Benchmark definition API
    "benchmark", "scorer", "ScorerInput",
    # Scoring primitives
    "exact_match", "multichoice_regex", "answer_line",
    "fuzzy_match", "numeric_match", "code_sandbox", "code_sandbox_async",
    "needs_judge",
]
