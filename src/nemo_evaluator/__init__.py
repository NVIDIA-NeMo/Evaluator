"""NeMo Evaluator -- environments, solvers, executors."""

__version__ = "0.4.0"

from nemo_evaluator.environments.base import EvalEnvironment, SeedResult, VerifyResult
from nemo_evaluator.environments.definitions import (
    ScorerInput,
    benchmark,
    scorer,
    # Scoring primitives
    answer_line,
    code_sandbox,
    exact_match,
    fuzzy_match,
    multichoice_regex,
    needs_judge,
    numeric_match,
)
from nemo_evaluator.environments.registry import get_environment, list_environments, register
from nemo_evaluator.runner.eval_loop import run_evaluation
from nemo_evaluator.runner.model_client import ModelClient
from nemo_evaluator.runner.solver import AgentSolver, ChatSolver, CompletionSolver, Solver, SolveResult

__all__ = [
    # Core
    "EvalEnvironment", "SeedResult", "VerifyResult",
    "register", "get_environment", "list_environments",
    "run_evaluation", "ModelClient",
    # Solver
    "Solver", "ChatSolver", "CompletionSolver", "AgentSolver", "SolveResult",
    # Benchmark definition API
    "benchmark", "scorer", "ScorerInput",
    "exact_match", "multichoice_regex", "answer_line",
    "fuzzy_match", "numeric_match", "code_sandbox", "needs_judge",
]
