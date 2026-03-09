"""NeMo Evaluator -- environments, solvers, executors."""

__version__ = "0.6.0"

from nemo_evaluator.environments.base import EvalEnvironment, SeedResult, VerifyResult
from nemo_evaluator.environments.define import benchmark, scorer
from nemo_evaluator.environments.registry import get_environment, list_environments, register
from nemo_evaluator.runner.eval_loop import run_evaluation
from nemo_evaluator.runner.model_client import ModelClient
from nemo_evaluator.runner.solver import (
    AgentSolver,
    ChatSolver,
    CompletionSolver,
    CrossEncoderSolver,
    EmbeddingSolver,
    Solver,
    SolveResult,
    VLMSolver,
)
from nemo_evaluator.scoring import (
    ScorerInput,
    answer_line,
    code_sandbox,
    exact_match,
    fuzzy_match,
    multichoice_regex,
    needs_judge,
    numeric_match,
)

__all__ = [
    # Core
    "EvalEnvironment", "SeedResult", "VerifyResult",
    "register", "get_environment", "list_environments",
    "run_evaluation", "ModelClient",
    # Solver
    "Solver", "ChatSolver", "CompletionSolver", "AgentSolver",
    "VLMSolver", "EmbeddingSolver", "CrossEncoderSolver", "SolveResult",
    # Benchmark definition API
    "benchmark", "scorer", "ScorerInput",
    # Scoring primitives
    "exact_match", "multichoice_regex", "answer_line",
    "fuzzy_match", "numeric_match", "code_sandbox", "needs_judge",
]
