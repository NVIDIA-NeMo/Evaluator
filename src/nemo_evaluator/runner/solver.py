"""Backward-compatibility shim -- solvers now live in nemo_evaluator.solvers."""
from nemo_evaluator.solvers import (  # noqa: F401
    AgentSolver,
    ChatSolver,
    CompletionSolver,
    CrossEncoderSolver,
    EmbeddingSolver,
    SandboxSolver,
    Solver,
    SolveResult,
    VLMSolver,
)
