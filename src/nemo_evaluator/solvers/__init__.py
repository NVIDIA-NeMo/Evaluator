"""Solver implementations: pluggable inference strategies for the eval loop."""

from nemo_evaluator.solvers.base import Solver, SolveResult
from nemo_evaluator.solvers.chat import ChatSolver
from nemo_evaluator.solvers.completion import CompletionSolver
from nemo_evaluator.solvers.nat import NatSolver
from nemo_evaluator.solvers.openclaw import OpenClawSolver
from nemo_evaluator.solvers.vlm import VLMSolver

__all__ = [
    "ChatSolver",
    "CompletionSolver",
    "HarborSolver",
    "NatSolver",
    "OpenClawSolver",
    "Solver",
    "SolveResult",
    "VLMSolver",
]


def __getattr__(name: str):
    if name == "HarborSolver":
        from nemo_evaluator.solvers.harbor import HarborSolver
        return HarborSolver
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
