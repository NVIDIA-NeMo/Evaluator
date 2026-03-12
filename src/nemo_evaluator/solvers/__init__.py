"""Solver implementations: pluggable inference strategies for the eval loop."""

from nemo_evaluator.solvers.base import Solver, SolveResult
from nemo_evaluator.solvers.chat import ChatSolver
from nemo_evaluator.solvers.completion import CompletionSolver
from nemo_evaluator.solvers.cross_encoder import CrossEncoderSolver
from nemo_evaluator.solvers.embedding import EmbeddingSolver
from nemo_evaluator.solvers.nat import NatSolver
from nemo_evaluator.solvers.openclaw import OpenClawSolver
from nemo_evaluator.solvers.sandbox import SandboxSolver
from nemo_evaluator.solvers.vlm import VLMSolver

AgentSolver = SandboxSolver

__all__ = [
    "AgentSolver",
    "ChatSolver",
    "CompletionSolver",
    "CrossEncoderSolver",
    "EmbeddingSolver",
    "NatSolver",
    "OpenClawSolver",
    "SandboxSolver",
    "Solver",
    "SolveResult",
    "VLMSolver",
]
