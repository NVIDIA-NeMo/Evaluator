"""Per-problem sandbox orchestration: protocol, backends, and lifecycle management."""

from nemo_evaluator.sandbox.base import (
    ExecResult,
    OutsideEndpoint,
    Sandbox,
    SandboxSpec,
)
from nemo_evaluator.sandbox.manager import SandboxManager

__all__ = [
    "ExecResult",
    "OutsideEndpoint",
    "Sandbox",
    "SandboxManager",
    "SandboxSpec",
]
