"""Per-problem sandbox orchestration: protocol, backends, and lifecycle management."""

from nemo_evaluator.sandbox.base import (
    ExecResult,
    OutsideEndpoint,
    Sandbox,
    SandboxSpec,
)
from nemo_evaluator.sandbox.strategies import LifecycleContext, NoSandbox, StatefulSandbox, StatelessSandbox, pick_lifecycle
from nemo_evaluator.sandbox.manager import SandboxManager

__all__ = [
    "ExecResult",
    "LifecycleContext",
    "NoSandbox",
    "OutsideEndpoint",
    "Sandbox",
    "SandboxManager",
    "SandboxSpec",
    "StatefulSandbox",
    "StatelessSandbox",
    "pick_lifecycle",
]
