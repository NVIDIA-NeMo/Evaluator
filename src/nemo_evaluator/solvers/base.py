"""Solver protocol: pluggable inference strategy for the eval loop."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from nemo_evaluator.environments.base import SeedResult
from nemo_evaluator.observability.types import ModelResponse

if TYPE_CHECKING:
    from nemo_evaluator.sandbox.base import Sandbox

logger = logging.getLogger(__name__)


@dataclass
class SolveResult:
    response: str
    model_response: ModelResponse | None = None
    trajectory: list[dict[str, Any]] = field(default_factory=list)


class Solver(Protocol):
    """Solvers MAY also accept ``sandbox: Sandbox | None = None`` in solve().

    The eval loop detects this via ``_solver_accepts_sandbox()`` introspection
    and passes the sandbox when available.
    """

    async def solve(self, task: SeedResult) -> SolveResult: ...
