"""Benchmark configuration schema."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .sandboxes import NoSandbox, SandboxConfig
from .scoring import ScoringConfig
from .solvers import (
    AgentSolverConfig,
    HarborSolverConfig,
    OpenClawSolverConfig,
    SolverConfig,
    ToolCallingSolverConfig,
)


class BenchmarkConfig(BaseModel):
    """Single benchmark entry."""

    model_config = ConfigDict(extra="forbid")

    name: str
    solver: SolverConfig
    repeats: int = 1
    max_problems: int | None = None
    max_concurrent: int = 32
    fewshot: int | None = None
    context_window: int | None = None
    timeout: float = 1800.0
    skip_failed: bool = False
    max_system_retries: int = 3

    verifier: str | None = None

    sandbox: SandboxConfig = Field(default_factory=NoSandbox)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)

    @model_validator(mode="after")
    def _solver_requires_sandbox(self) -> BenchmarkConfig:
        _NEEDS_SANDBOX = (
            HarborSolverConfig,
            AgentSolverConfig,
            OpenClawSolverConfig,
        )
        if isinstance(self.solver, _NEEDS_SANDBOX):
            if isinstance(self.sandbox, NoSandbox):
                raise ValueError(
                    f"solver type '{self.solver.type}' requires a sandbox (docker, apptainer, slurm, or ecs_fargate)"
                )
        if (
            isinstance(self.solver, ToolCallingSolverConfig)
            and self.solver.sandbox_tools
            and isinstance(self.sandbox, NoSandbox)
        ):
            raise ValueError(
                "tool_calling solver with sandbox_tools: true requires a sandbox "
                "(docker, apptainer, slurm, or ecs_fargate)"
            )
        return self
