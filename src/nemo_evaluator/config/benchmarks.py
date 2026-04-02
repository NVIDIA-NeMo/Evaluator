# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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

    instruction_template: str | None = None
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
