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
"""Solver configuration schemas (simple, harbor, agent, tool_calling, etc.)."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Discriminator, Field, Tag, field_validator, model_validator

from .services import GenerationConfig


class SimpleSolver(BaseModel):
    """Unified solver for chat completions, text completions, and VLM.
    The service's `protocol` field determines the API format."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["simple"] = "simple"
    service: str
    system_prompt: str | None = None
    generation: GenerationConfig | None = None
    image_detail: str = "auto"


class HarborSolverConfig(BaseModel):
    """Harbor agent solver — runs a Harbor agent inside a sandbox."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["harbor"] = "harbor"
    service: str
    agent: str
    agent_kwargs: dict[str, Any] = Field(default_factory=dict)
    container_env: dict[str, str] = Field(default_factory=dict)
    run_timeout: float | None = Field(
        default=None,
        description=(
            "Wall-clock timeout (seconds) for the agent.run() call. "
            "When the agent hangs (e.g. unreachable LLM), this prevents "
            "wasting the full benchmark timeout. Defaults to sandbox timeout."
        ),
    )
    cmd_timeout: float | None = Field(
        default=None,
        description=(
            "Hard ceiling (seconds) on any single terminal command. "
            "Prevents a long-running command from consuming the entire "
            "run_timeout budget. None means no ceiling (SDK default)."
        ),
    )
    timeout_strategy: Literal["override", "task", "max"] = Field(
        default="override",
        description=(
            "How to resolve agent timeout when the task defines its own "
            "timeout in task.toml. 'override': NEL timeout always wins. "
            "'task': use per-task timeout (NEL timeout as fallback). "
            "'max': use the larger of NEL and task timeout."
        ),
    )
    max_agent_timeout: float | None = Field(
        default=None,
        description=(
            "Hard ceiling (seconds) on agent timeout regardless of strategy. "
            "Useful with 'task' or 'max' strategy to cap runaway timeouts."
        ),
    )


class AgentSolverConfig(BaseModel):
    """Agent-as-library solver — imports and runs the agent in-process."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["agent"] = "agent"
    service: str
    framework: str = "harbor"
    agent: str
    agent_kwargs: dict[str, Any] = Field(default_factory=dict)


class ToolCallingSolverConfig(BaseModel):
    """Evaluator-native ReAct loop: model call -> parse tool_calls -> dispatch.

    At least one of ``resource_service`` (Gym HTTP tools) or ``sandbox_tools``
    (bash/file tools in sandbox) must be configured.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["tool_calling"] = "tool_calling"
    service: str
    resource_service: str | None = None
    sandbox_tools: bool = False
    max_turns: int = 50
    system_prompt: str | None = None
    generation: GenerationConfig | None = None
    tool_timeout: float = 180.0
    max_output_chars: int = 16384
    response_mode: Literal["last_message", "sandbox_artifact"] = "last_message"

    @model_validator(mode="after")
    def _check_has_tools(self) -> ToolCallingSolverConfig:
        if not self.resource_service and not self.sandbox_tools:
            raise ValueError(
                "tool_calling solver requires at least one tool source: "
                "set resource_service (Gym HTTP tools) and/or sandbox_tools: true"
            )
        return self


class GymDelegationSolverConfig(BaseModel):
    """Delegates entire agent loop to a Gym agent server (/run endpoint)."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["gym_delegation"] = "gym_delegation"
    service: str
    gym_service: str
    gym_agent: str | None = None
    trust_reward: bool = False


class NatSolverConfig(BaseModel):
    """NAT agent solver."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["nat"] = "nat"
    service: str


class OpenClawSolverConfig(BaseModel):
    """OpenClaw subprocess solver."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["openclaw"] = "openclaw"
    service: str
    thinking: str = "high"
    context_window: int = 131072
    max_concurrent: int = 4
    config_path: str | None = None
    skip_preflight: bool = False
    openclaw_bin: str = "openclaw"


class ContainerSolverConfig(BaseModel):
    """Container URI solver (NeMo Skills, etc.).

    The container receives a **legacy Evaluator** ``run_config.yaml`` with
    ``config.type``, ``target.api_endpoint``, and any extra ``params``
    defined here merged into ``config.params``.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["container"] = "container"
    service: str
    uri: str
    endpoint_type: str | None = None
    params: dict[str, Any] | None = None

    @field_validator("uri")
    @classmethod
    def _validate_uri(cls, v: str) -> str:
        if not v.startswith("container://"):
            raise ValueError("container solver uri must start with 'container://'")
        return v


class CustomSolverConfig(BaseModel):
    """Plugin solver — dynamically imported from class_path."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["custom"] = "custom"
    service: str | None = None
    class_path: str
    config: dict[str, Any] = Field(default_factory=dict)


def _solver_discriminator(v: Any) -> str:
    if isinstance(v, dict):
        t = v.get("type")
        if t is None:
            raise ValueError("Solver config must include a 'type' field")
        return t
    t = getattr(v, "type", None)
    if t is None:
        raise ValueError(f"Cannot determine solver type from {type(v).__name__}. Expected a dict with a 'type' field.")
    return t


SolverConfig = Annotated[
    Annotated[SimpleSolver, Tag("simple")]
    | Annotated[HarborSolverConfig, Tag("harbor")]
    | Annotated[AgentSolverConfig, Tag("agent")]
    | Annotated[ToolCallingSolverConfig, Tag("tool_calling")]
    | Annotated[GymDelegationSolverConfig, Tag("gym_delegation")]
    | Annotated[NatSolverConfig, Tag("nat")]
    | Annotated[OpenClawSolverConfig, Tag("openclaw")]
    | Annotated[ContainerSolverConfig, Tag("container")]
    | Annotated[CustomSolverConfig, Tag("custom")],
    Discriminator(_solver_discriminator),
]
