"""Solver configuration schemas (simple, harbor, agent, tool_calling, etc.)."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Discriminator, Field, Tag, field_validator, model_validator

from .services import GenerationConfig


class SimpleSolver(BaseModel):
    """Unified solver for chat completions, text completions, and VLM.
    The service's `protocol` field determines the API format."""

    type: Literal["simple"] = "simple"
    service: str
    system_prompt: str | None = None
    generation: GenerationConfig | None = None
    image_detail: str = "auto"


class HarborSolverConfig(BaseModel):
    """Harbor agent solver — runs a Harbor agent inside a sandbox."""

    type: Literal["harbor"] = "harbor"
    service: str
    agent: str
    agent_kwargs: dict[str, Any] = Field(default_factory=dict)
    container_env: dict[str, str] = Field(default_factory=dict)


class AgentSolverConfig(BaseModel):
    """Agent-as-library solver — imports agent into NEL process."""

    type: Literal["agent"] = "agent"
    service: str
    framework: str = "harbor"
    agent: str
    agent_kwargs: dict[str, Any] = Field(default_factory=dict)


class ToolCallingSolverConfig(BaseModel):
    """NEL-native ReAct loop: model call -> parse tool_calls -> dispatch.

    At least one of ``resource_service`` (Gym HTTP tools) or ``sandbox_tools``
    (bash/file tools in sandbox) must be configured.
    """

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

    type: Literal["gym_delegation"] = "gym_delegation"
    service: str
    gym_service: str
    gym_agent: str | None = None
    trust_reward: bool = False


class NatSolverConfig(BaseModel):
    """NAT agent solver."""

    type: Literal["nat"] = "nat"
    service: str


class OpenClawSolverConfig(BaseModel):
    """OpenClaw subprocess solver."""

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
