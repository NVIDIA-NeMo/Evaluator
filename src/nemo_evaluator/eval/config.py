"""Evaluation config schema (Pydantic).

Fully explicit, docker-compose-style configuration for evaluations.
No sugar, no shorthands, no implicit defaults for architectural choices.

Design:
  - Discriminated unions for services, solvers, sandboxes, clusters.
  - Services carry full URL + protocol + generation + interceptors.
  - Solver owns its service reference — benchmark is about WHAT to evaluate.
  - Composable scoring pipeline with multiple judges/metrics.
  - Named node pools for SLURM topology.
  - Named sandboxes for reuse across benchmarks.
  - Cross-reference validation at parse time.
"""

from __future__ import annotations

import os
import re
import warnings
from typing import Annotated, Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, Discriminator, Field, Tag, field_validator, model_validator


Protocol = Literal["chat_completions", "completions", "responses"]


# ═══════════════════════════════════════════════════════════════════════
#  SECTION 1 — SERVICES
# ═══════════════════════════════════════════════════════════════════════


class GenerationConfig(BaseModel):
    """Default generation parameters for a service.  Solvers may override
    these per-benchmark via their own `generation` field.

    Merge semantics: when a solver specifies generation, each non-None
    field overrides the corresponding service-level field.  Fields left
    as None inherit from the service's generation config."""

    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    max_tokens: int | None = Field(default=None, gt=0)
    stop: list[str] | None = None
    frequency_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    presence_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)

    def merge_onto(self, base: GenerationConfig) -> GenerationConfig:
        """Return a new config with self's non-None fields overriding base."""
        merged = {}
        for field_name in self.model_fields:
            override = getattr(self, field_name)
            merged[field_name] = override if override is not None else getattr(base, field_name)
        return GenerationConfig(**merged)


class InterceptorConfig(BaseModel):
    """A LiteLLM interceptor attached to a service's proxy."""

    name: str
    config: dict[str, Any] = Field(default_factory=dict)


class _ModelServerBase(BaseModel):
    """Shared fields for locally-deployed model servers (vllm, sglang,
    nim, docker_model).  NEL starts these servers and knows their URL."""

    model: str
    port: int = 8000
    protocol: Protocol
    tensor_parallel_size: int | None = None
    pipeline_parallel_size: int | None = None
    data_parallel_size: int | None = None
    num_nodes: int = 1
    gpus: list[int] | int | None = None
    image: str | None = None
    health_path: str = "/health"
    startup_timeout: float = 600.0
    extra_env: dict[str, str] = Field(default_factory=dict)
    extra_args: list[str] = Field(default_factory=list)
    setup_commands: list[str] = Field(default_factory=list)
    container_mounts: list[str] = Field(default_factory=list)
    reasoning_pattern: str | None = None
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    interceptors: list[InterceptorConfig] = Field(default_factory=list)
    proxy_verbose: bool = False
    depends_on: list[str] = Field(default_factory=list)
    node_pool: str | None = None

    @property
    def is_model_server(self) -> bool:
        return True

    @property
    def is_managed(self) -> bool:
        return True

    @property
    def base_url(self) -> str:
        return f"http://localhost:{self.port}/v1"


class VllmService(_ModelServerBase):
    type: Literal["vllm"] = "vllm"


class SglangService(_ModelServerBase):
    type: Literal["sglang"] = "sglang"


class NimService(_ModelServerBase):
    type: Literal["nim"] = "nim"


class DockerModelService(_ModelServerBase):
    type: Literal["docker_model"] = "docker_model"


class ExternalApiService(BaseModel):
    """Pre-deployed model / judge behind an HTTP endpoint.

    `url` is the FULL URL where requests are sent (always include path).
    `protocol` is the wire format (can decouple for experimental endpoints).
    """

    type: Literal["api"] = "api"
    url: str
    protocol: Protocol
    model: str | None = None
    api_key: str | None = None
    health_path: str | None = None
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    interceptors: list[InterceptorConfig] = Field(default_factory=list)
    proxy_verbose: bool = False

    @property
    def is_model_server(self) -> bool:
        return True

    @property
    def is_managed(self) -> bool:
        return False

    @property
    def base_url(self) -> str:
        return self.url

    @field_validator("url")
    @classmethod
    def _validate_url(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            raise ValueError(f"Service url must start with http:// or https://, got: {v!r}")
        return v

    @model_validator(mode="after")
    def _warn_url_protocol_mismatch(self) -> ExternalApiService:
        _SUFFIX_TO_PROTOCOL = [
            ("/chat/completions", "chat_completions"),
            ("/completions", "completions"),
            ("/responses", "responses"),
        ]
        path = urlparse(self.url).path
        for suffix, expected in _SUFFIX_TO_PROTOCOL:
            if path.endswith(suffix):
                if self.protocol != expected:
                    warnings.warn(
                        f"Service url '{self.url}' ends with '{suffix}' but "
                        f"protocol is '{self.protocol}' (expected '{expected}').",
                        UserWarning,
                        stacklevel=2,
                    )
                break
        return self


class GymResourceService(BaseModel):
    """A nemo-gym resource server (exposes /seed_session, /verify, tool
    endpoints).  Not a model — no protocol/generation/interceptors."""

    type: Literal["gym"] = "gym"
    url: str | None = None
    port: int = 8000
    image: str | None = None
    benchmark: str | None = None
    server_cmd: str | None = None
    startup_timeout: float = 120.0
    depends_on: list[str] = Field(default_factory=list)
    node_pool: str | None = None

    @property
    def is_model_server(self) -> bool:
        return False

    @property
    def is_managed(self) -> bool:
        return self.url is None

    @property
    def base_url(self) -> str:
        if self.url:
            return self.url
        return f"http://localhost:{self.port}"


class NatAgentService(BaseModel):
    """A NAT agent server."""

    type: Literal["nat"] = "nat"
    port: int = 8000
    image: str | None = None
    nat_config_file: str | None = None
    startup_timeout: float = 120.0
    depends_on: list[str] = Field(default_factory=list)
    node_pool: str | None = None

    @property
    def is_model_server(self) -> bool:
        return False

    @property
    def is_managed(self) -> bool:
        return True

    @property
    def base_url(self) -> str:
        return f"http://localhost:{self.port}"


class CustomService(BaseModel):
    """Plugin service — dynamically imported from class_path."""

    type: Literal["custom"] = "custom"
    class_path: str
    config: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_model_server(self) -> bool:
        return False

    @property
    def is_managed(self) -> bool:
        return True

    @property
    def base_url(self) -> str:
        return ""


def _service_discriminator(v: Any) -> str:
    if isinstance(v, dict):
        t = v.get("type")
        if t is None:
            raise ValueError("Service config must include a 'type' field")
        return t
    t = getattr(v, "type", None)
    if t is None:
        raise ValueError(f"Cannot determine service type from {type(v).__name__}. Expected a dict with a 'type' field.")
    return t


ServiceConfig = Annotated[
    Annotated[VllmService, Tag("vllm")]
    | Annotated[SglangService, Tag("sglang")]
    | Annotated[NimService, Tag("nim")]
    | Annotated[DockerModelService, Tag("docker_model")]
    | Annotated[ExternalApiService, Tag("api")]
    | Annotated[GymResourceService, Tag("gym")]
    | Annotated[NatAgentService, Tag("nat")]
    | Annotated[CustomService, Tag("custom")],
    Discriminator(_service_discriminator),
]

_MODEL_SERVICE_TYPES = (
    VllmService,
    SglangService,
    NimService,
    DockerModelService,
    ExternalApiService,
)


# ═══════════════════════════════════════════════════════════════════════
#  SECTION 2 — SANDBOXES
# ═══════════════════════════════════════════════════════════════════════


class _SandboxBase(BaseModel):
    """Shared sandbox fields used by the eval loop / lifecycle."""

    capture_cmd: str | None = None
    verify_timeout: float = 600.0


class DockerSandbox(_SandboxBase):
    type: Literal["docker"] = "docker"
    image: str | None = None
    image_template: str | None = None
    memory: str = "4g"
    cpus: float = 2.0
    timeout: float = 1800.0
    concurrency: int = 4
    network: Literal["bridge", "host", "none"] = "bridge"
    container_env: dict[str, str] = Field(default_factory=dict)


class SshSidecarConfig(BaseModel):
    sshd_port: int = 2222
    ssh_ready_timeout_sec: float = 120.0
    public_key_secret_arn: str
    private_key_secret_arn: str
    image: str | None = None
    exec_server_port: int | None = 5000


class EcsFargateSandbox(_SandboxBase):
    """ECS Fargate sandbox config.

    When ``region`` is set but ``cluster`` is omitted, NEL auto-discovers
    infrastructure from AWS SSM Parameter Store (written by Terraform).
    Any field explicitly set in YAML overrides the SSM default.
    """

    type: Literal["ecs_fargate"] = "ecs_fargate"
    image: str | None = None
    image_template: str | None = None
    timeout: float = 1800.0
    concurrency: int = 4
    container_env: dict[str, str] = Field(default_factory=dict)
    container_port: int | None = None

    region: str | None = None
    cluster: str | None = None
    subnets: list[str] = Field(default_factory=list)
    security_groups: list[str] = Field(default_factory=list)
    assign_public_ip: bool | None = None
    cpu: str = "4096"
    memory: str = "8192"
    ephemeral_storage_gib: int | None = None
    execution_role_arn: str | None = None
    task_role_arn: str | None = None
    log_group: str | None = None
    log_stream_prefix: str | None = None
    max_task_lifetime_sec: int | None = None
    ssh_sidecar: SshSidecarConfig | None = None
    s3_bucket: str | None = None
    s3_prefix: str | None = None
    ecr_repository: str | None = None
    codebuild_project: str | None = None
    codebuild_service_role: str | None = None
    dockerhub_secret_arn: str | None = None
    efs_filesystem_id: str | None = None
    efs_access_point_id: str | None = None

    ssm_project: str = "harbor"


class _SlurmSandboxBase(_SandboxBase):
    """Shared fields for SLURM-based sandboxes (Pyxis/Enroot and Apptainer)."""

    image: str | None = None
    image_template: str | None = None
    memory: str = "4g"
    cpus: float = 2.0
    timeout: float = 1800.0
    concurrency: int = 4
    node_pool: str | None = None
    slots_per_node: int = 4
    container_env: dict[str, str] = Field(default_factory=dict)


class SlurmSandbox(_SlurmSandboxBase):
    type: Literal["slurm"] = "slurm"


class ApptainerSandbox(_SlurmSandboxBase):
    type: Literal["apptainer"] = "apptainer"
    sif_cache_dir: str | None = None

    @model_validator(mode="after")
    def _sif_required_for_remote(self) -> ApptainerSandbox:
        if self.node_pool is not None and not self.sif_cache_dir:
            raise ValueError("sif_cache_dir required when node_pool is set (remote sandbox nodes)")
        return self


class NoSandbox(BaseModel):
    type: Literal["none"] = "none"

    capture_cmd: str | None = None
    verify_timeout: float = 600.0


class CustomSandbox(BaseModel):
    """Plugin sandbox backend — dynamically imported from class_path."""

    type: Literal["custom"] = "custom"
    class_path: str
    config: dict[str, Any] = Field(default_factory=dict)


def _sandbox_discriminator(v: Any) -> str:
    if isinstance(v, dict):
        t = v.get("type")
        if t is None:
            raise ValueError("Sandbox config must include a 'type' field")
        return t
    t = getattr(v, "type", None)
    if t is None:
        raise ValueError(f"Cannot determine sandbox type from {type(v).__name__}. Expected a dict with a 'type' field.")
    return t


SandboxConfig = Annotated[
    Annotated[DockerSandbox, Tag("docker")]
    | Annotated[EcsFargateSandbox, Tag("ecs_fargate")]
    | Annotated[SlurmSandbox, Tag("slurm")]
    | Annotated[ApptainerSandbox, Tag("apptainer")]
    | Annotated[NoSandbox, Tag("none")]
    | Annotated[CustomSandbox, Tag("custom")],
    Discriminator(_sandbox_discriminator),
]


# ═══════════════════════════════════════════════════════════════════════
#  SECTION 3 — SOLVERS
# ═══════════════════════════════════════════════════════════════════════


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


# ═══════════════════════════════════════════════════════════════════════
#  SECTION 4 — SCORING
# ═══════════════════════════════════════════════════════════════════════


class ScorerMetric(BaseModel):
    """A function-based scorer from the registry."""

    type: Literal["scorer"] = "scorer"
    name: str


class _JudgeBase(BaseModel):
    """Shared validation for judge metrics."""

    rubric: str | None = None
    rubric_file: str | None = None

    @model_validator(mode="after")
    def _rubric_xor(self) -> _JudgeBase:
        if self.rubric and self.rubric_file:
            raise ValueError("Specify 'rubric' (inline) or 'rubric_file' (path), not both.")
        return self


class JudgeMetric(_JudgeBase):
    """An LLM-as-judge metric."""

    type: Literal["judge"] = "judge"
    name: str
    service: str
    system_prompt: str | None = None
    max_score: float = 5.0
    swap_check: bool = False
    reference_free: bool = False
    temperature: float = 0.0
    timeout: float = 120.0
    max_retries: int = 2
    allow_self_judge: bool = False


class PairwiseJudgeMetric(_JudgeBase):
    """Pairwise comparison judge."""

    type: Literal["pairwise_judge"] = "pairwise_judge"
    name: str
    judge_service: str
    baseline_service: str
    max_score: float = 5.0
    swap_check: bool = True
    temperature: float = 0.0
    timeout: float = 120.0
    max_retries: int = 2


class EnsembleJudgeMetric(_JudgeBase):
    """Ensemble: multiple judge models score independently."""

    type: Literal["ensemble_judge"] = "ensemble_judge"
    name: str
    services: list[str] = Field(min_length=2)
    aggregation: Literal["mean", "median", "majority_vote"] = "mean"
    max_score: float = 5.0
    temperature: float = 0.0
    timeout: float = 120.0
    max_retries: int = 2


class SandboxMetric(BaseModel):
    """Sandbox-based evaluation (run command, check exit code)."""

    type: Literal["sandbox"] = "sandbox"
    name: str
    command: str | None = None
    verify_timeout: float = 600.0


class RewardModelMetric(BaseModel):
    """A reward-model-as-judge metric."""

    type: Literal["reward_model"] = "reward_model"
    name: str
    service: str
    mode: Literal["single", "multi_aspect", "pairwise"] = "single"
    aspect: str | None = None
    score_range: tuple[float, float] = (0.0, 1.0)
    threshold: float = 0.5
    timeout: float = 120.0
    max_retries: int = 2

    @model_validator(mode="after")
    def _validate_score_range(self) -> RewardModelMetric:
        lo, hi = self.score_range
        if lo >= hi:
            raise ValueError(f"score_range[0] ({lo}) must be less than score_range[1] ({hi})")
        return self


class CustomMetric(BaseModel):
    """Plugin metric — dynamically imported from class_path."""

    type: Literal["custom"] = "custom"
    name: str
    class_path: str
    config: dict[str, Any] = Field(default_factory=dict)


def _metric_discriminator(v: Any) -> str:
    if isinstance(v, dict):
        t = v.get("type")
        if t is None:
            raise ValueError("Metric config must include a 'type' field")
        return t
    t = getattr(v, "type", None)
    if t is None:
        raise ValueError(f"Cannot determine metric type from {type(v).__name__}. Expected a dict with a 'type' field.")
    return t


MetricConfig = Annotated[
    Annotated[ScorerMetric, Tag("scorer")]
    | Annotated[JudgeMetric, Tag("judge")]
    | Annotated[PairwiseJudgeMetric, Tag("pairwise_judge")]
    | Annotated[EnsembleJudgeMetric, Tag("ensemble_judge")]
    | Annotated[SandboxMetric, Tag("sandbox")]
    | Annotated[RewardModelMetric, Tag("reward_model")]
    | Annotated[CustomMetric, Tag("custom")],
    Discriminator(_metric_discriminator),
]


class ScoringConfig(BaseModel):
    """Scoring pipeline for a benchmark."""

    include_defaults: bool = True
    metrics: list[MetricConfig] = Field(default_factory=list)
    primary: str | None = None

    @model_validator(mode="after")
    def _validate_scoring(self) -> ScoringConfig:
        names = [m.name for m in self.metrics]
        dupes = [n for n in names if names.count(n) > 1]
        if dupes:
            raise ValueError(f"Duplicate metric names: {sorted(set(dupes))}. Each must be unique.")

        if self.primary is not None and self.metrics:
            metric_names = set(names)
            if self.primary not in metric_names:
                raise ValueError(
                    f"scoring.primary={self.primary!r} not found in metrics. Available: {sorted(metric_names)}"
                )
        if len(self.metrics) > 1 and self.primary is None:
            raise ValueError("scoring.primary is required when multiple metrics are defined.")
        return self


# ═══════════════════════════════════════════════════════════════════════
#  SECTION 5 — BENCHMARKS
# ═══════════════════════════════════════════════════════════════════════


class BenchmarkConfig(BaseModel):
    """Single benchmark entry."""

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


# ═══════════════════════════════════════════════════════════════════════
#  SECTION 6 — CLUSTER
# ═══════════════════════════════════════════════════════════════════════


class LocalCluster(BaseModel):
    type: Literal["local"] = "local"
    gpus: list[int] | None = None
    max_memory: str | None = None


class DockerCluster(BaseModel):
    type: Literal["docker"] = "docker"
    image: str | None = None
    container_mounts: list[str] = Field(default_factory=list)
    container_env: dict[str, str] = Field(default_factory=dict)
    mount_home: bool = True
    shm_size: str | None = None


class NodePool(BaseModel):
    """A named group of compute resources within a SLURM cluster."""

    partition: str
    nodes: int = 1
    ntasks_per_node: int = 1
    gres: str | None = None


class SlurmCluster(BaseModel):
    type: Literal["slurm"] = "slurm"
    account: str | None = None
    walltime: str = "04:00:00"
    node_pools: dict[str, NodePool] = Field(min_length=1)
    conda_env: str | None = None
    eval_image: str | None = None
    container_mounts: list[str] = Field(default_factory=list)
    container_env: dict[str, str] = Field(default_factory=dict)
    shm_size: str | None = None
    mount_home: bool = True
    auto_resume: bool = True
    max_retries: int = Field(default=3, ge=0)
    max_walltime: str | None = None
    shards: int | None = Field(default=None, ge=1)

    @field_validator("max_walltime")
    @classmethod
    def _validate_max_walltime(cls, v: str | None) -> str | None:
        if v is None:
            return v
        try:
            _parse_walltime(v)
        except (ValueError, TypeError):
            raise ValueError(
                f"max_walltime must be in SLURM time format (HH:MM:SS or D-HH:MM:SS), got: {v!r}"
            )
        return v

    hostname: str | None = None
    username: str | None = None


def _cluster_discriminator(v: Any) -> str:
    if isinstance(v, dict):
        t = v.get("type")
        if t is None:
            raise ValueError("Cluster config must include a 'type' field")
        return t
    t = getattr(v, "type", None)
    if t is None:
        raise ValueError(f"Cannot determine cluster type from {type(v).__name__}. Expected a dict with a 'type' field.")
    return t


ClusterConfig = Annotated[
    Annotated[LocalCluster, Tag("local")]
    | Annotated[DockerCluster, Tag("docker")]
    | Annotated[SlurmCluster, Tag("slurm")],
    Discriminator(_cluster_discriminator),
]


# ═══════════════════════════════════════════════════════════════════════
#  SECTION 7 — OUTPUT
# ═══════════════════════════════════════════════════════════════════════


class OutputConfig(BaseModel):
    dir: str = "./eval_results"
    timestamped: bool = True
    progress_interval: float = 60.0
    report: list[str] = Field(default_factory=lambda: ["markdown"])
    export: list[str] = Field(default_factory=list)
    export_config: dict[str, Any] = Field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════
#  SECTION 8 — TOP-LEVEL EVAL CONFIG
# ═══════════════════════════════════════════════════════════════════════


def _get_solver_service(solver: Any) -> str | None:
    """Extract the primary model service name from a solver config."""
    return getattr(solver, "service", None)


def _parse_walltime(walltime: str) -> int:
    """Parse HH:MM:SS or DD-HH:MM:SS walltime to seconds."""
    parts = walltime.split("-")
    days = 0
    if len(parts) == 2:
        days = int(parts[0])
        hms = parts[1]
    else:
        hms = parts[0]
    h, m, s = (int(x) for x in hms.split(":"))
    return days * 86400 + h * 3600 + m * 60 + s


class EvalConfig(BaseModel):
    services: dict[str, ServiceConfig]
    sandboxes: dict[str, SandboxConfig] = Field(default_factory=dict)
    benchmarks: list[BenchmarkConfig] = Field(min_length=1)
    cluster: ClusterConfig = Field(default_factory=LocalCluster)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @field_validator("services")
    @classmethod
    def _no_reserved_service_names(cls, v: dict[str, Any]) -> dict[str, Any]:
        reserved = {"default", "none", ""}
        bad = reserved & set(v.keys())
        if bad:
            raise ValueError(
                f"Reserved service names are not allowed: {sorted(bad)}. "
                f"Use semantic names (e.g. 'solver', 'judge', 'nemotron')."
            )
        return v

    @model_validator(mode="before")
    @classmethod
    def _resolve_sandbox_references(cls, data: Any) -> Any:
        """Resolve string sandbox references before Pydantic validates unions."""
        if isinstance(data, dict):
            named_sandboxes = data.get("sandboxes", {})
            for bench in data.get("benchmarks", []):
                if isinstance(bench, dict) and isinstance(bench.get("sandbox"), str):
                    name = bench["sandbox"]
                    if name not in named_sandboxes:
                        raise ValueError(
                            f"benchmark sandbox reference '{name}' not found "
                            f"in sandboxes. Available: {sorted(named_sandboxes.keys())}"
                        )
                    bench["sandbox"] = named_sandboxes[name]
        return data

    @model_validator(mode="after")
    def _validate_service_references(self) -> EvalConfig:
        """Comprehensive cross-reference validation."""
        available = set(self.services.keys())
        errors: list[str] = []

        for i, bench in enumerate(self.benchmarks):
            prefix = f"benchmarks[{i}]"
            solver = bench.solver

            solver_svc = _get_solver_service(solver)
            if solver_svc is not None and solver_svc not in available:
                errors.append(f"{prefix}.solver.service={solver_svc!r} not in services {sorted(available)}")

            if solver_svc is not None and solver_svc in available:
                svc = self.services[solver_svc]
                if isinstance(
                    solver,
                    (
                        SimpleSolver,
                        HarborSolverConfig,
                        AgentSolverConfig,
                        OpenClawSolverConfig,
                        ContainerSolverConfig,
                        ToolCallingSolverConfig,
                        GymDelegationSolverConfig,
                    ),
                ):
                    if not isinstance(svc, _MODEL_SERVICE_TYPES):
                        errors.append(
                            f"{prefix}.solver.service={solver_svc!r} is a "
                            f"'{svc.type}' service, but solver type "
                            f"'{solver.type}' requires a model service"
                        )
                elif isinstance(solver, NatSolverConfig):
                    if not isinstance(svc, NatAgentService):
                        errors.append(
                            f"{prefix}.solver.service={solver_svc!r} is a "
                            f"'{svc.type}' service, but solver type 'nat' "
                            f"requires a 'nat' service"
                        )

            if isinstance(solver, ToolCallingSolverConfig):
                if solver.resource_service is not None:
                    if solver.resource_service not in available:
                        errors.append(f"{prefix}.solver.resource_service={solver.resource_service!r} not in services")
                    elif not isinstance(self.services[solver.resource_service], GymResourceService):
                        errors.append(
                            f"{prefix}.solver.resource_service={solver.resource_service!r} must be a 'gym' service"
                        )
            elif isinstance(solver, GymDelegationSolverConfig):
                if solver.gym_service not in available:
                    errors.append(f"{prefix}.solver.gym_service={solver.gym_service!r} not in services")
                elif not isinstance(self.services[solver.gym_service], GymResourceService):
                    errors.append(f"{prefix}.solver.gym_service={solver.gym_service!r} must be a 'gym' service")

            if bench.verifier and bench.verifier not in available:
                errors.append(f"{prefix}.verifier={bench.verifier!r} not in services")

            for j, metric in enumerate(bench.scoring.metrics):
                mprefix = f"{prefix}.scoring.metrics[{j}]"

                if isinstance(metric, JudgeMetric):
                    if metric.service not in available:
                        errors.append(f"{mprefix}.service={metric.service!r} not in services")
                    elif solver_svc is not None and metric.service == solver_svc and not metric.allow_self_judge:
                        errors.append(
                            f"{mprefix}.service={metric.service!r} is the "
                            f"same as the solver's service. Use a separate "
                            f"service, or set allow_self_judge: true."
                        )
                elif isinstance(metric, PairwiseJudgeMetric):
                    for field_name in ("judge_service", "baseline_service"):
                        svc_name = getattr(metric, field_name)
                        if svc_name not in available:
                            errors.append(f"{mprefix}.{field_name}={svc_name!r} not in services")
                elif isinstance(metric, EnsembleJudgeMetric):
                    for k, svc_name in enumerate(metric.services):
                        if svc_name not in available:
                            errors.append(f"{mprefix}.services[{k}]={svc_name!r} not in services")
                elif isinstance(metric, RewardModelMetric):
                    if metric.service not in available:
                        errors.append(f"{mprefix}.service={metric.service!r} not in services")

        if errors:
            raise ValueError("Config validation errors:\n  " + "\n  ".join(errors))
        return self

    @model_validator(mode="after")
    def _validate_node_pools(self) -> EvalConfig:
        """Validate node_pool references and resource fit."""
        if not isinstance(self.cluster, SlurmCluster):
            return self
        pools = set(self.cluster.node_pools.keys())
        errors: list[str] = []

        for name, svc in self.services.items():
            pool_ref = getattr(svc, "node_pool", None)
            if pool_ref and pool_ref not in pools:
                errors.append(f"services.{name}.node_pool={pool_ref!r} not in cluster.node_pools {sorted(pools)}")
            tp = getattr(svc, "tensor_parallel_size", None)
            if tp and pool_ref and pool_ref in pools:
                pool = self.cluster.node_pools[pool_ref]
                if pool.gres:
                    match = re.search(r"gpu:(\d+)", pool.gres)
                    if match and int(match.group(1)) < tp:
                        errors.append(
                            f"services.{name} needs tensor_parallel_size={tp} "
                            f"but node_pool {pool_ref!r} only has {pool.gres}"
                        )

        for sb_name, sb in self.sandboxes.items():
            pool_ref = getattr(sb, "node_pool", None)
            if pool_ref and pool_ref not in pools:
                errors.append(f"sandboxes.{sb_name}.node_pool={pool_ref!r} not in cluster.node_pools {sorted(pools)}")

        for i, bench in enumerate(self.benchmarks):
            pool_ref = getattr(bench.sandbox, "node_pool", None)
            if pool_ref and pool_ref not in pools:
                errors.append(
                    f"benchmarks[{i}].sandbox.node_pool={pool_ref!r} not in cluster.node_pools {sorted(pools)}"
                )

        if errors:
            raise ValueError("Node pool errors:\n  " + "\n  ".join(errors))
        return self

    @model_validator(mode="after")
    def _validate_shards(self) -> EvalConfig:
        """SLURM --array is incompatible with heterogeneous jobs."""
        if not isinstance(self.cluster, SlurmCluster):
            return self
        if self.cluster.shards is None:
            return self
        pool_refs: set[str] = set()
        for svc in self.services.values():
            p = getattr(svc, "node_pool", None)
            if p:
                pool_refs.add(p)
        for sb in self.sandboxes.values():
            p = getattr(sb, "node_pool", None)
            if p:
                pool_refs.add(p)
        for bench in self.benchmarks:
            p = getattr(bench.sandbox, "node_pool", None)
            if p:
                pool_refs.add(p)
        if len(pool_refs) > 1:
            raise ValueError(
                f"shards={self.cluster.shards} requires a single node pool, "
                f"but {len(pool_refs)} distinct pools are referenced: "
                f"{sorted(pool_refs)}. SLURM --array is incompatible with "
                f"heterogeneous jobs."
            )
        if self.cluster.auto_resume:
            raise ValueError(
                "shards and auto_resume cannot be used together. "
                "Use SLURM --requeue for per-task retries in array mode."
            )
        return self

    @model_validator(mode="after")
    def _validate_dependency_graph(self) -> EvalConfig:
        """Ensure depends_on references exist and form a DAG."""
        available = set(self.services.keys())
        errors: list[str] = []
        graph: dict[str, list[str]] = {}

        for name, svc in self.services.items():
            deps = getattr(svc, "depends_on", [])
            graph[name] = deps
            for dep in deps:
                if dep not in available:
                    errors.append(f"services.{name}.depends_on references {dep!r}, not in services")
                if dep == name:
                    errors.append(f"services.{name}.depends_on contains self-reference")

        visited: set[str] = set()
        in_stack: set[str] = set()

        def _has_cycle(node: str) -> bool:
            if node in in_stack:
                return True
            if node in visited:
                return False
            visited.add(node)
            in_stack.add(node)
            for dep in graph.get(node, []):
                if _has_cycle(dep):
                    return True
            in_stack.discard(node)
            return False

        for svc_name in graph:
            if _has_cycle(svc_name):
                errors.append(f"Circular dependency detected involving service {svc_name!r}")
                break

        if errors:
            raise ValueError("Dependency errors:\n  " + "\n  ".join(errors))
        return self

    @model_validator(mode="after")
    def _validate_timeout_hierarchy(self) -> EvalConfig:
        """Warn if benchmark + startup timeouts exceed SLURM walltime."""
        if not isinstance(self.cluster, SlurmCluster):
            return self
        walltime_sec = _parse_walltime(self.cluster.walltime)
        max_startup = max(
            (getattr(svc, "startup_timeout", 0) for svc in self.services.values()),
            default=0,
        )
        for bench in self.benchmarks:
            total = max_startup + bench.timeout
            if total > walltime_sec:
                warnings.warn(
                    f"benchmark '{bench.name}': startup ({max_startup}s) + "
                    f"timeout ({bench.timeout}s) = {total}s exceeds "
                    f"walltime ({self.cluster.walltime} = {walltime_sec}s)",
                    UserWarning,
                    stacklevel=2,
                )
        return self

    # ── Convenience methods for runtime consumers ──

    def get_service(self, name: str) -> ServiceConfig:
        svc = self.services.get(name)
        if svc is None:
            raise KeyError(f"Unknown service {name!r}. Available: {sorted(self.services)}")
        return svc

    def get_model_url(self, service_name: str) -> str:
        return self.get_service(service_name).base_url

    def get_model_id(self, service_name: str) -> str:
        svc = self.get_service(service_name)
        return getattr(svc, "model", None) or ""

    def get_api_key(self, service_name: str) -> str | None:
        svc = self.get_service(service_name)
        return getattr(svc, "api_key", None)

    def managed_services(self) -> dict[str, ServiceConfig]:
        return {k: v for k, v in self.services.items() if v.is_managed}


# ═══════════════════════════════════════════════════════════════════════
#  SECTION 9 — PARSER (env-var expansion + validation)
# ═══════════════════════════════════════════════════════════════════════

_ENV_RE = re.compile(r"\$\{(\w+)(?::-(.*?))?\}")


def _expand_env(value: Any) -> Any:
    """Expand ${VAR} and ${VAR:-default} in strings.

    Strict: raises ValueError if a variable has no value and no default.
    Use ${VAR:-} to explicitly allow empty string fallback.
    """

    def _replace(m: re.Match) -> str:
        var_name = m.group(1)
        default = m.group(2)
        env_val = os.environ.get(var_name)
        if env_val is not None:
            return env_val
        if default is not None:
            return default
        raise ValueError(
            f"Environment variable ${{{var_name}}} is not set "
            f"and no default was provided. "
            f"Use ${{{var_name}:-default_value}} to set a fallback."
        )

    if isinstance(value, str):
        return _ENV_RE.sub(_replace, value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def parse_eval_config(raw: dict[str, Any]) -> EvalConfig:
    """Parse and validate a raw YAML dict, expanding env vars.

    This is the required entry point.  Do not call EvalConfig.model_validate()
    directly — env-var expansion would be skipped.
    """
    expanded = _expand_env(raw)
    return EvalConfig.model_validate(expanded)
