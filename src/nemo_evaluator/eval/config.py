"""Evaluation config schema (Pydantic)."""
from __future__ import annotations

import os
import re
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Endpoint types -- how the eval loop talks to the model / agent
# ---------------------------------------------------------------------------

class EndpointType(StrEnum):
    """Maps to a concrete Solver implementation."""

    harbor = "harbor"
    gym = "gym"
    chat = "chat"
    completions = "completions"
    vlm = "vlm"
    nat = "nat"
    openclaw = "openclaw"

    @property
    def manages_own_client(self) -> bool:
        return self in _AGENTIC_TYPES

    @property
    def modifies_sandbox(self) -> bool:
        return self in _SANDBOX_MODIFYING_TYPES


_AGENTIC_TYPES = frozenset({EndpointType.nat, EndpointType.openclaw, EndpointType.harbor, EndpointType.gym})
_SANDBOX_MODIFYING_TYPES = frozenset({EndpointType.harbor})

_ENV_RE = re.compile(r"\$\{(\w+)(?::-(.*?))?\}")


def _expand_env(value: Any) -> Any:
    """Recursively expand ${VAR} and ${VAR:-default} in strings."""
    if isinstance(value, str):
        def _repl(m: re.Match) -> str:
            return os.environ.get(m.group(1), m.group(2) or "")
        return _ENV_RE.sub(_repl, value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


# ---------------------------------------------------------------------------
# Sandbox config
# ---------------------------------------------------------------------------

class EcsFargateConfig(BaseModel):
    """AWS ECS Fargate sandbox backend configuration."""

    cluster: str
    subnets: list[str]
    security_groups: list[str]
    task_role_arn: str | None = None
    execution_role_arn: str | None = None
    ssh_key_name: str | None = None
    codebuild_project: str | None = None
    ecr_repo: str | None = None
    vcpus: float = 2.0
    memory_mb: int = 4096
    ephemeral_storage_gb: int = 21
    assign_public_ip: bool = True


class SandboxConfig(BaseModel):
    """Per-problem sandbox configuration for agent/code evaluations."""

    backend: Literal["docker", "slurm", "local", "ecs_fargate", "apptainer", "none"] = "none"
    image: str | None = None
    image_template: str | None = None
    memory: str = "4g"
    cpus: float = 2.0
    timeout: float = 1800.0
    concurrency: int = 4
    network: str = "bridge"
    sandbox_nodes: int = 0
    slots_per_node: int = 4

    agent_cmd: str | None = None
    capture_cmd: str | None = None
    verify_timeout: float = 600.0

    harbor_agent: str | None = None
    harbor_agent_kwargs: dict[str, Any] = Field(default_factory=dict)

    sif_cache_dir: str | None = None

    ecs: EcsFargateConfig | None = None

    @field_validator("image_template")
    @classmethod
    def _validate_template(cls, v: str | None) -> str | None:
        if v and ("{" in v or "}" in v):
            if re.search(r"\{[^}]*[!\[.]", v):
                raise ValueError("image_template supports only simple {key} placeholders")
        return v


# ---------------------------------------------------------------------------
# Benchmark config
# ---------------------------------------------------------------------------

class BenchmarkConfig(BaseModel):
    """Single benchmark entry. Name follows ``scheme://task`` or bare built-in name."""
    name: str
    model: str = "default"
    judge: str | None = None
    verifier: str | None = None
    repeats: int = 1
    max_problems: int | None = None
    max_concurrent: int = 32
    system_prompt: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    context_window: int | None = None
    fewshot: int | None = None
    endpoint_type: EndpointType = EndpointType.chat
    openclaw_config: str | None = None
    image_detail: str = "auto"
    sandbox: SandboxConfig | None = None

    # Gym solver fields
    gym_url: str | None = None
    gym_agent: str | None = None
    gym_trust_reward: bool = False

    @field_validator("endpoint_type", mode="before")
    @classmethod
    def _normalize_endpoint_type(cls, v: str) -> str:
        _LEGACY = {"nat_agent": "nat", "agent": "harbor", "sandbox": "harbor"}
        return _LEGACY.get(v, v)

    @model_validator(mode="after")
    def _check_harbor_requires_sandbox(self) -> "BenchmarkConfig":
        if self.endpoint_type is EndpointType.harbor:
            if not self.sandbox or self.sandbox.backend == "none":
                raise ValueError(
                    "endpoint_type='harbor' requires a sandbox with a real backend "
                    "(docker, slurm, apptainer, ecs_fargate, local)"
                )
            if not self.sandbox.harbor_agent:
                raise ValueError(
                    "endpoint_type='harbor' requires sandbox.harbor_agent to be set"
                )
        return self

    @model_validator(mode="after")
    def _check_gym_requires_url(self) -> "BenchmarkConfig":
        if self.endpoint_type is EndpointType.gym and not self.gym_url:
            raise ValueError("endpoint_type='gym' requires gym_url to be set")
        return self


# ---------------------------------------------------------------------------
# Model config (simple mode)
# ---------------------------------------------------------------------------

class ModelConfig(BaseModel):
    """Model under evaluation. Either connect to a running endpoint or deploy."""
    url: str | None = None
    id: str | None = None
    api_key: str | None = None

    deploy: Literal["vllm", "sglang", "nim", "docker"] | None = None
    name: str | None = None
    tensor_parallel_size: int | None = None
    pipeline_parallel_size: int | None = None
    num_nodes: int = 1
    port: int = 8000
    extra_env: dict[str, str] = Field(default_factory=dict)
    extra_args: list[str] = Field(default_factory=list)
    reasoning_pattern: str | None = None

    @model_validator(mode="after")
    def _check_endpoint_or_deploy(self) -> "ModelConfig":
        if not self.url and not self.deploy:
            raise ValueError("Model must specify either 'url' (pre-deployed) or 'deploy' (auto-deploy)")
        return self


# ---------------------------------------------------------------------------
# Service config (advanced mode)
# ---------------------------------------------------------------------------

class ServiceConfig(BaseModel):
    """Named infrastructure service: model server, judge, gym server, NAT agent."""
    type: Literal["vllm", "sglang", "nim", "docker", "api", "gym", "nat"]

    # Per-service container image override (takes precedence over containers.toml)
    image: str | None = None

    # Model server fields
    model: str | None = None
    url: str | None = None
    api_key: str | None = None
    port: int = 8000
    tensor_parallel_size: int | None = None
    pipeline_parallel_size: int | None = None
    num_nodes: int = 1
    gpus: list[int] | int | None = None
    health_path: str = "/v1/health/ready"
    startup_timeout: float = 600.0
    extra_env: dict[str, str] = Field(default_factory=dict)
    extra_args: list[str] = Field(default_factory=list)
    reasoning_pattern: str | None = None

    # Gym server fields
    benchmark: str | None = None
    server_cmd: str | None = None

    # NAT agent fields
    nat_config_file: str | None = None

    @property
    def is_model_server(self) -> bool:
        return self.type in ("vllm", "sglang", "nim", "docker", "api")

    @property
    def is_managed(self) -> bool:
        """Whether this service needs to be started/stopped."""
        return self.type not in ("api",)

    @property
    def base_url(self) -> str:
        if self.type == "api":
            return self.url or ""
        if self.type == "nat":
            return f"http://localhost:{self.port}"
        return f"http://localhost:{self.port}/v1"


# ---------------------------------------------------------------------------
# Cluster config
# ---------------------------------------------------------------------------

class ClusterConfig(BaseModel):
    """Execution environment."""
    type: Literal["local", "slurm", "docker"] = "local"
    hostname: str | None = None
    username: str | None = None
    account: str | None = None
    partition: str = "batch"
    nodes: int = 1
    ntasks_per_node: int = 1
    gres: str | None = None
    walltime: str = "04:00:00"
    conda_env: str | None = None
    container_image: str | None = None
    container_mounts: list[str] = Field(default_factory=list)
    container_env: dict[str, str] = Field(default_factory=dict)
    shm_size: str | None = None
    mount_home: bool = True
    auto_resume: bool = False
    max_resume_attempts: int = 3
    env_mode: Literal["colocated", "separated"] = "colocated"


# ---------------------------------------------------------------------------
# Output config
# ---------------------------------------------------------------------------

class OutputConfig(BaseModel):
    dir: str = "./eval_results"
    report: list[str] = Field(default_factory=lambda: ["markdown"])
    export: list[str] = Field(default_factory=list)
    export_config: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

class EvalConfig(BaseModel):
    """Top-level evaluation configuration.

    Simple mode:  ``model`` + ``benchmarks``
    Advanced mode: ``services`` + ``benchmarks``
    """
    # Simple mode
    model: ModelConfig | None = None

    # Advanced mode
    services: dict[str, ServiceConfig] | None = None

    benchmarks: list[BenchmarkConfig] = Field(min_length=1)

    cluster: ClusterConfig = Field(default_factory=ClusterConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @model_validator(mode="after")
    def _check_mode(self) -> "EvalConfig":
        if self.model and self.services:
            raise ValueError("Specify 'model' (simple) or 'services' (advanced), not both")
        if not self.model and not self.services:
            raise ValueError("Specify 'model' (simple) or 'services' (advanced)")
        return self

    @property
    def is_simple(self) -> bool:
        return self.model is not None

    def resolve_model_url(self, service_name: str = "default") -> str:
        """Return the base URL for a named service or the simple-mode model."""
        if self.is_simple:
            m = self.model
            if m.url:
                return m.url
            return f"http://localhost:{m.port}/v1"
        svc = self.services.get(service_name)
        if svc is None:
            raise KeyError(f"Unknown service {service_name!r}. Available: {list(self.services)}")
        return svc.base_url

    def resolve_model_id(self, service_name: str = "default") -> str:
        """Return the model ID for a named service or the simple-mode model."""
        if self.is_simple:
            return self.model.id or self.model.name or ""
        svc = self.services.get(service_name)
        if svc is None:
            raise KeyError(f"Unknown service {service_name!r}")
        return svc.model or ""

    def resolve_api_key(self, service_name: str = "default") -> str | None:
        if self.is_simple:
            return self.model.api_key
        svc = self.services.get(service_name)
        return svc.api_key if svc else None

    def managed_services(self) -> dict[str, ServiceConfig]:
        """Return services that need lifecycle management (start/stop)."""
        if not self.services:
            return {}
        return {k: v for k, v in self.services.items() if v.is_managed}

    def resolved_services(self) -> dict[str, ServiceConfig]:
        """Build the full services dict, auto-creating one in simple mode."""
        if self.services:
            return dict(self.services)

        if self.is_simple and self.model:
            m = self.model
            if m.deploy:
                svc = ServiceConfig(
                    type=m.deploy,
                    model=m.name or m.id,
                    port=m.port,
                    tensor_parallel_size=m.tensor_parallel_size,
                    pipeline_parallel_size=m.pipeline_parallel_size,
                    num_nodes=m.num_nodes,
                    extra_env=m.extra_env,
                    extra_args=list(m.extra_args),
                )
                return {"default": svc}
            svc = ServiceConfig(
                type="api",
                url=m.url,
                model=m.id,
                api_key=m.api_key,
            )
            return {"default": svc}

        return {}


def parse_eval_config(raw: dict[str, Any]) -> EvalConfig:
    """Parse and validate a raw YAML dict, expanding env vars."""
    expanded = _expand_env(raw)
    return EvalConfig.model_validate(expanded)
