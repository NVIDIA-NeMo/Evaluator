"""Unified evaluation config schema.

Two config modes:
  - Simple: `model:` + `benchmarks:` for single-model evaluation.
  - Advanced: `services:` + `benchmarks:` for multi-model / managed infrastructure.
"""
from __future__ import annotations

import os
import re
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

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
# Benchmark config
# ---------------------------------------------------------------------------

class BenchmarkConfig(BaseModel):
    """A single benchmark to evaluate.

    The `name` field uses the unified naming convention:
      - ``gsm8k``                 built-in BYOB benchmark
      - ``skills://mmlu-pro``     NeMo Skills benchmark
      - ``lm-eval/aime2025``      lm-eval-harness task
      - ``gym://host:port``       remote Gym environment
      - ``gym-managed://swebench`` managed Gym server
      - ``mteb://mteb-task``      MTEB embedding benchmark
      - ``container://image#task`` legacy container harness
    """
    name: str
    model: str = "default"
    judge: str | None = None
    repeats: int = 1
    max_problems: int | None = None
    max_concurrent: int = 32
    system_prompt: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    fewshot: int | None = None
    endpoint_type: Literal["chat", "completions", "vlm", "embedding"] = "chat"
    image_detail: str = "auto"
    reranker: str | None = None


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
    """Named infrastructure service: model server, judge, gym server."""
    type: Literal["vllm", "sglang", "nim", "docker", "api", "gym"]

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
    subproject: str | None = None
    conda_env: str | None = None
    mounts: dict[str, Any] = Field(default_factory=dict)
    container_image: str | None = None
    container_mounts: list[str] = Field(default_factory=list)
    container_env: dict[str, str] = Field(default_factory=dict)
    mount_home: bool = True
    max_walltime: str | None = None
    auto_resume: bool = False
    max_resume_attempts: int = 3


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


def parse_eval_config(raw: dict[str, Any]) -> EvalConfig:
    """Parse and validate a raw YAML dict, expanding env vars."""
    expanded = _expand_env(raw)
    return EvalConfig.model_validate(expanded)
