"""Executor protocol and config dataclasses."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class DeployConfig:
    type: str = "api"
    image: str | None = None
    model: str | None = None
    gpus: int = 1
    port: int = 8000
    health_path: str = "/v1/health/ready"
    startup_timeout: float = 600.0
    extra_env: dict[str, str] = field(default_factory=dict)
    extra_args: list[str] = field(default_factory=list)
    nodes: int = 1
    pipeline_parallel_size: int | None = None


@dataclass
class JudgeConfig:
    model_url: str | None = None
    model_id: str | None = None
    api_key: str | None = None
    deploy: DeployConfig | None = None


@dataclass
class RunConfig:
    env: str
    model_url: str | None = None
    model_id: str | None = None
    api_key: str | None = None
    deploy: DeployConfig | None = None
    judge: JudgeConfig | None = None
    repeats: int = 1
    max_problems: int | None = None
    system_prompt: str | None = None
    temperature: float = 0.0
    max_tokens: int = 2048
    top_p: float | None = None
    seed: int | None = None
    cache_dir: str | None = None
    output_dir: str = "./eval_results"

    # Executor-specific
    slurm_partition: str | None = None
    slurm_gpus: int | None = None
    slurm_nodes: int = 1
    slurm_time: str = "04:00:00"
    docker_network: str | None = None
    k8s_namespace: str = "default"


class Executor(Protocol):
    async def execute(self, config: RunConfig) -> dict[str, Any]:
        """Run the full eval pipeline: deploy model, run eval, collect results, tear down."""


_EXECUTOR_REGISTRY: dict[str, type] = {}


def _register_executor(name: str, cls: type) -> None:
    _EXECUTOR_REGISTRY[name] = cls


def get_executor(name: str = "local") -> Executor:
    if name not in _EXECUTOR_REGISTRY:
        _lazy_load()
    if name not in _EXECUTOR_REGISTRY:
        available = ", ".join(sorted(_EXECUTOR_REGISTRY))
        raise KeyError(f"Unknown executor {name!r}. Available: {available}")
    return _EXECUTOR_REGISTRY[name]()


def _lazy_load():
    from nemo_evaluator.executors.local import LocalExecutor
    _register_executor("local", LocalExecutor)

    from nemo_evaluator.executors.docker import DockerExecutor
    _register_executor("docker", DockerExecutor)

    from nemo_evaluator.executors.slurm import SlurmExecutor
    _register_executor("slurm", SlurmExecutor)
