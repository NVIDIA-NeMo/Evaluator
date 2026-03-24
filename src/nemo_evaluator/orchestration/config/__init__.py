"""Evaluation config schema (Pydantic).

Fully explicit, docker-compose-style configuration for evaluations.
No sugar, no shorthands, no implicit defaults for architectural choices.

Sub-modules:
  - services: model server + API + gym + NAT + custom service schemas
  - sandboxes: Docker, ECS Fargate, SLURM, Apptainer, custom sandbox schemas
  - solvers: simple, harbor, agent, tool_calling, etc. solver schemas
  - scoring: metric + scoring pipeline schemas
  - benchmarks: benchmark entry schema
  - clusters: local, Docker, SLURM cluster schemas
  - output: output configuration schema
  - eval_config: top-level EvalConfig + YAML parser
"""

from .benchmarks import BenchmarkConfig
from .clusters import (
    ClusterConfig,
    DockerCluster,
    LocalCluster,
    NodePool,
    SlurmCluster,
    _parse_walltime,
)
from .eval_config import EvalConfig, _expand_env, parse_eval_config
from .output import OutputConfig
from .sandboxes import (
    ApptainerSandbox,
    CustomSandbox,
    DockerSandbox,
    EcsFargateSandbox,
    NoSandbox,
    SandboxConfig,
    SlurmSandbox,
    SshSidecarConfig,
    _SandboxBase,
    _SlurmSandboxBase,
)
from .scoring import (
    CustomMetric,
    EnsembleJudgeMetric,
    JudgeMetric,
    MetricConfig,
    PairwiseJudgeMetric,
    RewardModelMetric,
    SandboxMetric,
    ScorerMetric,
    ScoringConfig,
)
from .services import (
    CustomService,
    DockerModelService,
    ExternalApiService,
    GenerationConfig,
    GymResourceService,
    InterceptorConfig,
    NatAgentService,
    NimService,
    Protocol,
    ServiceConfig,
    SglangService,
    VllmService,
    _MODEL_SERVICE_TYPES,
    _ModelServerBase,
)
from .solvers import (
    AgentSolverConfig,
    ContainerSolverConfig,
    CustomSolverConfig,
    GymDelegationSolverConfig,
    HarborSolverConfig,
    NatSolverConfig,
    OpenClawSolverConfig,
    SimpleSolver,
    SolverConfig,
    ToolCallingSolverConfig,
)

__all__ = [
    # Services
    "Protocol",
    "GenerationConfig",
    "InterceptorConfig",
    "_ModelServerBase",
    "VllmService",
    "SglangService",
    "NimService",
    "DockerModelService",
    "ExternalApiService",
    "GymResourceService",
    "NatAgentService",
    "CustomService",
    "ServiceConfig",
    "_MODEL_SERVICE_TYPES",
    # Sandboxes
    "_SandboxBase",
    "DockerSandbox",
    "SshSidecarConfig",
    "EcsFargateSandbox",
    "_SlurmSandboxBase",
    "SlurmSandbox",
    "ApptainerSandbox",
    "NoSandbox",
    "CustomSandbox",
    "SandboxConfig",
    # Solvers
    "SimpleSolver",
    "HarborSolverConfig",
    "AgentSolverConfig",
    "ToolCallingSolverConfig",
    "GymDelegationSolverConfig",
    "NatSolverConfig",
    "OpenClawSolverConfig",
    "ContainerSolverConfig",
    "CustomSolverConfig",
    "SolverConfig",
    # Scoring
    "ScorerMetric",
    "JudgeMetric",
    "PairwiseJudgeMetric",
    "EnsembleJudgeMetric",
    "SandboxMetric",
    "RewardModelMetric",
    "CustomMetric",
    "MetricConfig",
    "ScoringConfig",
    # Benchmarks
    "BenchmarkConfig",
    # Clusters
    "LocalCluster",
    "DockerCluster",
    "NodePool",
    "SlurmCluster",
    "ClusterConfig",
    "_parse_walltime",
    # Output
    "OutputConfig",
    # Top-level
    "EvalConfig",
    "parse_eval_config",
    "_expand_env",
]
