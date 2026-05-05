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
"""Evaluation config schema (Pydantic) and YAML composition engine.

Fully explicit, docker-compose-style configuration for evaluations.
No sugar, no shorthands, no implicit defaults for architectural choices.

Sub-modules:
  - compose: Hydra-style YAML composition (defaults, _base_, self-refs)
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
)
from .compose import compose_config
from .eval_config import EvalConfig, parse_eval_config
from .gate_policy import GatePolicy, default_policy, load_gate_policy
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
    ProxyConfig,
    ServiceConfig,
    SglangService,
    VllmService,
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
    # Composition
    "compose_config",
    # Services
    "Protocol",
    "GenerationConfig",
    "InterceptorConfig",
    "ProxyConfig",
    "VllmService",
    "SglangService",
    "NimService",
    "DockerModelService",
    "ExternalApiService",
    "GymResourceService",
    "NatAgentService",
    "CustomService",
    "ServiceConfig",
    # Sandboxes
    "DockerSandbox",
    "SshSidecarConfig",
    "EcsFargateSandbox",
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
    # Output
    "OutputConfig",
    # Gate policy
    "GatePolicy",
    "load_gate_policy",
    "default_policy",
    # Top-level
    "EvalConfig",
    "parse_eval_config",
]
