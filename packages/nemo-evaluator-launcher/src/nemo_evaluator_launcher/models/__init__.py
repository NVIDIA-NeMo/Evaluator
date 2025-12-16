# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Pydantic models for nemo-evaluator-launcher configuration.

This module provides type-safe configuration models for:
- Execution configurations (Local, SLURM, Lepton)
- Deployment configurations (vLLM, NIM, TRT-LLM, SGLang, Generic, None)
- Evaluation configurations (tasks and settings)
- Complete RunConfig structure

The models use Dict[str, Any] for nested complex structures where the schema
is not fully specified or may vary significantly.
"""

from nemo_evaluator_launcher.models.config import (
    ApiEndpointConfig,
    RunConfig,
    TargetConfig,
)
from nemo_evaluator_launcher.models.deployment import (
    BaseDeploymentConfig,
    DeploymentConfig,
    GenericDeploymentConfig,
    NIMDeploymentConfig,
    NoDeploymentConfig,
    SGLangDeploymentConfig,
    TRTLLMDeploymentConfig,
    VLLMDeploymentConfig,
)
from nemo_evaluator_launcher.models.evaluation import (
    EvaluationConfig,
    EvaluationTaskConfig,
)
from nemo_evaluator_launcher.models.execution import (
    BaseExecutorConfig,
    EnvVarsConfig,
    ExecutorConfig,
    LeptonExecutorConfig,
    LocalExecutorConfig,
    MountsConfig,
    SlurmExecutorConfig,
)

__all__ = [
    # Main config
    "RunConfig",
    "TargetConfig",
    "ApiEndpointConfig",
    # Execution configs
    "BaseExecutorConfig",
    "ExecutorConfig",
    "LocalExecutorConfig",
    "SlurmExecutorConfig",
    "LeptonExecutorConfig",
    "EnvVarsConfig",
    "MountsConfig",
    # Deployment configs
    "BaseDeploymentConfig",
    "DeploymentConfig",
    "NoDeploymentConfig",
    "GenericDeploymentConfig",
    "VLLMDeploymentConfig",
    "NIMDeploymentConfig",
    "TRTLLMDeploymentConfig",
    "SGLangDeploymentConfig",
    # Evaluation configs
    "EvaluationConfig",
    "EvaluationTaskConfig",
]
