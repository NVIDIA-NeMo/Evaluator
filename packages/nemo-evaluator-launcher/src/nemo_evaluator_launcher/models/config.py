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
"""Main configuration model for nemo-evaluator-launcher."""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from nemo_evaluator_launcher.models.deployment import DeploymentConfig
from nemo_evaluator_launcher.models.evaluation import EvaluationConfig
from nemo_evaluator_launcher.models.execution import ExecutorConfig


class ApiEndpointConfig(BaseModel):
    """Configuration for API endpoint connection."""

    url: Optional[str] = Field(
        default=None,
        description="API endpoint URL (e.g., https://api.nvidia.com/v1/chat/completions)",
    )
    model_id: Optional[str] = Field(
        default=None, description="Model identifier to use in API requests"
    )
    api_key_name: Optional[str] = Field(
        default=None, description="Name of environment variable containing the API key"
    )


class TargetConfig(BaseModel):
    """Target configuration for the model/endpoint to evaluate."""

    api_endpoint: Optional[ApiEndpointConfig] = Field(
        default=None, description="API endpoint configuration"
    )


class RunConfig(BaseModel):
    """Complete run configuration for nemo-evaluator-launcher.

    This is the top-level configuration that brings together:
    - execution: How to run (local, slurm, lepton)
    - deployment: What to deploy (vllm, nim, none, etc.)
    - target: Where to send requests (API endpoint config)
    - evaluation: What to evaluate (tasks and settings)
    """

    model_config = ConfigDict(extra="allow")

    execution: ExecutorConfig = Field(description="Execution configuration")
    deployment: DeploymentConfig = Field(description="Deployment configuration")
    target: Optional[TargetConfig] = Field(
        default=None, description="Target endpoint configuration"
    )
    evaluation: EvaluationConfig = Field(description="Evaluation tasks and settings")

    # Optional fields
    name: Optional[str] = Field(default=None, description="Configuration name")
    defaults: Optional[list[str]] = Field(
        default=None, description="Hydra defaults list"
    )
