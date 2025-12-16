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
"""Pydantic models for execution configurations."""

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class EnvVarsConfig(BaseModel):
    """Environment variables configuration for different execution phases."""

    model_config = ConfigDict(extra="allow")

    deployment: Optional[Dict[str, str]] = Field(
        default=None, description="Environment variables for deployment container"
    )
    evaluation: Optional[Dict[str, str]] = Field(
        default=None, description="Environment variables for evaluation container"
    )
    export: Optional[Dict[str, str]] = Field(
        default=None, description="Environment variables to export in shell"
    )


class MountsConfig(BaseModel):
    """Mount configuration for containers."""

    model_config = ConfigDict(extra="allow")

    deployment: Optional[Dict[str, str]] = Field(
        default=None,
        description="Mounts for deployment container (host_path: container_path)",
    )
    evaluation: Optional[Dict[str, str]] = Field(
        default=None,
        description="Mounts for evaluation container (host_path: container_path)",
    )
    mount_home: Optional[bool] = Field(
        default=None, description="Whether to mount home directory"
    )


class BaseExecutorConfig(BaseModel):
    """Base configuration for all executor types."""

    type: str = Field(description="Executor type (local, slurm, lepton)")
    output_dir: str = Field(description="Output directory for evaluation results")


class LocalExecutorConfig(BaseExecutorConfig):
    """Configuration for local execution using Docker."""

    type: Literal["local"] = "local"
    extra_docker_args: Optional[str] = Field(
        default="", description="Additional docker args"
    )
    mode: Optional[str] = Field(
        default="sequential", description="Execution mode: sequential or parallel"
    )
    env_vars: Optional[EnvVarsConfig] = Field(
        default=None, description="Environment variables config"
    )
    mounts: Optional[MountsConfig] = Field(
        default=None, description="Mount configuration"
    )
    auto_export: Optional[Dict[str, Any]] = Field(
        default=None, description="Auto-export configuration"
    )


class SlurmExecutorConfig(BaseExecutorConfig):
    """Configuration for SLURM cluster execution."""

    type: Literal["slurm"] = "slurm"
    hostname: str = Field(description="SLURM headnode hostname")
    username: Optional[str] = Field(default=None, description="SSH username")
    account: str = Field(description="SLURM account")
    partition: Optional[str] = Field(default="batch", description="SLURM partition")
    num_nodes: Optional[int] = Field(default=1, description="Number of nodes")
    ntasks_per_node: Optional[int] = Field(default=1, description="Tasks per node")
    gpus_per_node: Optional[int] = Field(default=None, description="GPUs per node")
    gres: Optional[str] = Field(default=None, description="Generic resources")
    walltime: Optional[str] = Field(default="01:00:00", description="Wall time")
    subproject: Optional[str] = Field(
        default="nemo-evaluator-launcher", description="Subproject name"
    )
    sbatch_comment: Optional[str] = Field(default=None, description="SLURM comment")
    env_vars: Optional[EnvVarsConfig] = Field(
        default=None, description="Environment variables config"
    )
    mounts: Optional[MountsConfig] = Field(
        default=None, description="Mount configuration"
    )
    deployment: Optional[Dict[str, Any]] = Field(
        default=None, description="Deployment-specific config"
    )
    proxy: Optional[Dict[str, Any]] = Field(
        default=None, description="Proxy configuration"
    )


class LeptonExecutorConfig(BaseExecutorConfig):
    """Configuration for Lepton platform execution."""

    type: Literal["lepton"] = "lepton"
    env_var_names: Optional[list[str]] = Field(
        default=None, description="Environment variable names"
    )
    evaluation_tasks: Optional[Dict[str, Any]] = Field(
        default=None, description="Evaluation tasks config"
    )
    lepton_platform: Dict[str, Any] = Field(description="Lepton platform configuration")
