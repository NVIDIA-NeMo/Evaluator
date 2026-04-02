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
"""Sandbox configuration schemas (Docker, ECS Fargate, SLURM, Apptainer, custom)."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Discriminator, Field, Tag, model_validator


class _SandboxBase(BaseModel):
    """Shared sandbox fields used by the eval loop / lifecycle."""

    model_config = ConfigDict(extra="forbid")

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
    model_config = ConfigDict(extra="forbid")

    sshd_port: int = 2222
    ssh_ready_timeout_sec: float = 300.0
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
    codebuild_build_timeout: int | None = None
    codebuild_compute_type: str | None = None
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
    model_config = ConfigDict(extra="forbid")

    type: Literal["none"] = "none"

    capture_cmd: str | None = None
    verify_timeout: float = 600.0


class CustomSandbox(BaseModel):
    """Plugin sandbox backend — dynamically imported from class_path."""

    model_config = ConfigDict(extra="forbid")

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
