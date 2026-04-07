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
"""Cluster configuration schemas (local, Docker, SLURM)."""

from __future__ import annotations

import re
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Discriminator, Field, Tag, field_validator


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


class LocalCluster(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["local"] = "local"
    gpus: list[int] | None = None
    max_memory: str | None = None
    container_env: dict[str, str] = Field(default_factory=dict)


class DockerCluster(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["docker"] = "docker"
    image: str | None = None
    container_mounts: list[str] = Field(default_factory=list)
    container_env: dict[str, str] = Field(default_factory=dict)
    mount_home: bool = True
    shm_size: str | None = None


class NodePool(BaseModel):
    """A named group of compute resources within a SLURM cluster."""

    model_config = ConfigDict(extra="forbid")

    partition: str
    nodes: int = 1
    ntasks_per_node: int = 1
    gres: str | None = None
    gpus_per_node: int | None = None


class SlurmCluster(BaseModel):
    model_config = ConfigDict(extra="forbid")

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
            raise ValueError(f"max_walltime must be in SLURM time format (HH:MM:SS or D-HH:MM:SS), got: {v!r}")
        return v

    sbatch_comment: str | None = None
    sbatch_extra_flags: dict[str, str | int | float | bool] = Field(default_factory=dict)

    @field_validator("sbatch_extra_flags")
    @classmethod
    def _validate_sbatch_flags(cls, v: dict) -> dict:
        for key, val in v.items():
            if not re.match(r"^[a-zA-Z][a-zA-Z0-9_-]*$", key):
                raise ValueError(f"Invalid sbatch flag key: {key!r}")
            if isinstance(val, str) and "\n" in val:
                raise ValueError(f"sbatch flag value for {key!r} must not contain newlines")
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
