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
"""Sandbox protocol: per-problem isolated execution environments.

A sandbox is infrastructure-only — it starts a container, provides
exec/upload/download, and cleans up.  What runs inside the container
(a script, a multi-turn agent, etc.) is the solver's concern.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol, Self


@dataclass
class ImageSpec:
    """One image that may need building before evaluation.

    ``image`` is the target Docker image name (e.g.
    ``swebench/sweb.eval.x86_64.astropy__astropy-12907:latest``).

    ``source`` carries benchmark-specific build context that the
    ``docker_build_fn`` needs — the manager never inspects it.
    Examples:
      - SWE-bench: ``{"instance_id": "astropy__astropy-12907"}``
      - Dockerfile: ``{"context_dir": "/path/to/ctx", "dockerfile": "Dockerfile.cuda"}``
      - Archive:    ``{"archive": "/tmp/build-ctx.tar.gz"}``
    """

    image: str
    source: dict[str, Any] = field(default_factory=dict)


@dataclass
class ImageBuildRequest:
    """Batch of images a benchmark needs, plus how to build the Docker originals.

    The benchmark says *what* it needs (``specs``) and *how* to build
    Docker source images (``docker_build_fn``).  The :class:`SandboxManager`
    decides the final format — Docker, SIF, ECR — based on the backend.

    ``docker_build_fn`` receives the **missing** subset of specs so it has
    full structured data (no reverse-engineering image names). It must raise
    on failure; the manager gates the call on Docker daemon availability.

    ``codebuild_buildspec_fn`` is an optional fallback for ECS Fargate when
    the local Docker daemon is unavailable (e.g. on SLURM).  It receives
    ``(missing_specs, ecr_repo, ecr_region, dockerhub_secret_arn)`` and
    returns a CodeBuild buildspec YAML string that builds the images and
    pushes them to ECR.
    """

    specs: list[ImageSpec]
    docker_build_fn: Callable[[list[ImageSpec]], None] | None = None
    codebuild_buildspec_fn: Callable[[list[ImageSpec], str, str | None, str | None], str] | None = None


@dataclass
class ExecResult:
    """Result of executing a command inside a sandbox."""

    stdout: str
    stderr: str
    return_code: int


@dataclass
class OutsideEndpoint:
    """A host-side URL that must be reachable from inside the sandbox.

    The sandbox rewrites *url* for its network topology and exposes the
    resolved address as the environment variable *env_var* inside the
    container.
    """

    url: str
    env_var: str


@dataclass
class VolumeMount:
    """Host bind mount or EFS mount.

    When ``efs_filesystem_id`` is set this is an EFS mount (ECS Fargate).
    Otherwise it is a host bind mount (Docker / Slurm / Apptainer).
    """

    host_path: str = ""
    container_path: str = ""
    readonly: bool = False
    efs_filesystem_id: str | None = None
    efs_root_directory: str | None = None
    efs_access_point_id: str | None = None

    @property
    def is_efs(self) -> bool:
        return self.efs_filesystem_id is not None


@dataclass
class SandboxSpec:
    """Per-problem sandbox requirements. Returned by env.seed() or sandbox_specs()."""

    image: str
    workdir: str = "/workspace"
    env: dict[str, str] = field(default_factory=dict)
    files: dict[str, str] = field(default_factory=dict)
    entrypoint: str | None = None
    volumes: list[VolumeMount] = field(default_factory=list)
    environment_dir: str | None = None
    """Local directory containing a Dockerfile / build context.
    Used by ECS Fargate to build and push images via CodeBuild."""


class Sandbox(Protocol):
    """Start/stop a container, exec commands, transfer files.

    All methods are async because sandbox operations are I/O-bound and
    multiple sandboxes may run concurrently.
    """

    @property
    def spec(self) -> SandboxSpec: ...

    async def start(self, *, outside_endpoints: list[OutsideEndpoint] | None = None) -> None: ...

    async def stop(self) -> None: ...

    async def exec(
        self,
        command: str,
        timeout_sec: float = 180,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        user: str | int | None = None,
    ) -> ExecResult: ...

    async def upload(self, local_path: Path, remote_path: str) -> None: ...

    async def download(self, remote_path: str, local_path: Path) -> None: ...

    def resolve_outside_endpoint(self, url: str) -> str:
        """Rewrite a host-side URL for use inside the sandbox.

        Performs network-topology rewriting only (e.g. ``localhost`` to
        ``host.docker.internal``).  The path component is preserved.
        For a fully resolved :class:`OutsideEndpoint` URL, use
        :meth:`resolved_endpoint_url`.
        """
        ...

    def resolved_endpoint_url(self, env_var: str) -> str | None:
        """Return the sandbox-resolved URL for a registered
        :class:`OutsideEndpoint`.

        Looks up the endpoint whose *env_var* matches and returns its URL
        rewritten for the sandbox's network topology.  Returns ``None``
        when no matching endpoint was registered.
        """
        return None

    @property
    def is_running(self) -> bool: ...

    @property
    def container_ip(self) -> str | None:
        """IP address of the container (bridge network). None for host/local."""
        ...

    async def __aenter__(self) -> Self: ...

    async def __aexit__(self, *exc: object) -> None: ...
