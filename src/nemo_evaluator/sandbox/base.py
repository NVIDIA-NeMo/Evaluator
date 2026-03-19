"""Sandbox protocol: per-problem isolated execution environments.

The sandbox is infrastructure-only: it knows nothing about agents, solvers,
or evaluation. It starts a container, provides exec/upload/download, and
cleans up. Whether we exec a Python script or start a multi-turn agent is
the solver's decision.
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
    """

    specs: list[ImageSpec]
    docker_build_fn: Callable[[list[ImageSpec]], None] | None = None


@dataclass
class ExecResult:
    """Result of executing a command inside a sandbox."""

    stdout: str
    stderr: str
    return_code: int


@dataclass
class OutsideEndpoint:
    """An orchestrator-side URL that should be reachable from inside the sandbox.

    The sandbox rewrites `url` for its network topology and injects the
    resolved address as `env_var` inside the container.
    """

    url: str
    env_var: str


@dataclass
class VolumeMount:
    """Host-to-container volume mount."""

    host_path: str
    container_path: str
    readonly: bool = False


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
    """Infrastructure-only: start/stop a container, exec commands, transfer files.

    All methods are async — sandbox operations are I/O-bound and the eval
    loop is async. Using sync calls would block the event loop when running
    concurrent sandboxes.
    """

    @property
    def spec(self) -> SandboxSpec: ...

    async def start(self, *, outside_endpoints: list[OutsideEndpoint] | None = None) -> None: ...

    async def stop(self) -> None: ...

    async def exec(
        self, command: str, timeout_sec: float = 180,
        *, cwd: str | None = None, env: dict[str, str] | None = None,
    ) -> ExecResult: ...

    async def upload(self, local_path: Path, remote_path: str) -> None: ...

    async def download(self, remote_path: str, local_path: Path) -> None: ...

    def resolve_outside_endpoint(self, url: str) -> str:
        """Rewrite orchestrator-side URL for use inside this sandbox.

        Bridge network: localhost -> host.docker.internal.
        Host/shared network: unchanged.
        """
        ...

    @property
    def is_running(self) -> bool: ...

    @property
    def container_ip(self) -> str | None:
        """IP address of the container (bridge network). None for host/local."""
        ...

    async def __aenter__(self) -> Self: ...

    async def __aexit__(self, *exc: object) -> None: ...
