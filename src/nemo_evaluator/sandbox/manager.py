"""Sandbox lifecycle manager: concurrency, pre-pull, emergency cleanup."""
from __future__ import annotations

import asyncio
import atexit
import logging
import signal
import subprocess
import sys
from typing import Any, Literal

from nemo_evaluator.environments.base import SeedResult
from nemo_evaluator.sandbox.base import OutsideEndpoint, Sandbox, SandboxSpec, VolumeMount

logger = logging.getLogger(__name__)


class SandboxManager:
    """Manages sandbox lifecycle: concurrency, pre-pull, emergency cleanup.

    Tracks active sandboxes and registers atexit/signal handlers so that
    a crash doesn't leave orphaned containers running.
    """

    def __init__(
        self,
        backend: Literal["docker", "slurm", "local", "ecs_fargate"],
        concurrency: int = 4,
        default_image: str | None = None,
        image_template: str | None = None,
        slurm_nodes: list[str] | None = None,
        slots_per_node: int = 4,
        **backend_kwargs: Any,
    ) -> None:
        self._backend = backend
        self._concurrency = concurrency
        self._sem = asyncio.Semaphore(concurrency)
        self._default_image = default_image
        self._image_template = image_template
        self._backend_kwargs = backend_kwargs
        self._active: set[Any] = set()
        self._pulled: set[str] = set()
        self._pulling: dict[str, asyncio.Event] = {}

        # SLURM multiplexing state
        self._slurm_nodes = slurm_nodes or []
        self._slots_per_node = slots_per_node
        self._slot_idx = 0

        atexit.register(self._sync_cleanup)
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

    # ------------------------------------------------------------------
    # Pre-pull
    # ------------------------------------------------------------------

    async def pre_pull(self, specs: list[SandboxSpec]) -> None:
        """Pull unique images in parallel before eval starts."""
        unique = {s.image for s in specs if s.image} - self._pulled
        if not unique:
            return
        logger.info("Pre-pulling %d sandbox images (concurrency=%d)",
                     len(unique), self._concurrency)
        pull_sem = asyncio.Semaphore(self._concurrency)

        async def _pull_one(img: str) -> None:
            async with pull_sem:
                try:
                    await self._ensure_pulled(img)
                except Exception:
                    logger.warning("Pre-pull failed for %s", img, exc_info=True)

        await asyncio.gather(*[_pull_one(img) for img in unique])

    # ------------------------------------------------------------------
    # Spec resolution
    # ------------------------------------------------------------------

    def resolve_spec(
        self,
        seed: SeedResult,
        extra_volumes: list[VolumeMount] | None = None,
    ) -> SandboxSpec | None:
        """Resolve sandbox spec: merge seed spec with config overrides.

        Priority: image_template overrides the image, but per-problem fields
        (workdir, env, files, entrypoint) from seed.sandbox_spec are preserved.
        Extra volumes (e.g. shared volume for stateless verification) are appended.
        """
        base = seed.sandbox_spec
        vols = extra_volumes or []

        if self._image_template:
            try:
                image = self._image_template.format_map(seed.metadata)
            except KeyError:
                image = base.image if base else None
            if image:
                return SandboxSpec(
                    image=image,
                    workdir=base.workdir if base else "/workspace",
                    env=dict(base.env) if base else {},
                    files=dict(base.files) if base else {},
                    entrypoint=base.entrypoint if base else None,
                    volumes=(list(base.volumes) if base else []) + vols,
                )

        if base:
            if vols:
                return SandboxSpec(
                    image=base.image,
                    workdir=base.workdir,
                    env=dict(base.env),
                    files=dict(base.files),
                    entrypoint=base.entrypoint,
                    volumes=list(base.volumes) + vols,
                )
            return base

        if self._default_image:
            return SandboxSpec(image=self._default_image, volumes=vols)
        return None

    # ------------------------------------------------------------------
    # On-demand image pull with dedup
    # ------------------------------------------------------------------

    async def _ensure_pulled(self, image: str) -> None:
        """Pull *image* if not already available, deduplicating concurrent requests."""
        if image in self._pulled:
            return
        if image in self._pulling:
            await self._pulling[image].wait()
            if image not in self._pulled:
                raise RuntimeError(f"Image pull failed for {image}")
            return

        event = asyncio.Event()
        self._pulling[image] = event
        try:
            await self._pull_image(image)
            self._pulled.add(image)
        finally:
            event.set()
            self._pulling.pop(image, None)

    async def _pull_image(self, image: str) -> None:
        """Pull a single image via the configured backend."""
        logger.info("Pulling image: %s", image)
        if self._backend == "docker":
            proc = await asyncio.create_subprocess_exec(
                "docker", "pull", image,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()
            if proc.returncode != 0:
                raise RuntimeError(f"docker pull {image} failed: {stderr.decode()[:500]}")
        elif self._backend == "slurm":
            for node in self._slurm_nodes:
                proc = await asyncio.create_subprocess_exec(
                    "srun", "--overlap", f"--nodelist={node}",
                    "--ntasks=1",
                    "enroot", "import", f"docker://{image}",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                _, stderr = await proc.communicate()
                if proc.returncode != 0:
                    raise RuntimeError(
                        f"enroot import {image} on {node} failed: {stderr.decode()[:500]}"
                    )

    # ------------------------------------------------------------------
    # Acquire / Release
    # ------------------------------------------------------------------

    async def acquire(
        self,
        spec: SandboxSpec,
        outside_endpoints: list[OutsideEndpoint] | None = None,
    ) -> Sandbox:
        await self._ensure_pulled(spec.image)
        await self._sem.acquire()
        try:
            sandbox = self._create(spec)
            await sandbox.start(outside_endpoints=outside_endpoints)
            self._active.add(sandbox)
            return sandbox
        except Exception:
            self._sem.release()
            raise

    async def release(self, sandbox: Sandbox) -> None:
        try:
            self._active.discard(sandbox)
            await sandbox.stop()
        finally:
            self._sem.release()

    async def shutdown(self) -> None:
        """Stop all active sandboxes."""
        for sb in list(self._active):
            try:
                await sb.stop()
            except Exception:
                pass
        self._active.clear()

    # ------------------------------------------------------------------
    # Backend dispatch
    # ------------------------------------------------------------------

    def _create(self, spec: SandboxSpec) -> Any:
        if self._backend == "docker":
            from nemo_evaluator.sandbox.docker import DockerSandbox
            return DockerSandbox(spec, **self._backend_kwargs)
        elif self._backend == "slurm":
            from nemo_evaluator.sandbox.slurm import SlurmSandbox
            node, slot = self._allocate_slot()
            return SlurmSandbox(spec, node=node, slot=slot, **self._backend_kwargs)
        elif self._backend == "ecs_fargate":
            from nemo_evaluator.sandbox.ecs_fargate import EcsFargateSandbox
            return EcsFargateSandbox(spec, **self._backend_kwargs)
        else:
            from nemo_evaluator.sandbox.local import LocalSandbox
            return LocalSandbox(spec)

    # ------------------------------------------------------------------
    # SLURM node multiplexing
    # ------------------------------------------------------------------

    def _allocate_slot(self) -> tuple[str, int]:
        """Round-robin across SLURM nodes, multiple slots per node."""
        if not self._slurm_nodes:
            raise RuntimeError(
                "SlurmSandbox requires slurm_nodes list. "
                "Set sandbox.sandbox_nodes in config or pass --sandbox-nodes."
            )
        total_slots = len(self._slurm_nodes) * self._slots_per_node
        idx = self._slot_idx % total_slots
        self._slot_idx += 1
        node_idx = idx // self._slots_per_node
        slot = idx % self._slots_per_node
        return self._slurm_nodes[node_idx], slot

    # ------------------------------------------------------------------
    # Emergency cleanup
    # ------------------------------------------------------------------

    def _signal_handler(self, signum: int, frame: Any) -> None:
        self._sync_cleanup()
        sys.exit(128 + signum)

    def _sync_cleanup(self) -> None:
        """Best-effort synchronous cleanup for atexit/signal."""
        for sb in list(self._active):
            try:
                if hasattr(sb, "_container_id") and sb._container_id:
                    subprocess.run(
                        ["docker", "rm", "-f", sb._container_id],
                        capture_output=True, timeout=5,
                    )
                elif hasattr(sb, "_sync_stop"):
                    sb._sync_stop()
                elif hasattr(sb, "_container_name") and hasattr(sb, "_running") and sb._running:
                    subprocess.run(
                        [
                            "srun", "--overlap",
                            f"--nodelist={sb._node}",
                            "--ntasks=1",
                            f"--container-name={sb._container_name}",
                            "kill", "1",
                        ],
                        capture_output=True, timeout=10,
                    )
            except Exception:
                pass
        self._active.clear()
