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
"""Sandbox lifecycle manager: concurrency, pre-pull, emergency cleanup."""

from __future__ import annotations

import asyncio
import atexit
import logging
import os
import signal
import subprocess
from pathlib import Path
from typing import Any, Literal

from nemo_evaluator.environments.base import SeedResult
from nemo_evaluator.sandbox.base import ImageBuildRequest, OutsideEndpoint, Sandbox, SandboxSpec, VolumeMount

logger = logging.getLogger(__name__)


class SandboxManager:
    """Manages sandbox lifecycle: concurrency, pre-pull, emergency cleanup.

    Tracks active sandboxes and registers atexit/signal handlers so that
    a crash doesn't leave orphaned containers running.
    """

    def __init__(
        self,
        backend: Literal["docker", "slurm", "local", "ecs_fargate", "apptainer"],
        concurrency: int = 4,
        default_image: str | None = None,
        image_template: str | None = None,
        slurm_nodes: list[str] | None = None,
        slots_per_node: int = 4,
        sif_cache_dir: str | None = None,
        global_env: dict[str, str] | None = None,
        **backend_kwargs: Any,
    ) -> None:
        self._backend = backend
        self._concurrency = concurrency
        self._sem = asyncio.Semaphore(concurrency)
        self._default_image = default_image
        self._image_template = image_template
        self._global_env = global_env or {}
        self._backend_kwargs = backend_kwargs
        self._active: set[Any] = set()
        self._pulled: set[str] = set()
        self._pulling: dict[str, asyncio.Event] = {}

        # SLURM multiplexing state
        self._slurm_nodes = slurm_nodes or []
        self._slots_per_node = slots_per_node
        self._slot_idx = 0

        # Apptainer SIF cache
        self._sif_cache_dir = sif_cache_dir

        atexit.register(self._sync_cleanup)
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

    # ------------------------------------------------------------------
    # Workspace transfer strategy
    # ------------------------------------------------------------------

    def get_transfer_strategy(self) -> Any:
        """Return the appropriate WorkspaceTransfer for this backend."""
        from nemo_evaluator.sandbox.transfer import (
            EfsTransfer,
            HostVolumeTransfer,
            LocalDirectTransfer,
            SandboxExecTransfer,
        )

        if self._backend == "local":
            return LocalDirectTransfer()

        if self._backend == "ecs_fargate":
            ecs_cfg = self._backend_kwargs.get("ecs_config")
            if ecs_cfg and getattr(ecs_cfg, "efs_filesystem_id", None):
                logger.info(
                    "Workspace transfer: EFS (fs=%s, ap=%s)",
                    ecs_cfg.efs_filesystem_id,
                    getattr(ecs_cfg, "efs_access_point_id", None),
                )
                return EfsTransfer(
                    filesystem_id=ecs_cfg.efs_filesystem_id,
                    access_point_id=getattr(ecs_cfg, "efs_access_point_id", None),
                )
            logger.warning(
                "Workspace transfer: SandboxExecTransfer (no EFS configured). "
                "This is slower and requires /output to be writable."
            )
            return SandboxExecTransfer()

        staging_base = self._resolve_staging_base()
        return HostVolumeTransfer(staging_base=staging_base)

    def _resolve_staging_base(self) -> str | None:
        """Return shared FS path for Slurm/Apptainer, None for Docker."""
        if self._backend == "slurm":
            return self._backend_kwargs.get("shared_fs_root")
        if self._backend == "apptainer":
            if self._slurm_nodes:
                return self._sif_cache_dir or "/tmp"
        return None

    # ------------------------------------------------------------------
    # Pre-pull
    # ------------------------------------------------------------------

    async def pre_pull(self, specs: list[SandboxSpec]) -> None:
        """Pull unique images in parallel before eval starts."""
        unique = {s.image for s in specs if s.image} - self._pulled
        if not unique:
            return
        logger.info("Pre-pulling %d sandbox images (concurrency=%d)", len(unique), self._concurrency)
        pull_sem = asyncio.Semaphore(self._concurrency)

        async def _pull_one(img: str) -> None:
            async with pull_sem:
                try:
                    await self._ensure_pulled(img)
                except Exception:
                    logger.warning("Pre-pull failed for %s", img, exc_info=True)

        await asyncio.gather(*[_pull_one(img) for img in unique])

    # ------------------------------------------------------------------
    # Image provisioning (from @image_builder declarations)
    # ------------------------------------------------------------------

    async def provision(self, requests: list[ImageBuildRequest]) -> None:
        """Ensure all declared images are available in the backend's format.

        1. Filter to specs not yet available in the target format.
        2. If Docker daemon is reachable, build Docker originals via
           the benchmark-provided ``docker_build_fn``.
        3. Convert to the backend's format (SIF, ECR push) with concurrency.

        For ECS Fargate with ``ecr_repository``, images are built and pushed
        via AWS CodeBuild (content-hash tagged for caching).
        """
        for req in requests:
            if self._backend == "ecs_fargate":
                ecs_cfg = self._backend_kwargs.get("ecs_config")
                if ecs_cfg and getattr(ecs_cfg, "ecr_repository", None):
                    codebuild_specs = [s for s in req.specs if s.source.get("task_dir")]
                    fallback_specs = [s for s in req.specs if not s.source.get("task_dir")]
                    if codebuild_specs:
                        await self._provision_ecs_codebuild(
                            ImageBuildRequest(specs=codebuild_specs),
                            ecs_cfg,
                        )
                    if fallback_specs:
                        await self._provision_ecs_local_push(
                            ImageBuildRequest(
                                specs=fallback_specs,
                                docker_build_fn=req.docker_build_fn,
                                codebuild_buildspec_fn=req.codebuild_buildspec_fn,
                            ),
                            ecs_cfg,
                        )
                    continue
                logger.info(
                    "All %d images assumed available for ecs_fargate backend "
                    "(no ecr_repository configured — images must already exist in registry)",
                    len(req.specs),
                )
                continue

            missing = [s for s in req.specs if not await self._image_available(s.image)]
            if not missing:
                logger.info("All %d images already available for %s backend", len(req.specs), self._backend)
                continue

            logger.info(
                "Provisioning %d images (%d cached) for %s backend",
                len(missing),
                len(req.specs) - len(missing),
                self._backend,
            )

            if req.docker_build_fn:
                if await self._docker_daemon_available():
                    docker_missing = [s for s in missing if not await self._docker_exists_async(s.image)]
                    if docker_missing:
                        loop = asyncio.get_running_loop()
                        await loop.run_in_executor(
                            None,
                            req.docker_build_fn,
                            docker_missing,
                        )
                else:
                    logger.info(
                        "Docker daemon not available — skipping Docker build, backend will pull from registry directly"
                    )

            sem = asyncio.Semaphore(self._concurrency)

            async def _convert(image: str) -> None:
                async with sem:
                    await self._ensure_backend_format(image)

            await asyncio.gather(*[_convert(s.image) for s in missing])

    async def _provision_ecs_codebuild(
        self,
        req: ImageBuildRequest,
        ecs_cfg: Any,
    ) -> None:
        """Build missing images via CodeBuild and push to ECR."""
        from dataclasses import replace as _dc_replace

        from nemo_evaluator.sandbox.ecs_fargate import ImageBuilder, _sanitize_id

        ecr_repo: str = ecs_cfg.ecr_repository

        candidates: list[tuple[Any, str, str]] = []
        for spec in req.specs:
            task_dir = spec.source.get("task_dir")
            if not task_dir:
                continue
            env_dir = str(Path(task_dir) / "environment")
            if not Path(env_dir).is_dir():
                logger.warning("Environment dir %s does not exist — skipping ECR build for %s", env_dir, spec.image)
                continue
            env_name = _sanitize_id(spec.image)
            tag = ImageBuilder.get_ecr_image_tag(env_dir, env_name)
            candidates.append((spec, env_dir, tag))

        loop = asyncio.get_running_loop()
        existing_tags = await loop.run_in_executor(
            None,
            ImageBuilder.list_ecr_tags,
            ecr_repo,
            ecs_cfg.region,
        )
        logger.info("ECR repo has %d existing tags; checking %d candidates", len(existing_tags), len(candidates))

        to_build: list[tuple[Any, str]] = []
        for spec, env_dir, tag in candidates:
            if tag in existing_tags:
                logger.info("ECR cache hit: %s:%s (%s)", ecr_repo, tag, spec.image)
                continue
            to_build.append((spec, env_dir))

        if not to_build:
            logger.info("All %d images cached in ECR for ecs_fargate backend", len(req.specs))
            return

        logger.info(
            "Building %d/%d images via CodeBuild → ECR for ecs_fargate backend",
            len(to_build),
            len(req.specs),
        )
        parallelism = min(self._concurrency, getattr(ecs_cfg, "build_parallelism", 50))
        sem = asyncio.Semaphore(parallelism)

        async def _build_one(spec: Any, env_dir: str) -> None:
            async with sem:
                per_task_cfg = _dc_replace(ecs_cfg, environment_dir=env_dir)
                await loop.run_in_executor(
                    None,
                    lambda cfg=per_task_cfg, name=_sanitize_id(spec.image): ImageBuilder.ensure_image_built(
                        cfg=cfg, environment_name=name
                    ),
                )

        await asyncio.gather(*[_build_one(s, d) for s, d in to_build])

    async def _provision_ecs_local_push(
        self,
        req: ImageBuildRequest,
        ecs_cfg: Any,
    ) -> None:
        """Build images locally via docker_build_fn and push to ECR.

        Mirrors the non-ECS provisioning flow (docker_build_fn → format
        conversion) but targets ECR as the destination.  Used for benchmarks
        like SWE-bench that build Docker images programmatically rather than
        from a task_dir/Dockerfile.
        """
        from nemo_evaluator.sandbox.ecs_fargate import ImageBuilder, _sanitize_id

        ecr_repo: str = ecs_cfg.ecr_repository
        loop = asyncio.get_running_loop()

        tag_map: dict[str, str] = {spec.image: _sanitize_id(spec.image) for spec in req.specs}

        existing_tags = await loop.run_in_executor(
            None,
            ImageBuilder.list_ecr_tags,
            ecr_repo,
            ecs_cfg.region,
        )

        missing = [s for s in req.specs if tag_map[s.image] not in existing_tags]
        cached = len(req.specs) - len(missing)
        if cached:
            logger.info("ECR cache hit for %d/%d images (local-push path)", cached, len(req.specs))

        if not missing:
            logger.info("All %d images cached in ECR — no build needed", len(req.specs))
            return

        if not req.docker_build_fn:
            logger.info(
                "%d images have no task_dir and no docker_build_fn — assuming they exist in a public registry",
                len(missing),
            )
            return

        if not await self._docker_daemon_available():
            if req.codebuild_buildspec_fn:
                await self._provision_ecs_codebuild_harness(
                    ImageBuildRequest(
                        specs=missing,
                        codebuild_buildspec_fn=req.codebuild_buildspec_fn,
                    ),
                    ecs_cfg,
                )
                return
            raise RuntimeError(
                f"Docker daemon required to build {len(missing)} images "
                f"locally before pushing to ECR, but is not available. "
                f"Hint: provide codebuild_buildspec_fn on ImageBuildRequest "
                f"to enable remote building via CodeBuild."
            )

        logger.info("Building %d images locally via docker_build_fn", len(missing))
        docker_missing = [s for s in missing if not await self._docker_exists_async(s.image)]
        if docker_missing:
            await loop.run_in_executor(None, req.docker_build_fn, docker_missing)

        logger.info("Logging into ECR %s", ecr_repo.split("/")[0])
        await loop.run_in_executor(
            None,
            ImageBuilder.ecr_docker_login,
            ecr_repo,
            ecs_cfg.region,
        )

        logger.info("Pushing %d images to ECR", len(missing))
        sem = asyncio.Semaphore(min(self._concurrency, 10))

        async def _push_one(spec: Any) -> None:
            async with sem:
                tag = tag_map[spec.image]
                try:
                    await loop.run_in_executor(
                        None,
                        ImageBuilder.docker_push_to_ecr,
                        spec.image,
                        ecr_repo,
                        tag,
                    )
                except Exception:
                    logger.exception("Failed to push %s to ECR", spec.image)

        await asyncio.gather(*[_push_one(s) for s in missing])

    async def _provision_ecs_codebuild_harness(
        self,
        req: ImageBuildRequest,
        ecs_cfg: Any,
    ) -> None:
        """Build images via CodeBuild when the local Docker daemon is unavailable.

        The benchmark's ``codebuild_buildspec_fn`` generates a self-contained
        buildspec that installs build tools, builds Docker images inside
        CodeBuild (which has privileged/DinD support), and pushes them to ECR.

        Images are batched to stay within CodeBuild timeout limits; batches
        run in parallel up to ``build_parallelism``.
        """
        from nemo_evaluator.sandbox.ecs_fargate import ImageBuilder, _sanitize_id

        assert req.codebuild_buildspec_fn is not None

        ecr_repo: str = ecs_cfg.ecr_repository
        ecr_region = ImageBuilder._ecr_region(ecr_repo, fallback=ecs_cfg.region)
        dockerhub_secret_arn = getattr(ecs_cfg, "dockerhub_secret_arn", None)
        loop = asyncio.get_running_loop()

        tag_map = {s.image: _sanitize_id(s.image) for s in req.specs}

        existing_tags = await loop.run_in_executor(
            None,
            ImageBuilder.list_ecr_tags,
            ecr_repo,
            ecs_cfg.region,
        )

        missing = [s for s in req.specs if tag_map[s.image] not in existing_tags]
        cached = len(req.specs) - len(missing)
        if cached:
            logger.info(
                "ECR cache hit for %d/%d images (codebuild-harness path)",
                cached,
                len(req.specs),
            )
        if not missing:
            logger.info("All %d images cached in ECR — no CodeBuild needed", len(req.specs))
            return

        batch_size = 15
        batches = [missing[i : i + batch_size] for i in range(0, len(missing), batch_size)]
        parallelism = min(
            len(batches),
            getattr(ecs_cfg, "build_parallelism", 50),
        )
        logger.info(
            "Building %d images via CodeBuild harness (%d batches, parallelism=%d)",
            len(missing),
            len(batches),
            parallelism,
        )

        sem = asyncio.Semaphore(parallelism)

        async def _build_batch(batch_idx: int, specs: list) -> None:
            async with sem:
                buildspec = req.codebuild_buildspec_fn(
                    specs,
                    ecr_repo,
                    ecr_region,
                    dockerhub_secret_arn,
                )
                label = f"harness-batch-{batch_idx + 1}-of-{len(batches)}"
                await loop.run_in_executor(
                    None,
                    lambda bs=buildspec, lb=label: ImageBuilder.run_buildspec_via_codebuild(
                        cfg=ecs_cfg,
                        buildspec=bs,
                        job_label=lb,
                    ),
                )

        await asyncio.gather(*[_build_batch(i, batch) for i, batch in enumerate(batches)])
        logger.info(
            "CodeBuild harness provisioning complete: %d images across %d batches",
            len(missing),
            len(batches),
        )

    # ------------------------------------------------------------------
    # Image availability checks (async — safe for the event loop)
    # ------------------------------------------------------------------

    async def _image_available(self, image: str) -> bool:
        """Check if *image* is ready in the target backend's format."""
        if self._backend in ("docker", "slurm"):
            return await self._docker_exists_async(image)
        elif self._backend == "apptainer":
            return self._sif_path(image).exists()
        return True

    @staticmethod
    async def _docker_exists_async(image: str) -> bool:
        proc = await asyncio.create_subprocess_exec(
            "docker",
            "image",
            "inspect",
            image,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.communicate()
        return proc.returncode == 0

    @staticmethod
    async def _docker_daemon_available() -> bool:
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker",
                "info",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.communicate()
            return proc.returncode == 0
        except FileNotFoundError:
            return False

    def _sif_path(self, image: str) -> Path:
        cache_dir = Path(self._sif_cache_dir or os.environ.get("APPTAINER_CACHEDIR", "/tmp/nel_sif_cache"))
        safe = image.replace("/", "_").replace(":", "__")
        return cache_dir / f"{safe}.sif"

    async def _ensure_backend_format(self, image: str) -> None:
        """Convert a Docker image to whatever the current backend needs."""
        if self._backend in ("docker", "slurm"):
            return
        elif self._backend == "apptainer":
            await self._docker_to_sif(image)
        elif self._backend == "ecs_fargate":
            logger.warning("ECR push not yet implemented for %s", image)

    async def _docker_to_sif(self, image: str) -> None:
        """Convert a Docker image to SIF, choosing the fastest source."""
        sif = self._sif_path(image)
        if sif.exists():
            return
        sif.parent.mkdir(parents=True, exist_ok=True)

        if await self._docker_exists_async(image):
            source = f"docker-daemon://{image}"
        else:
            source = f"docker://{image}"

        logger.info("Building SIF: %s -> %s", source, sif)
        proc = await asyncio.create_subprocess_exec(
            "apptainer",
            "build",
            str(sif),
            source,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            sif.unlink(missing_ok=True)
            raise RuntimeError(f"apptainer build failed for {image}: {stderr.decode()[:500]}")

    # ------------------------------------------------------------------
    # Spec resolution
    # ------------------------------------------------------------------

    def _merge_env(self, spec_env: dict[str, str]) -> dict[str, str]:
        """Merge global_env (from sandbox config's container_env) with per-seed env.

        Seed-provided env takes precedence over global env.
        """
        if not self._global_env:
            return spec_env
        conflicts = set(self._global_env) & set(spec_env)
        if conflicts:
            logger.debug(
                "Sandbox global_env keys %s overridden by seed env",
                sorted(conflicts),
            )
        return {**self._global_env, **spec_env}

    def resolve_spec(
        self,
        seed: SeedResult,
        extra_volumes: list[VolumeMount] | None = None,
        base_override: SandboxSpec | None = None,
    ) -> SandboxSpec | None:
        """Resolve sandbox spec: merge seed spec with config overrides.

        Priority: image_template overrides the image, but per-problem fields
        (workdir, env, files, entrypoint) from the base spec are preserved.
        Extra volumes (e.g. shared volume for stateless verification) are appended.
        ``global_env`` (from sandbox ``container_env``) is merged in, but
        seed-provided env takes precedence.

        Args:
            base_override: Use this spec instead of ``seed.sandbox_spec``
                (e.g. for verify specs that need the same image resolution).
        """
        base = base_override or seed.sandbox_spec
        vols = extra_volumes or []

        if self._image_template:
            try:
                image = self._image_template.format_map(seed.metadata)
            except KeyError as exc:
                logger.warning(
                    "image_template placeholder %s not in seed metadata keys %s; falling back to spec image",
                    exc,
                    sorted(seed.metadata),
                )
                image = base.image if base else None
            if image:
                return SandboxSpec(
                    image=image,
                    workdir=base.workdir if base else "/workspace",
                    env=self._merge_env(dict(base.env) if base else {}),
                    files=dict(base.files) if base else {},
                    entrypoint=base.entrypoint if base else None,
                    volumes=(list(base.volumes) if base else []) + vols,
                    environment_dir=base.environment_dir if base else None,
                )

        if base:
            merged_env = self._merge_env(dict(base.env))
            if vols or merged_env != base.env:
                return SandboxSpec(
                    image=base.image,
                    workdir=base.workdir,
                    env=merged_env,
                    files=dict(base.files),
                    entrypoint=base.entrypoint,
                    volumes=list(base.volumes) + vols,
                    environment_dir=base.environment_dir,
                )
            return base

        if self._default_image:
            return SandboxSpec(
                image=self._default_image,
                env=self._merge_env({}),
                volumes=vols,
            )
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
            inspect = await asyncio.create_subprocess_exec(
                "docker",
                "image",
                "inspect",
                image,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await inspect.communicate()
            if inspect.returncode == 0:
                logger.debug("Image %s already local, skipping pull", image)
                return
            proc = await asyncio.create_subprocess_exec(
                "docker",
                "pull",
                image,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()
            if proc.returncode != 0:
                raise RuntimeError(f"docker pull {image} failed: {stderr.decode()[:500]}")
        elif self._backend == "slurm":
            het_group = self._backend_kwargs.get("het_group")
            for node in self._slurm_nodes:
                srun_args = [
                    "srun",
                    "--overlap",
                    f"--nodelist={node}",
                    "--ntasks=1",
                ]
                if het_group is not None:
                    srun_args.append(f"--het-group={het_group}")
                proc = await asyncio.create_subprocess_exec(
                    *srun_args,
                    "enroot",
                    "import",
                    f"docker://{image}",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                _, stderr = await proc.communicate()
                if proc.returncode != 0:
                    raise RuntimeError(f"enroot import {image} on {node} failed: {stderr.decode()[:500]}")
        elif self._backend == "apptainer":
            if image.endswith(".sif") and Path(image).exists():
                return
            await self._docker_to_sif(image)

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
            het_group = self._backend_kwargs.get("het_group")
            filtered = {k: v for k, v in self._backend_kwargs.items() if k != "het_group"}
            return SlurmSandbox(spec, node=node, slot=slot, het_group=het_group, **filtered)
        elif self._backend == "ecs_fargate":
            from nemo_evaluator.sandbox.ecs_fargate import EcsFargateSandbox

            return EcsFargateSandbox(spec, **self._backend_kwargs)
        elif self._backend == "apptainer":
            from nemo_evaluator.sandbox.apptainer import ApptainerSandbox

            node = None
            if self._slurm_nodes:
                node, _ = self._allocate_slot()
            het_group = self._backend_kwargs.get("het_group")
            filtered = {k: v for k, v in self._backend_kwargs.items() if k != "het_group"}
            return ApptainerSandbox(
                spec,
                node=node,
                sif_cache_dir=self._sif_cache_dir,
                het_group=het_group,
                **filtered,
            )
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
                "SlurmSandbox requires slurm_nodes list. Set sandbox.sandbox_nodes in config or pass --sandbox-nodes."
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
        logger.info("Signal %d received — stopping sandboxes", signum)
        self._sync_cleanup()
        try:
            from nemo_evaluator.sandbox.ecs_fargate import _emergency_cleanup

            _emergency_cleanup()
        except Exception:
            pass
        raise KeyboardInterrupt()

    def _sync_cleanup(self) -> None:
        """Best-effort synchronous cleanup for atexit/signal."""
        for sb in list(self._active):
            try:
                if hasattr(sb, "_container_id") and sb._container_id:
                    subprocess.run(
                        ["docker", "rm", "-f", sb._container_id],
                        capture_output=True,
                        timeout=5,
                    )
                elif hasattr(sb, "_instance_name") and hasattr(sb, "_running") and sb._running:
                    cmd = ["apptainer", "instance", "stop", sb._instance_name]
                    if hasattr(sb, "_node") and sb._node:
                        srun_cmd = [
                            "srun",
                            "--overlap",
                            f"--nodelist={sb._node}",
                            "--ntasks=1",
                        ]
                        if hasattr(sb, "_het_group") and sb._het_group is not None:
                            srun_cmd.append(f"--het-group={sb._het_group}")
                        cmd = [*srun_cmd, *cmd]
                    subprocess.run(cmd, capture_output=True, timeout=10)
                elif hasattr(sb, "_sync_stop"):
                    sb._sync_stop()
                elif hasattr(sb, "_container_name") and hasattr(sb, "_running") and sb._running:
                    srun_cmd = [
                        "srun",
                        "--overlap",
                        f"--nodelist={sb._node}",
                        "--ntasks=1",
                    ]
                    if hasattr(sb, "_het_group") and sb._het_group is not None:
                        srun_cmd.append(f"--het-group={sb._het_group}")
                    subprocess.run(
                        [
                            *srun_cmd,
                            f"--container-name={sb._container_name}",
                            "kill",
                            "1",
                        ],
                        capture_output=True,
                        timeout=10,
                    )
            except Exception:
                pass
        self._active.clear()
