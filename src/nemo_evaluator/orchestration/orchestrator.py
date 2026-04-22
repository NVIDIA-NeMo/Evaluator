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
"""Local suite runner."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import threading
from pathlib import Path
from typing import IO, Any

from nemo_evaluator.config import (
    AgentSolverConfig,
    ApptainerSandbox,
    ContainerSolverConfig,
    DockerSandbox,
    EcsFargateSandbox,
    EvalConfig,
    GenerationConfig,
    GymDelegationSolverConfig,
    GymResourceService,
    HarborSolverConfig,
    InterceptorConfig,
    NatAgentService,
    NatSolverConfig,
    NoSandbox,
    OpenClawSolverConfig,
    SimpleSolver,
    SlurmSandbox,
    ToolCallingSolverConfig,
)
from nemo_evaluator.config.services import _MODEL_SERVICE_TYPES

logger = logging.getLogger(__name__)


def _drain_pipe_to_file(pipe: IO[bytes], log_path: Path) -> threading.Thread:
    """Read from a subprocess pipe and write to a log file in a background thread.

    Prevents pipe buffer deadlocks and creates per-service log files.
    """

    def _worker() -> None:
        try:
            with open(log_path, "wb") as f:
                while True:
                    chunk = pipe.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
                    f.flush()
        except (OSError, ValueError):
            pass

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return t


def _interceptor_specs(interceptors: list[InterceptorConfig]) -> list[dict]:
    return [{"name": ic.name, "config": ic.config} for ic in interceptors]


def _resolve_generation(config: EvalConfig, solver_cfg: Any) -> GenerationConfig:
    svc = config.get_service(solver_cfg.service)
    svc_gen = getattr(svc, "generation", GenerationConfig())
    solver_gen = getattr(solver_cfg, "generation", None)
    if solver_gen is not None:
        return solver_gen.merge_onto(svc_gen)
    return svc_gen


def _build_ecs_sandbox_config(cfg: EcsFargateSandbox) -> Any:
    from nemo_evaluator.sandbox.ecs_fargate import (
        EcsFargateConfig as SandboxEcsConfig,
        SshSidecarConfig as SandboxSshConfig,
    )

    ssm: dict[str, Any] = {}
    use_ssm = cfg.region is not None and cfg.cluster is None
    if use_ssm:
        from nemo_evaluator.sandbox.ecs_fargate import resolve_ecs_config_from_ssm

        ssm = resolve_ecs_config_from_ssm(cfg.region, cfg.ssm_project)  # type: ignore[arg-type]

    def _pick(yaml_val: Any, ssm_key: str, default: Any = None) -> Any:
        if yaml_val is not None:
            return yaml_val
        return ssm.get(ssm_key, default)

    cluster = _pick(cfg.cluster, "cluster", "")
    subnets = cfg.subnets or ssm.get("subnets", [])
    security_groups = cfg.security_groups or ssm.get("security_groups", [])
    assign_public_ip = _pick(cfg.assign_public_ip, "assign_public_ip", True)

    execution_role_arn = _pick(cfg.execution_role_arn, "execution_role_arn")
    task_role_arn = _pick(cfg.task_role_arn, "task_role_arn")
    log_group = _pick(cfg.log_group, "log_group")
    s3_bucket = _pick(cfg.s3_bucket, "s3_bucket")
    ecr_repository = _pick(cfg.ecr_repository, "ecr_repository")
    codebuild_service_role = _pick(cfg.codebuild_service_role, "codebuild_service_role")
    dockerhub_secret_arn = _pick(cfg.dockerhub_secret_arn, "dockerhub_secret_arn")
    efs_filesystem_id = _pick(cfg.efs_filesystem_id, "efs_filesystem_id")
    efs_access_point_id = _pick(cfg.efs_access_point_id, "efs_access_point_id")
    max_task_lifetime_sec = cfg.max_task_lifetime_sec or 14400

    ssh_sidecar = None
    ssm_ssh = ssm.get("ssh_sidecar", {})
    if cfg.ssh_sidecar:
        sc = cfg.ssh_sidecar
        pub_arn = sc.public_key_secret_arn or ssm_ssh.get("public_key_secret_arn", "")
        priv_arn = sc.private_key_secret_arn or ssm_ssh.get("private_key_secret_arn", "")
        if not pub_arn or not priv_arn:
            raise ValueError(
                "ssh_sidecar.private_key_secret_arn and "
                "ssh_sidecar.public_key_secret_arn are required "
                "(set in YAML or auto-discovered from SSM)."
            )
        ssh_sidecar = SandboxSshConfig(
            sshd_port=sc.sshd_port,
            ssh_ready_timeout_sec=sc.ssh_ready_timeout_sec,
            public_key_secret_arn=pub_arn,
            private_key_secret_arn=priv_arn,
            image=sc.image,
            exec_server_port=sc.exec_server_port,
        )
    elif ssm_ssh.get("public_key_secret_arn") and ssm_ssh.get("private_key_secret_arn"):
        ssh_sidecar = SandboxSshConfig(
            sshd_port=ssm_ssh.get("sshd_port", 2222),
            public_key_secret_arn=ssm_ssh["public_key_secret_arn"],
            private_key_secret_arn=ssm_ssh["private_key_secret_arn"],
            exec_server_port=ssm_ssh.get("exec_server_port", 5000),
        )

    return SandboxEcsConfig(
        region=cfg.region,
        cluster=cluster,
        subnets=subnets,
        security_groups=security_groups,
        assign_public_ip=assign_public_ip,
        image_template=cfg.image_template,
        container_port=cfg.container_port,
        cpu=cfg.cpu,
        memory=cfg.memory,
        ephemeral_storage_gib=cfg.ephemeral_storage_gib,
        execution_role_arn=execution_role_arn,
        task_role_arn=task_role_arn,
        log_group=log_group,
        log_stream_prefix=cfg.log_stream_prefix,
        max_task_lifetime_sec=max_task_lifetime_sec,
        ssh_sidecar=ssh_sidecar,
        s3_bucket=s3_bucket,
        s3_prefix=cfg.s3_prefix,
        ecr_repository=ecr_repository,
        codebuild_project=cfg.codebuild_project,
        codebuild_service_role=codebuild_service_role,
        codebuild_build_timeout=cfg.codebuild_build_timeout or 60,
        codebuild_compute_type=cfg.codebuild_compute_type or "BUILD_GENERAL1_MEDIUM",
        dockerhub_secret_arn=dockerhub_secret_arn,
        efs_filesystem_id=efs_filesystem_id,
        efs_access_point_id=efs_access_point_id,
    )


def _safe_name(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", s)


def _make_solver(
    bench: Any,
    config: EvalConfig,
    client: Any,
    model_url: str,
    model_id: str,
    api_key: str | None,
) -> Any:
    solver_cfg = bench.solver
    sb = bench.sandbox

    if isinstance(solver_cfg, SimpleSolver):
        svc = config.get_service(solver_cfg.service)
        protocol = svc.protocol

        if protocol == "completions":
            from nemo_evaluator.solvers import CompletionSolver

            gen = _resolve_generation(config, solver_cfg)
            return CompletionSolver(
                base_url=model_url,
                model=model_id,
                api_key=api_key,
                temperature=gen.temperature,
                max_tokens=gen.max_tokens,
                top_p=gen.top_p,
                seed=gen.seed,
                stop=gen.stop,
                frequency_penalty=gen.frequency_penalty,
                presence_penalty=gen.presence_penalty,
            )

        if solver_cfg.image_detail != "auto":
            from nemo_evaluator.solvers import VLMSolver

            return VLMSolver(client, system_prompt=solver_cfg.system_prompt, image_detail=solver_cfg.image_detail)

        from nemo_evaluator.solvers import ChatSolver

        return ChatSolver(client, system_prompt=solver_cfg.system_prompt)

    if isinstance(solver_cfg, (HarborSolverConfig, AgentSolverConfig)):
        from nemo_evaluator.solvers.harbor import HarborSolver

        if hasattr(sb, "model_fields_set") and "timeout" in sb.model_fields_set:
            import warnings

            warnings.warn(
                "sandbox.timeout is deprecated and has no effect. "
                "Use benchmark-level 'timeout' or solver 'run_timeout' instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        svc = config.get_service(solver_cfg.service)
        container_env = getattr(solver_cfg, "container_env", {})
        run_timeout = getattr(solver_cfg, "run_timeout", None)
        return HarborSolver(
            harbor_agent=solver_cfg.agent,
            harbor_agent_kwargs=solver_cfg.agent_kwargs,
            model_url=model_url,
            model_id=model_id,
            timeout=bench.timeout,
            run_timeout=run_timeout,
            api_key=api_key,
            container_env=container_env,
            max_input_tokens=getattr(svc, "max_input_tokens", None),
            max_output_tokens=getattr(svc, "max_output_tokens", None),
            cmd_timeout=getattr(solver_cfg, "cmd_timeout", None),
            timeout_strategy=getattr(solver_cfg, "timeout_strategy", "override"),
            max_agent_timeout=getattr(solver_cfg, "max_agent_timeout", None),
        )

    if isinstance(solver_cfg, GymDelegationSolverConfig):
        from nemo_evaluator.solvers.gym import GymSolver

        gym_svc = config.get_service(solver_cfg.gym_service)
        return GymSolver(
            gym_url=gym_svc.base_url,
            gym_agent=solver_cfg.gym_agent,
            trust_reward=solver_cfg.trust_reward,
            model_id=model_id,
            model_url=model_url,
            api_key=api_key,
            timeout=bench.timeout,
        )

    if isinstance(solver_cfg, ToolCallingSolverConfig):
        from nemo_evaluator.engine.model_client import ModelClient
        from nemo_evaluator.solvers.react import ReActSolver
        from nemo_evaluator.solvers.tool_backend import HttpToolBackend

        gen = _resolve_generation(config, solver_cfg)
        reasoning_pat = None
        svc = config.get_service(solver_cfg.service)
        reasoning_pat = getattr(svc, "reasoning_pattern", None)
        tc_client = ModelClient(
            base_url=model_url,
            model=model_id,
            api_key=api_key,
            temperature=gen.temperature,
            max_tokens=gen.max_tokens,
            top_p=gen.top_p,
            seed=gen.seed,
            stop=gen.stop,
            frequency_penalty=gen.frequency_penalty,
            presence_penalty=gen.presence_penalty,
            max_concurrent=bench.max_concurrent,
            reasoning_pattern=reasoning_pat,
        )

        http_backend = None
        if solver_cfg.resource_service:
            gym_svc = config.get_service(solver_cfg.resource_service)
            http_backend = HttpToolBackend(gym_svc.base_url)

        return ReActSolver(
            client=tc_client,
            http_backend=http_backend,
            use_sandbox_tools=solver_cfg.sandbox_tools,
            max_turns=solver_cfg.max_turns,
            system_prompt=solver_cfg.system_prompt,
            generation=solver_cfg.generation,
            tool_timeout=solver_cfg.tool_timeout,
            max_output_chars=solver_cfg.max_output_chars,
            response_mode=solver_cfg.response_mode,
        )

    if isinstance(solver_cfg, NatSolverConfig):
        from nemo_evaluator.solvers import NatSolver

        return NatSolver(
            nat_url=model_url,
            timeout=bench.timeout,
        )

    if isinstance(solver_cfg, OpenClawSolverConfig):
        from nemo_evaluator.solvers import OpenClawSolver

        uses_sandbox = not isinstance(sb, NoSandbox)
        gen = _resolve_generation(config, solver_cfg)
        # Sampling params (temperature, top_p, ...) are NOT passed to the
        # solver: the adapter proxy auto-injects services.generation.* into
        # every outbound request via a payload_modifier interceptor (see
        # _start_proxy).  Mirroring them into openclaw.json is redundant and
        # couples us to the OpenClaw JSON schema (broke with 2026.4.15).
        return OpenClawSolver(
            openclaw_bin=solver_cfg.openclaw_bin,
            thinking=solver_cfg.thinking,
            timeout=bench.timeout,
            model_url=model_url,
            model_id=model_id,
            api_key=api_key,
            context_window=solver_cfg.context_window,
            max_tokens=gen.max_tokens if gen else None,
            max_concurrent=solver_cfg.max_concurrent,
            config_path=solver_cfg.config_path,
            skip_preflight=solver_cfg.skip_preflight or uses_sandbox,
        )

    from nemo_evaluator.solvers import ChatSolver

    return ChatSolver(client, system_prompt=None)


def _start_model_service(svc: Any, cluster_env: dict[str, str] | None = None):
    import subprocess as _sp

    from nemo_evaluator.orchestration.model_server import DeployConfig, get_deployment

    if hasattr(svc, "setup_commands") and svc.setup_commands:
        for cmd in svc.setup_commands:
            logger.info("Running setup command: %s", cmd)
            _sp.check_call(cmd, shell=True)

    gpu_count = svc.gpus if isinstance(svc.gpus, int) else len(svc.gpus) if svc.gpus else 1

    deploy_cfg = DeployConfig(
        type=svc.type,
        model=svc.model,
        gpus=gpu_count,
        port=svc.port,
        health_path=svc.health_path,
        startup_timeout=svc.startup_timeout,
        extra_env=svc.extra_env,
        cluster_env=cluster_env or {},
        extra_args=list(svc.extra_args),
        nodes=svc.num_nodes,
        pipeline_parallel_size=svc.pipeline_parallel_size,
    )

    if svc.tensor_parallel_size:
        if svc.type == "vllm":
            deploy_cfg.extra_args.extend(["--tensor-parallel-size", str(svc.tensor_parallel_size)])
        elif svc.type == "sglang":
            deploy_cfg.extra_args.extend(["--tp-size", str(svc.tensor_parallel_size)])

    deployment = get_deployment(deploy_cfg)
    url = deployment.start()
    return deployment, url


def _start_gym_service(svc: GymResourceService):
    from nemo_evaluator.environments.gym import ManagedGymEnvironment

    gym = ManagedGymEnvironment(
        nel_benchmark=svc.benchmark,
        server_cmd=svc.server_cmd,
        port=svc.port,
        startup_timeout=svc.startup_timeout,
    )
    gym.start()
    return gym


class _NatServiceHandle:
    def __init__(self, port: int, config_file: str | None) -> None:
        import subprocess

        self.port = port
        self._config_file = config_file or "config.yml"
        self._process: subprocess.Popen | None = None
        self.endpoint = f"http://localhost:{port}"

    def start(self, startup_timeout: float = 120.0) -> None:
        import os
        import subprocess
        import time

        import urllib.error
        import urllib.request

        cmd = f"nat serve --config_file {self._config_file} --port {self.port} --host 0.0.0.0"
        logger.info("Starting NAT agent: %s", cmd)
        self._process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env={**os.environ},
        )
        deadline = time.monotonic() + startup_timeout
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                raise RuntimeError(f"NAT server exited with code {self._process.returncode} during startup")
            try:
                with urllib.request.urlopen(
                    f"{self.endpoint}/health",
                    timeout=2.0,
                ) as r:
                    if r.status == 200:
                        logger.info(
                            "NAT agent ready at %s (pid=%d)",
                            self.endpoint,
                            self._process.pid,
                        )
                        return
            except (urllib.error.URLError, OSError):
                pass
            time.sleep(1.0)
        self.stop()
        raise TimeoutError(f"NAT server not healthy within {startup_timeout}s")

    def stop(self) -> None:
        import signal
        import subprocess

        if self._process is None:
            return
        logger.info("Stopping NAT agent (pid=%d)", self._process.pid)
        try:
            self._process.send_signal(signal.SIGTERM)
            self._process.wait(timeout=10)
        except (subprocess.TimeoutExpired, OSError):
            self._process.kill()
            self._process.wait(timeout=5)
        self._process = None


class _ServiceHandle:
    def __init__(
        self,
        name: str,
        svc: Any,
        log_dir: Path | None = None,
        cluster_env: dict[str, str] | None = None,
    ) -> None:
        self.name = name
        self.svc = svc
        self._deployment = None
        self._gym = None
        self._nat = None
        self._drain_thread: threading.Thread | None = None
        self._log_dir = log_dir
        self._cluster_env = cluster_env or {}
        self.url: str = ""

    def _setup_log_drain(self) -> None:
        """Attach a pipe drain thread for the managed service's stdout."""
        if self._log_dir is None:
            return
        pipe = None
        if self._deployment and hasattr(self._deployment, "_process"):
            proc = self._deployment._process
            if proc and proc.stdout:
                pipe = proc.stdout
        elif self._nat and self._nat._process and self._nat._process.stdout:
            pipe = self._nat._process.stdout
        if pipe is not None:
            log_path = self._log_dir / f"server-{self.name}.log"
            self._drain_thread = _drain_pipe_to_file(pipe, log_path)
            logger.info("Draining %s stdout to %s", self.name, log_path)

    def start(self) -> str:
        if not self.svc.is_managed:
            self.url = self.svc.base_url
            return self.url

        if isinstance(self.svc, GymResourceService):
            self._gym = _start_gym_service(self.svc)
            self.url = self._gym.endpoint
            return self.url

        if isinstance(self.svc, NatAgentService):
            self._nat = _NatServiceHandle(
                self.svc.port,
                self.svc.nat_config_file,
            )
            self._nat.start(startup_timeout=self.svc.startup_timeout)
            self.url = self._nat.endpoint
            self._setup_log_drain()
            return self.url

        if isinstance(self.svc, _MODEL_SERVICE_TYPES) and self.svc.is_managed:
            self._deployment, self.url = _start_model_service(
                self.svc,
                cluster_env=self._cluster_env,
            )
            self._setup_log_drain()
            return self.url

        self.url = self.svc.base_url
        return self.url

    def stop(self) -> None:
        if self._deployment:
            self._deployment.stop()
            self._deployment = None
        if self._gym:
            self._gym.stop()
            self._gym = None
        if self._nat:
            self._nat.stop()
            self._nat = None


def _resolve_verifier_url(
    verifier_name: str,
    config: EvalConfig,
    handles: dict[str, _ServiceHandle],
) -> str:
    handle = handles.get(verifier_name)
    if handle and handle.url:
        return handle.url
    return config.get_model_url(verifier_name)


def _resolve_reasoning_pattern(
    config: EvalConfig,
    service_name: str | None,
) -> str | None:
    if service_name is None:
        return None
    svc = config.services.get(service_name)
    return getattr(svc, "reasoning_pattern", None) if svc else None


def _make_sandbox_manager(sb: Any) -> Any:
    if isinstance(sb, NoSandbox):
        return None

    from nemo_evaluator.sandbox.manager import SandboxManager

    sandbox_env = getattr(sb, "container_env", {}) or {}

    if isinstance(sb, DockerSandbox):
        return SandboxManager(
            backend="docker",
            concurrency=sb.concurrency,
            default_image=sb.image,
            image_template=sb.image_template,
            network=sb.network,
            memory=sb.memory,
            cpus=sb.cpus,
            global_env=sandbox_env,
        )

    if isinstance(sb, (SlurmSandbox, ApptainerSandbox)):
        het_group_raw = os.environ.get("NEL_SANDBOX_HET_GROUP")
        het_group = int(het_group_raw) if het_group_raw else None
        backend_kwargs: dict[str, Any] = {}

        if isinstance(sb, SlurmSandbox):
            backend_kwargs = {"shared_fs_root": None, "het_group": het_group}
        else:
            mem_mb = None
            if sb.memory:
                mem_str = sb.memory.strip().lower()
                if mem_str.endswith("g"):
                    mem_mb = int(float(mem_str[:-1]) * 1024)
                elif mem_str.endswith("m"):
                    mem_mb = int(float(mem_str[:-1]))
            backend_kwargs = {"memory_mb": mem_mb, "het_group": het_group}

        slurm_nodes: list[str] | None = None
        raw = os.environ.get("NEL_SANDBOX_NODES", "")
        if raw:
            slurm_nodes = raw.split(",")

        return SandboxManager(
            backend=sb.type,
            concurrency=sb.concurrency,
            default_image=sb.image,
            image_template=sb.image_template,
            slurm_nodes=slurm_nodes,
            slots_per_node=sb.slots_per_node,
            sif_cache_dir=getattr(sb, "sif_cache_dir", None),
            global_env=sandbox_env,
            **backend_kwargs,
        )

    if isinstance(sb, EcsFargateSandbox):
        return SandboxManager(
            backend="ecs_fargate",
            concurrency=sb.concurrency,
            default_image=sb.image,
            image_template=sb.image_template,
            ecs_config=_build_ecs_sandbox_config(sb),
            global_env=sandbox_env,
        )

    return None


def _make_judge_client(
    config: EvalConfig,
    judge_name: str,
    handles: dict[str, _ServiceHandle],
) -> Any:
    from nemo_evaluator.engine.model_client import ModelClient

    handle = handles.get(judge_name)
    if handle and handle.url:
        url = handle.url
    else:
        url = config.get_model_url(judge_name)
    mid = config.get_model_id(judge_name)
    api_key = config.get_api_key(judge_name)
    return ModelClient(
        base_url=url,
        model=mid,
        api_key=api_key,
        temperature=0.0,
        max_tokens=2048,
    )


_MANAGES_OWN_CLIENT = (
    GymDelegationSolverConfig,
    ToolCallingSolverConfig,
    HarborSolverConfig,
    AgentSolverConfig,
    NatSolverConfig,
    OpenClawSolverConfig,
    ContainerSolverConfig,
)


def _resolve_service_connection(
    config: EvalConfig,
    handles: dict[str, _ServiceHandle],
    service_name: str | None,
) -> tuple[str, str, str | None, Any]:
    """Return ``(model_url, model_id, api_key, svc)`` for *service_name*."""
    if not service_name:
        return "", "", None, None
    handle = handles.get(service_name)
    url = handle.url if (handle and handle.url) else config.get_model_url(service_name)
    return url, config.get_model_id(service_name), config.get_api_key(service_name), config.get_service(service_name)


def _start_proxy(
    model_url: str,
    model_id: str,
    api_key: str | None,
    svc: Any,
) -> tuple[str, Any]:
    """Start the adapter proxy when *model_url* is set.

    Returns ``(effective_url, proxy_handle | None)``.
    """
    if not model_url:
        return model_url, None

    from nemo_evaluator.adapters.proxy import start_adapter_proxy

    proxy_kwargs: dict[str, Any] = dict(upstream_url=model_url, model_id=model_id, api_key=api_key)
    proxy_cfg = getattr(svc, "proxy", None) if svc else None
    interceptor_specs: list[dict[str, Any]] = []

    if proxy_cfg is not None and proxy_cfg.needs_proxy:
        interceptor_specs = _interceptor_specs(proxy_cfg.interceptors)
        proxy_kwargs.update(
            verbose=proxy_cfg.verbose,
            extra_body=proxy_cfg.extra_body or None,
            request_timeout=proxy_cfg.request_timeout,
            max_retries=proxy_cfg.max_retries,
            retry_on_status=proxy_cfg.retry_on_status,
            max_concurrent_upstream=proxy_cfg.max_concurrent_upstream,
        )

    gen = getattr(svc, "generation", None)
    if gen is not None:
        overrides = {k: v for k, v in gen.model_dump().items() if v is not None}
        if overrides:
            interceptor_specs.append({"name": "payload_modifier", "config": {"params_to_add": overrides}})

    if interceptor_specs:
        proxy_kwargs["interceptor_specs"] = interceptor_specs

    handle = start_adapter_proxy(**proxy_kwargs)
    return handle.url, handle


def _build_environment(
    bench: Any,
    config: EvalConfig,
    handles: dict[str, _ServiceHandle],
) -> Any:
    """Create the evaluation environment, optionally wrapping with a verifier."""
    from nemo_evaluator.environments.registry import get_environment

    # Under shuffle we must load the full dataset so the permutation draws
    # from the complete index space; max_problems is applied post-shuffle.
    num_examples = None if bench.shuffle_seed is not None else bench.max_problems
    env = get_environment(
        bench.name,
        num_examples=num_examples,
        num_fewshot=bench.fewshot,
        params=bench.params,
    )
    if not bench.verifier:
        return env

    from nemo_evaluator.environments.composite import CompositeEnvironment
    from nemo_evaluator.environments.gym import GymEnvironment

    verifier_url = _resolve_verifier_url(bench.verifier, config, handles)
    return CompositeEnvironment(
        seed_env=env,
        verify_env=GymEnvironment(verifier_url, protocol="native"),
    )


def _create_client_and_solver(
    bench: Any,
    config: EvalConfig,
    solver_cfg: Any,
    service_name: str | None,
    model_url: str,
    model_id: str,
    api_key: str | None,
    concurrency: int,
    reasoning_pat: str | None,
) -> tuple[Any, Any]:
    """Return ``(client | None, solver)`` based on the solver config type."""
    from nemo_evaluator.engine.model_client import ModelClient

    if isinstance(solver_cfg, _MANAGES_OWN_CLIENT):
        client = None
    else:
        gen = _resolve_generation(config, solver_cfg) if service_name else GenerationConfig()
        client = ModelClient(
            base_url=model_url,
            model=model_id,
            api_key=api_key,
            temperature=gen.temperature,
            max_tokens=gen.max_tokens,
            top_p=gen.top_p,
            seed=gen.seed,
            stop=gen.stop,
            frequency_penalty=gen.frequency_penalty,
            presence_penalty=gen.presence_penalty,
            max_concurrent=concurrency,
            reasoning_pattern=reasoning_pat,
        )
    solver = _make_solver(bench, config, client, model_url, model_id, api_key)
    return client, solver


def _build_batch_config(
    model_url: str,
    model_id: str,
    api_key: str | None,
    solver_cfg: Any,
    svc: Any,
) -> dict[str, Any]:
    """Build the config dict for ``env.run_batch()``."""
    cfg: dict[str, Any] = {"base_url": model_url, "model": model_id, "api_key": api_key}
    if not isinstance(solver_cfg, ContainerSolverConfig):
        return cfg

    from nemo_evaluator.environments.container import _NEL_PROTOCOL_TO_LEGACY_TYPE

    protocol = getattr(svc, "protocol", "chat_completions") if svc else "chat_completions"
    cfg["endpoint_type"] = solver_cfg.endpoint_type or _NEL_PROTOCOL_TO_LEGACY_TYPE.get(protocol, "chat")
    if solver_cfg.params:
        cfg["params"] = solver_cfg.params
    return cfg


def _finalize_batch_result(
    batch_result: dict[str, Any],
    shard_info: tuple[int, int] | None,
    bench_name: str,
    env: Any,
    output_dir: Path,
) -> dict[str, Any]:
    """Strip API keys, persist artifacts, and return *batch_result*."""
    from nemo_evaluator.engine.artifacts import write_all

    if shard_info:
        logger.warning(
            "Shard env vars detected (idx=%d, total=%d) but benchmark '%s' uses run_batch() which is not shardable.",
            shard_info[0],
            shard_info[1],
            bench_name,
        )
    for key in ("api_key", "api-key"):
        batch_result.get("config", {}).pop(key, None)

    name = getattr(env, "name", bench_name)
    task_dir = output_dir / _safe_name(name)
    task_dir.mkdir(parents=True, exist_ok=True)
    write_all(batch_result, task_dir)
    return batch_result


def _find_judge_client(
    bench: Any,
    config: EvalConfig,
    handles: dict[str, _ServiceHandle],
) -> Any:
    """Return a judge ``ModelClient`` for the first ``JudgeMetric``, or *None*."""
    from nemo_evaluator.config import JudgeMetric

    for metric in bench.scoring.metrics:
        if isinstance(metric, JudgeMetric):
            return _make_judge_client(config, metric.service, handles)
    return None


# ------------------------------------------------------------------
# Main single-benchmark orchestration
# ------------------------------------------------------------------


async def _run_single_benchmark(
    bench: Any,
    config: EvalConfig,
    handles: dict[str, _ServiceHandle],
    output_dir: Path,
    *,
    resume: bool = False,
) -> dict[str, Any]:
    from nemo_evaluator.engine.artifacts import write_all
    from nemo_evaluator.engine.eval_loop import run_evaluation
    from nemo_evaluator.engine.sharding import shard_from_env
    from nemo_evaluator.observability.progress import ConsoleProgress
    from nemo_evaluator.templates import resolve_template_path

    solver_cfg = bench.solver
    service_name: str | None = getattr(solver_cfg, "service", None)

    model_url, model_id, api_key, svc = _resolve_service_connection(config, handles, service_name)
    model_url, proxy_handle = _start_proxy(model_url, model_id, api_key, svc)

    judge_client = None
    try:
        env = _build_environment(bench, config, handles)
        _client, solver = _create_client_and_solver(
            bench,
            config,
            solver_cfg,
            service_name,
            model_url,
            model_id,
            api_key,
            bench.max_concurrent,
            _resolve_reasoning_pattern(config, service_name),
        )

        shard_info = shard_from_env()
        batch_config = _build_batch_config(model_url, model_id, api_key, solver_cfg, svc)
        batch_result = await env.run_batch(solver=solver, config=batch_config)
        if batch_result is not None:
            return _finalize_batch_result(batch_result, shard_info, bench.name, env, output_dir)

        judge_client = _find_judge_client(bench, config, handles)

        bench_name = getattr(env, "name", bench.name)
        task_dir = output_dir / _safe_name(bench_name)
        task_dir.mkdir(parents=True, exist_ok=True)

        scorer_names = [m.name for m in bench.scoring.metrics if getattr(m, "type", None) == "scorer"]

        bundle = await run_evaluation(
            env,
            solver,
            n_repeats=bench.repeats,
            max_problems=bench.max_problems,
            max_concurrent=bench.max_concurrent,
            config={
                "benchmark": bench_name,
                "model": model_id,
                "base_url": model_url,
                "repeats": bench.repeats,
                "max_problems": bench.max_problems,
                "scorers": scorer_names or None,
                "_sandbox_config": bench.sandbox,
            },
            progress=ConsoleProgress(log_interval=config.output.progress_interval),
            judge_client=judge_client,
            sandbox_manager=_make_sandbox_manager(bench.sandbox),
            model_url=model_url,
            step_log_dir=task_dir,
            resume=resume,
            skip_failed=bench.skip_failed,
            max_system_retries=bench.max_system_retries,
            shard_info=shard_info,
            instruction_template=resolve_template_path(bench.instruction_template),
            shuffle_seed=bench.shuffle_seed,
        )

        write_all(bundle, task_dir)
        return bundle
    finally:
        if judge_client is not None and hasattr(judge_client, "close"):
            await judge_client.close()
        if proxy_handle is not None:
            await proxy_handle.async_stop()


def _load_prior_bundle(task_dir: str) -> dict[str, Any]:
    import json

    d = Path(task_dir)
    bundle_files = sorted(d.glob("eval-*.json"))
    if not bundle_files:
        return {"benchmark": {"name": d.name}, "_resumed": True}
    return json.loads(bundle_files[0].read_text(encoding="utf-8"))


def run_local(
    config: EvalConfig,
    *,
    resume: bool = False,
) -> list[dict[str, Any]]:
    import click

    from nemo_evaluator.engine.checkpoint import CheckpointManager

    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    cluster_env = getattr(config.cluster, "container_env", {}) or {}
    saved_env: dict[str, str | None] = {k: os.environ.get(k) for k in cluster_env}
    os.environ.update(cluster_env)

    output_dir = Path(config.output.dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_dir = output_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    ckpt = CheckpointManager(output_dir)
    if not resume:
        ckpt.clear()

    handles: dict[str, _ServiceHandle] = {}
    bundles: list[dict[str, Any]] = []

    for name, svc in config.services.items():
        if svc.is_managed:
            handles[name] = _ServiceHandle(
                name,
                svc,
                log_dir=log_dir,
                cluster_env=cluster_env,
            )

    try:
        for name, handle in handles.items():
            click.echo(f"Starting service: {name} ({handle.svc.type})")
            url = handle.start()
            click.echo(f"  {name} ready at {url}")

        for i, bench in enumerate(config.benchmarks):
            n = len(config.benchmarks)
            click.echo(f"\n{'=' * 60}\n  Benchmark {i + 1}/{n}: {bench.name}\n{'=' * 60}")

            if ckpt.is_completed(bench.name):
                prior = ckpt.get_completed_result(bench.name)
                click.echo("  Skipping (already completed)")
                bundles.append(_load_prior_bundle(prior["bundle_path"]))
                continue

            if resume and ckpt.has_partial_progress(bench.name):
                progress = ckpt.get_progress(bench.name)
                if progress:
                    click.echo(f"  Resuming ({progress['verified']} verified, {progress['inferred']} inferred)")

            try:
                bundle = asyncio.run(
                    _run_single_benchmark(
                        bench,
                        config,
                        handles,
                        output_dir,
                        resume=resume,
                    )
                )

                bench_name = bundle.get("benchmark", {}).get("name", bench.name)
                task_dir = output_dir / _safe_name(bench_name)
                ckpt.mark_completed(bench.name, str(task_dir))
                bundles.append(bundle)

                bm = bundle.get("benchmark", {})
                click.echo(f"\n  {bm.get('name', '?')}: ", nl=False)
                for k, v in bm.get("scores", {}).items():
                    if isinstance(v, dict) and "value" in v:
                        click.echo(f"{k}={v['value']:.4f} ", nl=False)
                click.echo()

            except KeyboardInterrupt:
                click.echo("\n  Interrupted — shutting down gracefully", err=True)
                break

            except Exception as exc:
                logger.error(
                    "Benchmark %s failed: %s",
                    bench.name,
                    exc,
                    exc_info=True,
                )
                ckpt.mark_failed(bench.name, str(exc))
                click.echo(f"  FAILED: {exc}", err=True)
                bundles.append(
                    {
                        "benchmark": {"name": bench.name, "samples": 0},
                        "_failed": True,
                        "_error": str(exc),
                    }
                )

        summary = ckpt.summary
        completed, failed = summary["completed"], summary["failed"]
        if failed > 0:
            click.echo(f"\n{completed} completed, {failed} failed", err=True)
            click.echo(
                "Re-run with --resume to retry failed benchmarks.",
                err=True,
            )

        _generate_reports(config, output_dir, in_memory_bundles=bundles)

    finally:
        for name, handle in reversed(list(handles.items())):
            logger.info("Stopping service: %s", name)
            handle.stop()
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    return bundles


def _generate_reports(
    config: EvalConfig,
    output_dir: Path,
    *,
    in_memory_bundles: list[dict[str, Any]] | None = None,
) -> None:
    import click

    from nemo_evaluator.reports.eval import RENDERERS, build_table, load_bundles

    bundle_files = sorted(output_dir.rglob("eval-*.json"))
    if not bundle_files:
        click.echo("No bundles found for report generation.")
        return

    bundles = load_bundles(bundle_files)
    if not bundles:
        return

    table = build_table(bundles)

    extensions = {
        "markdown": "md",
        "html": "html",
        "csv": "csv",
        "json": "json",
        "latex": "tex",
    }

    for fmt in config.output.report:
        renderer = RENDERERS.get(fmt)
        if renderer is None:
            logger.warning("Unknown report format: %s", fmt)
            continue
        ext = extensions.get(fmt, fmt)
        path = output_dir / f"report.{ext}"
        path.write_text(renderer(table), encoding="utf-8")
        click.echo(f"Report: {path}")

    export_bundles = in_memory_bundles if in_memory_bundles else bundles

    for exporter_name in config.output.export:
        try:
            from nemo_evaluator.engine.exporters import get_exporter

            exporter_kwargs = config.output.export_config.get(
                exporter_name,
                {},
            )
            exporter = get_exporter(exporter_name, **exporter_kwargs)
            exporter.export(
                export_bundles,
                config={"output_dir": str(output_dir)},
            )
            click.echo(f"Exported to: {exporter_name}")
        except Exception as exc:
            logger.error("Export to %s failed: %s", exporter_name, exc)
            click.echo(f"Export to {exporter_name} failed: {exc}", err=True)
