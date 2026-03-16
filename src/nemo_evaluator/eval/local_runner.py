"""Local suite runner: start services, run benchmarks, generate reports, cleanup."""
from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from typing import Any

from nemo_evaluator.eval.config import BenchmarkConfig, EndpointType, EvalConfig, ServiceConfig

logger = logging.getLogger(__name__)


def _safe_name(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", s)


def _make_solver(bench: BenchmarkConfig, client: Any, model_url: str,
                 model_id: str, api_key: str | None) -> Any:
    """Select the correct solver based on benchmark endpoint_type."""
    from nemo_evaluator.solvers import (
        ChatSolver,
        CompletionSolver,
        EmbeddingSolver,
        SandboxSolver,
        VLMSolver,
    )

    ep = bench.endpoint_type
    sb = bench.sandbox

    match ep:
        case EndpointType.sandbox:
            return SandboxSolver(
                agent_cmd=sb.agent_cmd if sb else "agent",
                model_url=model_url,
                model_id=model_id,
                api_key=api_key,
                setup_cmd=sb.agent_setup_cmd if sb else None,
                invocation_template=sb.agent_invocation_template if sb else None,
                timeout=sb.timeout if sb else 1800.0,
            )
        case EndpointType.nat:
            from nemo_evaluator.solvers import NatSolver
            nat_url = model_url.rsplit("/v1", 1)[0] if "/v1" in model_url else model_url
            return NatSolver(
                nat_url=nat_url,
                timeout=sb.timeout if sb else 600.0,
            )
        case EndpointType.openclaw:
            from nemo_evaluator.solvers import OpenClawSolver
            uses_sandbox = sb is not None and sb.backend != "none"
            return OpenClawSolver(
                openclaw_bin=sb.agent_cmd if sb and sb.agent_cmd else "openclaw",
                thinking="high",
                timeout=sb.timeout if sb else 600.0,
                model_url=model_url,
                model_id=model_id,
                api_key=api_key,
                context_window=bench.context_window or 131_072,
                max_tokens=bench.max_tokens,
                max_concurrent=bench.max_concurrent,
                config_path=bench.openclaw_config,
                temperature=bench.temperature,
                top_p=bench.top_p,
                skip_preflight=uses_sandbox,
            )
        case EndpointType.vlm:
            return VLMSolver(client, system_prompt=bench.system_prompt,
                             image_detail=bench.image_detail)
        case EndpointType.completions:
            return CompletionSolver(
                base_url=model_url, model=model_id, api_key=api_key,
                temperature=bench.temperature or 0.0,
                max_tokens=bench.max_tokens or 2048,
            )
        case EndpointType.embedding:
            return EmbeddingSolver(client)
        case _:
            return ChatSolver(client, system_prompt=bench.system_prompt)


def _start_model_service(svc: ServiceConfig):
    """Start a model server process and return the deployment handle."""
    from nemo_evaluator.eval.deployment import DeployConfig, get_deployment

    deploy_cfg = DeployConfig(
        type=svc.type,
        model=svc.model,
        gpus=svc.gpus if isinstance(svc.gpus, int) else len(svc.gpus) if svc.gpus else 1,
        port=svc.port,
        health_path=svc.health_path,
        startup_timeout=svc.startup_timeout,
        extra_env=svc.extra_env,
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


def _start_gym_service(svc: ServiceConfig):
    """Start a managed Gym environment server."""
    from nemo_evaluator.environments.gym import ManagedGymEnvironment

    gym = ManagedGymEnvironment(
        nel_benchmark=svc.benchmark,
        server_cmd=svc.server_cmd,
        port=svc.port,
    )
    gym.start()
    return gym


class _NatServiceHandle:
    """Manages a NAT agent server subprocess."""

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

        import httpx

        cmd = f"nat serve --config_file {self._config_file} --port {self.port} --host 0.0.0.0"
        logger.info("Starting NAT agent: %s", cmd)
        self._process = subprocess.Popen(
            cmd, shell=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            env={**os.environ},
        )
        deadline = time.monotonic() + startup_timeout
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                raise RuntimeError(f"NAT server exited with code {self._process.returncode} during startup")
            try:
                r = httpx.get(f"{self.endpoint}/health", timeout=2.0)
                if r.status_code == 200:
                    logger.info("NAT agent ready at %s (pid=%d)", self.endpoint, self._process.pid)
                    return
            except (httpx.ConnectError, httpx.TimeoutException):
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
    """Wraps a running service for uniform lifecycle management."""

    def __init__(self, name: str, svc: ServiceConfig) -> None:
        self.name = name
        self.svc = svc
        self._deployment = None
        self._gym = None
        self._nat = None
        self.url: str = ""

    def start(self) -> str:
        if self.svc.type == "api":
            self.url = self.svc.url or ""
            return self.url

        if self.svc.type == "gym":
            self._gym = _start_gym_service(self.svc)
            self.url = self._gym.endpoint
            return self.url

        if self.svc.type == "nat":
            self._nat = _NatServiceHandle(self.svc.port, self.svc.nat_config_file)
            self._nat.start(startup_timeout=self.svc.startup_timeout)
            self.url = self._nat.endpoint
            return self.url

        self._deployment, self.url = _start_model_service(self.svc)
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


def _warn_incompatible(bench: BenchmarkConfig, env: Any) -> None:
    """Emit warnings for known-bad solver + environment combinations."""
    ep = bench.endpoint_type
    name = bench.name

    from nemo_evaluator.environments.harbor import HarborEnvironment
    from nemo_evaluator.environments.mteb import MTEBEnvironment

    if isinstance(env, HarborEnvironment) and not ep.modifies_sandbox:
        logger.warning(
            "Benchmark %r uses Harbor (requires sandbox interaction) but "
            "endpoint_type=%r does not modify sandbox state. "
            "Only 'sandbox' runs inside the container. "
            "Expect reward=0.0 for all problems.",
            name, ep,
        )

    if isinstance(env, MTEBEnvironment) and ep is not EndpointType.embedding:
        logger.warning(
            "Benchmark %r uses MTEB (embedding tasks) but endpoint_type=%r "
            "-- only 'embedding' produces valid results.",
            name, ep,
        )

    if ep is EndpointType.embedding and not isinstance(env, MTEBEnvironment):
        logger.warning(
            "Benchmark %r uses endpoint_type='embedding' but environment is "
            "%s -- embedding solvers return vectors, not text. Results will "
            "be meaningless for text-based scoring.",
            name, type(env).__name__,
        )


async def _run_single_benchmark(
    bench: BenchmarkConfig,
    config: EvalConfig,
    handles: dict[str, _ServiceHandle],
    output_dir: Path,
    judge_clients: dict[str, Any],
    *,
    resume: bool = False,
) -> dict[str, Any]:
    """Run a single benchmark and return the bundle."""
    from nemo_evaluator.environments.registry import get_environment
    from nemo_evaluator.observability.progress import ConsoleProgress
    from nemo_evaluator.runner.artifacts import write_all
    from nemo_evaluator.runner.eval_loop import run_evaluation
    from nemo_evaluator.runner.model_client import ModelClient

    handle = handles.get(bench.model)
    if handle and handle.url:
        model_url = handle.url
    else:
        model_url = config.resolve_model_url(bench.model)
    model_id = config.resolve_model_id(bench.model)
    api_key = config.resolve_api_key(bench.model)

    env = get_environment(bench.name, num_examples=bench.max_problems,
                          num_fewshot=bench.fewshot)

    _warn_incompatible(bench, env)

    reasoning_pat = _resolve_reasoning_pattern(config, bench.model)

    concurrency = bench.max_concurrent
    if bench.endpoint_type.manages_own_client:
        client = None
        solver = _make_solver(bench, client, model_url, model_id, api_key)
    else:
        client = ModelClient(
            base_url=model_url,
            model=model_id,
            api_key=api_key,
            temperature=bench.temperature or 0.0,
            max_tokens=bench.max_tokens or 2048,
            max_concurrent=concurrency,
            reasoning_pattern=reasoning_pat,
        )
        solver = _make_solver(bench, client, model_url, model_id, api_key)

    # run_batch() environments own the full loop — step logging not applicable
    batch_config = {"base_url": model_url, "model": model_id, "api_key": api_key}
    batch_result = await env.run_batch(solver=solver, config=batch_config)
    if batch_result is not None:
        for _key in ("api_key", "api-key"):
            batch_result.get("config", {}).pop(_key, None)
        bench_name = getattr(env, "name", bench.name)
        safe = _safe_name(bench_name)
        task_dir = output_dir / safe
        task_dir.mkdir(parents=True, exist_ok=True)
        write_all(batch_result, task_dir)
        return batch_result

    judge_client = _make_judge_client(config, bench.judge) if bench.judge else None

    sandbox_mgr = None
    if bench.sandbox and bench.sandbox.backend != "none":
        from nemo_evaluator.sandbox.manager import SandboxManager

        sb = bench.sandbox
        backend_kwargs: dict[str, Any] = {}
        if sb.backend == "docker":
            backend_kwargs = {"network": sb.network, "memory": sb.memory, "cpus": sb.cpus}
        elif sb.backend == "slurm":
            backend_kwargs = {"shared_fs_root": None}
        elif sb.backend == "ecs_fargate":
            backend_kwargs = {"ecs_config": sb.ecs}

        slurm_nodes: list[str] | None = None
        if sb.backend == "slurm" and sb.sandbox_nodes > 0:
            import os
            raw = os.environ.get("NEL_SANDBOX_NODES", "")
            slurm_nodes = raw.split(",") if raw else None

        sandbox_mgr = SandboxManager(
            backend=sb.backend,
            concurrency=sb.concurrency,
            default_image=sb.image,
            image_template=sb.image_template,
            slurm_nodes=slurm_nodes,
            slots_per_node=sb.slots_per_node,
            **backend_kwargs,
        )

    bench_name = getattr(env, "name", bench.name)
    safe = _safe_name(bench_name)
    task_dir = output_dir / safe
    task_dir.mkdir(parents=True, exist_ok=True)

    run_config: dict[str, Any] = {
        "benchmark": bench_name,
        "model": model_id,
        "base_url": model_url,
        "repeats": bench.repeats,
        "max_problems": bench.max_problems,
    }
    if sb:
        run_config["_sandbox_config"] = sb

    bundle = await run_evaluation(
        env, solver,
        n_repeats=bench.repeats,
        max_problems=bench.max_problems,
        max_concurrent=concurrency,
        config=run_config,
        progress=ConsoleProgress(),
        judge_client=judge_client,
        sandbox_manager=sandbox_mgr,
        model_url=model_url,
        step_log_dir=task_dir,
        resume=resume,
    )

    write_all(bundle, task_dir)
    return bundle


def _make_judge_client(config: EvalConfig, judge_name: str) -> Any:
    """Create a fresh ModelClient for a judge service (safe for per-benchmark event loops)."""
    from nemo_evaluator.runner.model_client import ModelClient

    url = config.resolve_model_url(judge_name)
    mid = config.resolve_model_id(judge_name)
    api_key = config.resolve_api_key(judge_name)
    return ModelClient(base_url=url, model=mid, api_key=api_key,
                       temperature=0.0, max_tokens=2048)


def _resolve_reasoning_pattern(config: EvalConfig, service_name: str) -> str | None:
    """Extract reasoning_pattern from config, handling both simple and advanced modes."""
    if config.is_simple and config.model:
        return config.model.reasoning_pattern
    if config.services:
        svc = config.services.get(service_name)
        if svc:
            return svc.reasoning_pattern
    return None


def _load_prior_bundle(task_dir: str) -> dict[str, Any]:
    """Load an eval bundle JSON from a prior completed benchmark run."""
    import json

    d = Path(task_dir)
    bundle_files = sorted(d.glob("eval-*.json"))
    if not bundle_files:
        return {"benchmark": {"name": d.name}, "_resumed": True}
    return json.loads(bundle_files[0].read_text(encoding="utf-8"))


def run_local(config: EvalConfig, *, resume: bool = False) -> list[dict[str, Any]]:
    """Execute the full evaluation suite locally.

    1. Start all managed services
    2. Run each benchmark (with checkpoint/resume and failure isolation)
    3. Generate reports from completed benchmarks
    4. Stop all services
    """
    import click

    from nemo_evaluator.runner.checkpoint import CheckpointManager

    output_dir = Path(config.output.dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt = CheckpointManager(output_dir)
    if not resume:
        ckpt.clear()

    handles: dict[str, _ServiceHandle] = {}
    bundles: list[dict[str, Any]] = []

    for name, svc in config.resolved_services().items():
        if svc.is_managed:
            handles[name] = _ServiceHandle(name, svc)

    try:
        for name, handle in handles.items():
            click.echo(f"Starting service: {name} ({handle.svc.type})")
            url = handle.start()
            click.echo(f"  {name} ready at {url}")

        for i, bench in enumerate(config.benchmarks):
            n = len(config.benchmarks)
            click.echo(f"\n{'='*60}\n  Benchmark {i+1}/{n}: {bench.name}\n{'='*60}")

            if ckpt.is_completed(bench.name):
                prior = ckpt.get_completed_result(bench.name)
                click.echo("  Skipping (already completed)")
                bundles.append(_load_prior_bundle(prior["bundle_path"]))
                continue

            if resume and ckpt.has_partial_progress(bench.name):
                progress = ckpt.get_progress(bench.name)
                if progress:
                    click.echo(f"  Resuming ({progress['verified']} verified, "
                               f"{progress['inferred']} inferred)")

            try:
                bundle = asyncio.run(_run_single_benchmark(
                    bench, config, handles, output_dir, {},
                    resume=resume,
                ))

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

            except Exception as exc:
                logger.error("Benchmark %s failed: %s", bench.name, exc, exc_info=True)
                ckpt.mark_failed(bench.name, str(exc))
                click.echo(f"  FAILED: {exc}", err=True)
                bundles.append({
                    "benchmark": {"name": bench.name, "samples": 0},
                    "_failed": True,
                    "_error": str(exc),
                })

        summary = ckpt.summary
        completed, failed = summary["completed"], summary["failed"]
        if failed > 0:
            click.echo(f"\n{completed} completed, {failed} failed", err=True)
            click.echo("Re-run with --resume to retry failed benchmarks.", err=True)

        _generate_reports(config, output_dir)

    finally:
        for name, handle in reversed(list(handles.items())):
            logger.info("Stopping service: %s", name)
            handle.stop()

    return bundles


def _generate_reports(config: EvalConfig, output_dir: Path) -> None:
    """Generate all requested report formats and run export plugins."""
    import click

    from nemo_evaluator.cli.report import RENDERERS, _build_table, _load_bundles

    bundle_files = sorted(output_dir.rglob("eval-*.json"))
    if not bundle_files:
        click.echo("No bundles found for report generation.")
        return

    bundles = _load_bundles(bundle_files)
    if not bundles:
        return

    table = _build_table(bundles)

    extensions = {
        "markdown": "md", "html": "html", "csv": "csv",
        "json": "json", "latex": "tex",
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

    for exporter_name in config.output.export:
        try:
            from nemo_evaluator.runner.exporters import get_exporter
            exporter_kwargs = config.output.export_config.get(exporter_name, {})
            exporter = get_exporter(exporter_name, **exporter_kwargs)
            exporter.export(bundles, config={"output_dir": str(output_dir)})
            click.echo(f"Exported to: {exporter_name}")
        except Exception as exc:
            logger.error("Export to %s failed: %s", exporter_name, exc)
            click.echo(f"Export to {exporter_name} failed: {exc}", err=True)
