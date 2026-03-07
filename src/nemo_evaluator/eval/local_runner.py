"""Local suite runner: start services, run benchmarks, generate reports, cleanup."""
from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from typing import Any

from nemo_evaluator.eval.config import BenchmarkConfig, EvalConfig, ServiceConfig

logger = logging.getLogger(__name__)


def _safe_name(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", s)


def _start_model_service(svc: ServiceConfig):
    """Start a model server process and return the deployment handle."""
    from nemo_evaluator.executors.base import DeployConfig
    from nemo_evaluator.runner.deployment import get_deployment

    deploy_cfg = DeployConfig(
        type=svc.type,
        model=svc.model,
        gpus=svc.gpus if isinstance(svc.gpus, int) else len(svc.gpus) if svc.gpus else 1,
        port=svc.port,
        health_path=svc.health_path,
        startup_timeout=svc.startup_timeout,
        extra_env=svc.extra_env,
        extra_args=list(svc.extra_args),
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


class _ServiceHandle:
    """Wraps a running service for uniform lifecycle management."""

    def __init__(self, name: str, svc: ServiceConfig) -> None:
        self.name = name
        self.svc = svc
        self._deployment = None
        self._gym = None
        self.url: str = ""

    def start(self) -> str:
        if self.svc.type == "api":
            self.url = self.svc.url or ""
            return self.url

        if self.svc.type == "gym":
            self._gym = _start_gym_service(self.svc)
            self.url = self._gym.endpoint
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


async def _run_single_benchmark(
    bench: BenchmarkConfig,
    config: EvalConfig,
    handles: dict[str, _ServiceHandle],
    output_dir: Path,
    judge_clients: dict[str, Any],
) -> dict[str, Any]:
    """Run a single benchmark and return the bundle."""
    from nemo_evaluator.environments.registry import get_environment
    from nemo_evaluator.observability.progress import ConsoleProgress
    from nemo_evaluator.runner.artifacts import write_all
    from nemo_evaluator.runner.eval_loop import run_evaluation
    from nemo_evaluator.runner.model_client import ModelClient
    from nemo_evaluator.runner.solver import ChatSolver

    model_url = config.resolve_model_url(bench.model)
    model_id = config.resolve_model_id(bench.model)
    api_key = config.resolve_api_key(bench.model)

    env = get_environment(bench.name, num_examples=bench.max_problems)

    client = ModelClient(
        base_url=model_url,
        model=model_id,
        api_key=api_key,
        temperature=bench.temperature or 0.0,
        max_tokens=bench.max_tokens or 2048,
    )
    solver = ChatSolver(client, system_prompt=bench.system_prompt)

    judge_client = None
    if bench.judge and bench.judge in judge_clients:
        judge_client = judge_clients[bench.judge]

    bench_name = getattr(env, "name", bench.name)
    run_config = {
        "benchmark": bench_name,
        "model": model_id,
        "base_url": model_url,
        "repeats": bench.repeats,
        "max_problems": bench.max_problems,
    }

    bundle = await run_evaluation(
        env, solver,
        n_repeats=bench.repeats,
        max_problems=bench.max_problems,
        config=run_config,
        progress=ConsoleProgress(),
        judge_client=judge_client,
    )

    safe = _safe_name(bench_name)
    task_dir = output_dir / safe
    write_all(bundle, task_dir)
    return bundle


def _build_judge_clients(
    config: EvalConfig,
    benchmarks: list[BenchmarkConfig],
) -> dict[str, Any]:
    """Pre-create ModelClient instances for any judge services referenced by benchmarks."""
    from nemo_evaluator.runner.model_client import ModelClient

    judge_names = {b.judge for b in benchmarks if b.judge}
    clients: dict[str, Any] = {}

    for name in judge_names:
        url = config.resolve_model_url(name)
        mid = config.resolve_model_id(name)
        api_key = config.resolve_api_key(name)
        clients[name] = ModelClient(
            base_url=url, model=mid, api_key=api_key,
            temperature=0.0, max_tokens=2048,
        )

    return clients


def run_local(config: EvalConfig) -> list[dict[str, Any]]:
    """Execute the full evaluation suite locally.

    1. Start all managed services
    2. Run each benchmark sequentially
    3. Generate reports
    4. Stop all services
    """
    import click

    output_dir = Path(config.output.dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    handles: dict[str, _ServiceHandle] = {}
    bundles: list[dict[str, Any]] = []

    # Simple mode: auto-deploy if model.deploy is set
    if config.is_simple and config.model.deploy:
        from nemo_evaluator.eval.config import ServiceConfig
        auto_svc = ServiceConfig(
            type=config.model.deploy,
            model=config.model.name or config.model.id,
            port=config.model.port,
            extra_env=config.model.extra_env,
            extra_args=list(config.model.extra_args),
        )
        if config.model.tensor_parallel_size:
            auto_svc.tensor_parallel_size = config.model.tensor_parallel_size
        handles["default"] = _ServiceHandle("default", auto_svc)

    # Advanced mode: start all managed services
    if config.services:
        for name, svc in config.services.items():
            if svc.is_managed:
                handles[name] = _ServiceHandle(name, svc)

    try:
        for name, handle in handles.items():
            click.echo(f"Starting service: {name} ({handle.svc.type})")
            url = handle.start()
            click.echo(f"  {name} ready at {url}")

        judge_clients = _build_judge_clients(config, config.benchmarks)

        for i, bench in enumerate(config.benchmarks):
            n = len(config.benchmarks)
            click.echo(f"\n{'='*60}\n  Benchmark {i+1}/{n}: {bench.name}\n{'='*60}")

            bundle = asyncio.run(_run_single_benchmark(
                bench, config, handles, output_dir, judge_clients,
            ))
            bundles.append(bundle)

            bm = bundle.get("benchmark", {})
            click.echo(f"\n  {bm.get('name', '?')}: ", nl=False)
            for k, v in bm.get("scores", {}).items():
                if isinstance(v, dict) and "value" in v:
                    click.echo(f"{k}={v['value']:.4f} ", nl=False)
            click.echo()

        _generate_reports(config, output_dir)

    finally:
        for name, handle in reversed(list(handles.items())):
            logger.info("Stopping service: %s", name)
            handle.stop()

    return bundles


def _generate_reports(config: EvalConfig, output_dir: Path) -> None:
    """Generate all requested report formats."""
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
