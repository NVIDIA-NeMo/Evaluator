from __future__ import annotations

import asyncio
import logging

import click

DEFAULT_OUTPUT_DIR = "./eval_results"
DEFAULT_MAX_TOKENS = 2048


@click.command("run")
@click.argument("config_file", required=False, type=click.Path(exists=True))
@click.option("--env", "-e", "env_uri", help="Environment: gsm8k, lm-eval/aime25, skills://mmlu-pro, gym://host:port")
@click.option("--benchmark", "-b", "env_uri_compat", hidden=True)
@click.option("--adapter", "-a", "adapter_compat", hidden=True)
@click.option("--model-url", envvar="NEMO_MODEL_URL")
@click.option("--model-id", envvar="NEMO_MODEL_ID")
@click.option("--api-key", envvar="NEMO_API_KEY")
@click.option("--repeats", "-n", type=int, default=1)
@click.option("--max-problems", type=int, default=None)
@click.option("--system-prompt", type=str, default=None)
@click.option("--temperature", type=float, default=0.0)
@click.option("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
@click.option("--top-p", type=float, default=None)
@click.option("--seed", type=int, default=None)
@click.option("--cache-dir", envvar="NEL_CACHE_DIR", default=None)
@click.option("--output-dir", "-o", default=DEFAULT_OUTPUT_DIR)
# Executor options
@click.option("--executor", type=click.Choice(["local", "docker", "slurm"]), default="local")
# Deployment options
@click.option("--deploy", type=click.Choice(["api", "nim", "vllm", "sglang", "docker"]), default=None)
@click.option("--deploy-image", default=None, help="Docker image for model server")
@click.option("--deploy-model", default=None, help="Model name/path for deployment")
@click.option("--deploy-gpus", type=int, default=1)
@click.option("--deploy-port", type=int, default=8000)
# Judge options
@click.option("--judge-url", default=None, help="Judge model API URL")
@click.option("--judge-model", default=None, help="Judge model ID")
@click.option("--judge-api-key", default=None)
# SLURM options
@click.option("--slurm-partition", default=None)
@click.option("--slurm-gpus", type=int, default=None)
@click.option("--slurm-time", default="04:00:00")
# Progress
@click.option("--progress/--no-progress", default=True)
@click.option("--verbose", "-v", is_flag=True)
def run_cmd(config_file, env_uri, env_uri_compat, adapter_compat,
            model_url, model_id, api_key,
            repeats, max_problems, system_prompt, temperature, max_tokens,
            top_p, seed, cache_dir, output_dir,
            executor, deploy, deploy_image, deploy_model, deploy_gpus, deploy_port,
            judge_url, judge_model, judge_api_key,
            slurm_partition, slurm_gpus, slurm_time,
            progress, verbose):
    """Run evaluation. Specify --env, or a config file."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    resolved_env = env_uri or env_uri_compat or adapter_compat

    # Config file mode
    if config_file and not resolved_env:
        _run_config_file(config_file, model_url, model_id, api_key, repeats,
                         max_problems, system_prompt, temperature, max_tokens,
                         top_p, seed, cache_dir, output_dir, progress)
        return

    if not resolved_env:
        raise click.ClickException("Specify --env or a config file.")

    # Build RunConfig
    from nemo_evaluator.executors.base import DeployConfig, JudgeConfig, RunConfig, get_executor

    deploy_cfg = None
    if deploy:
        deploy_cfg = DeployConfig(
            type=deploy, image=deploy_image, model=deploy_model,
            gpus=deploy_gpus, port=deploy_port,
        )

    judge_cfg = None
    if judge_url or judge_model:
        judge_cfg = JudgeConfig(model_url=judge_url, model_id=judge_model, api_key=judge_api_key)

    run_config = RunConfig(
        env=resolved_env,
        model_url=model_url, model_id=model_id, api_key=api_key,
        deploy=deploy_cfg, judge=judge_cfg,
        repeats=repeats, max_problems=max_problems,
        system_prompt=system_prompt, temperature=temperature,
        max_tokens=max_tokens, top_p=top_p, seed=seed,
        cache_dir=cache_dir, output_dir=output_dir,
        slurm_partition=slurm_partition, slurm_gpus=slurm_gpus,
        slurm_time=slurm_time,
    )

    exec_instance = get_executor(executor)
    bundle = asyncio.run(exec_instance.execute(run_config))

    # Print summary
    if bundle and "benchmark" in bundle:
        _print_summary(bundle)


def _run_config_file(config_file, model_url, model_id, api_key, repeats,
                     max_problems, system_prompt, temperature, max_tokens,
                     top_p, seed, cache_dir, output_dir, progress_flag):
    """Handle YAML config file with multiple tasks."""
    from pathlib import Path
    from typing import Any

    import yaml

    from nemo_evaluator.environments.registry import get_environment
    from nemo_evaluator.observability.progress import ConsoleProgress, NoOpProgress
    from nemo_evaluator.runner.artifacts import write_all
    from nemo_evaluator.runner.eval_loop import run_evaluation
    from nemo_evaluator.runner.model_client import ModelClient
    from nemo_evaluator.runner.solver import ChatSolver

    config = yaml.safe_load(Path(config_file).read_text()) or {}
    from nemo_evaluator.config_schema import parse_config
    eval_cfg = parse_config(config)

    tasks = [t.model_dump(exclude_none=True) for t in eval_cfg.tasks]
    global_url = model_url or eval_cfg.model_url
    global_mid = model_id or eval_cfg.model_id

    all_bundles: list[dict[str, Any]] = []
    for i, task in enumerate(tasks):
        env_name = task.get("env") or task.get("benchmark") or task.get("adapter")
        if not env_name:
            continue
        url = task.get("model_url") or global_url
        mid = task.get("model_id") or global_mid
        if not url or not mid:
            raise click.ClickException("Model URL and ID required.")

        env = get_environment(env_name, num_examples=max_problems or task.get("max_problems"))
        client = ModelClient(
            base_url=url, model=mid, api_key=api_key,
            temperature=temperature, max_tokens=max_tokens,
            top_p=top_p, seed=seed, cache_dir=cache_dir,
        )
        solver = ChatSolver(client, system_prompt=system_prompt)
        n = repeats or task.get("repeats", 1)
        pg = ConsoleProgress() if progress_flag else NoOpProgress()

        if len(tasks) > 1:
            click.echo(f"\n{'='*60}\n  Task {i+1}/{len(tasks)}: {env_name}\n{'='*60}")

        bundle = asyncio.run(run_evaluation(
            env, solver, n_repeats=n, max_problems=max_problems,
            config={"benchmark": getattr(env, "name", env_name), "model": mid},
            progress=pg,
        ))

        import re
        safe = re.sub(r"[^a-zA-Z0-9_.-]", "_", getattr(env, "name", env_name))
        task_dir = output_dir if len(tasks) == 1 else f"{output_dir}/{safe}"
        write_all(bundle, task_dir)
        all_bundles.append(bundle)
        _print_summary(bundle)


def _print_summary(bundle):
    bm = bundle.get("benchmark", {})
    name = bm.get("name", "?")
    click.echo(f"\n{name}")
    click.echo(f"  problems={bm.get('samples', 0)}")
    for k, v in bm.get("scores", {}).items():
        if isinstance(v, dict) and "value" in v:
            ci_lo = v.get("ci_lower")
            ci_hi = v.get("ci_upper")
            if ci_lo is not None and ci_hi is not None:
                click.echo(f"  {k}: {v['value']:.4f}  [{ci_lo:.4f}, {ci_hi:.4f}]")
            else:
                click.echo(f"  {k}: {v['value']:.4f}")
