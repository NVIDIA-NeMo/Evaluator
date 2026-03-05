from __future__ import annotations

import asyncio
import logging
import signal
import sys
from pathlib import Path
from typing import Any

import click
import yaml


@click.command("run")
@click.argument("config_file", required=False, type=click.Path(exists=True))
@click.option("--benchmark", "-b", help="Benchmark name")
@click.option("--adapter", "-a", help="Adapter URI: gym://host:port, pi://env, skills://benchmark")
@click.option("--model-url", envvar="NEMO_MODEL_URL", help="Model endpoint URL (or set NEMO_MODEL_URL)")
@click.option("--model-id", envvar="NEMO_MODEL_ID", help="Model identifier (or set NEMO_MODEL_ID)")
@click.option("--api-key", envvar="NEMO_API_KEY")
@click.option("--repeats", "-n", type=int, default=None)
@click.option("--max-problems", type=int, default=None)
@click.option("--system-prompt", type=str, default=None)
@click.option("--output-dir", "-o", default="./eval_results")
@click.option("--progress/--no-progress", default=True)
@click.option("--verbose", "-v", is_flag=True)
def run_cmd(config_file, benchmark, adapter, model_url, model_id, api_key,
            repeats, max_problems, system_prompt, output_dir, progress, verbose):
    """Run evaluation from config, benchmark name, or adapter URI."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = {}
    if config_file:
        try:
            config = yaml.safe_load(Path(config_file).read_text()) or {}
        except Exception as e:
            raise click.ClickException(f"Failed to parse config file: {e}")

    eval_cfg = config.get("evaluation", config)
    tasks = eval_cfg.get("tasks", [])

    if benchmark or adapter:
        tasks = [{"benchmark": benchmark, "adapter": adapter}]
    elif not tasks:
        raise click.ClickException("Specify --benchmark, --adapter, or a config file with tasks.")

    global_url = model_url or eval_cfg.get("model_url")
    global_mid = model_id or eval_cfg.get("model_id")
    global_repeats = repeats
    global_max = max_problems

    from nemo_evaluator.observability.progress import ConsoleProgress, NoOpProgress
    from nemo_evaluator.runner.artifacts import write_all
    from nemo_evaluator.runner.eval_loop import run_evaluation
    from nemo_evaluator.runner.model_client import ModelClient
    from nemo_evaluator.runner.sharding import get_shard_range, shard_from_env

    all_bundles: list[dict[str, Any]] = []
    partial_bundle: dict[str, Any] | None = None

    def _handle_sigterm(*_):
        if partial_bundle:
            click.echo("\nInterrupted -- writing partial results...", err=True)
            try:
                write_all(partial_bundle, output_dir)
            except Exception:
                pass
        sys.exit(1)

    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigterm)

    for i, task in enumerate(tasks):
        adapter_uri = task.get("adapter")
        bench = task.get("benchmark") or task.get("resource_server")
        if not bench and not adapter_uri:
            click.echo(f"Task {i}: no benchmark or adapter, skipping", err=True)
            continue

        url = task.get("model_url") or global_url
        mid = task.get("model_id") or global_mid
        if not url or not mid:
            raise click.ClickException(
                "Model URL and ID are required. Set --model-url/--model-id or NEMO_MODEL_URL/NEMO_MODEL_ID."
            )

        n = global_repeats or task.get("repeats", 1)
        mp = global_max or task.get("max_problems")
        sys_prompt = system_prompt or task.get("system_prompt")

        try:
            env = _resolve(bench, adapter_uri)
        except Exception as e:
            raise click.ClickException(f"Failed to resolve environment: {e}")

        client = ModelClient(base_url=url, model=mid, api_key=api_key, temperature=0.0)

        run_config = {"benchmark": bench or adapter_uri, "model": mid, "base_url": url,
                      "repeats": n, "max_problems": mp, "adapter": adapter_uri}

        shard_info = shard_from_env()
        problem_range = None
        if shard_info:
            shard_idx, total_shards = shard_info
            total = asyncio.run(_get_ds_size(env, mp))
            problem_range = get_shard_range(total, shard_idx, total_shards)
            run_config["shard"] = {"idx": shard_idx, "total": total_shards, "range": list(problem_range)}

        pg = ConsoleProgress() if progress else NoOpProgress()
        if len(tasks) > 1:
            click.echo(f"\n{'='*60}\n  Task {i+1}/{len(tasks)}: {bench or adapter_uri}\n{'='*60}")

        bundle = asyncio.run(run_evaluation(
            env, client, n_repeats=n, max_problems=mp,
            system_prompt=sys_prompt, config=run_config,
            progress=pg, problem_range=problem_range,
        ))
        partial_bundle = bundle

        task_dir = output_dir if len(tasks) == 1 else f"{output_dir}/{_safe_name(bench or adapter_uri)}"
        paths = write_all(bundle, task_dir)

        _print_summary(bundle, run_config, n, paths, task_dir)
        all_bundles.append(bundle)

    if len(all_bundles) > 1:
        click.echo(f"\n{'='*60}\n  Summary: {len(all_bundles)} tasks\n{'='*60}")
        for b in all_bundles:
            name = b.get("config", {}).get("benchmark", "?")
            scores = b.get("benchmark", {}).get("scores", {})
            p1 = scores.get("pass@1", {})
            if isinstance(p1, dict) and "value" in p1:
                click.echo(f"  {name}: pass@1={p1['value']:.4f}")


def _print_summary(bundle, run_config, n, paths, output_dir):
    click.echo(f"\n{run_config['benchmark']}")
    bm = bundle.get("benchmark", {})
    click.echo(f"  problems={bm.get('samples', 0)}  repeats={n}")
    for k, v in bm.get("scores", {}).items():
        if isinstance(v, dict) and "value" in v:
            ci_lo = v.get("ci_lower")
            ci_hi = v.get("ci_upper")
            if ci_lo is not None and ci_hi is not None:
                click.echo(f"  {k}: {v['value']:.4f}  [{ci_lo:.4f}, {ci_hi:.4f}]")
            else:
                click.echo(f"  {k}: {v['value']:.4f}")
    rt = bm.get("scores", {}).get("runtime", {})
    if isinstance(rt, dict) and rt.get("total_tokens"):
        click.echo(f"  tokens={rt['total_tokens']:,}  speed={rt.get('steps_per_second', 0):.1f}/s")
    fa = bm.get("scores", {}).get("failures", {})
    if isinstance(fa, dict) and fa.get("total_failures", 0) > 0:
        click.echo(f"  failures={fa['total_failures']} ({fa.get('failure_rate', 0):.1%})")
    click.echo(f"\n  Output: {output_dir}/")
    for name, p in paths.items():
        click.echo(f"    {name}: {p.name}")


def _safe_name(s: str) -> str:
    import re
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", s)


async def _get_ds_size(env, max_problems):
    from nemo_evaluator.adapters.base import EnvironmentAdapter
    if isinstance(env, EnvironmentAdapter):
        total = await env.dataset_size()
    else:
        total = len(env)
    return min(total, max_problems) if max_problems else total


_ADAPTER_SCHEMES: dict[str, Any] = {}


def _init_builtin_adapters():
    if _ADAPTER_SCHEMES:
        return

    def _gym(rest):
        from nemo_evaluator.adapters.gym import GymAdapter
        return GymAdapter(f"http://{rest}")

    def _pi(rest):
        from nemo_evaluator.adapters.pi import PIAdapter
        return PIAdapter(rest)

    def _skills(rest):
        from nemo_evaluator.adapters.skills import SkillsAdapter
        return SkillsAdapter(rest)

    _ADAPTER_SCHEMES.update({"gym": _gym, "pi": _pi, "skills": _skills})


def register_adapter(scheme: str):
    """Register a custom adapter scheme (e.g. ``register_adapter("mycloud")``).

    The decorated function receives the part after ``scheme://`` and returns an adapter instance.
    """
    def decorator(factory):
        _ADAPTER_SCHEMES[scheme] = factory
        return factory
    return decorator


def _resolve(bench: str | None, adapter_uri: str | None):
    if adapter_uri:
        _init_builtin_adapters()
        for scheme, factory in _ADAPTER_SCHEMES.items():
            prefix = f"{scheme}://"
            if adapter_uri.startswith(prefix):
                return factory(adapter_uri[len(prefix):])

        known = ", ".join(f"{s}://" for s in sorted(_ADAPTER_SCHEMES))
        raise click.ClickException(
            f"Unknown adapter URI scheme: {adapter_uri!r}. Known: {known}"
        )

    from nemo_evaluator.environments.registry import get_environment
    try:
        return get_environment(bench)
    except KeyError:
        raise click.ClickException(f"Unknown benchmark: {bench!r}. Run 'nel list-environments'.")
