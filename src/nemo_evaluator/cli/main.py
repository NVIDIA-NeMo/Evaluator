"""CLI entrypoint: nel {run, serve, validate, list-environments}."""

import click

from nemo_evaluator.cli.harness import harness_group
from nemo_evaluator.cli.regression import regression_cmd
from nemo_evaluator.cli.report import report_cmd
from nemo_evaluator.cli.run import run_cmd
from nemo_evaluator.cli.serve import serve_cmd
from nemo_evaluator.cli.slurm import slurm_cmd
from nemo_evaluator.cli.validate import validate_cmd


@click.group()
@click.version_option(package_name="nemo-evaluator")
def cli():
    """NeMo Evaluator CLI."""


cli.add_command(run_cmd, "run")
cli.add_command(serve_cmd, "serve")
cli.add_command(validate_cmd, "validate")
cli.add_command(regression_cmd, "regression")
cli.add_command(report_cmd, "report")
cli.add_command(harness_group, "harness")
cli.add_command(slurm_cmd, "slurm")


@cli.command("list-environments")
def list_envs():
    """List all registered benchmark environments."""
    from nemo_evaluator.environments.registry import list_environments

    names = list_environments()
    if not names:
        click.echo("No environments registered.")
        return
    click.echo("Available environments:")
    for name in names:
        click.echo(f"  - {name}")


@cli.command("list-harnesses")
def list_harnesses():
    """List known legacy evaluator container harnesses."""
    from nemo_evaluator.adapters.container import HARNESS_IMAGES

    click.echo("Legacy evaluator harnesses:")
    for name, image in sorted(HARNESS_IMAGES.items()):
        click.echo(f"  {name:30s} {image}")


@cli.command("list-skills")
@click.option("--data-dir", help="NeMo Skills data directory")
def list_skills(data_dir):
    """List available NeMo Skills benchmarks (requires nemo-skills)."""
    try:
        from nemo_evaluator.adapters.skills import list_skills_benchmarks
    except ImportError:
        click.echo("nemo-skills is not installed. Install: pip install nemo-skills", err=True)
        return

    benchmarks = list_skills_benchmarks(data_dir)
    if not benchmarks:
        click.echo("No prepared Skills benchmarks found. Run: ns prepare_data <benchmark>")
        return
    click.echo("Available Skills benchmarks:")
    for b in benchmarks:
        splits = ", ".join(b["splits"])
        click.echo(f"  {b['benchmark']:30s} type={b['metrics_type']:15s} splits=[{splits}]")


@cli.command("container-eval")
@click.argument("task_type")
@click.option("--model-url", required=True, help="Model endpoint URL")
@click.option("--model-id", required=True, help="Model identifier")
@click.option("--harness", help="Override harness (default: inferred from task)")
@click.option("--image", help="Override container image")
@click.option("--limit-samples", type=int, help="Limit number of samples")
@click.option("--temperature", type=float, default=0.0)
@click.option("--output-dir", "-o", default="./eval_results")
@click.option("--timeout", type=int, default=3600)
def container_eval_cmd(task_type, model_url, model_id, harness, image,
                       limit_samples, temperature, output_dir, timeout):
    """Run evaluation via legacy evaluator container."""
    import json
    import logging

    from nemo_evaluator.adapters.container import ContainerConfig, run_container_eval

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = ContainerConfig(
        task_type=task_type,
        model_url=model_url,
        model_id=model_id,
        harness=harness,
        image=image,
        limit_samples=limit_samples,
        temperature=temperature,
        timeout=timeout,
    )

    click.echo(f"Running container eval: {task_type}")
    bundle = run_container_eval(cfg, output_dir)

    click.echo("\nScores:")
    for key, score in bundle.get("scores", {}).items():
        if isinstance(score, dict) and "value" in score:
            click.echo(f"  {key}: {score['value']}")

    stats = bundle.get("response_stats", {})
    if stats.get("count", 0) > 0:
        click.echo(f"\nResponse stats: {stats['count']} requests, "
                    f"avg_latency={stats.get('avg_latency_ms', 0):.0f}ms")

    from pathlib import Path
    bundle_path = Path(output_dir) / "container_bundle.json"
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    with open(bundle_path, "w") as f:
        json.dump(bundle, f, indent=2, default=str)
    click.echo(f"\nBundle: {bundle_path}")


if __name__ == "__main__":
    cli()
