"""nel eval — orchestrated evaluation with lifecycle management.

Subcommands:
  run     Start evaluation (foreground, background, or submit to cluster)
  status  Check a running evaluation
  stop    Stop/cancel a running evaluation
  report  Generate reports from completed results
"""
from __future__ import annotations

import logging
from pathlib import Path

import click


@click.group("eval")
def eval_cmd():
    """Evaluate model(s) on benchmark(s)."""


# ---------------------------------------------------------------------------
# nel eval run
# ---------------------------------------------------------------------------

@eval_cmd.command("run")
@click.argument("config_file", required=False, type=click.Path(exists=True))
@click.option("--bench", "-b", help="Benchmark name (quick single-benchmark mode)")
@click.option("--model-url", envvar="NEMO_MODEL_URL")
@click.option("--model-id", envvar="NEMO_MODEL_ID")
@click.option("--api-key", envvar="NEMO_API_KEY")
@click.option("--repeats", "-n", type=int, default=1)
@click.option("--max-problems", type=int, default=None)
@click.option("--system-prompt", type=str, default=None)
@click.option("--temperature", type=float, default=0.0)
@click.option("--max-tokens", type=int, default=2048)
@click.option("--output-dir", "-o", default="./eval_results")
@click.option("--dry-run", is_flag=True, help="Generate SLURM/Docker scripts without running")
@click.option("--submit", is_flag=True, help="Submit to cluster via SSH")
@click.option("--background", is_flag=True, help="Run locally in background, write PID file")
@click.option("--no-progress", is_flag=True, hidden=True)
@click.option("--verbose", "-v", is_flag=True)
def eval_run(config_file, bench, model_url, model_id, api_key,
             repeats, max_problems, system_prompt, temperature, max_tokens,
             output_dir, dry_run, submit, background, no_progress, verbose):
    """Start evaluation. Accepts a config YAML or --bench for quick mode."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if config_file:
        config = _load_config(config_file)
    elif bench:
        config = _build_quick_config(
            bench, model_url, model_id, api_key, repeats,
            max_problems, system_prompt, temperature, max_tokens, output_dir,
        )
    else:
        raise click.ClickException("Specify a config file or --bench <name>.")

    cluster_type = config.cluster.type

    if cluster_type == "slurm":
        _handle_slurm(config, dry_run, submit)
    elif cluster_type == "local":
        if background:
            _run_background(config)
        else:
            _run_foreground(config)
    else:
        raise click.ClickException(f"Unsupported cluster type: {cluster_type}")


def _load_config(config_file: str):
    import yaml

    from nemo_evaluator.eval.config import parse_eval_config

    raw = yaml.safe_load(Path(config_file).read_text()) or {}
    return parse_eval_config(raw)


def _build_quick_config(bench, model_url, model_id, api_key, repeats,
                        max_problems, system_prompt, temperature, max_tokens, output_dir):
    from nemo_evaluator.eval.config import (
        BenchmarkConfig,
        EvalConfig,
        ModelConfig,
        OutputConfig,
    )

    if not model_url:
        raise click.ClickException("--model-url is required for quick mode.")
    if not model_id:
        raise click.ClickException("--model-id is required for quick mode.")

    return EvalConfig(
        model=ModelConfig(url=model_url, id=model_id, api_key=api_key),
        benchmarks=[BenchmarkConfig(
            name=bench, repeats=repeats, max_problems=max_problems,
            system_prompt=system_prompt, temperature=temperature,
            max_tokens=max_tokens,
        )],
        output=OutputConfig(dir=output_dir),
    )


def _handle_slurm(config, dry_run: bool, submit: bool):
    from nemo_evaluator.eval.slurm_gen import write_sbatch

    script_path = write_sbatch(config)
    click.echo(f"Generated: {script_path}")

    if dry_run:
        click.echo(f"\nTo submit locally:  sbatch {script_path}")
        if config.cluster.hostname:
            click.echo(f"To submit via SSH:  nel eval run {script_path} --submit")
        return

    if submit and config.cluster.hostname:
        from nemo_evaluator.eval.ssh import submit_eval
        meta = submit_eval(
            script_path=script_path,
            hostname=config.cluster.hostname,
            remote_dir=config.output.dir,
            username=config.cluster.username,
        )
        click.echo(f"SLURM job submitted: {meta['job_id']}")
        click.echo(f"Check status: nel eval status -o {config.output.dir}")
        return

    import subprocess
    result = subprocess.run(["sbatch", str(script_path)], capture_output=True, text=True)
    if result.returncode != 0:
        raise click.ClickException(f"sbatch failed: {result.stderr}")
    click.echo(result.stdout.strip())


def _run_foreground(config):
    from nemo_evaluator.eval.lifecycle import write_local_pid
    from nemo_evaluator.eval.local_runner import run_local

    write_local_pid(config.output.dir)
    try:
        bundles = run_local(config)
        click.echo(f"\nCompleted {len(bundles)} benchmark(s). Results: {config.output.dir}")
    finally:
        pid_file = Path(config.output.dir) / "nel.pid"
        if pid_file.exists():
            pid_file.unlink()


def _run_background(config):
    import os
    import sys

    from nemo_evaluator.eval.lifecycle import write_local_pid

    pid = os.fork()
    if pid > 0:
        write_local_pid(config.output.dir, pid)
        click.echo(f"Background evaluation started (PID {pid})")
        click.echo(f"Check status: nel eval status -o {config.output.dir}")
        click.echo(f"Stop:         nel eval stop -o {config.output.dir}")
        return

    os.setsid()
    sys.stdin.close()

    log_path = Path(config.output.dir) / "nel_eval.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fd = open(log_path, "w")
    os.dup2(log_fd.fileno(), 1)
    os.dup2(log_fd.fileno(), 2)

    try:
        from nemo_evaluator.eval.local_runner import run_local
        run_local(config)
    finally:
        pid_file = Path(config.output.dir) / "nel.pid"
        if pid_file.exists():
            pid_file.unlink()
        log_fd.close()
        os._exit(0)


# ---------------------------------------------------------------------------
# nel eval status
# ---------------------------------------------------------------------------

@eval_cmd.command("status")
@click.option("--output-dir", "-o", default="./eval_results", help="Output directory to check")
@click.option("--job-id", default=None, help="SLURM job ID (with --host)")
@click.option("--host", default=None, help="SLURM login hostname")
@click.option("--user", default=None, help="SSH username")
def eval_status(output_dir, job_id, host, user):
    """Check evaluation status."""
    if job_id and host:
        from nemo_evaluator.eval.ssh import check_job_status
        info = check_job_status(host, job_id, user)
        for k, v in info.items():
            click.echo(f"  {k}: {v}")
        return

    from nemo_evaluator.eval.lifecycle import status
    state = status(output_dir)
    click.echo(f"Executor: {state.executor}")
    click.echo(f"Running:  {state.running}")
    for k, v in state.details.items():
        click.echo(f"  {k}: {v}")


# ---------------------------------------------------------------------------
# nel eval stop
# ---------------------------------------------------------------------------

@eval_cmd.command("stop")
@click.option("--output-dir", "-o", default="./eval_results", help="Output directory to check")
@click.option("--job-id", default=None, help="SLURM job ID (with --host)")
@click.option("--host", default=None, help="SLURM login hostname")
@click.option("--user", default=None, help="SSH username")
def eval_stop(output_dir, job_id, host, user):
    """Stop/cancel a running evaluation."""
    if job_id and host:
        from nemo_evaluator.eval.ssh import cancel_job
        cancel_job(host, job_id, user)
        click.echo(f"Cancelled SLURM job {job_id}")
        return

    from nemo_evaluator.eval.lifecycle import stop
    if stop(output_dir):
        click.echo("Evaluation stopped.")
    else:
        click.echo("Could not stop evaluation (may already be finished).", err=True)


# ---------------------------------------------------------------------------
# nel eval report
# ---------------------------------------------------------------------------

@eval_cmd.command("report")
@click.argument("results_dir", type=click.Path(exists=True))
@click.option("--format", "-f", "fmt", type=click.Choice(["markdown", "html", "csv", "json", "latex"]), default="markdown")
@click.option("--output", "-o", type=click.Path(), default=None)
@click.option("--all-formats", is_flag=True, help="Generate all report formats")
def eval_report(results_dir, fmt, output, all_formats):
    """Generate evaluation reports from completed results."""
    from nemo_evaluator.cli.report import RENDERERS, _build_table, _load_bundles

    results = Path(results_dir)
    bundle_files = sorted(results.rglob("eval-*.json"))

    if not bundle_files:
        raise click.ClickException(f"No eval bundles found in {results_dir}")

    bundles = _load_bundles(bundle_files)
    if not bundles:
        raise click.ClickException("No valid bundles loaded.")

    table = _build_table(bundles)

    if all_formats:
        ext_map = {"markdown": "md", "html": "html", "csv": "csv", "json": "json", "latex": "tex"}
        for f, renderer in RENDERERS.items():
            ext = ext_map.get(f, f)
            path = results / f"report.{ext}"
            path.write_text(renderer(table), encoding="utf-8")
            click.echo(f"Report: {path}")
        return

    rendered = RENDERERS[fmt](table)
    if output:
        Path(output).write_text(rendered, encoding="utf-8")
        click.echo(f"Report written to {output}")
    else:
        click.echo(rendered)
