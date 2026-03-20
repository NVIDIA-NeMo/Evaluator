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
@click.option("--output-dir", "-o", default=None, help="Output directory")
@click.option("--dry-run", is_flag=True, help="Generate scripts without running")
@click.option("--submit", is_flag=True, help="Submit to cluster via SSH")
@click.option("--background", is_flag=True, help="Run locally in background")
@click.option("--resume", is_flag=True, help="Resume a partially completed evaluation")
@click.option("--override", "-O", multiple=True, help="Override: key=value (dot notation)")
@click.option("--verbose", "-v", is_flag=True)
def eval_run(config_file, bench, model_url, model_id, api_key,
             repeats, max_problems, system_prompt, temperature, max_tokens,
             output_dir, dry_run, submit, background, resume, override, verbose):
    """Start evaluation. Accepts a config YAML or --bench for quick mode."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if config_file:
        config = _load_config(config_file, overrides=override)
        if output_dir is not None:
            config.output.dir = output_dir
    elif bench:
        config = _build_quick_config(
            bench, model_url, model_id, api_key, repeats,
            max_problems, system_prompt, temperature, max_tokens,
            output_dir or "./eval_results",
        )
    else:
        raise click.ClickException("Specify a config file or --bench <name>.")

    import os

    from nemo_evaluator.executors import get_executor

    executor_type = config.cluster.type
    if os.environ.get("NEL_INNER_EXECUTION") == "1":
        logging.getLogger(__name__).warning(
            "NEL_INNER_EXECUTION=1 — forcing local executor (config says %s)",
            executor_type,
        )
        executor_type = "local"

    try:
        executor = get_executor(executor_type)
    except KeyError as e:
        raise click.ClickException(str(e))
    executor.run(config, dry_run=dry_run, resume=resume,
                 background=background, submit=submit)


def _load_config(config_file: str, overrides: tuple[str, ...] = ()):
    import yaml

    from nemo_evaluator.eval.config import parse_eval_config

    raw = yaml.safe_load(Path(config_file).read_text()) or {}
    for ov in overrides:
        _apply_override(raw, ov)
    return parse_eval_config(raw)


def _apply_override(data: dict, override: str) -> None:
    if "=" not in override:
        raise click.ClickException(
            f"Invalid override format: {override!r}. Expected key=value"
        )
    key, value = override.split("=", 1)
    parts = key.strip().split(".")

    parsed_value: object = value
    if value.lower() in ("true", "false"):
        parsed_value = value.lower() == "true"
    else:
        try:
            parsed_value = int(value)
        except ValueError:
            try:
                parsed_value = float(value)
            except ValueError:
                pass

    d = data
    for p in parts[:-1]:
        if isinstance(d, list):
            try:
                d = d[int(p)]
            except (ValueError, IndexError):
                raise click.ClickException(
                    f"Cannot index list with {p!r} in override {key!r}"
                )
        else:
            d = d.setdefault(p, {})
    last = parts[-1]
    if isinstance(d, list):
        try:
            d[int(last)] = parsed_value
        except (ValueError, IndexError):
            raise click.ClickException(
                f"Cannot index list with {last!r} in override {key!r}"
            )
    else:
        d[last] = parsed_value


def _build_quick_config(bench, model_url, model_id, api_key, repeats,
                        max_problems, system_prompt, temperature, max_tokens,
                        output_dir):
    from nemo_evaluator.eval.config import (
        BenchmarkConfig,
        EvalConfig,
        ExternalApiService,
        GenerationConfig,
        OutputConfig,
        SimpleSolver,
    )

    if not model_url:
        raise click.ClickException("--model-url is required for quick mode.")
    if not model_id:
        raise click.ClickException("--model-id is required for quick mode.")

    url = model_url.rstrip("/")
    if not url.endswith(("/chat/completions", "/completions", "/responses")):
        url = url + "/chat/completions"

    return EvalConfig(
        services={
            "model": ExternalApiService(
                type="api",
                url=url,
                protocol="chat_completions",
                model=model_id,
                api_key=api_key,
                generation=GenerationConfig(
                    temperature=temperature,
                    max_tokens=max_tokens,
                ),
            ),
        },
        benchmarks=[BenchmarkConfig(
            name=bench,
            solver=SimpleSolver(
                type="simple",
                service="model",
                system_prompt=system_prompt,
            ),
            repeats=repeats,
            max_problems=max_problems,
        )],
        output=OutputConfig(dir=output_dir),
    )


# ---------------------------------------------------------------------------
# nel eval status
# ---------------------------------------------------------------------------

@eval_cmd.command("status")
@click.option("--output-dir", "-o", default="./eval_results")
@click.option("--job-id", default=None, help="SLURM job ID")
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

    from nemo_evaluator.executors import detect_executor

    ex = detect_executor(output_dir)
    if ex is None:
        raise click.ClickException(f"No evaluation metadata found in {output_dir}")
    state = ex.status(output_dir)
    click.echo(f"Executor: {state.executor}")
    click.echo(f"Running:  {state.running}")
    for k, v in state.details.items():
        click.echo(f"  {k}: {v}")


# ---------------------------------------------------------------------------
# nel eval stop
# ---------------------------------------------------------------------------

@eval_cmd.command("stop")
@click.option("--output-dir", "-o", default="./eval_results")
@click.option("--job-id", default=None, help="SLURM job ID")
@click.option("--host", default=None, help="SLURM login hostname")
@click.option("--user", default=None, help="SSH username")
def eval_stop(output_dir, job_id, host, user):
    """Stop/cancel a running evaluation."""
    if job_id and host:
        from nemo_evaluator.eval.ssh import cancel_job

        cancel_job(host, job_id, user)
        click.echo(f"Cancelled SLURM job {job_id}")
        return

    from nemo_evaluator.executors import detect_executor

    ex = detect_executor(output_dir)
    if ex is None:
        raise click.ClickException(f"No evaluation metadata found in {output_dir}")
    if ex.stop(output_dir):
        click.echo("Evaluation stopped.")
    else:
        click.echo("Could not stop evaluation (may already be finished).", err=True)


# ---------------------------------------------------------------------------
# nel eval report
# ---------------------------------------------------------------------------

@eval_cmd.command("report")
@click.argument("results_dir", type=click.Path(exists=True))
@click.option("--format", "-f", "fmt",
              type=click.Choice(["markdown", "html", "csv", "json", "latex"]),
              default="markdown")
@click.option("--output", "-o", type=click.Path(), default=None)
@click.option("--all-formats", is_flag=True)
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
        ext_map = {
            "markdown": "md", "html": "html", "csv": "csv",
            "json": "json", "latex": "tex",
        }
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


# ---------------------------------------------------------------------------
# nel eval merge
# ---------------------------------------------------------------------------

@eval_cmd.command("merge")
@click.argument("output_dir", type=click.Path(exists=True))
@click.option("--repeats", "-n", type=int, default=None,
              help="Override n_repeats (auto-detected from shard bundles if omitted)")
def eval_merge(output_dir, repeats):
    """Merge results from sharded evaluation runs.

    OUTPUT_DIR should contain shard_0/, shard_1/, ... subdirectories
    produced by a SLURM array job with shards configured.
    """
    import json

    root = Path(output_dir)
    shard_dirs = sorted(root.glob("shard_*"))
    if not shard_dirs:
        raise click.ClickException(
            f"No shard_*/ directories found in {output_dir}. "
            f"Expected shard_0/, shard_1/, etc."
        )

    actual = set()
    for d in shard_dirs:
        try:
            actual.add(int(d.name.split("_", 1)[1]))
        except (ValueError, IndexError):
            pass
    max_idx = max(actual) if actual else 0
    missing = set(range(max_idx + 1)) - actual
    if missing:
        click.echo(
            f"Warning: missing shards: {sorted(missing)}. "
            f"Merge will produce partial results.",
            err=True,
        )

    benchmarks: dict[str, list[Path]] = {}
    for sd in shard_dirs:
        for bench_dir in sorted(sd.iterdir()):
            if bench_dir.is_dir() and (bench_dir / "results.jsonl").exists():
                benchmarks.setdefault(bench_dir.name, []).append(bench_dir)

    if not benchmarks:
        raise click.ClickException(
            "No benchmark results (results.jsonl) found in any shard directory."
        )

    n_repeats = repeats
    if n_repeats is None:
        for dirs in benchmarks.values():
            for d in dirs:
                for bp in d.glob("eval-*.json"):
                    try:
                        cfg = json.loads(bp.read_text(encoding="utf-8")).get("config", {})
                        if "repeats" in cfg:
                            n_repeats = int(cfg["repeats"])
                            break
                    except (json.JSONDecodeError, ValueError):
                        pass
                if n_repeats is not None:
                    break
            if n_repeats is not None:
                break
    if n_repeats is None:
        n_repeats = 1
        click.echo("Warning: could not detect n_repeats from bundles, defaulting to 1.", err=True)

    from nemo_evaluator.runner.sharding import merge_results

    for bench_name, dirs in sorted(benchmarks.items()):
        merged_dir = root / bench_name
        click.echo(f"Merging {bench_name}: {len(dirs)} shards -> {merged_dir}")
        _dedup_shard_results(dirs)
        bundle = merge_results(dirs, str(merged_dir), n_repeats=n_repeats)
        scores = bundle.get("benchmark", {}).get("scores", {})
        for k, v in scores.items():
            if isinstance(v, dict) and "value" in v:
                click.echo(f"  {k}: {v['value']:.4f}")

    click.echo(f"\nMerge complete. Results in {root}/")


def _dedup_shard_results(shard_dirs: list[Path]) -> None:
    """Deduplicate results.jsonl by (problem_idx, repeat), keeping last occurrence."""
    import json

    for d in shard_dirs:
        rp = d / "results.jsonl"
        if not rp.exists():
            continue
        seen: dict[tuple[int, int], str] = {}
        for line in rp.read_text(encoding="utf-8").strip().split("\n"):
            if not line:
                continue
            try:
                r = json.loads(line)
                key = (r.get("problem_idx", -1), r.get("repeat", 0))
                seen[key] = line
            except json.JSONDecodeError:
                pass
        rp.write_text("\n".join(seen.values()) + "\n", encoding="utf-8")
