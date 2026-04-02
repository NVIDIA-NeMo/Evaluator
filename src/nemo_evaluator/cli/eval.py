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
@click.option("--max-tokens", type=int, default=None)
@click.option("--output-dir", "-o", default=None, help="Output directory")
@click.option("--dry-run", is_flag=True, help="Generate scripts without running")
@click.option("--submit", is_flag=True, help="Submit to cluster via SSH")
@click.option("--background", is_flag=True, help="Run locally in background")
@click.option("--resume", is_flag=True, help="Resume a partially completed evaluation")
@click.option("--override", "-O", multiple=True, help="Override: key=value (dot notation)")
@click.option("--verbose", "-v", is_flag=True)
def eval_run(
    config_file,
    bench,
    model_url,
    model_id,
    api_key,
    repeats,
    max_problems,
    system_prompt,
    temperature,
    max_tokens,
    output_dir,
    dry_run,
    submit,
    background,
    resume,
    override,
    verbose,
):
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
            bench,
            model_url,
            model_id,
            api_key,
            repeats,
            max_problems,
            system_prompt,
            temperature,
            max_tokens,
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
    executor.run(config, dry_run=dry_run, resume=resume, background=background, submit=submit)


def _load_config(config_file: str, overrides: tuple[str, ...] = ()):
    from nemo_evaluator.config import parse_eval_config
    from nemo_evaluator.config.compose import compose_config, _prune_nulls

    raw = compose_config(config_file)
    for ov in overrides:
        _apply_override(raw, ov)
    _prune_nulls(raw)
    return parse_eval_config(raw)


def _apply_override(data: dict, override: str) -> None:
    if "=" not in override:
        raise click.ClickException(f"Invalid override format: {override!r}. Expected key=value")
    key, value = override.split("=", 1)
    parts = key.strip().split(".")

    parsed_value: object = value
    if value in ("null", "Null", "NULL", "~"):
        parsed_value = None
    elif value.lower() in ("true", "false"):
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
                raise click.ClickException(f"Cannot index list with {p!r} in override {key!r}")
        else:
            d = d.setdefault(p, {})
    last = parts[-1]
    if isinstance(d, list):
        try:
            d[int(last)] = parsed_value
        except (ValueError, IndexError):
            raise click.ClickException(f"Cannot index list with {last!r} in override {key!r}")
    else:
        d[last] = parsed_value


def _build_quick_config(
    bench, model_url, model_id, api_key, repeats, max_problems, system_prompt, temperature, max_tokens, output_dir
):
    from nemo_evaluator.config import (
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
        benchmarks=[
            BenchmarkConfig(
                name=bench,
                solver=SimpleSolver(
                    type="simple",
                    service="model",
                    system_prompt=system_prompt,
                ),
                repeats=repeats,
                max_problems=max_problems,
            )
        ],
        output=OutputConfig(dir=output_dir),
    )


# ---------------------------------------------------------------------------
# Shared run resolution helper
# ---------------------------------------------------------------------------


def _resolve_run_or_fail(run_id=None, output_dir=None, job_id=None, host=None):
    """Resolve a RunMeta from --run-id, --output-dir, or legacy --job-id."""
    from nemo_evaluator.run_store import resolve_run

    meta = resolve_run(run_id=run_id, output_dir=output_dir, job_id=job_id, host=host)
    if meta is not None:
        return meta

    if job_id or host:
        from nemo_evaluator.executors.slurm_executor import resolve_job

        legacy = resolve_job(job_id=job_id, host=host, output_dir=output_dir)
        if legacy is not None:
            from nemo_evaluator.run_store import RunMeta

            return RunMeta(
                run_id=f"legacy-{legacy.get('job_id', 'unknown')}",
                executor="slurm",
                output_dir=legacy.get("remote_dir", str(output_dir or "")),
                started_at=legacy.get("submitted_at", ""),
                config_summary="",
                details=legacy,
            )

    hint = f"run-id={run_id}" if run_id else f"job-id={job_id}" if job_id else f"output-dir={output_dir}"
    raise click.ClickException(f"No run metadata found for {hint}.")


def _get_executor_for_run(run_meta):
    """Get the executor instance for a RunMeta."""
    from nemo_evaluator.executors import get_executor

    return get_executor(run_meta.executor)


# ---------------------------------------------------------------------------
# nel eval status
# ---------------------------------------------------------------------------


@eval_cmd.command("status")
@click.option("--run-id", "-r", default=None, help="NEL run ID")
@click.option("--output-dir", "-o", default=None)
@click.option("--job-id", default=None, help="SLURM job ID (legacy)")
@click.option("--host", default=None, help="SLURM login hostname (legacy)")
@click.option("--watch", "-w", is_flag=True, default=False, help="Refresh every 10s until Ctrl+C")
def eval_status(run_id, output_dir, job_id, host, watch):
    """Check evaluation status."""
    import time

    def _fetch_status():
        if run_id or output_dir or job_id:
            run_meta = _resolve_run_or_fail(run_id, output_dir, job_id, host)
            ex = _get_executor_for_run(run_meta)
            state = ex.status(run_meta.output_dir)
            lines = [f"Run:      {run_meta.run_id}"]
        else:
            from nemo_evaluator.executors import detect_executor

            ex = detect_executor("./eval_results")
            if ex is None:
                raise click.ClickException("No evaluation metadata found. Use -r <run_id> or -o <dir>.")
            state = ex.status("./eval_results")
            lines = []

        lines += [
            f"Executor: {state.executor}",
            f"Running:  {state.running}",
        ]
        for k, v in state.details.items():
            lines.append(f"  {k}: {v}")
        return state.running, "\n".join(lines)

    if not watch:
        _, text = _fetch_status()
        click.echo(text)
        return

    try:
        while True:
            still_running, text = _fetch_status()
            ts = time.strftime("%H:%M:%S")
            click.clear()
            click.echo(text)
            click.echo(f"\n[{ts}] Refreshing in 60s... (Ctrl+C to stop)")
            if not still_running:
                click.echo("Run finished.")
                break
            time.sleep(60)
    except KeyboardInterrupt:
        click.echo()


# ---------------------------------------------------------------------------
# nel eval stop
# ---------------------------------------------------------------------------


@eval_cmd.command("stop")
@click.option("--run-id", "-r", default=None, help="NEL run ID")
@click.option("--output-dir", "-o", default=None)
@click.option("--job-id", default=None, help="SLURM job ID (legacy)")
@click.option("--host", default=None, help="SLURM login hostname (legacy)")
@click.option("--shard", "-s", "shard_idx", type=int, default=None, help="Cancel specific shard (0-indexed)")
def eval_stop(run_id, output_dir, job_id, host, shard_idx):
    """Stop/cancel a running evaluation."""
    if run_id or output_dir or job_id:
        run_meta = _resolve_run_or_fail(run_id, output_dir, job_id, host)
        ex = _get_executor_for_run(run_meta)
        if shard_idx is not None and run_meta.executor != "slurm":
            raise click.ClickException("--shard is only supported for SLURM runs.")
        stop_kwargs = {"shard_idx": shard_idx} if shard_idx is not None else {}
        if ex.stop(run_meta.output_dir, **stop_kwargs):
            label = f" shard {shard_idx}" if shard_idx is not None else ""
            click.echo(f"Stopped{label} run {run_meta.run_id} ({run_meta.executor})")
        else:
            click.echo("Could not stop (may already be finished).", err=True)
        return

    from nemo_evaluator.executors import detect_executor

    ex = detect_executor("./eval_results")
    if ex is None:
        raise click.ClickException("No evaluation metadata found. Use -r <run_id> or -o <dir>.")
    if ex.stop("./eval_results"):
        click.echo("Evaluation stopped.")
    else:
        click.echo("Could not stop evaluation (may already be finished).", err=True)


# ---------------------------------------------------------------------------
# nel eval jobs
# ---------------------------------------------------------------------------


@eval_cmd.command("jobs")
@click.option("--offline", is_flag=True, help="Skip live status checks")
def eval_jobs(offline):
    """List all tracked evaluation runs."""
    from nemo_evaluator.executors import get_executor
    from nemo_evaluator.run_store import list_runs

    runs = list_runs()
    if not runs:
        click.echo("No tracked runs.")
        return

    header = f"{'RUN_ID':<26} {'EXECUTOR':<10} {'STATE':<20} {'BENCHMARKS':<30} {'OUTPUT_DIR'}"
    click.echo(header)
    click.echo("-" * len(header))

    for run in runs:
        state = "\u2014"
        if not offline:
            try:
                ex = get_executor(run.executor)
                ps = ex.status(run.output_dir)
                state = "running" if ps.running else "stopped"
                st = ps.details.get("state") or ps.details.get("status")
                if st:
                    state = str(st)
                elif ps.details.get("shards"):
                    state = str(ps.details["shards"]).split(",")[0]
            except Exception:
                state = "ERR"

        started = run.started_at
        if started and len(started) > 19:
            started = started[:19]

        summary = run.config_summary or "\u2014"
        if len(summary) > 28:
            summary = summary[:25] + "..."

        odir = run.output_dir
        if len(odir) > 60:
            odir = "..." + odir[-57:]

        if len(state) > 18:
            state = state[:15] + "..."
        click.echo(f"{run.run_id:<26} {run.executor:<10} {state:<20} {summary:<30} {odir}")


# ---------------------------------------------------------------------------
# nel eval logs
# ---------------------------------------------------------------------------


@eval_cmd.command("logs")
@click.option("--run-id", "-r", default=None, help="NEL run ID")
@click.option("--output-dir", "-o", default=None)
@click.option("--job-id", default=None, help="SLURM job ID (legacy)")
@click.option("--host", default=None, help="SLURM login hostname (legacy)")
@click.option("--follow", "-f", is_flag=True, help="Stream logs (tail -f)")
@click.option("--tail", "-n", "tail_lines", type=int, default=None, help="Last N lines")
@click.option("--shard", "-s", "shard_idx", type=int, default=None, help="Shard index for sharded runs")
def eval_logs(run_id, output_dir, job_id, host, follow, tail_lines, shard_idx):
    """View evaluation logs."""
    run_meta = _resolve_run_or_fail(run_id, output_dir, job_id, host)
    ex = _get_executor_for_run(run_meta)

    if follow and run_meta.executor == "slurm":
        import subprocess
        import sys

        details = run_meta.details
        hostname = details.get("hostname", "")
        username = details.get("username") or None
        rdir = details.get("remote_dir", run_meta.output_dir)
        job_ids = details.get("job_ids", [])
        is_sharded_run = len(job_ids) > 1

        if is_sharded_run and shard_idx is None:
            from nemo_evaluator.executors.slurm_executor import (
                _RUNNING_STATES,
                _resolve_shard_latest_ids,
            )
            from nemo_evaluator.executors.ssh import batch_check_job_status

            latest_ids = _resolve_shard_latest_ids(hostname, rdir, job_ids, username)
            all_status = batch_check_job_status(hostname, latest_ids, username)
            running_shards = []
            for i, lid in enumerate(latest_ids):
                info = all_status.get(lid, {})
                state = info.get("state", "UNKNOWN")
                click.echo(f"  shard {i}: {state} (job {lid})", err=True)
                if state in _RUNNING_STATES:
                    running_shards.append(i)

            if len(running_shards) == 1:
                shard_idx = running_shards[0]
                click.echo(f"\nAuto-selected shard {shard_idx} (only running shard).", err=True)
            elif len(running_shards) == 0:
                click.echo("\nNo shards currently running. Use --shard N to view a completed shard.", err=True)
                return
            else:
                click.echo(
                    f"\nMultiple running shards: {running_shards}. Use --shard N to pick one.",
                    err=True,
                )
                return

        if is_sharded_run:
            if shard_idx < 0 or shard_idx >= len(job_ids):
                raise click.ClickException(
                    f"Shard index {shard_idx} out of range. This run has {len(job_ids)} shards (0-{len(job_ids) - 1})."
                )
            from nemo_evaluator.executors.slurm_executor import _resolve_latest_job_id

            shard_rdir = f"{rdir}/shard_{shard_idx}"
            latest_id = _resolve_latest_job_id(hostname, shard_rdir, job_ids[shard_idx], username)
            log_file = f"{shard_rdir}/logs/slurm-{latest_id}.log"
        else:
            from nemo_evaluator.executors.slurm_executor import _resolve_latest_job_id_from_meta

            latest_id = _resolve_latest_job_id_from_meta(details)
            log_file = f"{rdir}/logs/slurm-{latest_id}.log"

        target = f"{username}@{hostname}" if username else hostname
        n_arg = f"-n {tail_lines}" if tail_lines else "-n 50"
        cmd = ["ssh", target, f"tail {n_arg} -f {log_file}"]
        click.echo(f"Streaming: {target}:{log_file}", err=True)
        try:
            subprocess.run(cmd, check=False)
        except KeyboardInterrupt:
            sys.exit(0)
        return

    if follow and run_meta.executor == "docker":
        import subprocess
        import sys

        container_id = run_meta.details.get("container_id", "")
        cmd = ["docker", "logs", "-f"]
        if tail_lines:
            cmd.extend(["--tail", str(tail_lines)])
        cmd.append(container_id)
        try:
            subprocess.run(cmd, check=False)
        except KeyboardInterrupt:
            sys.exit(0)
        return

    log_kwargs: dict = {"follow": follow, "tail": tail_lines}
    if shard_idx is not None:
        log_kwargs["shard_idx"] = shard_idx
    content = ex.logs(run_meta.output_dir, **log_kwargs)
    if content is None:
        click.echo("No logs found.", err=True)
    else:
        click.echo(content)


# ---------------------------------------------------------------------------
# nel eval resume
# ---------------------------------------------------------------------------


@eval_cmd.command("resume")
@click.option("--run-id", "-r", default=None, help="NEL run ID")
@click.option("--output-dir", "-o", default=None)
@click.option("--job-id", default=None, help="SLURM job ID (legacy)")
@click.option("--host", default=None, help="SLURM login hostname (legacy)")
@click.option("--continue-attempts", is_flag=True, help="Keep existing retry counter (default: reset)")
@click.option("--shard", "-s", "shard_idx", type=int, default=None, help="Resume specific shard only")
def eval_resume(run_id, output_dir, job_id, host, continue_attempts, shard_idx):
    """Resume a failed/timed-out evaluation."""
    run_meta = _resolve_run_or_fail(run_id, output_dir, job_id, host)
    ex = _get_executor_for_run(run_meta)
    click.echo(f"Resuming run {run_meta.run_id} ({run_meta.executor})")
    ex.resume_run(run_meta, continue_attempts=continue_attempts, shard_idx=shard_idx)


# ---------------------------------------------------------------------------
# nel eval clean
# ---------------------------------------------------------------------------


@eval_cmd.command("clean")
@click.option("--older-than", default=None, help="Remove entries older than duration (e.g. 7d, 4w, 24h)")
@click.option(
    "--executor", "filter_executor", default=None, help="Remove only entries for this executor (slurm, docker, local)"
)
@click.option("--dry-run", is_flag=True, help="Show what would be removed")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
def eval_clean(older_than, filter_executor, dry_run, yes):
    """Clean stale entries from the run store."""
    import re
    from datetime import datetime, timedelta, timezone

    from nemo_evaluator.run_store import list_runs, remove_run

    runs = list_runs()
    if not runs:
        click.echo("No tracked runs to clean.")
        return

    cutoff = None
    if older_than:
        m = re.match(r"^(\d+)\s*([hdwm])$", older_than.strip())
        if not m:
            raise click.ClickException(f"Invalid duration: {older_than!r}. Use e.g. 7d, 4w, 24h")
        amount, unit = int(m.group(1)), m.group(2)
        delta = {
            "h": timedelta(hours=amount),
            "d": timedelta(days=amount),
            "w": timedelta(weeks=amount),
            "m": timedelta(days=amount * 30),
        }[unit]
        cutoff = datetime.now(timezone.utc) - delta

    to_remove = []
    for run in runs:
        if filter_executor and run.executor != filter_executor:
            continue

        if cutoff and run.started_at:
            try:
                sub = datetime.fromisoformat(run.started_at)
                if sub > cutoff:
                    continue
            except ValueError:
                pass

        to_remove.append(run)

    if not to_remove:
        click.echo("Nothing to clean.")
        return

    click.echo(f"Found {len(to_remove)} entries to remove:")
    for run in to_remove:
        started = run.started_at[:19] if run.started_at and len(run.started_at) > 19 else run.started_at
        click.echo(f"  {run.run_id}  {run.executor:<8}  {started}  {run.config_summary}")

    if dry_run:
        click.echo("(dry-run \u2014 nothing removed)")
        return

    if not yes:
        click.confirm("Remove these entries?", abort=True)

    removed = 0
    for run in to_remove:
        if remove_run(run.run_id):
            removed += 1
    click.echo(f"Removed {removed} entries.")


# ---------------------------------------------------------------------------
# nel eval report
# ---------------------------------------------------------------------------


@eval_cmd.command("report")
@click.argument("results_dir", type=click.Path(exists=True))
@click.option(
    "--format", "-f", "fmt", type=click.Choice(["markdown", "html", "csv", "json", "latex"]), default="markdown"
)
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
            "markdown": "md",
            "html": "html",
            "csv": "csv",
            "json": "json",
            "latex": "tex",
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
@click.option(
    "--repeats", "-n", type=int, default=None, help="Override n_repeats (auto-detected from shard bundles if omitted)"
)
def eval_merge(output_dir, repeats):
    """Merge results from sharded evaluation runs.

    OUTPUT_DIR should contain shard_0/, shard_1/, ... subdirectories
    produced by a SLURM array job with shards configured.
    """
    import json

    root = Path(output_dir)
    shard_dirs = sorted(root.glob("shard_*"))
    if not shard_dirs:
        raise click.ClickException(f"No shard_*/ directories found in {output_dir}. Expected shard_0/, shard_1/, etc.")

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
            f"Warning: missing shards: {sorted(missing)}. Merge will produce partial results.",
            err=True,
        )

    benchmarks: dict[str, list[Path]] = {}
    for sd in shard_dirs:
        for results_file in sd.rglob("results.jsonl"):
            benchmarks.setdefault(results_file.parent.name, []).append(results_file.parent)

    if not benchmarks:
        raise click.ClickException("No benchmark results (results.jsonl) found in any shard directory.")

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

    from nemo_evaluator.engine.sharding import merge_results

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
