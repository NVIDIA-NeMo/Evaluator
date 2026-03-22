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
# Shared job resolution helper
# ---------------------------------------------------------------------------

def _resolve_or_fail(job_id, host, output_dir):
    from nemo_evaluator.executors.slurm import resolve_job

    meta = resolve_job(job_id=job_id, host=host, output_dir=output_dir)
    if meta is None:
        hint = f"job-id={job_id}" if job_id else f"output-dir={output_dir}"
        raise click.ClickException(
            f"No job metadata found for {hint}. "
            f"Verify SSH access with: ssh <host> hostname"
        )
    return meta


def _resolve_latest_job_id(meta: dict) -> str:
    """Follow .nel_job_chain on the remote host to get the latest job ID."""
    remote_dir = meta.get("remote_dir", "")
    hostname = meta.get("hostname", "")
    if not remote_dir or not hostname:
        return meta["job_id"]
    try:
        from nemo_evaluator.eval.ssh import ssh_run

        chain = ssh_run(
            hostname,
            f"tail -1 {remote_dir}/.nel_job_chain 2>/dev/null",
            username=meta.get("username") or None,
            timeout=10.0,
        ).strip()
        return chain if chain else meta["job_id"]
    except Exception:
        return meta["job_id"]


# ---------------------------------------------------------------------------
# nel eval status
# ---------------------------------------------------------------------------

@eval_cmd.command("status")
@click.option("--output-dir", "-o", default=None)
@click.option("--job-id", default=None, help="SLURM job ID (bare number or full path)")
@click.option("--host", default=None, help="SLURM login hostname")
@click.option("--user", default=None, help="SSH username")
def eval_status(output_dir, job_id, host, user):
    """Check evaluation status. Follows auto-resume job chains."""
    if job_id or output_dir:
        meta = _resolve_or_fail(job_id, host, output_dir or "./eval_results")
        latest_id = _resolve_latest_job_id(meta)

        from nemo_evaluator.eval.ssh import check_job_status

        target_host = meta["hostname"]
        info = check_job_status(target_host, latest_id, meta.get("username") or user)

        if latest_id != meta["job_id"]:
            click.echo(f"Original job: {meta['job_id']}  (chain → {latest_id})")
        for k, v in info.items():
            click.echo(f"  {k}: {v}")
        if meta.get("remote_dir"):
            click.echo(f"  remote_dir: {meta['remote_dir']}")
        return

    from nemo_evaluator.executors import detect_executor

    ex = detect_executor("./eval_results")
    if ex is None:
        raise click.ClickException("No evaluation metadata found. Use --job-id or -o.")
    state = ex.status("./eval_results")
    click.echo(f"Executor: {state.executor}")
    click.echo(f"Running:  {state.running}")
    for k, v in state.details.items():
        click.echo(f"  {k}: {v}")


# ---------------------------------------------------------------------------
# nel eval stop
# ---------------------------------------------------------------------------

@eval_cmd.command("stop")
@click.option("--output-dir", "-o", default=None)
@click.option("--job-id", default=None, help="SLURM job ID")
@click.option("--host", default=None, help="SLURM login hostname")
@click.option("--user", default=None, help="SSH username")
def eval_stop(output_dir, job_id, host, user):
    """Stop/cancel a running evaluation."""
    if job_id or output_dir:
        meta = _resolve_or_fail(job_id, host, output_dir or "./eval_results")
        latest_id = _resolve_latest_job_id(meta)

        from nemo_evaluator.eval.ssh import cancel_job

        cancel_job(meta["hostname"], latest_id, meta.get("username") or user)
        click.echo(f"Cancelled SLURM job {latest_id}")
        return

    from nemo_evaluator.executors import detect_executor

    ex = detect_executor("./eval_results")
    if ex is None:
        raise click.ClickException("No evaluation metadata found. Use --job-id or -o.")
    if ex.stop("./eval_results"):
        click.echo("Evaluation stopped.")
    else:
        click.echo("Could not stop evaluation (may already be finished).", err=True)


# ---------------------------------------------------------------------------
# nel eval jobs
# ---------------------------------------------------------------------------

@eval_cmd.command("jobs")
@click.option("--offline", is_flag=True, help="Skip live status checks via SSH")
def eval_jobs(offline):
    """List all tracked SLURM jobs from local store."""
    import json

    from nemo_evaluator.executors.slurm import _jobs_store

    jobs_dir = _jobs_store()
    if not jobs_dir.is_dir():
        click.echo("No tracked jobs.")
        return

    rows = []
    for meta_path in sorted(jobs_dir.glob("*/slurm_job.json")):
        try:
            meta = json.loads(meta_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        rows.append(meta)

    if not rows:
        click.echo("No tracked jobs.")
        return

    rows.sort(key=lambda m: m.get("submitted_at", ""), reverse=True)

    header = f"{'JOB_ID':<12} {'STATE':<12} {'HOST':<30} {'SUBMITTED':<22} {'REMOTE_DIR'}"
    click.echo(header)
    click.echo("-" * len(header))

    for meta in rows:
        state = "—"
        if not offline:
            try:
                from nemo_evaluator.eval.ssh import check_job_status

                latest_id = _resolve_latest_job_id(meta)
                info = check_job_status(
                    meta["hostname"], latest_id, meta.get("username") or None,
                )
                state = info.get("state", "UNKNOWN")
            except Exception:
                state = "SSH_ERR"

        submitted = meta.get("submitted_at", "—")
        if submitted != "—" and len(submitted) > 19:
            submitted = submitted[:19]

        rdir = meta.get("remote_dir", "")
        if len(rdir) > 60:
            rdir = "..." + rdir[-57:]
        click.echo(f"{meta['job_id']:<12} {state:<12} {meta.get('hostname', '?'):<30} {submitted:<22} {rdir}")


# ---------------------------------------------------------------------------
# nel eval logs
# ---------------------------------------------------------------------------

@eval_cmd.command("logs")
@click.option("--output-dir", "-o", default=None)
@click.option("--job-id", default=None, help="SLURM job ID")
@click.option("--host", default=None, help="SLURM login hostname")
@click.option("--follow", "-f", is_flag=True, help="Stream logs (tail -f)")
@click.option("--tail", "-n", "tail_lines", type=int, default=None, help="Last N lines")
def eval_logs(output_dir, job_id, host, follow, tail_lines):
    """View remote SLURM job logs."""
    import subprocess
    import sys

    meta = _resolve_or_fail(job_id, host, output_dir)
    latest_id = _resolve_latest_job_id(meta)
    target_host = meta["hostname"]
    target = f"{meta['username']}@{target_host}" if meta.get("username") else target_host
    rdir = meta["remote_dir"]
    log_file = f"{rdir}/slurm-{latest_id}.log"

    if latest_id != meta["job_id"]:
        click.echo(f"Following chain: {meta['job_id']} → {latest_id}", err=True)

    if follow:
        n_arg = f"-n {tail_lines}" if tail_lines else "-n 50"
        cmd = ["ssh", target, f"tail {n_arg} -f {log_file}"]
        click.echo(f"Streaming: {target}:{log_file}", err=True)
        try:
            subprocess.run(cmd, check=False)
        except KeyboardInterrupt:
            sys.exit(0)
    elif tail_lines:
        from nemo_evaluator.eval.ssh import ssh_run

        out = ssh_run(target_host, f"tail -n {tail_lines} {log_file}",
                       username=meta.get("username") or None, timeout=30.0)
        click.echo(out)
    else:
        from nemo_evaluator.eval.ssh import ssh_run

        out = ssh_run(target_host, f"cat {log_file}",
                       username=meta.get("username") or None, timeout=60.0)
        click.echo(out)


# ---------------------------------------------------------------------------
# nel eval resume
# ---------------------------------------------------------------------------

@eval_cmd.command("resume")
@click.option("--output-dir", "-o", default=None)
@click.option("--job-id", default=None, help="SLURM job ID to resume")
@click.option("--host", default=None, help="SLURM login hostname")
@click.option("--continue-attempts", is_flag=True,
              help="Keep existing attempt counter (default: reset to 0)")
def eval_resume(output_dir, job_id, host, continue_attempts):
    """Manually resubmit a failed/timed-out SLURM evaluation."""
    import shlex

    meta = _resolve_or_fail(job_id, host, output_dir)
    target_host = meta["hostname"]
    rdir = meta["remote_dir"]
    script = f"{rdir}/nel_eval.sbatch"

    from nemo_evaluator.eval.ssh import ssh_run

    if not continue_attempts:
        ssh_run(target_host, f"echo 0 > {shlex.quote(rdir)}/.nel_attempt",
                username=meta.get("username") or None, timeout=10.0)
        click.echo("Reset attempt counter to 0.")

    output = ssh_run(target_host, f"sbatch {shlex.quote(script)}",
                     username=meta.get("username") or None, timeout=30.0)
    click.echo(f"Resubmitted: {output.strip()}")

    import re

    m = re.search(r"(\d+)", output)
    if m:
        new_jid = m.group(1)
        click.echo(f"New job ID: {new_jid}")
        click.echo(f"Tail logs:  nel eval logs --job-id {new_jid} -f")
        click.echo(f"Status:     nel eval status --job-id {new_jid}")


# ---------------------------------------------------------------------------
# nel eval clean
# ---------------------------------------------------------------------------

@eval_cmd.command("clean")
@click.option("--older-than", default=None,
              help="Remove entries older than duration (e.g. 7d, 4w, 24h)")
@click.option("--state", "filter_state", default=None,
              help="Remove only entries with this SLURM state (e.g. COMPLETED, FAILED)")
@click.option("--dry-run", is_flag=True, help="Show what would be removed")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
def eval_clean(older_than, filter_state, dry_run, yes):
    """Clean stale entries from the local SLURM jobs store."""
    import json
    import re
    import shutil
    from datetime import datetime, timedelta, timezone

    from nemo_evaluator.executors.slurm import _jobs_store

    jobs_dir = _jobs_store()
    if not jobs_dir.is_dir():
        click.echo("No tracked jobs to clean.")
        return

    cutoff = None
    if older_than:
        m = re.match(r"^(\d+)\s*([hdwm])$", older_than.strip())
        if not m:
            raise click.ClickException(
                f"Invalid duration: {older_than!r}. Use e.g. 7d, 4w, 24h"
            )
        amount, unit = int(m.group(1)), m.group(2)
        delta = {"h": timedelta(hours=amount), "d": timedelta(days=amount),
                 "w": timedelta(weeks=amount), "m": timedelta(days=amount * 30)}[unit]
        cutoff = datetime.now(timezone.utc) - delta

    to_remove: list[tuple[Path, dict]] = []
    for meta_path in sorted(jobs_dir.glob("*/slurm_job.json")):
        try:
            meta = json.loads(meta_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue

        if cutoff and meta.get("submitted_at"):
            try:
                sub = datetime.fromisoformat(meta["submitted_at"])
                if sub > cutoff:
                    continue
            except ValueError:
                pass

        if filter_state:
            try:
                from nemo_evaluator.eval.ssh import check_job_status

                info = check_job_status(
                    meta["hostname"], meta["job_id"],
                    meta.get("username") or None,
                )
                if info.get("state", "").upper() != filter_state.upper():
                    continue
            except Exception:
                continue

        to_remove.append((meta_path.parent, meta))

    if not to_remove:
        click.echo("Nothing to clean.")
        return

    click.echo(f"Found {len(to_remove)} entries to remove:")
    for dirp, meta in to_remove:
        click.echo(f"  {meta['job_id']}  {meta.get('hostname', '?')}  {meta.get('submitted_at', '?')}")

    if dry_run:
        click.echo("(dry-run — nothing removed)")
        return

    if not yes:
        click.confirm("Remove these entries?", abort=True)

    for dirp, _meta in to_remove:
        shutil.rmtree(dirp, ignore_errors=True)
    click.echo(f"Removed {len(to_remove)} entries.")


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
