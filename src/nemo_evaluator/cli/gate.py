from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import click

from nemo_evaluator.config import load_gate_policy
from nemo_evaluator.engine import gate_runs, write_gate_report

_NO_COLOR = os.environ.get("NO_COLOR") is not None


def _style(text: str, **kwargs) -> str:
    if _NO_COLOR:
        return text
    return click.style(text, **kwargs)


def _verdict_color(verdict: str) -> str:
    if verdict == "NO-GO":
        return "red"
    if verdict == "INCONCLUSIVE":
        return "yellow"
    return "green"


def _resolve_results_dir(path_str: str) -> Path:
    """Accept a results directory or a single eval bundle path."""
    p = Path(path_str)
    if p.is_dir():
        return p
    if p.is_file() and p.name.startswith("eval-") and p.suffix == ".json":
        return p.parent
    raise click.BadParameter(
        f"{p} must be a results directory or an eval-*.json bundle path"
    )


@click.command("gate")
@click.argument("baseline", type=click.Path(exists=True))
@click.argument("candidate", type=click.Path(exists=True))
@click.option(
    "--policy",
    "-p",
    "policy_path",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Gate policy YAML file.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default=None,
    help="Write JSON gate report to file.",
)
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["text", "json"]),
    default="text",
    help="Output format. 'json' writes structured JSON to stdout.",
)
@click.option(
    "--strict/--no-strict",
    default=True,
    help="Exit non-zero on NO-GO (exit 1) or INCONCLUSIVE (exit 2). Default: strict. Use --no-strict for interactive exploration.",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Show per-benchmark reasons and warnings.",
)
@click.option(
    "--report",
    "report_path",
    type=click.Path(),
    default=None,
    help="Write Markdown damage report to this path.",
)
@click.option(
    "--no-report",
    is_flag=True,
    help="Suppress auto-generated Markdown damage report.",
)
def gate_cmd(
    baseline: str,
    candidate: str,
    policy_path: str,
    output: str | None,
    fmt: str,
    strict: bool,
    verbose: bool,
    report_path: str | None,
    no_report: bool,
) -> None:
    """Apply a multi-benchmark release policy to baseline and candidate results.

    Verdicts: GO (all pass), NO-GO (any breach/missing), INCONCLUSIVE (insufficient evidence).
    To investigate a specific benchmark, use 'nel compare'.
    """
    try:
        baseline_dir = _resolve_results_dir(baseline)
        candidate_dir = _resolve_results_dir(candidate)
        policy = load_gate_policy(policy_path)
        report = gate_runs(baseline_dir, candidate_dir, policy)
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

    if output:
        write_gate_report(report, output)

    if fmt == "json":
        click.echo(json.dumps(report.to_dict(), indent=2, default=str))
        _exit_strict(strict, report.verdict)
        return

    _render_text(report, baseline_dir, candidate_dir, Path(policy_path), verbose, output)

    # Markdown damage report
    if not no_report:
        from nemo_evaluator.engine.gate_report import write_gate_report as write_md_report

        if report_path:
            md_path = write_md_report(report.to_dict(), report_path)
            click.echo(f"\nMarkdown damage report written to: {md_path}")
        elif output:
            # Auto-generate next to JSON output
            md_path = write_md_report(report.to_dict(), Path(output).with_suffix(".md"))
            click.echo(f"Markdown damage report written to: {md_path}")

    _exit_strict(strict, report.verdict)


def _render_text(
    report,
    baseline_dir: Path,
    candidate_dir: Path,
    policy_path: Path,
    verbose: bool,
    output: str | None,
) -> None:
    color = _verdict_color(report.verdict)

    click.echo()
    click.echo(_style(report.verdict, fg=color, bold=True))
    click.echo(f"Policy:    {policy_path}")
    click.echo(f"Baseline:  {baseline_dir}")
    click.echo(f"Candidate: {candidate_dir}")

    if report.verdict_reasons:
        click.echo()
        click.echo("VERDICT REASONS")
        for reason in report.verdict_reasons:
            click.echo(f"  - {reason}")

    # Always show missing benchmarks (not gated by --verbose)
    if report.missing:
        click.echo()
        click.echo(_style("MISSING BENCHMARKS", fg="red", bold=True))
        for name in report.missing:
            click.echo(f"  - {name}")

    if report.warnings and verbose:
        click.echo()
        click.echo(_style("WARNINGS", fg="yellow", bold=True))
        for warning in report.warnings:
            click.echo(f"  - {warning}")

    if report.benchmarks:
        click.echo()
        click.echo("BENCHMARKS")
        click.echo("  name                 tier         status                 metric       delta")
        click.echo("  -------------------------------------------------------------------------------")
        for result in sorted(report.benchmarks, key=lambda item: item.benchmark):
            delta = f"{result.delta * 100:+.1f}pp" if result.delta is not None else "n/a"
            metric = result.metric or "n/a"
            click.echo(
                f"  {result.benchmark:<20} {result.tier:<12} {result.status:<22} "
                f"{metric:<12} {delta}"
            )
            if verbose or result.status != "PASS":
                for reason in result.reasons:
                    click.echo(f"    - {reason}")

    if output:
        click.echo()
        click.echo(f"JSON report written to: {output}")


def _exit_strict(strict: bool, verdict: str) -> None:
    if not strict:
        return
    if verdict == "NO-GO":
        sys.exit(1)
    if verdict == "INCONCLUSIVE":
        sys.exit(2)
