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
from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from nemo_evaluator.reports.regression import RENDERERS


# ── Verdict severity for multi-benchmark worst-case ──────────────────

_VERDICT_SEVERITY = {"PASS": 0, "INCONCLUSIVE": 1, "WARN": 2, "BLOCK": 3}


# ── Bundle resolution helpers ────────────────────────────────────────


def _resolve_single_bundle(path_str: str) -> str | None:
    """Try to resolve *path_str* to exactly one eval-*.json bundle.

    Returns the bundle path as a string, or ``None`` when the path is a
    directory containing multiple benchmark subdirs (multi-benchmark mode).
    """
    p = Path(path_str)
    if p.is_file():
        return str(p)
    if p.is_dir():
        bundles = sorted(p.glob("eval-*.json"))
        if len(bundles) == 1:
            return str(bundles[0])
        if len(bundles) > 1:
            raise click.BadParameter(
                f"Directory {p} contains {len(bundles)} eval-*.json bundles. "
                f"Specify one: {', '.join(b.name for b in bundles[:3])}"
            )
        nested = sorted(p.glob("*/eval-*.json"))
        if len(nested) == 1:
            return str(nested[0])
        if len(nested) > 1:
            return None  # multi-benchmark directory
        raise click.BadParameter(f"No eval-*.json bundle found in {p} or its subdirectories")
    raise click.BadParameter(f"Path {p} does not exist")


def _exit_strict(strict: bool, verdict: str):
    if not strict:
        return
    if verdict == "BLOCK":
        sys.exit(1)
    if verdict in ("WARN", "INCONCLUSIVE"):
        sys.exit(2)


# ── Single-benchmark comparison ──────────────────────────────────────


def _compare_single(
    baseline_path: str,
    candidate_path: str,
    *,
    threshold: float,
    reward_threshold: float,
    test_method: str,
    fmt: str,
    output: str | None,
    report_path: str | None,
    no_report: bool,
    compact: bool,
    verbose: bool,
    show_flips: bool,
) -> str:
    """Run comparison for a single benchmark pair. Returns the verdict."""
    from nemo_evaluator.engine.comparison import compare_runs, write_regression

    report = compare_runs(
        baseline_path,
        candidate_path,
        reward_threshold=reward_threshold,
        min_effect=threshold,
        test=test_method,
    )

    if report_path is None and not no_report:
        report_path = str(Path(candidate_path).parent / "regression_report.md")

    opts = {"compact": compact, "verbose": verbose, "show_flips": show_flips}
    rendered = RENDERERS[fmt](report, **opts)

    if output:
        write_regression(report, output)
        click.echo(f"\nJSON report written to: {output}")

    if fmt == "json":
        click.echo(rendered)
        if report_path:
            from nemo_evaluator.reports.regression import write_report

            write_report(report, report_path)
            click.echo(f"Report written to: {report_path}", err=True)
        return report.get("verdict", "PASS")

    click.echo(rendered)

    if report_path:
        from nemo_evaluator.reports.regression import write_report

        rp = write_report(report, report_path)
        click.echo(f"Markdown report written to: {rp}")

    return report.get("verdict", "PASS")


# ── Multi-benchmark comparison ───────────────────────────────────────


def _compare_multi(
    baseline_dir: str,
    candidate_dir: str,
    *,
    threshold: float,
    reward_threshold: float,
    test_method: str,
    fmt: str,
    output: str | None,
    no_report: bool,
    compact: bool,
    verbose: bool,
    show_flips: bool,
) -> str:
    """Run comparison for every matched benchmark pair. Returns worst verdict."""
    from nemo_evaluator.engine.bundles import discover_bundles, match_bundles
    from nemo_evaluator.engine.comparison import compare_runs
    from nemo_evaluator.reports.regression import write_report

    base_bundles = discover_bundles(Path(baseline_dir))
    cand_bundles = discover_bundles(Path(candidate_dir))

    if not base_bundles:
        raise click.ClickException(f"No eval bundles found in baseline directory: {baseline_dir}")
    if not cand_bundles:
        raise click.ClickException(f"No eval bundles found in candidate directory: {candidate_dir}")

    matched, base_only, cand_only = match_bundles(base_bundles, cand_bundles)

    if base_only:
        click.echo(
            click.style(
                f"Warning: baseline-only benchmarks (no candidate): {', '.join(sorted(base_only))}", fg="yellow"
            ),
            err=True,
        )
    if cand_only:
        click.echo(
            click.style(
                f"Warning: candidate-only benchmarks (no baseline): {', '.join(sorted(cand_only))}", fg="yellow"
            ),
            err=True,
        )
    if not matched:
        raise click.ClickException(
            "No matching benchmarks between baseline and candidate. "
            f"Baseline has: {', '.join(sorted(base_bundles))}. "
            f"Candidate has: {', '.join(sorted(cand_bundles))}."
        )

    opts = {"compact": compact, "verbose": verbose, "show_flips": show_flips}
    worst_verdict = "PASS"
    all_reports: dict[str, dict] = {}

    for bench_name in sorted(matched):
        base_path, cand_path = matched[bench_name]

        report = compare_runs(
            str(base_path),
            str(cand_path),
            reward_threshold=reward_threshold,
            min_effect=threshold,
            test=test_method,
        )

        all_reports[bench_name] = report
        verdict = report.get("verdict", "PASS")
        if _VERDICT_SEVERITY.get(verdict, 0) > _VERDICT_SEVERITY.get(worst_verdict, 0):
            worst_verdict = verdict

        if fmt != "json":
            header = f"{'─' * 20}  {bench_name}  {'─' * 20}"
            click.echo(click.style(header, bold=True))
            click.echo(RENDERERS[fmt](report, **opts))
            click.echo()

        if not no_report:
            md_path = Path(cand_path).parent / "regression_report.md"
            rp = write_report(report, str(md_path))
            if fmt != "json":
                click.echo(f"  Markdown report: {rp}")

    if fmt == "json":
        click.echo(json.dumps(all_reports, indent=2, default=str))

    if output:
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(all_reports, indent=2, default=str), encoding="utf-8")
        click.echo(f"\nJSON report written to: {output}")

    return worst_verdict


# ── CLI command ──────────────────────────────────────────────────────


@click.command("compare")
@click.argument("baseline", type=click.Path(exists=True))
@click.argument("candidate", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), default=None, help="Write full report JSON to file.")
@click.option(
    "--max-drop",
    "-t",
    "threshold",
    type=float,
    default=0.05,
    help="Maximum allowed absolute drop in any metric (0-1 scale, default: 0.05 = 5%%).",
)
@click.option(
    "--strict/--no-strict",
    default=True,
    help="Exit non-zero on BLOCK (exit 1) or WARN/INCONCLUSIVE (exit 2). Default: strict.",
)
@click.option(
    "--correct-above",
    "reward_threshold",
    type=float,
    default=0.0,
    help="Reward above this counts as 'correct' for flip analysis. Use 0.5 for judge scores. Default: >0.",
)
@click.option("--show-flips", is_flag=True, help="Show per-sample flip list with details.")
@click.option("--compact", is_flag=True, help="Short output for Slack / CI logs (~8 lines).")
@click.option(
    "--format",
    "fmt",
    type=click.Choice(sorted(RENDERERS)),
    default="text",
    help="Output format.",
)
@click.option("--verbose", is_flag=True, help="Show statistical details (p-values, methods, power).")
@click.option(
    "--report",
    "report_path",
    type=click.Path(),
    default=None,
    help="Write Markdown report to this path (default: auto-generated next to candidate bundle).",
)
@click.option("--no-report", is_flag=True, help="Suppress auto-generated Markdown report.")
@click.option(
    "--test",
    "test_method",
    type=click.Choice(["auto", "mcnemar", "sign", "permutation"]),
    default="auto",
    help="Statistical test: auto (detect from data), mcnemar (N=1 binary), sign (N>1 averaged), permutation (continuous).",
)
def compare_cmd(
    baseline,
    candidate,
    output,
    threshold,
    strict,
    reward_threshold,
    show_flips,
    compact,
    fmt,
    verbose,
    report_path,
    no_report,
    test_method,
):
    """Compare two evaluation runs and report regressions.

    BASELINE and CANDIDATE can be eval-*.json bundle files, single-benchmark
    directories, or multi-benchmark run directories.  When both sides contain
    multiple benchmarks, each matched pair is compared and results are printed
    with per-benchmark headers.

    \b
    Examples:
      nel compare ./baseline/ ./candidate/
      nel compare ./baseline/ ./candidate/ --show-flips
      nel compare ./baseline/ ./candidate/ --strict --max-drop 0.01
      nel compare ./baseline/eval-base.json ./cand/eval-cand.json
    """
    base_single = _resolve_single_bundle(baseline)
    cand_single = _resolve_single_bundle(candidate)

    if base_single is not None and cand_single is not None:
        verdict = _compare_single(
            base_single,
            cand_single,
            threshold=threshold,
            reward_threshold=reward_threshold,
            test_method=test_method,
            fmt=fmt,
            output=output,
            report_path=report_path,
            no_report=no_report,
            compact=compact,
            verbose=verbose,
            show_flips=show_flips,
        )
    else:
        if report_path:
            click.echo(
                click.style(
                    "Warning: --report is ignored in multi-benchmark mode "
                    "(each benchmark gets its own regression_report.md).",
                    fg="yellow",
                ),
                err=True,
            )
        verdict = _compare_multi(
            baseline,
            candidate,
            threshold=threshold,
            reward_threshold=reward_threshold,
            test_method=test_method,
            fmt=fmt,
            output=output,
            no_report=no_report,
            compact=compact,
            verbose=verbose,
            show_flips=show_flips,
        )

    _exit_strict(strict, verdict)
