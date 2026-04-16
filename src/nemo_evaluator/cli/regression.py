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

import sys

import click

from nemo_evaluator.reports.regression import RENDERERS


def _resolve_bundle(path_str: str) -> str:
    """Accept a directory or a bundle .json file. If directory, find the eval-*.json inside."""
    from pathlib import Path

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
        raise click.BadParameter(f"No eval-*.json bundle found in {p}")
    raise click.BadParameter(f"Path {p} does not exist")


def _exit_strict(strict: bool, verdict: str):
    if not strict:
        return
    if verdict == "BLOCK":
        sys.exit(1)
    if verdict in ("WARN", "INCONCLUSIVE"):
        sys.exit(2)


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
    help="Exit non-zero on BLOCK (exit 1) or WARN/INCONCLUSIVE (exit 2). Default: strict. Use --no-strict for interactive exploration. For multi-benchmark policies, see 'nel gate'.",
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

    BASELINE and CANDIDATE can be eval-*.json bundle files OR directories
    containing one (produced by 'nel eval run').

    \b
    Examples:
      nel compare ./baseline/ ./candidate/
      nel compare ./baseline/ ./candidate/ --show-flips
      nel compare ./baseline/ ./candidate/ --strict --max-drop 0.01
      nel compare ./baseline/eval-base.json ./cand/eval-cand.json
    """
    from pathlib import Path
    from nemo_evaluator.engine.comparison import compare_runs, write_regression

    baseline = _resolve_bundle(baseline)
    candidate = _resolve_bundle(candidate)

    report = compare_runs(
        baseline,
        candidate,
        reward_threshold=reward_threshold,
        min_effect=threshold,
        test=test_method,
    )

    if report_path is None and not no_report:
        report_path = str(Path(candidate).parent / "regression_report.md")

    # Render the chosen format
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
        return

    click.echo(rendered)

    # Sidecar markdown (independent of --format)
    if report_path:
        from nemo_evaluator.reports.regression import write_report

        rp = write_report(report, report_path)
        click.echo(f"Markdown report written to: {rp}")

    _exit_strict(strict, report.get("verdict", "PASS"))
