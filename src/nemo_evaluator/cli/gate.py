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
from pathlib import Path

import click

from nemo_evaluator.config import load_gate_policy
from nemo_evaluator.engine import gate_runs, write_gate_report
from nemo_evaluator.reports.gate import RENDERERS


def _resolve_results_dir(path_str: str) -> Path:
    """Accept a results directory or a single eval bundle path."""
    p = Path(path_str)
    if p.is_dir():
        return p
    if p.is_file() and p.name.startswith("eval-") and p.suffix == ".json":
        return p.parent
    raise click.BadParameter(f"{p} must be a results directory or an eval-*.json bundle path")


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
    type=click.Choice(sorted(RENDERERS)),
    default="text",
    help="Output format.",
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
    except (FileNotFoundError, ValueError, TypeError) as exc:
        raise click.ClickException(str(exc)) from exc

    if output:
        write_gate_report(report, output)

    report_dict = report.to_dict()

    opts = {
        "verbose": verbose,
        "policy_path": str(policy_path),
        "baseline_dir": str(baseline_dir),
        "candidate_dir": str(candidate_dir),
        "output": output or "",
    }
    rendered = RENDERERS[fmt](report_dict, **opts)
    click.echo(rendered)

    if fmt == "json":
        _exit_strict(strict, report.verdict)
        return

    # Sidecar markdown
    if not no_report:
        from nemo_evaluator.reports.gate import write_gate_markdown

        if report_path:
            md_path = write_gate_markdown(report_dict, report_path)
            click.echo(f"\nMarkdown damage report written to: {md_path}")
        elif output:
            md_path = write_gate_markdown(report_dict, Path(output).with_suffix(".md"))
            click.echo(f"Markdown damage report written to: {md_path}")

    _exit_strict(strict, report.verdict)


def _exit_strict(strict: bool, verdict: str) -> None:
    if not strict:
        return
    if verdict == "NO-GO":
        sys.exit(1)
    if verdict == "INCONCLUSIVE":
        sys.exit(2)
