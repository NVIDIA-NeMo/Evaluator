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


@click.command("regression")
@click.argument("baseline", type=click.Path(exists=True))
@click.argument("candidate", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), default=None, help="Write report JSON to file")
@click.option("--threshold", "-t", type=float, default=0.05, help="Fail if any metric drops more than this")
@click.option("--strict", is_flag=True, help="Exit non-zero on regression beyond threshold")
def regression_cmd(baseline, candidate, output, threshold, strict):
    """Compare two evaluation bundles and report regressions.

    BASELINE and CANDIDATE are paths to eval-*.json bundle files.
    """
    from nemo_evaluator.engine.comparison import compare_runs, write_regression

    report = compare_runs(baseline, candidate)

    click.echo(f"Baseline:  {report['baseline']['run_id']}")
    click.echo(f"Candidate: {report['candidate']['run_id']}")
    click.echo()

    regressions = []
    for metric, delta in report.get("score_deltas", {}).items():
        d = delta["delta"]
        marker = "  "
        if d < -threshold:
            marker = "! "
            regressions.append(metric)
        elif d > threshold:
            marker = "+ "

        overlap = "overlap" if delta.get("ci_overlap") else "NO overlap"
        click.echo(
            f"{marker}{metric}: {delta['baseline']:.4f} -> {delta['candidate']:.4f}  "
            f"(delta={d:+.4f}, {delta['relative_pct']:+.1f}%, CI {overlap})"
        )

    rt = report.get("runtime_deltas", {})
    if rt:
        click.echo()
        for k, v in rt.items():
            click.echo(f"  {k}: {v['baseline']} -> {v['candidate']}  (delta={v['delta']:+.2f})")

    cats = report.get("category_deltas", {})
    if cats:
        click.echo()
        click.echo("Per-category deltas:")
        for cat, v in sorted(cats.items()):
            click.echo(f"  {cat}: {v['baseline']:.4f} -> {v['candidate']:.4f}  (delta={v['delta']:+.4f})")

    if output:
        write_regression(report, output)
        click.echo(f"\nReport written to: {output}")

    if regressions:
        click.echo(f"\nREGRESSIONS detected (>{threshold:.0%} drop): {', '.join(regressions)}")
        if strict:
            sys.exit(1)
    else:
        click.echo(f"\nNo regressions beyond {threshold:.0%} threshold.")
