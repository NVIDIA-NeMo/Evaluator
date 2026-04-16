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
"""nel report -- aggregate multiple evaluation bundles into comparison tables."""

from __future__ import annotations

from pathlib import Path

import click

from nemo_evaluator.reports.eval import RENDERERS, build_table, load_bundles


@click.command("report")
@click.argument("bundle_paths", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--format", "-f", "fmt", type=click.Choice(sorted(RENDERERS)), default="markdown")
@click.option("--output", "-o", type=click.Path(), default=None)
def report_cmd(bundle_paths, fmt, output):
    """Generate a multi-benchmark evaluation report from bundle JSON files."""
    paths = [Path(p) for p in bundle_paths]
    expanded: list[Path] = []
    for p in paths:
        if p.is_dir():
            top = sorted(p.glob("eval-*.json"))
            if top:
                expanded.extend(top)
            else:
                expanded.extend(sorted(p.glob("*/eval-*.json")))
        else:
            expanded.append(p)

    if not expanded:
        raise click.ClickException(
            "No eval-*.json bundle files found. Pass benchmark directories or bundle files directly."
        )

    bundles = load_bundles(expanded)
    if not bundles:
        raise click.ClickException("No valid bundles loaded.")

    table = build_table(bundles)
    rendered = RENDERERS[fmt](table)

    if output:
        Path(output).write_text(rendered, encoding="utf-8")
        click.echo(f"Report written to {output}")
    else:
        click.echo(rendered)
