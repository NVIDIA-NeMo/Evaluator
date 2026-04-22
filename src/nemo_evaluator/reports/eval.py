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
"""Evaluation report renderers: markdown, html, latex, csv, json.

Every renderer follows the signature ``fn(table_dict, **opts) -> str``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def load_bundles(paths: list[Path]) -> list[dict[str, Any]]:
    bundles = []
    for p in paths:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            data["_source"] = str(p)
            bundles.append(data)
        except Exception as e:
            logger.warning("Skipping %s: %s", p, e)
    return bundles


def build_table(bundles: list[dict[str, Any]]) -> dict[str, Any]:
    benchmarks: dict[str, dict[str, Any]] = {}
    model_name = ""

    for b in bundles:
        bm = b.get("benchmark", {})
        name = bm.get("name", "unknown")
        model_name = model_name or b.get("config", {}).get("model", "")

        scores = bm.get("scores", {})
        row: dict[str, Any] = {"samples": bm.get("samples", 0), "repeats": bm.get("repeats", 1)}

        for metric, val in scores.items():
            if isinstance(val, dict) and "value" in val:
                row[metric] = val

        cats = bm.get("categories", {})
        if cats:
            row["categories"] = cats

        rt = scores.get("runtime", {})
        if isinstance(rt, dict):
            row["total_tokens"] = rt.get("total_tokens", 0)
            row["latency_p50_ms"] = rt.get("latency_percentiles_ms", {}).get("p50", 0)

        benchmarks[name] = row

    return {"model": model_name, "benchmarks": benchmarks, "n_benchmarks": len(benchmarks)}


def _primary_metric(row: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Pick the best headline metric: mean_reward if available, else pass@1."""
    mr = row.get("mean_reward", {})
    if isinstance(mr, dict) and "value" in mr:
        return "mean_reward", mr
    p1 = row.get("pass@1", {})
    if isinstance(p1, dict) and "value" in p1:
        return "pass@1", p1
    return "", {}


def render_markdown(table: dict[str, Any], **_) -> str:
    lines = [f"# Evaluation Report: {table['model']}\n"]

    header = "| Benchmark | Scorer | Samples | Score | CI (95%) |"
    sep = "|-----------|--------|---------|-------|----------|"
    lines.extend([header, sep])

    for name, row in sorted(table["benchmarks"].items()):
        samples = row.get("samples", "-")
        scorer_metrics = {
            k: v for k, v in row.items() if k.startswith("scorer:") and isinstance(v, dict) and "value" in v
        }

        metric_name, metric = _primary_metric(row)
        if metric:
            val = f"{metric['value']:.4f}"
            lo = metric.get("ci_lower", "")
            hi = metric.get("ci_upper", "")
            ci = f"[{lo:.4f}, {hi:.4f}]" if lo and hi else ""
            label = f"{metric_name} (native)" if scorer_metrics else metric_name
            lines.append(f"| {name} | {label} | {samples} | {val} | {ci} |")
        elif not scorer_metrics:
            lines.append(f"| {name} | - | {samples} | - | |")

        for sname, sval in sorted(scorer_metrics.items()):
            slabel = sname.removeprefix("scorer:")
            val = f"{sval['value']:.4f}"
            lo = sval.get("ci_lower", "")
            hi = sval.get("ci_upper", "")
            ci = f"[{lo:.4f}, {hi:.4f}]" if lo != "" and hi != "" else ""
            correct = sval.get("correct", "")
            total = sval.get("total", "")
            detail = f" ({correct}/{total})" if correct != "" and total != "" else ""
            lines.append(f"| {name} | {slabel} | {samples} | {val}{detail} | {ci} |")

    lines.append("")

    for name, row in sorted(table["benchmarks"].items()):
        cats = row.get("categories", {})
        if cats:
            lines.append(f"\n## {name} -- Category Breakdown\n")
            lines.append("| Category | Score | N |")
            lines.append("|----------|-------|---|")
            for cat, info in sorted(cats.items()):
                score = info.get("mean_reward", info.get("pass@1", "-"))
                n = info.get("n", info.get("n_samples", "-"))
                lines.append(f"| {cat} | {score} | {n} |")

    return "\n".join(lines)


def render_latex(table: dict[str, Any], **_) -> str:
    lines = [
        "\\begin{table}[h]",
        "\\centering",
        f"\\caption{{Evaluation results: {table['model']}}}",
        "\\begin{tabular}{lrccc}",
        "\\toprule",
        "Benchmark & Samples & Score & Metric & 95\\% CI \\\\",
        "\\midrule",
    ]

    for name, row in sorted(table["benchmarks"].items()):
        metric_name, metric = _primary_metric(row)
        if metric:
            val = f"{metric['value']:.4f}"
            lo = metric.get("ci_lower", "")
            hi = metric.get("ci_upper", "")
            ci = f"[{lo:.4f}, {hi:.4f}]" if lo and hi else ""
        else:
            val = "--"
            ci = ""
        lines.append(f"{name} & {row.get('samples', '--')} & {val} & {metric_name} & {ci} \\\\")

    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    return "\n".join(lines)


def render_csv(table: dict[str, Any], **_) -> str:
    lines = ["benchmark,samples,repeats,metric,score,ci_lower,ci_upper"]
    for name, row in sorted(table["benchmarks"].items()):
        metric_name, metric = _primary_metric(row)
        val = metric.get("value", "") if metric else ""
        lo = metric.get("ci_lower", "") if metric else ""
        hi = metric.get("ci_upper", "") if metric else ""
        lines.append(f"{name},{row.get('samples', '')},{row.get('repeats', '')},{metric_name},{val},{lo},{hi}")
    return "\n".join(lines)


def render_html(table: dict[str, Any], **_) -> str:
    from datetime import datetime, timezone

    model = table.get("model", "Unknown")
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    n = table.get("n_benchmarks", 0)

    rows_html = []
    for name, row in sorted(table["benchmarks"].items()):
        scorer_metrics = {
            k: v for k, v in row.items() if k.startswith("scorer:") and isinstance(v, dict) and "value" in v
        }

        metric_name, metric = _primary_metric(row)
        tokens = row.get("total_tokens", "")
        latency = row.get("latency_p50_ms", "")
        if latency:
            latency = f"{latency:.0f}"

        if metric:
            val = f"{metric['value']:.4f}"
            lo = metric.get("ci_lower", "")
            hi = metric.get("ci_upper", "")
            ci = f"[{lo:.4f}, {hi:.4f}]" if lo and hi else "&mdash;"
            label = f"{metric_name} (native)" if scorer_metrics else metric_name
        else:
            val = "&mdash;"
            ci = "&mdash;"
            label = "&mdash;"

        if metric or not scorer_metrics:
            rows_html.append(
                f"<tr><td>{name}</td><td>{label}</td><td>{row.get('samples', '')}</td>"
                f"<td><strong>{val}</strong></td>"
                f"<td>{ci}</td><td>{tokens}</td><td>{latency}</td></tr>"
            )

        for sname, sval in sorted(scorer_metrics.items()):
            slabel = sname.removeprefix("scorer:")
            sval_str = f"{sval['value']:.4f}"
            c = sval.get("correct", "")
            t = sval.get("total", "")
            detail = f" ({c}/{t})" if c != "" and t != "" else ""
            lo = sval.get("ci_lower", "")
            hi = sval.get("ci_upper", "")
            sci = f"[{lo:.4f}, {hi:.4f}]" if lo != "" and hi != "" else "&mdash;"
            rows_html.append(
                f"<tr><td>{name}</td><td>{slabel}</td><td>{row.get('samples', '')}</td>"
                f"<td><strong>{sval_str}{detail}</strong></td>"
                f"<td>{sci}</td><td></td><td></td></tr>"
            )

    extra_sections = []
    for name, row in sorted(table["benchmarks"].items()):
        cats = row.get("categories", {})
        if cats:
            cat_rows = []
            for cat, info in sorted(cats.items()):
                score = info.get("mean_reward", info.get("pass@1", "&mdash;"))
                n_samples = info.get("n", info.get("n_samples", "&mdash;"))
                cat_rows.append(f"<tr><td>{cat}</td><td>{score}</td><td>{n_samples}</td></tr>")
            extra_sections.append(
                f"<h2>{name} &mdash; Category Breakdown</h2>"
                f"<table><thead><tr><th>Category</th><th>Score</th><th>N</th></tr></thead>"
                f"<tbody>{''.join(cat_rows)}</tbody></table>"
            )

    return f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Evaluation Report: {model}</title>
<style>
  :root {{ --bg: #ffffff; --fg: #1a1a2e; --accent: #0f3460; --border: #e0e0e0;
           --row-alt: #f8f9fa; --header-bg: #0f3460; --header-fg: #ffffff; }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --bg: #1a1a2e; --fg: #e0e0e0; --accent: #4e9af1; --border: #2d2d44;
             --row-alt: #16213e; --header-bg: #16213e; --header-fg: #e0e0e0; }}
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
          background: var(--bg); color: var(--fg); max-width: 960px; margin: 0 auto;
          padding: 2rem 1rem; line-height: 1.6; }}
  h1 {{ font-size: 1.5rem; margin-bottom: 0.25rem; }}
  .meta {{ color: #888; font-size: 0.875rem; margin-bottom: 1.5rem; }}
  h2 {{ font-size: 1.15rem; margin: 2rem 0 0.75rem; }}
  table {{ width: 100%; border-collapse: collapse; margin-bottom: 1.5rem; font-size: 0.9rem; }}
  th {{ background: var(--header-bg); color: var(--header-fg); text-align: left;
        padding: 0.6rem 0.75rem; font-weight: 600; }}
  td {{ padding: 0.5rem 0.75rem; border-bottom: 1px solid var(--border); }}
  tr:nth-child(even) {{ background: var(--row-alt); }}
  strong {{ color: var(--accent); }}
  footer {{ margin-top: 3rem; font-size: 0.8rem; color: #888; border-top: 1px solid var(--border);
            padding-top: 1rem; }}
</style>
</head>
<body>
<h1>Evaluation Report: {model}</h1>
<p class="meta">{n} benchmark(s) &middot; {ts}</p>
<table>
<thead>
<tr><th>Benchmark</th><th>Scorer</th><th>Samples</th><th>Score</th><th>95% CI</th><th>Tokens</th><th>P50 ms</th></tr>
</thead>
<tbody>
{"".join(rows_html)}
</tbody>
</table>
{"".join(extra_sections)}
<footer>Generated by NeMo Evaluator</footer>
</body>
</html>"""


def render_json(table: dict[str, Any], **_) -> str:
    return json.dumps(table, indent=2, default=str)


# ── Registry ──────────────────────────────────────────────────────────

RENDERERS = {
    "markdown": render_markdown,
    "html": render_html,
    "latex": render_latex,
    "csv": render_csv,
    "json": render_json,
}
