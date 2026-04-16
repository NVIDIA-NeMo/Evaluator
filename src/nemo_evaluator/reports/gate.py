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
"""Gate report renderers: text (terminal), markdown, and JSON.

Every renderer follows the signature ``fn(report_dict, **opts) -> str``.

Text-renderer opts
------------------
- ``verbose``        (bool) — show per-benchmark reasons and warnings
- ``policy_path``    (str)  — display path in header
- ``baseline_dir``   (str)  — display path in header
- ``candidate_dir``  (str)  — display path in header
- ``output``         (str)  — display "JSON written to" message
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nemo_evaluator.reports._formatting import style, verdict_color


# ── Text renderer ─────────────────────────────────────────────────────


def render_text(report_dict: dict[str, Any], **opts) -> str:
    """Render a gate report as styled terminal text."""
    verbose = opts.get("verbose", False)
    policy_path = opts.get("policy_path", "")
    baseline_dir = opts.get("baseline_dir", "")
    candidate_dir = opts.get("candidate_dir", "")
    output = opts.get("output", "")

    verdict = report_dict.get("verdict", "ERROR")
    verdict_reasons = report_dict.get("verdict_reasons", [])
    missing = report_dict.get("missing", [])
    warnings = report_dict.get("warnings", [])
    benchmarks = report_dict.get("benchmarks", [])
    color = verdict_color(verdict)

    lines: list[str] = []

    lines.append("")
    lines.append(style(verdict, fg=color, bold=True))
    lines.append(f"Policy:    {policy_path}")
    lines.append(f"Baseline:  {baseline_dir}")
    lines.append(f"Candidate: {candidate_dir}")

    if verdict_reasons:
        lines.append("")
        lines.append("VERDICT REASONS")
        for reason in verdict_reasons:
            lines.append(f"  - {reason}")

    if missing:
        lines.append("")
        lines.append(style("MISSING BENCHMARKS", fg="red", bold=True))
        for name in missing:
            lines.append(f"  - {name}")

    if warnings and verbose:
        lines.append("")
        lines.append(style("WARNINGS", fg="yellow", bold=True))
        for warning in warnings:
            lines.append(f"  - {warning}")

    if benchmarks:
        lines.append("")
        lines.append("BENCHMARKS")
        lines.append("  name                 tier         status                 metric       delta")
        lines.append("  -------------------------------------------------------------------------------")
        for result in sorted(benchmarks, key=lambda item: item.get("benchmark", "")):
            delta = f"{result['delta'] * 100:+.1f}pp" if result.get("delta") is not None else "n/a"
            metric = result.get("metric") or "n/a"
            bm = result.get("benchmark", "?")
            tier = result.get("tier", "?")
            status = result.get("status", "?")
            lines.append(f"  {bm:<20} {tier:<12} {status:<22} {metric:<12} {delta}")
            reasons = result.get("reasons", [])
            if verbose or status != "PASS":
                for reason in reasons:
                    lines.append(f"    - {reason}")

    if output:
        lines.append("")
        lines.append(f"JSON report written to: {output}")

    return "\n".join(lines)


# ── Markdown renderer ─────────────────────────────────────────────────


def render_markdown(report_dict: dict[str, Any], **_) -> str:
    """Generate a Markdown damage report from a GateReport dict."""
    lines: list[str] = []
    _w = lines.append

    verdict = report_dict.get("verdict", "ERROR")
    reasons = report_dict.get("verdict_reasons", [])
    warnings = report_dict.get("warnings", [])
    benchmarks = report_dict.get("benchmarks", [])
    missing = report_dict.get("missing", [])

    _w(f"# Quality Gate Report: {verdict}")
    _w("")
    _w(f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    _w("")

    _w("## Verdict")
    _w("")
    _w(f"**{verdict}**")
    _w("")
    if reasons:
        for r in reasons:
            _w(f"- {r}")
        _w("")

    if warnings:
        _w("## Warnings")
        _w("")
        for w in warnings:
            _w(f"- {w}")
        _w("")

    if benchmarks:
        _w("## Per-Benchmark Results")
        _w("")
        _w("| Benchmark | Tier | Status | Metric | Baseline | Candidate | Delta (pp) | 95% CI | N |")
        _w("|---|---|---|---|---|---|---|---|---|")

        for b in sorted(benchmarks, key=lambda x: x.get("benchmark", "")):
            name = b.get("benchmark", "?")
            tier = b.get("tier", "?")
            status = b.get("status", "?")
            metric = b.get("metric", "") or ""
            base_s = b.get("baseline_score")
            cand_s = b.get("candidate_score")
            delta = b.get("delta")
            ci_lo = b.get("delta_ci_lower")
            ci_hi = b.get("delta_ci_upper")
            n = b.get("n_paired", 0)

            base_str = f"{base_s:.4f}" if base_s is not None else "—"
            cand_str = f"{cand_s:.4f}" if cand_s is not None else "—"
            delta_str = f"{delta * 100:+.1f}" if delta is not None else "—"
            ci_str = f"[{ci_lo:.4f}, {ci_hi:.4f}]" if ci_lo is not None and ci_hi is not None else "—"
            n_str = f"{n:,}" if n > 0 else "—"

            _w(
                f"| {name} | {tier} | {status} | {metric} | {base_str} | {cand_str} | {delta_str} | {ci_str} | {n_str} |"
            )

        _w("")

    non_pass = [b for b in benchmarks if b.get("status") != "PASS"]
    if non_pass:
        _w("## Details")
        _w("")
        for b in non_pass:
            name = b.get("benchmark", "?")
            status = b.get("status", "?")
            b_reasons = b.get("reasons", [])
            rel_drop = b.get("relative_drop_pct")

            _w(f"### {name} — {status}")
            _w("")
            if b_reasons:
                for r in b_reasons:
                    _w(f"- {r}")
            if rel_drop is not None:
                _w(f"- Relative drop: {rel_drop:+.1f}%")
            _w("")

    if missing:
        _w("## Missing Benchmarks")
        _w("")
        for m in missing:
            _w(f"- `{m}`")
        _w("")

    _w("---")
    _w("*Report generated by NeMo Evaluator `nel gate`*")

    return "\n".join(lines)


# ── JSON renderer ─────────────────────────────────────────────────────


def render_json(report_dict: dict[str, Any], **_) -> str:
    return json.dumps(report_dict, indent=2, default=str)


# ── Convenience I/O ───────────────────────────────────────────────────


def write_gate_markdown(report_dict: dict[str, Any], output_path: str | Path) -> Path:
    """Write a Markdown damage report to disk."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(report_dict), encoding="utf-8")
    return path


# ── Registry ──────────────────────────────────────────────────────────

RENDERERS = {
    "text": render_text,
    "markdown": render_markdown,
    "json": render_json,
}
