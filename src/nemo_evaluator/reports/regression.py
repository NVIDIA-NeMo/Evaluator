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
"""Regression report renderers: text (terminal), markdown, and JSON.

Every renderer follows the signature ``fn(report_dict, **opts) -> str``.

Text-renderer opts
------------------
- ``compact``    (bool) — short output for Slack / CI
- ``verbose``    (bool) — show statistical details
- ``show_flips`` (bool) — show per-sample flip list
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nemo_evaluator.metrics.paired_tests import mde_estimate
from nemo_evaluator.reports._formatting import bar, style, verdict_color

_TEST_KEY_MAP = {"mcnemar": "mcnemar", "sign": "sign_test", "permutation": "permutation_test"}


# ── Helpers (used by text renderer) ───────────────────────────────────


def _test_significant(report: dict) -> bool | None:
    """Extract significance from whichever statistical test was used."""
    test = report.get("test_used")
    key = _TEST_KEY_MAP.get(test, test) if test else None
    result = report.get(key) if key else None
    return result.get("significant") if isinstance(result, dict) else None


def _cap_verdict(delta: float, threshold: float, sig: bool | None) -> tuple[str, str]:
    if delta < -threshold and sig:
        return "BROKE \u26a0", "red"
    if delta < -threshold:
        return "WARN", "yellow"
    return "HELD", "green"


def _plain_english_verdict(verdict: str, s: dict, mcnemar: dict | None) -> str:
    nr = s.get("n_regressions", 0)
    ni = s.get("n_improvements", 0)
    np_ = s.get("n_paired", 0)
    nd = mcnemar.get("n_discordant", 0) if mcnemar else 0

    if verdict == "BLOCK":
        return f"{nr} problems regressed vs {ni} improved — statistically significant, exceeds practical threshold"
    if verdict == "WARN":
        return f"{nr} problems regressed vs {ni} improved — statistically significant, but within practical threshold"
    if verdict == "INCONCLUSIVE":
        return f"only {nd} flips in {np_} samples — not enough data to rule out small regressions"
    if nr == 0:
        return f"no flips detected in {np_} paired samples"
    return f"{nr} flip(s) out of {np_} paired samples — within normal variation"


def _broke_categories(report: dict, threshold: float, sig: bool | None) -> list[str]:
    cats = report.get("category_deltas", {})
    return [cat for cat, v in sorted(cats.items()) if v["delta"] < -threshold and sig]


def build_summary_sentence(
    summary: dict[str, Any],
    category_deltas: dict[str, Any],
    threshold: float = 0.05,
) -> str | None:
    """One-sentence narrative for Slack copy-paste / executive summary."""
    n_paired = summary.get("n_paired")
    if not n_paired:
        return None

    n_reg = summary.get("n_regressions", 0)

    if not category_deltas:
        if n_reg == 0:
            return "No regressions detected across all problems."
        return None

    held = [c for c, v in sorted(category_deltas.items()) if v["delta"] >= -threshold]
    broke = [c for c, v in sorted(category_deltas.items()) if v["delta"] < -threshold]

    if not broke and n_reg == 0:
        return f"All capabilities held ({', '.join(held)})."

    if not broke and n_reg > 0:
        return f"All capabilities held. {n_reg} problem(s) flipped but within normal variation for {n_paired} samples."

    parts = []
    if held:
        parts.append(f"Safe for {', '.join(held)}.")

    cat_bd = summary.get("category_breakdown", {})
    for cat in broke:
        delta_pct = abs(category_deltas[cat]["delta"]) * 100
        cat_reg = cat_bd.get(cat, {}).get("regressions", 0)
        parts.append(f"{cat} regresses {delta_pct:.1f}% ({cat_reg} problems).")

    return " ".join(parts) if parts else None


# ── Text renderer ─────────────────────────────────────────────────────


def _render_flip_entry(lines: list[str], f_entry: dict, color: str, verbose: bool) -> None:
    cat = f"  [{f_entry.get('category')}]" if f_entry.get("category") else ""
    exp = f_entry.get("expected_answer") or ""
    if len(exp) > 30:
        exp = exp[:27] + "..."

    lines.append(
        style(
            f"  #{f_entry['problem_idx']:<5}  "
            f"{f_entry['baseline_reward']:.1f} \u2192 {f_entry['candidate_reward']:.1f}",
            fg=color,
        )
        + f"  expected={exp!r:<32s}{cat}"
    )

    if verbose:
        cand_resp = f_entry.get("candidate_response")
        if cand_resp:
            lines.append(f"          got: {cand_resp[:72]}")


def _render_compact_lines(
    verdict: str,
    v_color: str,
    base_id: str,
    cand_id: str,
    s: dict,
    report: dict,
    threshold: float,
    warnings: list[str],
) -> list[str]:
    lines: list[str] = []
    nr = s.get("n_regressions", 0)
    ni = s.get("n_improvements", 0)
    np_ = s.get("n_paired", 0)
    sig = "significant" if _test_significant(report) else "not significant"

    lines.append(
        style(verdict, fg=v_color, bold=True) + f" — {nr} regressions, {ni} improvements ({np_} samples, {sig})"
    )
    lines.append(f"  Baseline:  {base_id}")
    lines.append(f"  Candidate: {cand_id}")

    cats = report.get("category_deltas", {})
    if cats:
        for cat, v in sorted(cats.items()):
            d = v["delta"]
            if d < -threshold:
                lines.append(
                    style(
                        f"  {cat}: {v['baseline'] * 100:.1f}% -> {v['candidate'] * 100:.1f}%  ({d * 100:+.1f}%)",
                        fg="red",
                    )
                )

    for w in warnings:
        lines.append(style(f"  WARNING: {w}", fg="yellow"))

    return lines


def render_text(report: dict[str, Any], **opts) -> str:
    """Render a regression report as styled terminal text.

    Accepts ``compact``, ``verbose``, ``show_flips`` via ``**opts``.
    """
    compact = opts.get("compact", False)
    verbose = opts.get("verbose", False)
    show_flips = opts.get("show_flips", False)

    mcnemar = report.get("mcnemar")
    flip = report.get("flip_report")
    verdict = report.get("verdict", "PASS")
    verdict_reasons = report.get("verdict_reasons", [])
    warnings = report.get("warnings", [])
    s = flip.get("summary", {}) if flip else {}
    v_color = verdict_color(verdict)
    threshold = report.get("threshold", 0.05)

    base_id = report.get("baseline", {}).get("run_id", "unknown")
    cand_id = report.get("candidate", {}).get("run_id", "unknown")

    if compact:
        return "\n".join(_render_compact_lines(verdict, v_color, base_id, cand_id, s, report, threshold, warnings))

    lines: list[str] = []

    # One-liner summary
    n_reg = s.get("n_regressions", 0)
    n_imp = s.get("n_improvements", 0)
    n_paired = s.get("n_paired", 0)
    sig_word = "significant" if _test_significant(report) else "not significant"
    lines.append("")
    lines.append(
        style(
            f"{verdict} — {n_reg} regressions, {n_imp} improvements out of {n_paired} paired samples — {sig_word}",
            fg=v_color,
            bold=True,
        )
    )

    # Header
    lines.append("")
    lines.append("REGRESSION REPORT")
    lines.append("\u2550" * 60)
    lines.append(f"Baseline:  {base_id}")
    lines.append(f"Candidate: {cand_id}")
    lines.append(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

    # Warnings
    for w in warnings:
        lines.append(style(f"  WARNING: {w}", fg="yellow"))

    # Verdict detail
    if verdict_reasons:
        lines.append("")
        reason_text = "; ".join(verdict_reasons)
        if verbose:
            lines.append(style(f"VERDICT: {verdict}", fg=v_color, bold=True) + f"  ({reason_text})")
        else:
            plain = _plain_english_verdict(verdict, s, mcnemar)
            lines.append(style(f"VERDICT: {verdict}", fg=v_color, bold=True) + f"  ({plain})")

    # Capability impact profile
    cats = report.get("category_deltas", {})
    score_deltas = report.get("score_deltas", {})
    test_sig = _test_significant(report)

    if cats:
        lines.append("")
        lines.append(style("CAPABILITY IMPACT PROFILE", bold=True))
        lines.append("\u2500" * 60)

        max_name_len = max(len(c) for c in cats) if cats else 10
        for cat, v in sorted(cats.items()):
            base_val = v["baseline"]
            cand_val = v["candidate"]
            delta = v["delta"]
            label, color = _cap_verdict(delta, threshold, test_sig)
            line = (
                f"  {cat:<{max_name_len}}  {bar(cand_val)}  "
                f"{base_val * 100:5.1f}% \u2192 {cand_val * 100:5.1f}%  "
                f"({delta * 100:+.1f}%)  "
            )
            lines.append(line + style(label, fg=color, bold=True))

    elif score_deltas:
        lines.append("")
        lines.append(style("SCORE DELTAS", bold=True))
        lines.append("\u2500" * 60)
        for metric, delta_info in score_deltas.items():
            d = delta_info["delta"]
            base_val = delta_info["baseline"]
            cand_val = delta_info["candidate"]
            overlap = "overlap" if delta_info.get("ci_overlap") else "NO overlap"
            label, color = _cap_verdict(d, threshold, test_sig)
            line = (
                f"  {metric:<20}  {bar(cand_val)}  "
                f"{base_val * 100:5.1f}% \u2192 {cand_val * 100:5.1f}%  "
                f"({d * 100:+.1f}%, CI {overlap})  "
            )
            lines.append(line + style(label, fg=color, bold=True))

    # Flip summary
    if flip:
        n = max(1, s.get("n_paired", 1))
        scale = 30.0 / n
        sc = s.get("n_stable_correct", 0)
        sw = s.get("n_stable_wrong", 0)
        nr = s.get("n_regressions", 0)
        ni = s.get("n_improvements", 0)

        sc_bar = "\u2588" * max(1, round(sc * scale))
        sw_bar = "\u2591" * max(1, round(sw * scale)) if sw else ""
        rg_bar = "\u2593" * max(1, round(nr * scale)) if nr else ""
        im_bar = "\u2592" * max(1, round(ni * scale)) if ni else ""

        reg_rate = f" ({nr / n * 100:.1f}%)" if n > 0 and nr > 0 else ""
        imp_rate = f" ({ni / n * 100:.1f}%)" if n > 0 and ni > 0 else ""

        lines.append("")
        lines.append(style("FLIP SUMMARY", bold=True) + f"  ({n} paired samples)")
        lines.append("\u2500" * 60)
        lines.append(style(f"  Stable correct:  {sc:>4}  {sc_bar}", fg="green") + "  (both got it right)")
        lines.append(f"  Stable wrong:    {sw:>4}  {sw_bar}  (both got it wrong)")
        lines.append(
            style(f"  Regressions:     {nr:>4}  {rg_bar}", fg="red") + f"  (baseline right, candidate wrong){reg_rate}"
        )
        lines.append(
            style(f"  Improvements:    {ni:>4}  {im_bar}", fg="green")
            + f"  (baseline wrong, candidate right){imp_rate}"
        )

        cat_bd = s.get("category_breakdown")
        if cat_bd:
            lines.append("")
            lines.append("  By category:")
            for cat, counts in sorted(cat_bd.items()):
                r = counts.get("regressions", 0)
                i = counts.get("improvements", 0)
                parts = []
                if r:
                    parts.append(style(f"{r} regressed", fg="red"))
                if i:
                    parts.append(style(f"{i} improved", fg="green"))
                lines.append(f"    {cat}: {', '.join(parts)}")

    # Per-sample flips
    if show_flips and flip:
        if flip.get("regressions"):
            nr = s.get("n_regressions", 0)
            np_ = s.get("n_paired", 0)
            lines.append("")
            lines.append(
                style(
                    f"REGRESSIONS ({nr} of {np_} problems, {nr / np_ * 100:.1f}%)"
                    if np_
                    else f"REGRESSIONS ({nr} problems)",
                    fg="red",
                    bold=True,
                )
            )
            lines.append("\u2500" * 60)
            for f_entry in flip["regressions"]:
                _render_flip_entry(lines, f_entry, "red", verbose)

        if flip.get("improvements"):
            ni = s.get("n_improvements", 0)
            np_ = s.get("n_paired", 0)
            lines.append("")
            lines.append(
                style(
                    f"IMPROVEMENTS ({ni} of {np_} problems, {ni / np_ * 100:.1f}%)"
                    if np_
                    else f"IMPROVEMENTS ({ni} problems)",
                    fg="green",
                    bold=True,
                )
            )
            lines.append("\u2500" * 60)
            for f_entry in flip["improvements"]:
                _render_flip_entry(lines, f_entry, "green", verbose)

    # Verbose: statistical details
    if verbose:
        test_used = report.get("test_used")
        test_reason = report.get("test_reason")
        lines.append("")
        lines.append("STATISTICAL DETAILS")
        lines.append("\u2500" * 60)
        if test_used:
            lines.append(f"  Test selected: {test_used}")
        if test_reason:
            lines.append(f"  Reason: {test_reason}")

        if mcnemar:
            lines.append("  Test: McNemar exact binomial (one-sided, H1: regressions > improvements)")
            lines.append(f"  p-value: {mcnemar.get('p_value')}")
            lines.append(f"  Method: {mcnemar.get('method')}")
            lines.append(f"  Discordant pairs: {mcnemar.get('n_discordant')}")
            lines.append(f"  Effect size: {mcnemar.get('effect_size')} (net regression rate over paired samples)")
            if mcnemar.get("ci_lower") is not None:
                lines.append(f"  95% CI: [{mcnemar['ci_lower']:.4f}, {mcnemar['ci_upper']:.4f}]")

        sign_result = report.get("sign_test")
        if sign_result:
            lines.append("  Test: Sign test (one-sided, H1: baseline > candidate)")
            lines.append(f"  p-value: {sign_result.get('p_value')}")
            lines.append(
                f"  Regressions (d>0): {sign_result.get('n_positive')}, Improvements (d<0): {sign_result.get('n_negative')}, Ties: {sign_result.get('n_ties')}"
            )

        perm_result = report.get("permutation_test")
        if perm_result:
            lines.append(
                f"  Test: Permutation test (one-sided, {perm_result.get('n_permutations', 10000)} permutations)"
            )
            lines.append(f"  p-value: {perm_result.get('p_value')}")
            lines.append(f"  Observed mean diff: {perm_result.get('observed_mean_diff')}")
            lines.append(f"  Effect size (Cohen's d): {perm_result.get('effect_size')}")

    # Summary sentence
    summary_sentence = build_summary_sentence(s, cats, threshold)
    if summary_sentence:
        lines.append("")
        lines.append(style(f"Summary: {summary_sentence}", bold=True))

    # Final verdict banner
    lines.append("")
    if verdict == "BLOCK":
        broke_cats = _broke_categories(report, threshold, test_sig)
        if broke_cats:
            lines.append(style(f"BLOCKED: {', '.join(broke_cats)} regressed beyond threshold.", fg="red", bold=True))
        else:
            lines.append(style("BLOCKED: significant regression detected.", fg="red", bold=True))
    elif verdict == "WARN":
        lines.append(
            style("WARNING: statistically significant change, but below practical threshold.", fg="yellow", bold=True)
        )
    elif verdict == "INCONCLUSIVE":
        lines.append(
            style(
                "INCONCLUSIVE: not enough data to detect regressions at the configured threshold.",
                fg="yellow",
                bold=True,
            )
        )
    else:
        nr = s.get("n_regressions", 0)
        if nr > 0:
            lines.append(
                style(
                    f"No significant regressions. ({nr} flip(s) within normal variation for {s.get('n_paired', 0)} samples.)",
                    fg="green",
                )
            )
        else:
            lines.append(style("No regressions detected.", fg="green"))

    return "\n".join(lines)


# ── Markdown renderer ─────────────────────────────────────────────────


def _render_flip_detail(
    _w,
    f_entry: dict[str, Any],
) -> None:
    pid = f_entry.get("problem_idx", "?")
    cat = f_entry.get("category") or "uncategorized"
    exp = f_entry.get("expected_answer") or ""
    b_resp = f_entry.get("baseline_response") or "(not captured)"
    c_resp = f_entry.get("candidate_response") or "(not captured)"

    exp_safe = exp.replace("|", "\\|")
    b_safe = b_resp.replace("|", "\\|")
    c_safe = c_resp.replace("|", "\\|")

    _w(f"### Problem #{pid} — {cat}")
    _w("")
    _w(f"**Expected:** `{exp_safe}`")
    _w("")
    _w("| Baseline | Candidate |")
    _w("|---|---|")
    _w(f"| {f_entry.get('baseline_reward', 0):.1f} (reward) | {f_entry.get('candidate_reward', 0):.1f} (reward) |")
    _w(f"| {b_safe} | {c_safe} |")
    _w("")


def render_markdown(report: dict[str, Any], **_) -> str:
    """Generate a full Markdown regression report from a comparison dict."""
    lines: list[str] = []
    _w = lines.append

    base_id = report.get("baseline", {}).get("run_id", "unknown")
    cand_id = report.get("candidate", {}).get("run_id", "unknown")
    verdict = report.get("verdict", "PASS")
    reasons = report.get("verdict_reasons", [])
    warnings = report.get("warnings", [])
    mcnemar = report.get("mcnemar", {})
    flip = report.get("flip_report", {})
    cats = report.get("category_deltas", {})
    scores = report.get("score_deltas", {})
    s = flip.get("summary", {}) if flip else {}
    threshold = report.get("threshold", 0.05)

    _w(f"# Regression Report: {verdict}")
    _w("")
    _w("| | |")
    _w("|---|---|")
    _w(f"| **Baseline** | `{base_id}` |")
    _w(f"| **Candidate** | `{cand_id}` |")
    _w(f"| **Generated** | {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} |")
    _w(f"| **Paired samples** | {s.get('n_paired', 'n/a')} |")
    _w(f"| **Verdict** | **{verdict}** |")
    if reasons:
        _w(f"| **Reason** | {'; '.join(reasons)} |")
    _w("")

    if warnings:
        _w("## Warnings")
        _w("")
        for w in warnings:
            _w(f"> ⚠️ {w}")
        _w("")

    # Executive Summary
    _w("## Executive Summary")
    _w("")
    n_reg = s.get("n_regressions", 0)
    n_imp = s.get("n_improvements", 0)
    n_paired = s.get("n_paired", 0)
    n_stable_c = s.get("n_stable_correct", 0)
    n_stable_w = s.get("n_stable_wrong", 0)

    if n_paired > 0:
        _w(f"Out of **{n_paired}** paired evaluation problems:")
        _w("")
        _w(f"- ✅ **{n_stable_c}** stable correct ({n_stable_c / n_paired * 100:.1f}%) — both models got it right")
        _w(f"- ⬜ **{n_stable_w}** stable wrong ({n_stable_w / n_paired * 100:.1f}%) — both models got it wrong")
        _w(f"- 🔴 **{n_reg}** regressions ({n_reg / n_paired * 100:.1f}%) — baseline right, candidate wrong")
        _w(f"- 🟢 **{n_imp}** improvements ({n_imp / n_paired * 100:.1f}%) — baseline wrong, candidate right")
        _w("")

        if verdict == "BLOCK":
            _w(
                f"**The candidate model shows a statistically significant regression.** "
                f"{n_reg} problems that the baseline solved correctly are now answered incorrectly. "
                f"This exceeds both the statistical significance threshold and the practical effect threshold."
            )
        elif verdict == "WARN":
            _w(
                "**A statistically significant change was detected**, but the practical effect size "
                "is below the configured threshold. Review the specific flips below before shipping."
            )
        elif verdict == "INCONCLUSIVE":
            _w(
                f"**Not enough data to rule out small regressions.** With only {mcnemar.get('n_discordant', 0)} "
                f"discordant pairs, this evaluation cannot reliably detect regressions below "
                f"~{mde_estimate(mcnemar.get('n_discordant', 0)) * 100:.1f}%. Consider adding more evaluation samples."
            )
        else:
            if n_reg > 0:
                _w(
                    f"**No significant regression detected.** {n_reg} problem(s) flipped, "
                    f"but this is within normal variation for {n_paired} samples."
                )
            else:
                _w("**No regressions detected.** Both models produce identical outcomes on all paired samples.")

    summary = build_summary_sentence(s, cats, threshold)
    if summary:
        _w("")
        _w(f"> **{summary}**")
    _w("")

    # Capability Impact Profile
    if cats:
        _w("## Capability Impact Profile")
        _w("")
        _w("| Capability | Baseline | Candidate | Delta | Status |")
        _w("|---|---|---|---|---|")
        for cat, v in sorted(cats.items()):
            b = v["baseline"]
            c = v["candidate"]
            d = v["delta"]
            status = "🔴 BROKE" if d < -threshold else ("🟡 WARN" if d < -threshold / 5 else "🟢 HELD")
            _w(f"| {cat} | {b * 100:.1f}% | {c * 100:.1f}% | {d * 100:+.1f}% | {status} |")
        _w("")

    # Score Deltas
    if scores:
        _w("## Score Deltas")
        _w("")
        _w("| Metric | Baseline | Candidate | Delta | Relative | CI |")
        _w("|---|---|---|---|---|---|")
        for metric, d in sorted(scores.items()):
            overlap = "overlap" if d.get("ci_overlap") else "**NO overlap**"
            _w(
                f"| {metric} | {d['baseline']:.4f} | {d['candidate']:.4f} | "
                f"{d['delta']:+.4f} | {d['relative_pct']:+.1f}% | {overlap} |"
            )
        _w("")

    # Statistical Details
    if mcnemar:
        _w("## Statistical Analysis")
        _w("")
        _w("### McNemar Paired Test")
        _w("")
        _w("| | |")
        _w("|---|---|")
        _w("| **Test** | McNemar exact binomial (one-sided) |")
        _w("| **Hypothesis** | H₁: P(regression) > P(improvement) |")
        _w(f"| **Discordant pairs** | {mcnemar.get('n_discordant', 0)} |")
        p = mcnemar.get("p_value")
        _w(f"| **p-value** | {p:.6f} |" if p is not None else "| **p-value** | n/a (scipy not installed) |")
        _w(f"| **Significant** | {'Yes' if mcnemar.get('significant') else 'No'} (α=0.05) |")
        _w(f"| **Method** | {mcnemar.get('method', 'n/a')} |")
        es = mcnemar.get("effect_size")
        if es is not None:
            _w(f"| **Effect size** | {es:.4f} (net regression rate) |")
        ci_lo = mcnemar.get("ci_lower")
        ci_hi = mcnemar.get("ci_upper")
        if ci_lo is not None and ci_hi is not None:
            _w(f"| **95% CI** | [{ci_lo:.4f}, {ci_hi:.4f}] |")
        _w("")

        _w("### Contingency Table")
        _w("")
        ct = flip.get("contingency", {})
        _w("| | Candidate Correct | Candidate Wrong |")
        _w("|---|---|---|")
        _w(
            f"| **Baseline Correct** | {ct.get('both_correct', 0)} | {ct.get('baseline_only_correct', 0)} (regressions) |"
        )
        _w(f"| **Baseline Wrong** | {ct.get('candidate_only_correct', 0)} (improvements) | {ct.get('both_wrong', 0)} |")
        _w("")

    # Category Breakdown of Flips
    cat_bd = s.get("category_breakdown")
    if cat_bd:
        _w("## Regressions by Category")
        _w("")
        _w("| Category | Regressions | Improvements | Net |")
        _w("|---|---|---|---|")
        for cat, counts in sorted(cat_bd.items()):
            r = counts.get("regressions", 0)
            i = counts.get("improvements", 0)
            net = r - i
            flag = " 🔴" if net > 0 else ""
            _w(f"| {cat} | {r} | {i} | {net:+d}{flag} |")
        _w("")

    # Full Flip Tables
    regressions = flip.get("regressions", []) if flip else []
    improvements = flip.get("improvements", []) if flip else []

    if regressions:
        _w(f"## Regressed Problems ({len(regressions)})")
        _w("")
        _w("These problems were answered correctly by the baseline but incorrectly by the candidate.")
        _w("")
        for f_entry in regressions:
            _render_flip_detail(_w, f_entry)

    if improvements:
        _w(f"## Improved Problems ({len(improvements)})")
        _w("")
        _w("These problems were answered incorrectly by the baseline but correctly by the candidate.")
        _w("")
        for f_entry in improvements:
            _render_flip_detail(_w, f_entry)

    # Investigation Guidance
    if regressions:
        _w("## Investigation Guidance")
        _w("")

        if cat_bd:
            worst_cats = sorted(
                cat_bd.items(), key=lambda x: x[1].get("regressions", 0) - x[1].get("improvements", 0), reverse=True
            )
            worst = worst_cats[0] if worst_cats else None
            if worst and worst[1].get("regressions", 0) > 0:
                _w(
                    f"**Most affected category:** `{worst[0]}` "
                    f"({worst[1]['regressions']} regressions, {worst[1].get('improvements', 0)} improvements)"
                )
                _w("")

        _w("**Suggested next steps:**")
        _w("")
        _w(
            f"1. **Examine the {min(5, len(regressions))} regressed problems above.** "
            "What do they have in common? (length, difficulty, topic, format)"
        )
        _w(
            "2. **Check if regressions cluster on 'easy' problems** (problems the baseline always gets right). "
            "Easy-problem regressions indicate systematic failure, not noise."
        )
        _w(
            "3. **Compare model responses** for regressed problems. Are they near-misses (formatting), "
            "refusals, or completely wrong outputs?"
        )
        _w("4. **Run with `n_repeats > 1`** to distinguish deterministic failures from stochastic ones.")
        _w("")

        _w("**To extract regressed problem IDs for further analysis:**")
        _w("```bash")
        _w("nel compare baseline.json candidate.json --format json | jq '.flip_report.regressions[].problem_idx'")
        _w("```")
        _w("")

    _w("---")
    _w(f"*Generated by nemo-evaluator `nel compare` • {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*")

    return "\n".join(lines)


# ── JSON renderer ─────────────────────────────────────────────────────


def render_json(report: dict[str, Any], **_) -> str:
    """Serialize the report dict as indented JSON."""
    return json.dumps(report, indent=2, default=str)


# ── Convenience I/O ───────────────────────────────────────────────────


def write_report(report: dict[str, Any], output_path: str | Path) -> Path:
    """Write the full Markdown report to a file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(report), encoding="utf-8")
    return path


# ── Registry ──────────────────────────────────────────────────────────

RENDERERS = {
    "text": render_text,
    "markdown": render_markdown,
    "json": render_json,
}
