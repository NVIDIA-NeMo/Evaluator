"""Generate a full Markdown regression report for human review.

The CLI dashboard (`nel regression`) is a 5-second scan. This module
produces a document that a researcher can sit with for 30 minutes:
full flip tables with model responses, category drill-downs, statistical
details, and actionable investigation guidance.

Usage::

    nel regression baseline.json candidate.json --output-report report.md
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nemo_evaluator.engine.comparison import build_summary_sentence, mde_estimate


def generate_report(report: dict[str, Any]) -> str:
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

    # ── Header ─────────────────────────────────────────────────────
    _w(f"# Regression Report: {verdict}")
    _w("")
    _w(f"| | |")
    _w(f"|---|---|")
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

    # ── Executive Summary ──────────────────────────────────────────
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
            _w(f"**The candidate model shows a statistically significant regression.** "
               f"{n_reg} problems that the baseline solved correctly are now answered incorrectly. "
               f"This exceeds both the statistical significance threshold and the practical effect threshold.")
        elif verdict == "WARN":
            _w(f"**A statistically significant change was detected**, but the practical effect size "
               f"is below the configured threshold. Review the specific flips below before shipping.")
        elif verdict == "INCONCLUSIVE":
            _w(f"**Not enough data to rule out small regressions.** With only {mcnemar.get('n_discordant', 0)} "
               f"discordant pairs, this evaluation cannot reliably detect regressions below "
               f"~{mde_estimate(mcnemar.get('n_discordant', 0)) * 100:.1f}%. Consider adding more evaluation samples.")
        else:
            if n_reg > 0:
                _w(f"**No significant regression detected.** {n_reg} problem(s) flipped, "
                   f"but this is within normal variation for {n_paired} samples.")
            else:
                _w("**No regressions detected.** Both models produce identical outcomes on all paired samples.")

    # Summary sentence
    summary = build_summary_sentence(verdict, s, cats, threshold)
    if summary:
        _w("")
        _w(f"> **{summary}**")
    _w("")

    # ── Capability Impact Profile ──────────────────────────────────
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

    # ── Score Deltas ───────────────────────────────────────────────
    if scores:
        _w("## Score Deltas")
        _w("")
        _w("| Metric | Baseline | Candidate | Delta | Relative | CI |")
        _w("|---|---|---|---|---|---|")
        for metric, d in sorted(scores.items()):
            overlap = "overlap" if d.get("ci_overlap") else "**NO overlap**"
            _w(f"| {metric} | {d['baseline']:.4f} | {d['candidate']:.4f} | "
               f"{d['delta']:+.4f} | {d['relative_pct']:+.1f}% | {overlap} |")
        _w("")

    # ── Statistical Details ────────────────────────────────────────
    if mcnemar:
        _w("## Statistical Analysis")
        _w("")
        _w("### McNemar Paired Test")
        _w("")
        _w(f"| | |")
        _w(f"|---|---|")
        _w(f"| **Test** | McNemar exact binomial (one-sided) |")
        _w(f"| **Hypothesis** | H₁: P(regression) > P(improvement) |")
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
        _w(f"| **Baseline Correct** | {ct.get('both_correct', 0)} | {ct.get('baseline_only_correct', 0)} (regressions) |")
        _w(f"| **Baseline Wrong** | {ct.get('candidate_only_correct', 0)} (improvements) | {ct.get('both_wrong', 0)} |")
        _w("")

    # ── Category Breakdown of Flips ────────────────────────────────
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

    # ── Full Flip Tables ───────────────────────────────────────────
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

    # ── Investigation Guidance ─────────────────────────────────────
    if regressions:
        _w("## Investigation Guidance")
        _w("")

        # Find most affected categories
        if cat_bd:
            worst_cats = sorted(cat_bd.items(), key=lambda x: x[1].get("regressions", 0) - x[1].get("improvements", 0), reverse=True)
            worst = worst_cats[0] if worst_cats else None
            if worst and worst[1].get("regressions", 0) > 0:
                _w(f"**Most affected category:** `{worst[0]}` "
                   f"({worst[1]['regressions']} regressions, {worst[1].get('improvements', 0)} improvements)")
                _w("")

        _w("**Suggested next steps:**")
        _w("")
        _w(f"1. **Examine the {min(5, len(regressions))} regressed problems above.** "
           "What do they have in common? (length, difficulty, topic, format)")
        _w("2. **Check if regressions cluster on 'easy' problems** (problems the baseline always gets right). "
           "Easy-problem regressions indicate systematic failure, not noise.")
        _w("3. **Compare model responses** for regressed problems. Are they near-misses (formatting), "
           "refusals, or completely wrong outputs?")
        _w("4. **Run with `n_repeats > 1`** to distinguish deterministic failures from stochastic ones.")
        _w("")

        _w("**To extract regressed problem IDs for further analysis:**")
        _w("```bash")
        _w("nel regression baseline.json candidate.json --format json | jq '.flip_report.regressions[].problem_idx'")
        _w("```")
        _w("")

    # ── Footer ─────────────────────────────────────────────────────
    _w("---")
    _w(f"*Generated by nemo-evaluator `nel regression` • "
       f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*")

    return "\n".join(lines)


def write_report(report: dict[str, Any], output_path: str | Path) -> Path:
    """Write the full Markdown report to a file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(generate_report(report), encoding="utf-8")
    return path


def _render_flip_detail(
    _w,
    f_entry: dict[str, Any],
) -> None:
    """Render a single flipped problem with side-by-side model responses."""
    pid = f_entry.get("problem_idx", "?")
    cat = f_entry.get("category") or "uncategorized"
    exp = f_entry.get("expected_answer") or ""
    b_resp = f_entry.get("baseline_response") or "(not captured)"
    c_resp = f_entry.get("candidate_response") or "(not captured)"

    # Escape pipe chars for markdown tables
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


