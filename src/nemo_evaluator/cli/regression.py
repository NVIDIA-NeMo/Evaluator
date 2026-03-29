from __future__ import annotations

import os
import sys
from datetime import datetime, timezone

import click

# Item #14: respect NO_COLOR env var
_NO_COLOR = os.environ.get("NO_COLOR") is not None


def _style(text: str, **kwargs) -> str:
    if _NO_COLOR:
        return text
    return click.style(text, **kwargs)


def _bar(value: float, width: int = 20) -> str:
    filled = round(value * width)
    return "\u2588" * filled + "\u2591" * (width - filled)


def _verdict_color(verdict: str) -> str:
    if verdict in ("BLOCK",):
        return "red"
    if verdict in ("WARN", "INCONCLUSIVE"):
        return "yellow"
    return "green"


def _cap_verdict(delta: float, threshold: float, mcnemar_sig: bool | None) -> tuple[str, str]:
    if delta < -threshold and mcnemar_sig:
        return "BROKE \u26a0", "red"
    if delta < -threshold:
        return "WARN", "yellow"
    return "HELD", "green"


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


@click.command("regression")
@click.argument("baseline", type=click.Path(exists=True))
@click.argument("candidate", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), default=None, help="Write full report JSON to file.")
@click.option(
    "--max-drop", "-t", "threshold", type=float, default=0.05,
    help="Maximum allowed absolute drop in any metric (0-1 scale, default: 0.05 = 5%%).",
)
@click.option("--strict", is_flag=True, help="Exit non-zero on BLOCK (exit 1) or WARN (exit 2).")
@click.option(
    "--correct-above", "reward_threshold", type=float, default=0.0,
    help="Reward above this counts as 'correct' for flip analysis. Use 0.5 for judge scores. Default: >0.",
)
@click.option("--show-flips", is_flag=True, help="Show per-sample flip list with details.")
@click.option("--compact", is_flag=True, help="Short output for Slack / CI logs (~8 lines).")
@click.option("--format", "fmt", type=click.Choice(["text", "json"]), default="text",
              help="Output format. 'json' writes structured JSON to stdout.")
@click.option("--verbose", is_flag=True, help="Show statistical details (p-values, methods, power).")
@click.option("--report", "report_path", type=click.Path(), default=None,
              help="Write Markdown report to this path (default: auto-generated next to candidate bundle).")
@click.option("--no-report", is_flag=True, help="Suppress auto-generated Markdown report.")
def regression_cmd(baseline, candidate, output, threshold, strict, reward_threshold, show_flips, compact, fmt, verbose, report_path, no_report):
    """Compare two evaluation runs and report regressions.

    BASELINE and CANDIDATE can be eval-*.json bundle files OR directories
    containing one (produced by 'nel eval run').

    \b
    Examples:
      nel regression ./baseline/ ./candidate/
      nel regression ./baseline/ ./candidate/ --show-flips
      nel regression ./baseline/ ./candidate/ --strict --max-drop 0.01
      nel regression ./baseline/eval-base.json ./cand/eval-cand.json
    """
    from pathlib import Path
    from nemo_evaluator.engine.comparison import compare_runs, write_regression

    baseline = _resolve_bundle(baseline)
    candidate = _resolve_bundle(candidate)

    report = compare_runs(
        baseline, candidate,
        reward_threshold=reward_threshold,
        min_effect=threshold,
    )

    # Item #4: auto-generate Markdown report next to candidate bundle
    if report_path is None and not no_report:
        report_path = str(Path(candidate).parent / "regression_report.md")

    # Item #9: --format json writes to stdout
    if fmt == "json":
        import json as _json
        click.echo(_json.dumps(report, indent=2, default=str))
        if report_path:
            from nemo_evaluator.engine.regression_report import write_report
            write_report(report, report_path)
            click.echo(f"Report written to: {report_path}", err=True)
        return

    mcnemar = report.get("mcnemar")
    flip = report.get("flip_report")
    verdict = report.get("verdict", "PASS")
    verdict_reasons = report.get("verdict_reasons", [])
    warnings = report.get("warnings", [])
    s = flip.get("summary", {}) if flip else {}
    v_color = _verdict_color(verdict)

    base_id = report["baseline"]["run_id"]
    cand_id = report["candidate"]["run_id"]

    # Item #15: --compact mode for Slack (~8 lines)
    if compact:
        _render_compact(verdict, v_color, base_id, cand_id, s, mcnemar, report, threshold, warnings)
        _exit_strict(strict, verdict)
        return

    regressions = []

    # ── One-liner summary as first line (item #7) ──────────────────
    n_reg = s.get("n_regressions", 0)
    n_imp = s.get("n_improvements", 0)
    n_paired = s.get("n_paired", 0)
    sig_word = "significant" if mcnemar and mcnemar.get("significant") else "not significant"
    click.echo()
    click.echo(_style(
        f"{verdict} — {n_reg} regressions, {n_imp} improvements out of {n_paired} paired samples — {sig_word}",
        fg=v_color, bold=True,
    ))

    # ── Header ─────────────────────────────────────────────────────
    click.echo()
    click.echo("REGRESSION REPORT")
    click.echo("\u2550" * 60)
    click.echo(f"Baseline:  {base_id}")
    click.echo(f"Candidate: {cand_id}")
    # Item #4 from PM: add timestamp
    click.echo(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

    # ── Warnings ───────────────────────────────────────────────────
    for w in warnings:
        click.echo(_style(f"  WARNING: {w}", fg="yellow"))

    # ── Verdict detail ─────────────────────────────────────────────
    if verdict_reasons:
        click.echo()
        reason_text = "; ".join(verdict_reasons)
        if verbose:
            click.echo(_style(f"VERDICT: {verdict}", fg=v_color, bold=True) + f"  ({reason_text})")
        else:
            # Item #6: plain English, hide p-values
            plain = _plain_english_verdict(verdict, s, mcnemar)
            click.echo(_style(f"VERDICT: {verdict}", fg=v_color, bold=True) + f"  ({plain})")

    # ── Capability Impact Profile ──────────────────────────────────
    cats = report.get("category_deltas", {})
    score_deltas = report.get("score_deltas", {})
    mcnemar_sig = mcnemar.get("significant") if mcnemar else None

    if cats:
        click.echo()
        click.echo(_style("CAPABILITY IMPACT PROFILE", bold=True))
        click.echo("\u2500" * 60)

        max_name_len = max(len(c) for c in cats) if cats else 10
        for cat, v in sorted(cats.items()):
            base_val = v["baseline"]
            cand_val = v["candidate"]
            delta = v["delta"]
            label, color = _cap_verdict(delta, threshold, mcnemar_sig)
            line = (
                f"  {cat:<{max_name_len}}  {_bar(cand_val)}  "
                f"{base_val * 100:5.1f}% \u2192 {cand_val * 100:5.1f}%  "
                f"({delta * 100:+.1f}%)  "
            )
            click.echo(line + _style(label, fg=color, bold=True))

    elif score_deltas:
        click.echo()
        click.echo(_style("SCORE DELTAS", bold=True))
        click.echo("\u2500" * 60)
        for metric, delta in score_deltas.items():
            d = delta["delta"]
            base_val = delta["baseline"]
            cand_val = delta["candidate"]
            overlap = "overlap" if delta.get("ci_overlap") else "NO overlap"
            label, color = _cap_verdict(d, threshold, mcnemar_sig)
            if d < -threshold:
                regressions.append(metric)
            line = (
                f"  {metric:<20}  {_bar(cand_val)}  "
                f"{base_val * 100:5.1f}% \u2192 {cand_val * 100:5.1f}%  "
                f"({d * 100:+.1f}%, CI {overlap})  "
            )
            click.echo(line + _style(label, fg=color, bold=True))

    if cats:
        for metric, delta in score_deltas.items():
            if delta["delta"] < -threshold:
                regressions.append(metric)

    # ── Flip summary ───────────────────────────────────────────────
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

        # Item #12: show regression rate with denominator
        reg_rate = f" ({nr / n * 100:.1f}%)" if n > 0 and nr > 0 else ""
        imp_rate = f" ({ni / n * 100:.1f}%)" if n > 0 and ni > 0 else ""

        click.echo()
        click.echo(_style("FLIP SUMMARY", bold=True) + f"  ({n} paired samples)")
        click.echo("\u2500" * 60)
        click.echo(_style(f"  Stable correct:  {sc:>4}  {sc_bar}", fg="green") + "  (both got it right)")
        click.echo(f"  Stable wrong:    {sw:>4}  {sw_bar}  (both got it wrong)")
        click.echo(_style(f"  Regressions:     {nr:>4}  {rg_bar}", fg="red") + f"  (baseline right, candidate wrong){reg_rate}")
        click.echo(_style(f"  Improvements:    {ni:>4}  {im_bar}", fg="green") + f"  (baseline wrong, candidate right){imp_rate}")

        # Item #13: category breakdown subtotals
        cat_bd = s.get("category_breakdown")
        if cat_bd:
            click.echo()
            click.echo("  By category:")
            for cat, counts in sorted(cat_bd.items()):
                r = counts.get("regressions", 0)
                i = counts.get("improvements", 0)
                parts = []
                if r:
                    parts.append(_style(f"{r} regressed", fg="red"))
                if i:
                    parts.append(_style(f"{i} improved", fg="green"))
                click.echo(f"    {cat}: {', '.join(parts)}")

    # ── Per-sample flips ───────────────────────────────────────────
    if show_flips and flip:
        if flip.get("regressions"):
            nr = s.get("n_regressions", 0)
            np_ = s.get("n_paired", 0)
            click.echo()
            click.echo(_style(
                f"REGRESSIONS ({nr} of {np_} problems, {nr / np_ * 100:.1f}%)" if np_ else f"REGRESSIONS ({nr} problems)",
                fg="red", bold=True,
            ))
            click.echo("\u2500" * 60)
            for f_entry in flip["regressions"]:
                _render_flip_entry(f_entry, "red", verbose)

        if flip.get("improvements"):
            ni = s.get("n_improvements", 0)
            np_ = s.get("n_paired", 0)
            click.echo()
            click.echo(_style(
                f"IMPROVEMENTS ({ni} of {np_} problems, {ni / np_ * 100:.1f}%)" if np_ else f"IMPROVEMENTS ({ni} problems)",
                fg="green", bold=True,
            ))
            click.echo("\u2500" * 60)
            for f_entry in flip["improvements"]:
                _render_flip_entry(f_entry, "green", verbose)

    # ── Verbose: statistical details ───────────────────────────────
    if verbose and mcnemar:
        click.echo()
        click.echo("STATISTICAL DETAILS")
        click.echo("\u2500" * 60)
        click.echo(f"  Test: McNemar exact binomial (one-sided, H1: regressions > improvements)")
        click.echo(f"  p-value: {mcnemar.get('p_value')}")
        click.echo(f"  Method: {mcnemar.get('method')}")
        click.echo(f"  Discordant pairs: {mcnemar.get('n_discordant')}")
        click.echo(f"  Effect size: {mcnemar.get('effect_size')} (net regression rate over paired samples)")
        if mcnemar.get("ci_lower") is not None:
            click.echo(f"  95% CI: [{mcnemar['ci_lower']:.4f}, {mcnemar['ci_upper']:.4f}]")

    # ── Output / exit ──────────────────────────────────────────────
    if output:
        write_regression(report, output)
        click.echo(f"\nJSON report written to: {output}")

    if report_path:
        from nemo_evaluator.engine.regression_report import write_report
        rp = write_report(report, report_path)
        click.echo(f"Markdown report written to: {rp}")

    # ── Summary sentence (item #5) ───────────────────────────────
    summary_sentence = _build_summary_sentence(verdict, s, cats, threshold)
    if summary_sentence:
        click.echo()
        click.echo(_style(f"Summary: {summary_sentence}", bold=True))

    click.echo()
    # Item #3: clarify "no regressions" vs flip count
    if verdict == "BLOCK":
        broke_cats = _broke_categories(report, threshold, mcnemar_sig)
        if broke_cats:
            click.echo(_style(f"BLOCKED: {', '.join(broke_cats)} regressed beyond threshold.", fg="red", bold=True))
        else:
            click.echo(_style("BLOCKED: significant regression detected.", fg="red", bold=True))
    elif verdict == "WARN":
        click.echo(_style("WARNING: statistically significant change, but below practical threshold.", fg="yellow", bold=True))
    elif verdict == "INCONCLUSIVE":
        click.echo(_style("INCONCLUSIVE: not enough data to detect regressions at the configured threshold.", fg="yellow", bold=True))
    else:
        nr = s.get("n_regressions", 0)
        if nr > 0:
            click.echo(_style(
                f"No significant regressions. ({nr} flip(s) within normal variation for {s.get('n_paired', 0)} samples.)",
                fg="green",
            ))
        else:
            click.echo(_style("No regressions detected.", fg="green"))

    _exit_strict(strict, verdict)


# ── Helpers ────────────────────────────────────────────────────────


def _render_compact(verdict, v_color, base_id, cand_id, s, mcnemar, report, threshold, warnings):
    """Item #15: compact mode for Slack / CI (~8 lines)."""
    nr = s.get("n_regressions", 0)
    ni = s.get("n_improvements", 0)
    np_ = s.get("n_paired", 0)
    sig = "significant" if mcnemar and mcnemar.get("significant") else "not significant"

    click.echo(_style(f"{verdict}", fg=v_color, bold=True) + f" — {nr} regressions, {ni} improvements ({np_} samples, {sig})")
    click.echo(f"  Baseline:  {base_id}")
    click.echo(f"  Candidate: {cand_id}")

    cats = report.get("category_deltas", {})
    if cats:
        for cat, v in sorted(cats.items()):
            d = v["delta"]
            if d < -threshold:
                click.echo(_style(f"  {cat}: {v['baseline'] * 100:.1f}% -> {v['candidate'] * 100:.1f}%  ({d * 100:+.1f}%)", fg="red"))

    for w in warnings:
        click.echo(_style(f"  WARNING: {w}", fg="yellow"))


def _render_flip_entry(f_entry: dict, color: str, verbose: bool):
    cat = f"  [{f_entry.get('category')}]" if f_entry.get("category") else ""
    exp = f_entry.get("expected_answer") or ""
    if len(exp) > 30:
        exp = exp[:27] + "..."

    click.echo(_style(
        f"  #{f_entry['problem_idx']:<5}  "
        f"{f_entry['baseline_reward']:.1f} \u2192 {f_entry['candidate_reward']:.1f}",
        fg=color,
    ) + f"  expected={exp!r:<32s}{cat}")

    if verbose:
        cand_resp = f_entry.get("candidate_response")
        if cand_resp:
            click.echo(f"          got: {cand_resp[:72]}")


def _plain_english_verdict(verdict: str, s: dict, mcnemar: dict | None) -> str:
    """Item #6: human-readable verdict explanation without statistical jargon."""
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


def _broke_categories(report: dict, threshold: float, mcnemar_sig: bool | None) -> list[str]:
    cats = report.get("category_deltas", {})
    return [cat for cat, v in sorted(cats.items()) if v["delta"] < -threshold and mcnemar_sig]


def _build_summary_sentence(verdict: str, s: dict, cats: dict, threshold: float) -> str | None:
    """Generate a one-sentence narrative for Slack copy-paste."""
    if not s.get("n_paired"):
        return None

    n_reg = s.get("n_regressions", 0)
    n_paired = s.get("n_paired", 0)

    if not cats:
        if n_reg == 0:
            return "No regressions detected across all problems."
        return None  # no categories to narrate

    held = [c for c, v in sorted(cats.items()) if v["delta"] >= -threshold]
    broke = [c for c, v in sorted(cats.items()) if v["delta"] < -threshold]

    if not broke and n_reg == 0:
        return f"All capabilities held ({', '.join(held)})."

    if not broke and n_reg > 0:
        return (f"All capabilities held. {n_reg} problem(s) flipped but within normal variation "
                f"for {n_paired} samples.")

    # Build the narrative: what's safe, what broke
    parts = []
    if held:
        parts.append(f"Safe for {', '.join(held)}.")

    for cat in broke:
        v = cats[cat]
        delta_pct = abs(v["delta"]) * 100
        # Find regressed problem IDs in this category from category_breakdown
        cat_bd = s.get("category_breakdown", {})
        cat_reg = cat_bd.get(cat, {}).get("regressions", 0)
        parts.append(f"{cat} regresses {delta_pct:.1f}% ({cat_reg} problems).")

    return " ".join(parts) if parts else None


def _exit_strict(strict: bool, verdict: str):
    if not strict:
        return
    # Item #8: exit 1 for BLOCK, exit 2 for WARN/INCONCLUSIVE
    if verdict == "BLOCK":
        sys.exit(1)
    if verdict in ("WARN", "INCONCLUSIVE"):
        sys.exit(2)
