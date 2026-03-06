"""nel report -- aggregate multiple evaluation bundles into comparison tables."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import click

logger = logging.getLogger(__name__)


def _load_bundles(paths: list[Path]) -> list[dict[str, Any]]:
    bundles = []
    for p in paths:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            data["_source"] = str(p)
            bundles.append(data)
        except Exception as e:
            logger.warning("Skipping %s: %s", p, e)
    return bundles


def _build_table(bundles: list[dict[str, Any]]) -> dict[str, Any]:
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


def _render_markdown(table: dict[str, Any]) -> str:
    lines = [f"# Evaluation Report: {table['model']}\n"]

    header = "| Benchmark | Samples | pass@1 | CI (95%) |"
    sep = "|-----------|---------|--------|----------|"
    lines.extend([header, sep])

    for name, row in sorted(table["benchmarks"].items()):
        p1 = row.get("pass@1", {})
        if isinstance(p1, dict) and "value" in p1:
            val = f"{p1['value']:.4f}"
            lo = p1.get("ci_lower", "")
            hi = p1.get("ci_upper", "")
            ci = f"[{lo:.4f}, {hi:.4f}]" if lo and hi else ""
        else:
            val = "-"
            ci = ""
        lines.append(f"| {name} | {row.get('samples', '-')} | {val} | {ci} |")

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


def _render_latex(table: dict[str, Any]) -> str:
    lines = [
        "\\begin{table}[h]",
        "\\centering",
        f"\\caption{{Evaluation results: {table['model']}}}",
        "\\begin{tabular}{lrcc}",
        "\\toprule",
        "Benchmark & Samples & pass@1 & 95\\% CI \\\\",
        "\\midrule",
    ]

    for name, row in sorted(table["benchmarks"].items()):
        p1 = row.get("pass@1", {})
        if isinstance(p1, dict) and "value" in p1:
            val = f"{p1['value']:.4f}"
            lo = p1.get("ci_lower", "")
            hi = p1.get("ci_upper", "")
            ci = f"[{lo:.4f}, {hi:.4f}]" if lo and hi else ""
        else:
            val = "--"
            ci = ""
        lines.append(f"{name} & {row.get('samples', '--')} & {val} & {ci} \\\\")

    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    return "\n".join(lines)


def _render_csv(table: dict[str, Any]) -> str:
    lines = ["benchmark,samples,repeats,pass@1,ci_lower,ci_upper"]
    for name, row in sorted(table["benchmarks"].items()):
        p1 = row.get("pass@1", {})
        val = p1.get("value", "") if isinstance(p1, dict) else ""
        lo = p1.get("ci_lower", "") if isinstance(p1, dict) else ""
        hi = p1.get("ci_upper", "") if isinstance(p1, dict) else ""
        lines.append(f"{name},{row.get('samples', '')},{row.get('repeats', '')},{val},{lo},{hi}")
    return "\n".join(lines)


RENDERERS = {
    "markdown": _render_markdown,
    "latex": _render_latex,
    "csv": _render_csv,
    "json": lambda t: json.dumps(t, indent=2, default=str),
}


@click.command("report")
@click.argument("bundle_paths", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--format", "-f", "fmt", type=click.Choice(list(RENDERERS)), default="markdown")
@click.option("--output", "-o", type=click.Path(), default=None)
def report_cmd(bundle_paths, fmt, output):
    """Generate a multi-benchmark evaluation report from bundle JSON files."""
    paths = [Path(p) for p in bundle_paths]
    expanded: list[Path] = []
    for p in paths:
        if p.is_dir():
            expanded.extend(sorted(p.glob("eval-*.json")))
        else:
            expanded.append(p)

    if not expanded:
        raise click.ClickException("No bundle files found.")

    bundles = _load_bundles(expanded)
    if not bundles:
        raise click.ClickException("No valid bundles loaded.")

    table = _build_table(bundles)
    rendered = RENDERERS[fmt](table)

    if output:
        Path(output).write_text(rendered, encoding="utf-8")
        click.echo(f"Report written to {output}")
    else:
        click.echo(rendered)
