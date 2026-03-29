"""Multi-benchmark accuracy quality gate.

Discovers eval bundles across baseline/candidate directories, matches
them by benchmark name, applies per-benchmark policy thresholds to
paired evaluation data, and aggregates to a single GO / NO-GO /
INCONCLUSIVE verdict.

The gate computes its own threshold-based verdicts from per-item paired
deltas and confidence intervals.  ``compare_runs()`` output is attached
as supporting evidence (flips, McNemar, category breakdowns) but does
**not** drive the gate decision.
"""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from nemo_evaluator.config.gate_policy import (
    Direction,
    GatePolicy,
    ResolvedBenchmarkPolicy,
)
from nemo_evaluator.engine.comparison import (
    compare_runs,
    load_paired_records,
)

logger = logging.getLogger(__name__)

_MIN_PAIRED_ITEMS = 10
_GATE_REPORT_VERSION = 1


# ── Dataclasses ───────────────────────────────────────────────────────


@dataclass
class BenchmarkGateResult:
    """Per-benchmark result within a gate run."""

    benchmark: str = ""
    tier: str = ""
    status: str = "PASS"  # PASS | BREACH | INSUFFICIENT_EVIDENCE | MISSING
    metric: str = ""
    baseline_score: float | None = None
    candidate_score: float | None = None
    delta: float | None = None
    relative_drop_pct: float | None = None
    delta_ci_lower: float | None = None
    delta_ci_upper: float | None = None
    n_paired: int = 0
    max_drop_threshold: float = 0.0
    max_relative_drop_threshold: float | None = None
    direction: str = "higher_is_better"
    absolute_breached: bool = False
    relative_breached: bool = False
    reasons: list[str] = field(default_factory=list)
    regression_report: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("regression_report", None)
        return d


@dataclass
class GateReport:
    """Aggregate gate report across all benchmarks."""

    report_version: int = _GATE_REPORT_VERSION
    verdict: str = "GO"  # GO | NO-GO | INCONCLUSIVE
    verdict_reasons: list[str] = field(default_factory=list)
    benchmarks: list[BenchmarkGateResult] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_version": self.report_version,
            "verdict": self.verdict,
            "verdict_reasons": self.verdict_reasons,
            "benchmarks": [b.to_dict() for b in self.benchmarks],
            "missing": self.missing,
            "warnings": self.warnings,
        }


# ── Public API ────────────────────────────────────────────────────────


def gate_runs(
    baseline_dir: str | Path,
    candidate_dir: str | Path,
    policy: GatePolicy,
) -> GateReport:
    """Run multi-benchmark accuracy quality gate.

    Parameters
    ----------
    baseline_dir, candidate_dir:
        Directories containing evaluation results.  Each may hold
        eval-*.json bundles directly (flat) or in subdirectories (nested).
    policy:
        Gate policy defining per-benchmark thresholds and tiers.

    Returns
    -------
    GateReport with per-benchmark results and aggregate verdict.
    """
    baseline_dir = Path(baseline_dir)
    candidate_dir = Path(candidate_dir)

    base_bundles = _discover_bundles(baseline_dir)
    cand_bundles = _discover_bundles(candidate_dir)

    report = GateReport()

    # Check for missing required benchmarks in either direction
    missing = _check_missing(policy, base_bundles, cand_bundles)
    report.missing = sorted(missing)
    for name in report.missing:
        report.benchmarks.append(BenchmarkGateResult(
            benchmark=name,
            tier=policy.resolve(name).tier.value,
            status="MISSING",
            reasons=[f"Required benchmark {name!r} not found in evaluation results"],
        ))

    # Match and evaluate
    matched, unmatched_b, unmatched_c = _match_bundles(base_bundles, cand_bundles)

    if unmatched_b:
        report.warnings.append(f"Baseline-only benchmarks (no candidate): {', '.join(sorted(unmatched_b))}")
    if unmatched_c:
        report.warnings.append(f"Candidate-only benchmarks (no baseline): {', '.join(sorted(unmatched_c))}")

    for bench_name in sorted(matched):
        if bench_name in missing:
            continue  # already recorded as MISSING
        base_path, cand_path = matched[bench_name]
        resolved = policy.resolve(bench_name)
        result = _evaluate_benchmark(bench_name, base_path, cand_path, resolved)
        report.benchmarks.append(result)

    report.verdict, report.verdict_reasons = _aggregate_verdict(report.benchmarks, report.missing)
    return report


def write_gate_report(report: GateReport, output_path: str | Path) -> Path:
    """Serialize GateReport to JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2, default=str), encoding="utf-8")
    return path


# ── Bundle discovery ──────────────────────────────────────────────────


def _discover_bundles(directory: Path) -> dict[str, Path]:
    """Find eval-*.json bundles, mapping benchmark_name → bundle_path.

    Supports flat (bundles in root) and nested (one subdir per benchmark)
    layouts.  Raises ValueError on duplicate benchmark names.
    """
    bundles: dict[str, Path] = {}

    def _register(p: Path) -> None:
        name = _extract_benchmark_name(p)
        if name is None:
            return
        if name in bundles:
            raise ValueError(
                f"Duplicate benchmark {name!r} in {directory}: "
                f"found in {bundles[name]} and {p}"
            )
        bundles[name] = p

    # Direct bundles
    for p in sorted(directory.glob("eval-*.json")):
        _register(p)

    # Subdirectory bundles
    for sub in sorted(directory.iterdir()):
        if not sub.is_dir():
            continue
        sub_bundles = sorted(sub.glob("eval-*.json"))
        if len(sub_bundles) == 1:
            _register(sub_bundles[0])
        elif len(sub_bundles) > 1:
            logger.warning(
                "Directory %s contains %d eval bundles; skipping.",
                sub, len(sub_bundles),
            )

    return bundles


def _extract_benchmark_name(bundle_path: Path) -> str | None:
    try:
        data = json.loads(bundle_path.read_text(encoding="utf-8"))
        return data.get("benchmark", {}).get("name")
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to read bundle %s: %s", bundle_path, e)
        return None


# ── Matching ──────────────────────────────────────────────────────────


def _match_bundles(
    base: dict[str, Path],
    cand: dict[str, Path],
) -> tuple[dict[str, tuple[Path, Path]], set[str], set[str]]:
    common = set(base) & set(cand)
    matched = {name: (base[name], cand[name]) for name in common}
    return matched, set(base) - common, set(cand) - common


def _check_missing(
    policy: GatePolicy,
    base_bundles: dict[str, Path],
    cand_bundles: dict[str, Path],
) -> set[str]:
    """Required benchmarks missing from either baseline or candidate."""
    required = policy.required_benchmarks()
    available = set(base_bundles) & set(cand_bundles)
    return required - available


# ── Per-benchmark evaluation ──────────────────────────────────────────


def _evaluate_benchmark(
    bench_name: str,
    base_path: Path,
    cand_path: Path,
    policy: ResolvedBenchmarkPolicy,
) -> BenchmarkGateResult:
    """Evaluate a single benchmark against policy thresholds."""
    result = BenchmarkGateResult(
        benchmark=bench_name,
        tier=policy.tier.value,
        direction=policy.direction.value,
        max_drop_threshold=policy.max_drop,
        max_relative_drop_threshold=policy.max_relative_drop,
    )

    # Get supporting evidence from compare_runs
    try:
        reg_report = compare_runs(str(base_path), str(cand_path))
        result.regression_report = reg_report
    except Exception as e:
        logger.warning("compare_runs failed for %s: %s", bench_name, e)
        reg_report = None

    # Load per-item records
    base_records = load_paired_records(base_path.parent)
    cand_records = load_paired_records(cand_path.parent)

    if not base_records or not cand_records:
        result.status = "INSUFFICIENT_EVIDENCE"
        result.reasons.append("Per-sample results (results.jsonl) missing or empty")
        # Fall back to bundle-level score_deltas for informational purposes
        _populate_from_score_deltas(result, reg_report, policy)
        return result

    # Aggregate repeats if requested
    if policy.repeats_aggregation == "mean":
        base_records = _aggregate_repeats(base_records)
        cand_records = _aggregate_repeats(cand_records)

    # Pair items
    paired_keys = sorted(set(base_records) & set(cand_records))
    result.n_paired = len(paired_keys)

    if result.n_paired < _MIN_PAIRED_ITEMS:
        result.status = "INSUFFICIENT_EVIDENCE"
        result.reasons.append(
            f"Only {result.n_paired} paired items (minimum {_MIN_PAIRED_ITEMS})"
        )
        _populate_from_score_deltas(result, reg_report, policy)
        return result

    # Select metric and compute scores from per-item data
    metric = policy.metric or _select_metric(reg_report)
    result.metric = metric

    base_rewards = [float(base_records[k].get("reward", 0)) for k in paired_keys]
    cand_rewards = [float(cand_records[k].get("reward", 0)) for k in paired_keys]

    result.baseline_score = float(np.mean(base_rewards))
    result.candidate_score = float(np.mean(cand_rewards))

    # Compute paired deltas and CI
    deltas = [c - b for b, c in zip(base_rewards, cand_rewards)]
    delta_mean = float(np.mean(deltas))
    result.delta = round(delta_mean, 6)

    if result.baseline_score != 0:
        result.relative_drop_pct = round(100 * delta_mean / result.baseline_score, 2)

    # Normal CI on paired deltas
    ci = _paired_delta_ci(deltas)
    result.delta_ci_lower = ci[0]
    result.delta_ci_upper = ci[1]

    # Apply direction: determine if this is a regression
    is_regression = _is_regression(delta_mean, policy.direction)

    if not is_regression:
        result.status = "PASS"
        result.reasons.append("No regression detected (metric improved or unchanged)")
        return result

    # Regression magnitude (always positive for threshold comparison)
    regression_magnitude = abs(delta_mean)

    # Absolute threshold check
    if regression_magnitude > policy.max_drop:
        result.absolute_breached = True
        result.reasons.append(
            f"Absolute drop {regression_magnitude:.4f} exceeds threshold {policy.max_drop}"
        )

    # Relative threshold check (direction-aware, only on regressions)
    if policy.max_relative_drop is not None and result.baseline_score is not None:
        apply_relative = (
            policy.relative_guard_below is None
            or result.baseline_score < policy.relative_guard_below
        )
        if apply_relative and result.baseline_score != 0:
            relative_drop = regression_magnitude / abs(result.baseline_score)
            if relative_drop > policy.max_relative_drop:
                result.relative_breached = True
                result.reasons.append(
                    f"Relative drop {relative_drop * 100:.1f}% exceeds threshold "
                    f"{policy.max_relative_drop * 100:.1f}% "
                    f"(baseline {result.baseline_score:.4f} < guard {policy.relative_guard_below})"
                )

    if result.absolute_breached or result.relative_breached:
        result.status = "BREACH"
    else:
        result.status = "PASS"
        result.reasons.append("Regression within tolerance")

    return result


def _is_regression(delta: float, direction: Direction) -> bool:
    """Return True if the delta represents a regression given the metric direction."""
    if direction == Direction.higher_is_better:
        return delta < 0  # score went down
    return delta > 0  # score went up (bad for lower-is-better)


def _paired_delta_ci(
    deltas: list[float],
    confidence: float = 0.95,
) -> tuple[float | None, float | None]:
    """Compute normal CI on paired deltas without requiring scipy."""
    n = len(deltas)
    if n < 2:
        return None, None

    arr = np.array(deltas, dtype=np.float64)
    mean = float(arr.mean())
    se = float(arr.std(ddof=1) / math.sqrt(n))

    # z-value for 95% CI (avoid scipy dependency for core gate logic)
    z = 1.96 if confidence == 0.95 else 1.645 if confidence == 0.90 else 1.96

    return round(mean - z * se, 6), round(mean + z * se, 6)


def _select_metric(reg_report: dict[str, Any] | None) -> str:
    """Auto-detect primary metric from score_deltas keys.

    Prefers mean_reward, then pass@1, then first available scorer metric.
    """
    if reg_report is None:
        return "mean_reward"
    score_deltas = reg_report.get("score_deltas", {})
    if "mean_reward" in score_deltas:
        return "mean_reward"
    if "pass@1" in score_deltas:
        return "pass@1"
    for key in score_deltas:
        if key.startswith("scorer:"):
            return key
    return "mean_reward"


def _populate_from_score_deltas(
    result: BenchmarkGateResult,
    reg_report: dict[str, Any] | None,
    policy: ResolvedBenchmarkPolicy,
) -> None:
    """Best-effort: fill result fields from bundle-level score deltas."""
    if reg_report is None:
        return
    metric = policy.metric or _select_metric(reg_report)
    result.metric = metric
    sd = reg_report.get("score_deltas", {}).get(metric, {})
    if sd:
        result.baseline_score = sd.get("baseline")
        result.candidate_score = sd.get("candidate")
        result.delta = sd.get("delta")
        result.relative_drop_pct = sd.get("relative_pct")


# ── Repeat aggregation ────────────────────────────────────────────────


def _aggregate_repeats(
    records: dict[tuple[int, int], dict[str, Any]],
) -> dict[tuple[int, int], dict[str, Any]]:
    """Collapse multiple repeats per problem by averaging reward.

    Returns records keyed by ``(problem_idx, 0)``.
    """
    by_problem: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for (pid, _rep), record in records.items():
        by_problem[pid].append(record)

    aggregated: dict[tuple[int, int], dict[str, Any]] = {}
    for pid, reps in by_problem.items():
        reward = sum(float(r.get("reward", 0)) for r in reps) / len(reps)
        template = reps[0].copy()
        template["reward"] = reward
        template["repeat"] = 0
        template["_aggregated_from"] = len(reps)
        aggregated[(pid, 0)] = template

    return aggregated


# ── Verdict aggregation ───────────────────────────────────────────────


def _aggregate_verdict(
    benchmarks: list[BenchmarkGateResult],
    missing: list[str],
) -> tuple[str, list[str]]:
    """Compute GO / NO-GO / INCONCLUSIVE from per-benchmark results.

    Rules:
      - Any critical/supporting BREACH or MISSING → NO-GO
      - Any critical/supporting INSUFFICIENT_EVIDENCE (none BREACH/MISSING) → INCONCLUSIVE
      - All critical/supporting PASS → GO
      - Advisory benchmarks never affect the verdict
    """
    reasons: list[str] = []

    gated = [b for b in benchmarks if b.tier in ("critical", "supporting")]

    breached = [b for b in gated if b.status == "BREACH"]
    missing_results = [b for b in gated if b.status == "MISSING"]
    insufficient = [b for b in gated if b.status == "INSUFFICIENT_EVIDENCE"]

    if breached or missing_results:
        if breached:
            names = [f"{b.benchmark} [{b.tier}]" for b in breached]
            reasons.append(f"BREACH: {', '.join(names)}")
        if missing_results:
            names = [f"{b.benchmark} [{b.tier}]" for b in missing_results]
            reasons.append(f"MISSING: {', '.join(names)}")
        return "NO-GO", reasons

    if insufficient:
        names = [f"{b.benchmark} [{b.tier}]" for b in insufficient]
        reasons.append(f"INSUFFICIENT_EVIDENCE: {', '.join(names)}")
        return "INCONCLUSIVE", reasons

    passed = [b for b in gated if b.status == "PASS"]
    if passed:
        reasons.append(f"All {len(passed)} gated benchmark(s) passed")
    else:
        reasons.append("No gated benchmarks evaluated")

    return "GO", reasons
