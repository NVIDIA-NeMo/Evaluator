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
from nemo_evaluator.engine.bundles import (
    discover_bundles,
    match_bundles,
)
from nemo_evaluator.engine.comparison import (
    compare_runs,
    load_paired_records,
)

logger = logging.getLogger(__name__)

_MIN_PAIRED_ITEMS = 10
_GATE_REPORT_VERSION = 1
_SUPPORTED_GATED_METRICS = frozenset({"mean_reward", "pass@1"})


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
        return asdict(self)


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
    policy.validate_for_gate(set(_SUPPORTED_GATED_METRICS))

    base_bundles = discover_bundles(baseline_dir)
    cand_bundles = discover_bundles(candidate_dir)

    report = GateReport()

    # Check for missing required benchmarks in either direction
    missing = _check_missing(policy, base_bundles, cand_bundles)
    report.missing = sorted(missing)
    for name in report.missing:
        report.benchmarks.append(
            BenchmarkGateResult(
                benchmark=name,
                tier=policy.resolve(name).tier.value,
                status="MISSING",
                reasons=[f"Required benchmark {name!r} not found in evaluation results"],
            )
        )

    # Match and evaluate
    matched, unmatched_b, unmatched_c = match_bundles(base_bundles, cand_bundles)

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

    report.verdict, report.verdict_reasons = _aggregate_verdict(report.benchmarks)
    return report


def write_gate_report(report: GateReport, output_path: str | Path) -> Path:
    """Serialize GateReport to JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2, default=str), encoding="utf-8")
    return path


# ── Matching ──────────────────────────────────────────────────────────


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
    except (FileNotFoundError, ValueError, TypeError, KeyError) as e:
        logger.warning("compare_runs failed for %s: %s", bench_name, e)
        reg_report = None

    # Load per-item records
    base_records = load_paired_records(base_path.parent)
    cand_records = load_paired_records(cand_path.parent)

    if not base_records or not cand_records:
        result.status = "INSUFFICIENT_EVIDENCE"
        result.reasons.append("Per-sample results (results.jsonl) missing or empty")
        _populate_from_bundle_scores(result, base_path, cand_path, reg_report, policy)
        return result

    metric = policy.metric or _select_metric(reg_report)
    result.metric = metric

    base_values = _metric_problem_values(base_records, metric)
    cand_values = _metric_problem_values(cand_records, metric)
    if base_values is None or cand_values is None:
        result.status = "INSUFFICIENT_EVIDENCE"
        result.reasons.append(f"Metric {metric!r} is not supported for paired gating")
        _populate_from_bundle_scores(result, base_path, cand_path, reg_report, policy)
        return result

    paired_keys = sorted(set(base_values) & set(cand_values))
    result.n_paired = len(paired_keys)

    if result.n_paired < _MIN_PAIRED_ITEMS:
        result.status = "INSUFFICIENT_EVIDENCE"
        result.reasons.append(f"Only {result.n_paired} paired items (minimum {_MIN_PAIRED_ITEMS})")
        _populate_from_bundle_scores(result, base_path, cand_path, reg_report, policy)
        return result

    base_metric = [base_values[k] for k in paired_keys]
    cand_metric = [cand_values[k] for k in paired_keys]

    result.baseline_score = round(float(np.mean(base_metric)), 6)
    result.candidate_score = round(float(np.mean(cand_metric)), 6)

    deltas = [c - b for b, c in zip(base_metric, cand_metric)]
    delta_mean = float(np.mean(deltas))
    result.delta = round(delta_mean, 6)

    if result.baseline_score != 0:
        result.relative_drop_pct = round(100 * delta_mean / result.baseline_score, 2)

    # CI on paired deltas — use clustered CI when category metadata available
    clusters = _extract_clusters(base_records, paired_keys)
    ci = _paired_delta_ci(deltas, clusters=clusters)
    result.delta_ci_lower = ci[0]
    result.delta_ci_upper = ci[1]

    _apply_thresholds(result, policy, allow_point_estimate=False)
    return result


def _paired_delta_ci(
    deltas: list[float],
    confidence: float = 0.95,
    clusters: list[str] | None = None,
) -> tuple[float | None, float | None]:
    """Compute CI on paired deltas. Uses clustered SE when cluster labels provided."""
    n = len(deltas)
    if n < 2:
        return None, None

    # Try clustered CI if cluster labels are available and non-trivial
    if clusters and len(set(clusters)) > 1:
        try:
            from nemo_evaluator.metrics.confidence import clustered_ci

            ci = clustered_ci(deltas, clusters, confidence=confidence)
            return round(ci.ci_lower, 6), round(ci.ci_upper, 6)
        except ImportError:
            pass  # scipy not installed, fall through to normal CI
        except (ValueError, TypeError) as exc:
            logger.warning("Clustered CI failed: %s; falling back to normal CI", exc)

    arr = np.array(deltas, dtype=np.float64)
    mean = float(arr.mean())
    se = float(arr.std(ddof=1) / math.sqrt(n))

    # Use t-distribution when scipy is available (correct at small N), z-fallback otherwise
    try:
        from scipy.stats import t as t_dist

        z = float(t_dist.ppf(1 - (1 - confidence) / 2, df=n - 1))
    except ImportError:
        z = 1.96 if confidence == 0.95 else 1.645 if confidence == 0.90 else 1.96

    return round(mean - z * se, 6), round(mean + z * se, 6)


def _extract_clusters(
    records: dict[tuple[int, int], dict[str, Any]],
    paired_keys: list[int],
) -> list[str] | None:
    """Extract category/cluster labels for paired problems, if available."""
    by_problem: dict[int, dict[str, Any]] = {}
    for (pid, _repeat), rec in records.items():
        if pid not in by_problem:
            by_problem[pid] = rec
    clusters = []
    for pid in paired_keys:
        record = by_problem.get(pid)
        if record is None:
            return None
        cat = record.get("metadata", {}).get("category") or record.get("scoring_details", {}).get("category")
        if cat is None:
            return None  # all problems must have categories for clustered CI
        clusters.append(str(cat))
    return clusters


def _metric_problem_values(
    records: dict[tuple[int, int], dict[str, Any]],
    metric: str,
) -> dict[int, float] | None:
    """Return per-problem values for a supported gated metric."""
    by_problem: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for (pid, _repeat), record in records.items():
        by_problem[pid].append(record)

    values: dict[int, float] = {}
    if metric == "mean_reward":
        for pid, reps in by_problem.items():
            rewards = [float(r.get("reward", 0.0)) for r in reps]
            values[pid] = float(np.mean(rewards))
        return values

    if metric == "pass@1":
        for pid, reps in by_problem.items():
            successes = sum(1 for r in reps if float(r.get("reward", 0.0)) > 0)
            values[pid] = successes / len(reps) if reps else 0.0
        return values

    return None


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


def _load_bundle_score(bundle_path: Path, metric: str) -> dict[str, Any] | None:
    try:
        data = json.loads(bundle_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to read bundle %s: %s", bundle_path, e)
        return None
    score = data.get("benchmark", {}).get("scores", {}).get(metric)
    return score if isinstance(score, dict) else None


def _populate_from_bundle_scores(
    result: BenchmarkGateResult,
    base_path: Path,
    cand_path: Path,
    reg_report: dict[str, Any] | None,
    policy: ResolvedBenchmarkPolicy,
) -> None:
    metric = policy.metric or _select_metric(reg_report)
    result.metric = metric
    base_score = _load_bundle_score(base_path, metric)
    cand_score = _load_bundle_score(cand_path, metric)

    if not base_score or not cand_score:
        _populate_from_score_deltas(result, reg_report, policy)
        return

    result.baseline_score = base_score.get("value")
    result.candidate_score = cand_score.get("value")
    if result.baseline_score is None or result.candidate_score is None:
        _populate_from_score_deltas(result, reg_report, policy)
        return

    result.delta = round(float(result.candidate_score) - float(result.baseline_score), 6)
    if result.baseline_score != 0:
        result.relative_drop_pct = round(100 * result.delta / result.baseline_score, 2)

    base_lo = base_score.get("ci_lower")
    base_hi = base_score.get("ci_upper")
    cand_lo = cand_score.get("ci_lower")
    cand_hi = cand_score.get("ci_upper")
    if None not in (base_lo, base_hi, cand_lo, cand_hi):
        result.delta_ci_lower = round(float(cand_lo) - float(base_hi), 6)
        result.delta_ci_upper = round(float(cand_hi) - float(base_lo), 6)

    allow_point_estimate = result.tier == "advisory"
    _apply_thresholds(result, policy, allow_point_estimate=allow_point_estimate)


def _populate_from_score_deltas(
    result: BenchmarkGateResult,
    reg_report: dict[str, Any] | None,
    policy: ResolvedBenchmarkPolicy,
) -> None:
    """Final fallback when bundle score entries are unavailable."""
    if reg_report is None:
        return
    metric = policy.metric or _select_metric(reg_report)
    result.metric = metric
    sd = reg_report.get("score_deltas", {}).get(metric, {})
    if not sd:
        return
    result.baseline_score = sd.get("baseline")
    result.candidate_score = sd.get("candidate")
    result.delta = sd.get("delta")
    result.relative_drop_pct = sd.get("relative_pct")
    allow_point_estimate = result.tier == "advisory"
    _apply_thresholds(result, policy, allow_point_estimate=allow_point_estimate)


def _damage_from_delta(delta: float, direction: Direction) -> float:
    return -delta if direction == Direction.higher_is_better else delta


def _damage_interval(
    delta_ci_lower: float | None,
    delta_ci_upper: float | None,
    direction: Direction,
) -> tuple[float | None, float | None]:
    if delta_ci_lower is None or delta_ci_upper is None:
        return None, None
    if direction == Direction.higher_is_better:
        return round(-delta_ci_upper, 6), round(-delta_ci_lower, 6)
    return delta_ci_lower, delta_ci_upper


def _apply_thresholds(
    result: BenchmarkGateResult,
    policy: ResolvedBenchmarkPolicy,
    *,
    allow_point_estimate: bool,
) -> None:
    """Apply absolute and relative thresholds to the selected metric evidence."""
    if result.delta is None:
        if not result.reasons:
            result.reasons.append("No metric delta available for gate decision")
        return

    damage = _damage_from_delta(float(result.delta), policy.direction)
    damage_ci_lower, damage_ci_upper = _damage_interval(
        result.delta_ci_lower,
        result.delta_ci_upper,
        policy.direction,
    )

    if policy.max_relative_drop is not None and result.baseline_score not in (None, 0) and damage > 0:
        apply_relative = (
            policy.relative_guard_below is None or float(result.baseline_score) < policy.relative_guard_below
        )
        if apply_relative:
            relative_drop = damage / abs(float(result.baseline_score))
            if relative_drop > policy.max_relative_drop:
                result.relative_breached = True
                result.status = "BREACH"
                result.reasons.append(
                    f"Relative drop {relative_drop * 100:.1f}% exceeds threshold {policy.max_relative_drop * 100:.1f}%"
                )
                return

    if damage_ci_lower is not None and damage_ci_upper is not None:
        if damage_ci_lower > policy.max_drop:
            result.absolute_breached = True
            result.status = "BREACH"
            result.reasons.append(
                f"95% CI on damage [{damage_ci_lower:.4f}, {damage_ci_upper:.4f}] "
                f"exceeds threshold {policy.max_drop:.4f}"
            )
            return
        if damage_ci_upper <= policy.max_drop:
            result.status = "PASS"
            result.reasons.append(
                f"95% CI on damage [{damage_ci_lower:.4f}, {damage_ci_upper:.4f}] "
                f"is within threshold {policy.max_drop:.4f}"
            )
            return
        result.status = "INSUFFICIENT_EVIDENCE"
        result.reasons.append(
            f"95% CI on damage [{damage_ci_lower:.4f}, {damage_ci_upper:.4f}] straddles threshold {policy.max_drop:.4f}"
        )
        return

    if not allow_point_estimate:
        result.status = "INSUFFICIENT_EVIDENCE"
        result.reasons.append("No confidence interval available for required benchmark")
        return

    if damage > policy.max_drop:
        result.absolute_breached = True
        result.status = "BREACH"
        result.reasons.append(f"Point estimate damage {damage:.4f} exceeds threshold {policy.max_drop:.4f}")
        return

    result.status = "PASS"
    result.reasons.append(f"Point estimate damage {damage:.4f} is within threshold {policy.max_drop:.4f}")


# ── Verdict aggregation ───────────────────────────────────────────────


def _aggregate_verdict(
    benchmarks: list[BenchmarkGateResult],
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
        reasons.append("Action: re-run with more items, add repeats, or relax thresholds for these benchmarks.")
        return "INCONCLUSIVE", reasons

    passed = [b for b in gated if b.status == "PASS"]
    if passed:
        reasons.append(f"All {len(passed)} gated benchmark(s) passed")
    else:
        reasons.append("No gated benchmarks evaluated")

    return "GO", reasons
