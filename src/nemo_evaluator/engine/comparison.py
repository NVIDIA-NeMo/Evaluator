"""Paired regression analysis: McNemar test, sign test, permutation test, flip report, and verdict logic.

Public API
----------
- ``compare_runs``  — compare two eval bundles (file paths)
- ``compare_results`` — compare pre-loaded records (in-memory)
- ``build_flip_report`` — 2x2 contingency table + per-sample flips
- ``mcnemar_test`` — one-sided McNemar's exact test for degradation (N=1 binary)
- ``sign_test`` — one-sided sign test for degradation (N>1 averaged to continuous)
- ``permutation_test`` — permutation test on paired differences (continuous scores)
- ``detect_test`` — auto-detect appropriate test from data shape
- ``write_regression`` — serialize report to JSON
"""

from __future__ import annotations

import json
import logging
import math
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

logger = logging.getLogger(__name__)

_SIGNIFICANCE_THRESHOLD = 0.05
_MIN_EFFECT_SIZE = 0.005  # 0.5% practical threshold
_REPORT_VERSION = 1


# ── Dataclasses (public API) ──────────────────────────────────────────


@dataclass
class McNemarResult:
    p_value: float | None = None
    significant: bool | None = None
    method: str | None = None
    n_discordant: int = 0
    effect_size: float | None = None  # (b - c) / n_paired
    ci_lower: float | None = None  # 95% CI on regression rate difference
    ci_upper: float | None = None
    hypothesis: str = "one-sided (degradation)"

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class FlipEntry:
    problem_idx: int
    repeat: int
    expected_answer: str | None = None
    baseline_reward: float = 0.0
    candidate_reward: float = 0.0
    category: str | None = None
    baseline_response: str | None = None
    candidate_response: str | None = None
    scoring_details: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None}


@dataclass
class FlipSummary:
    n_paired: int = 0
    n_regressions: int = 0
    n_improvements: int = 0
    n_stable_correct: int = 0
    n_stable_wrong: int = 0
    regression_rate: float | None = None  # n_regressions / n_paired
    improvement_rate: float | None = None
    category_breakdown: dict[str, dict[str, int]] | None = None


@dataclass
class FlipReport:
    contingency: dict[str, int] = field(default_factory=dict)
    regressions: list[FlipEntry] = field(default_factory=list)
    improvements: list[FlipEntry] = field(default_factory=list)
    summary: FlipSummary = field(default_factory=FlipSummary)


@dataclass
class RegressionReport:
    """Typed regression report. Use .to_dict() for JSON serialization."""
    report_version: int = _REPORT_VERSION
    baseline: dict[str, Any] = field(default_factory=dict)
    candidate: dict[str, Any] = field(default_factory=dict)
    score_deltas: dict[str, Any] = field(default_factory=dict)
    runtime_deltas: dict[str, Any] = field(default_factory=dict)
    category_deltas: dict[str, Any] | None = None
    flip_report: FlipReport | None = None
    mcnemar: McNemarResult | None = None
    sign_test_result: SignTestResult | None = None
    permutation_result: PermutationResult | None = None
    test_used: str | None = None  # "mcnemar", "sign", "permutation"
    test_reason: str | None = None  # why this test was selected
    verdict: str | None = None  # PASS / WARN / BLOCK / INCONCLUSIVE
    verdict_reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    threshold: float = _MIN_EFFECT_SIZE  # user's --max-drop value

    def is_regression(self) -> bool:
        if self.mcnemar and self.mcnemar.significant:
            s = self.flip_report.summary if self.flip_report else None
            if s and s.n_regressions > s.n_improvements:
                return True
        return False

    def worst_category(self) -> str | None:
        if not self.category_deltas:
            return None
        return min(self.category_deltas, key=lambda c: self.category_deltas[c].get("delta", 0))

    def flip_list(self, direction: str = "regressions") -> list[FlipEntry]:
        if not self.flip_report:
            return []
        return self.flip_report.regressions if direction == "regressions" else self.flip_report.improvements

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "report_version": self.report_version,
            "baseline": self.baseline,
            "candidate": self.candidate,
            "score_deltas": self.score_deltas,
            "runtime_deltas": self.runtime_deltas,
            "verdict": self.verdict,
            "verdict_reasons": self.verdict_reasons,
            "threshold": self.threshold,
        }
        if self.warnings:
            d["warnings"] = self.warnings
        if self.category_deltas:
            d["category_deltas"] = self.category_deltas
        if self.flip_report:
            d["flip_report"] = {
                "contingency": self.flip_report.contingency,
                "regressions": [f.to_dict() for f in self.flip_report.regressions],
                "improvements": [f.to_dict() for f in self.flip_report.improvements],
                "summary": asdict(self.flip_report.summary),
            }
        if self.mcnemar:
            d["mcnemar"] = self.mcnemar.to_dict()
        if self.sign_test_result:
            d["sign_test"] = self.sign_test_result.to_dict()
        if self.permutation_result:
            d["permutation_test"] = self.permutation_result.to_dict()
        if self.test_used:
            d["test_used"] = self.test_used
        if self.test_reason:
            d["test_reason"] = self.test_reason
        return d


# ── Public API ─────────────────────────────────────────────────────────


def compare_runs(
    baseline_path: str | Path,
    candidate_path: str | Path,
    reward_threshold: float = 0.0,
    alpha: float = _SIGNIFICANCE_THRESHOLD,
    min_effect: float = _MIN_EFFECT_SIZE,
    test: Literal["auto", "mcnemar", "sign", "permutation"] = "auto",
) -> dict[str, Any]:
    """Compare two eval run bundles. Returns regression report as dict.

    Parameters
    ----------
    baseline_path, candidate_path:
        Paths to eval-*.json bundle files.
    reward_threshold:
        Reward above this value counts as 'correct' for flip analysis.
    alpha:
        Significance level for statistical test (default 0.05).
    min_effect:
        Minimum practical effect size for BLOCK verdict (default 0.005 = 0.5%).
    test:
        Statistical test to use. "auto" detects from data shape:
        binary rewards → McNemar, non-binary → permutation test.
    """
    base = _load_bundle(baseline_path)
    cand = _load_bundle(candidate_path)

    report = RegressionReport(
        baseline={"run_id": base.get("run_id", "unknown"), "config": base.get("config", {})},
        candidate={"run_id": cand.get("run_id", "unknown"), "config": cand.get("config", {})},
        threshold=min_effect,
    )

    # Benchmark name validation (#17)
    b_name = base.get("benchmark", {}).get("name")
    c_name = cand.get("benchmark", {}).get("name")
    if b_name and c_name and b_name != c_name:
        report.warnings.append(f"Benchmark mismatch: baseline={b_name!r}, candidate={c_name!r}")

    # Score deltas (remove Mann-Whitney — item #5, replaced by McNemar)
    b_scores = base.get("benchmark", {}).get("scores", {})
    c_scores = cand.get("benchmark", {}).get("scores", {})
    for metric in sorted(set(list(b_scores.keys()) + list(c_scores.keys()))):
        bv = b_scores.get(metric, {})
        cv = c_scores.get(metric, {})
        if not (isinstance(bv, dict) and isinstance(cv, dict)):
            continue
        b_val = bv.get("value")
        c_val = cv.get("value")
        if b_val is None or c_val is None:
            continue
        try:
            b_val = float(b_val)
            c_val = float(c_val)
        except (TypeError, ValueError):
            continue

        delta = c_val - b_val
        report.score_deltas[metric] = {
            "baseline": b_val,
            "candidate": c_val,
            "delta": round(delta, 4),
            "relative_pct": round(100 * delta / b_val, 2) if b_val != 0 else 0,
            "ci_overlap": _ci_overlap(bv, cv),
        }

    # Runtime deltas
    b_rt = b_scores.get("runtime", {})
    c_rt = c_scores.get("runtime", {})
    if isinstance(b_rt, dict) and isinstance(c_rt, dict):
        for k in ["total_tokens", "steps_per_second", "tokens_per_second"]:
            if k in b_rt and k in c_rt:
                try:
                    report.runtime_deltas[k] = {
                        "baseline": b_rt[k],
                        "candidate": c_rt[k],
                        "delta": round(float(c_rt[k]) - float(b_rt[k]), 2),
                    }
                except (TypeError, ValueError):
                    pass

    # Category deltas
    b_cats = base.get("benchmark", {}).get("categories", {})
    c_cats = cand.get("benchmark", {}).get("categories", {})
    if b_cats or c_cats:
        cat_deltas = {}
        all_cats = (
            set(list(b_cats.keys()) + list(c_cats.keys()))
            if isinstance(b_cats, dict) and isinstance(c_cats, dict)
            else set()
        )
        for cat in all_cats:
            bm = b_cats.get(cat, {}).get("mean_reward", 0)
            cm = c_cats.get(cat, {}).get("mean_reward", 0)
            cat_deltas[cat] = {"baseline": bm, "candidate": cm, "delta": round(cm - bm, 4)}
        report.category_deltas = cat_deltas

    # Paired analysis (#2: warn when missing)
    base_dir = Path(baseline_path).parent
    cand_dir = Path(candidate_path).parent
    base_has_results = (base_dir / "results.jsonl").exists()
    cand_has_results = (cand_dir / "results.jsonl").exists()

    if not base_has_results or not cand_has_results:
        missing = []
        if not base_has_results:
            missing.append(f"baseline ({base_dir / 'results.jsonl'})")
        if not cand_has_results:
            missing.append(f"candidate ({cand_dir / 'results.jsonl'})")
        report.warnings.append(
            f"Per-sample results missing: {', '.join(missing)}. "
            "McNemar paired test and flip analysis skipped. "
            "Only aggregate score deltas are available."
        )
    else:
        base_records = load_paired_records(base_dir)
        cand_records = load_paired_records(cand_dir)
        if base_records and cand_records:
            # Aggregate repeats before analysis (mean reward per problem)
            base_agg = aggregate_repeats(base_records)
            cand_agg = aggregate_repeats(cand_records)

            # Auto-detect or use explicit test selection
            selected_test = test if test != "auto" else detect_test(base_agg, cand_agg)
            report.test_used = selected_test

            if selected_test == "mcnemar":
                report.test_reason = "all per-problem rewards are binary (0.0 or 1.0)"
                report.flip_report = build_flip_report(
                    base_agg, cand_agg, threshold=reward_threshold,
                )
                report.mcnemar = mcnemar_test(report.flip_report.contingency, report.flip_report.summary.n_paired)
            else:
                report.test_reason = "per-problem rewards are continuous (N>1 repeats averaged or non-binary scores)"
                # Build flip report for display (still useful for category breakdown)
                report.flip_report = build_flip_report(
                    base_agg, cand_agg, threshold=reward_threshold,
                )
                # Compute paired deltas for sign/permutation test
                paired_keys = sorted(set(base_agg) & set(cand_agg))
                paired_deltas = [
                    float(base_agg[k].get("reward", 0)) - float(cand_agg[k].get("reward", 0))
                    for k in paired_keys
                ]
                if selected_test == "sign":
                    report.sign_test_result = sign_test(paired_deltas)
                else:  # permutation
                    report.permutation_result = permutation_test(paired_deltas)
        elif not base_records or not cand_records:
            empty = []
            if not base_records:
                empty.append(f"baseline ({base_dir / 'results.jsonl'})")
            if not cand_records:
                empty.append(f"candidate ({cand_dir / 'results.jsonl'})")
            report.warnings.append(
                f"Per-sample results empty or unparseable: {', '.join(empty)}. "
                "McNemar paired test and flip analysis skipped. "
                "Only aggregate score deltas are available."
            )

    # Verdict logic (#4: effect size gating, #27: INCONCLUSIVE)
    report.verdict, report.verdict_reasons = _compute_verdict(report, min_effect)

    return report.to_dict()


def compare_results(
    baseline_records: dict[tuple[int, int], dict[str, Any]],
    candidate_records: dict[tuple[int, int], dict[str, Any]],
    reward_threshold: float = 0.0,
    alpha: float = _SIGNIFICANCE_THRESHOLD,
    min_effect: float = _MIN_EFFECT_SIZE,
    test: Literal["auto", "mcnemar", "sign", "permutation"] = "auto",
) -> dict[str, Any]:
    """Compare pre-loaded paired records in memory (no file I/O).

    Parameters are the same as ``compare_runs`` but accept dicts keyed by
    ``(problem_idx, repeat)`` instead of file paths.  Useful for library
    integration (e.g. ``modelopt[eval]``).
    """
    base_agg = aggregate_repeats(baseline_records)
    cand_agg = aggregate_repeats(candidate_records)

    report = RegressionReport()
    selected_test = test if test != "auto" else detect_test(base_agg, cand_agg)
    report.test_used = selected_test

    flip = build_flip_report(base_agg, cand_agg, threshold=reward_threshold)
    report.flip_report = flip

    if selected_test == "mcnemar":
        report.test_reason = "all per-problem rewards are binary (0.0 or 1.0)"
        report.mcnemar = mcnemar_test(flip.contingency, flip.summary.n_paired)
    else:
        report.test_reason = "per-problem rewards are continuous (N>1 repeats averaged or non-binary scores)"
        paired_keys = sorted(set(base_agg) & set(cand_agg))
        paired_deltas = [
            float(base_agg[k].get("reward", 0)) - float(cand_agg[k].get("reward", 0))
            for k in paired_keys
        ]
        if selected_test == "sign":
            report.sign_test_result = sign_test(paired_deltas)
        else:
            report.permutation_result = permutation_test(paired_deltas)

    report.verdict, report.verdict_reasons = _compute_verdict(report, min_effect)
    return report.to_dict()


def load_paired_records(task_dir: Path) -> dict[tuple[int, int], dict[str, Any]]:
    """Load results.jsonl preserving the (problem_idx, repeat) pairing key."""
    results_path = task_dir / "results.jsonl"
    if not results_path.exists():
        return {}
    records: dict[tuple[int, int], dict[str, Any]] = {}
    for line in results_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
            pid = record.get("problem_idx")
            rep = record.get("repeat", 0)
            if pid is not None:
                records[(int(pid), int(rep))] = record
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
    return records


def build_flip_report(
    base_records: dict[tuple[int, int], dict[str, Any]],
    cand_records: dict[tuple[int, int], dict[str, Any]],
    threshold: float = 0.0,
) -> FlipReport:
    """Build a 2x2 contingency table and per-sample flip lists from paired results."""
    paired_keys = sorted(set(base_records) & set(cand_records))

    both_correct = 0
    baseline_only = 0
    candidate_only = 0
    both_wrong = 0
    regressions: list[FlipEntry] = []
    improvements: list[FlipEntry] = []
    cat_regressions: Counter = Counter()
    cat_improvements: Counter = Counter()

    for key in paired_keys:
        br = base_records[key]
        cr = cand_records[key]
        b_ok = float(br.get("reward", 0)) > threshold
        c_ok = float(cr.get("reward", 0)) > threshold

        entry = _make_flip_entry(key, br, cr)

        if b_ok and c_ok:
            both_correct += 1
        elif b_ok and not c_ok:
            baseline_only += 1
            regressions.append(entry)
            if entry.category:
                cat_regressions[entry.category] += 1
        elif not b_ok and c_ok:
            candidate_only += 1
            improvements.append(entry)
            if entry.category:
                cat_improvements[entry.category] += 1
        else:
            both_wrong += 1

    n_paired = len(paired_keys)

    # Category breakdown (#13)
    all_cats = sorted(set(list(cat_regressions.keys()) + list(cat_improvements.keys())))
    cat_breakdown = {
        cat: {"regressions": cat_regressions.get(cat, 0), "improvements": cat_improvements.get(cat, 0)}
        for cat in all_cats
    } if all_cats else None

    return FlipReport(
        contingency={
            "both_correct": both_correct,
            "baseline_only_correct": baseline_only,
            "candidate_only_correct": candidate_only,
            "both_wrong": both_wrong,
        },
        regressions=regressions,
        improvements=improvements,
        summary=FlipSummary(
            n_paired=n_paired,
            n_regressions=baseline_only,
            n_improvements=candidate_only,
            n_stable_correct=both_correct,
            n_stable_wrong=both_wrong,
            regression_rate=round(baseline_only / n_paired, 4) if n_paired else None,
            improvement_rate=round(candidate_only / n_paired, 4) if n_paired else None,
            category_breakdown=cat_breakdown,
        ),
    )


def mcnemar_test(
    contingency: dict[str, int],
    n_paired: int = 0,
) -> McNemarResult:
    """One-sided McNemar's test for degradation (H1: regressions > improvements).

    Uses exact binomial test for all sample sizes (item #24: drop chi-squared).
    """
    b = contingency["baseline_only_correct"]  # regressions
    c = contingency["candidate_only_correct"]  # improvements
    n_discordant = b + c

    # Effect size: net regression rate over all paired samples (#3 from statistician)
    effect_size = round((b - c) / n_paired, 6) if n_paired > 0 else None

    # 95% CI on effect size using Wald interval (#25)
    ci_lower = ci_upper = None
    if n_paired > 0 and n_discordant > 0:
        p_b = b / n_paired
        p_c = c / n_paired
        se = math.sqrt((p_b + p_c - (p_b - p_c) ** 2) / n_paired) if n_paired > 1 else 0
        if se > 0:
            ci_lower = round((p_b - p_c) - 1.96 * se, 6)
            ci_upper = round((p_b - p_c) + 1.96 * se, 6)

    result = McNemarResult(
        n_discordant=n_discordant,
        effect_size=effect_size,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
    )

    if n_discordant == 0:
        result.p_value = 1.0
        result.significant = False
        result.method = "exact"
        return result

    try:
        # Item #1: one-sided test (alternative="greater") per RFC-0004
        # Item #24: use exact binomial for all n (drop chi-squared branch)
        from scipy.stats import binomtest

        res = binomtest(b, n_discordant, 0.5, alternative="greater")
        result.p_value = round(float(res.pvalue), 6)
        result.method = "exact_binomial"
        result.significant = result.p_value < _SIGNIFICANCE_THRESHOLD
    except ImportError:
        logger.debug("scipy not installed; skipping McNemar test (pip install nemo-evaluator[stats])")

    return result


@dataclass
class SignTestResult:
    """Result of one-sided sign test on paired differences."""
    n_positive: int = 0  # baseline > candidate (regressions)
    n_negative: int = 0  # candidate > baseline (improvements)
    n_ties: int = 0
    p_value: float | None = None
    significant: bool | None = None
    method: str = "sign_test"

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class PermutationResult:
    """Result of permutation test on paired differences."""
    observed_mean_diff: float = 0.0
    p_value: float | None = None
    significant: bool | None = None
    n_permutations: int = 10_000
    effect_size: float | None = None  # Cohen's d paired
    method: str = "permutation"

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


def sign_test(
    paired_deltas: list[float],
) -> SignTestResult:
    """One-sided sign test for degradation (H1: baseline > candidate).

    Counts positive vs negative paired differences, ignoring ties (zeros).
    This is the correct generalization of McNemar to continuous data —
    when N>1 repeats are averaged, scores become continuous and McNemar
    no longer applies.
    """
    n_positive = sum(1 for d in paired_deltas if d > 0)  # regressions
    n_negative = sum(1 for d in paired_deltas if d < 0)  # improvements
    n_ties = sum(1 for d in paired_deltas if d == 0)

    result = SignTestResult(
        n_positive=n_positive,
        n_negative=n_negative,
        n_ties=n_ties,
    )

    n_non_tied = n_positive + n_negative
    if n_non_tied == 0:
        result.p_value = 1.0
        result.significant = False
        return result

    try:
        from scipy.stats import binomtest

        res = binomtest(n_positive, n_non_tied, 0.5, alternative="greater")
        result.p_value = round(float(res.pvalue), 6)
        result.significant = result.p_value < _SIGNIFICANCE_THRESHOLD
    except ImportError:
        logger.debug("scipy not installed; skipping sign test (pip install nemo-evaluator[stats])")

    return result


def permutation_test(
    paired_deltas: list[float],
    n_permutations: int = 10_000,
    seed: int = 42,
) -> PermutationResult:
    """One-sided permutation test on paired differences (H1: mean diff > 0 = baseline better).

    Under H0, the signs of the paired differences are exchangeable.
    Randomly flip signs n_permutations times, compare observed mean to
    the permutation distribution. Uses only numpy (no scipy required).
    """
    deltas = np.array(paired_deltas, dtype=np.float64)
    observed_mean = float(deltas.mean())
    n = len(deltas)

    if n == 0:
        return PermutationResult(observed_mean_diff=0.0, p_value=1.0, significant=False)

    # Cohen's d for paired samples
    std = float(deltas.std(ddof=1)) if n > 1 else 0.0
    effect_size = round(observed_mean / std, 6) if std > 0 else None

    rng = np.random.default_rng(seed)
    # Generate random sign flips: shape (n_permutations, n)
    signs = rng.choice([-1, 1], size=(n_permutations, n))
    perm_means = (signs * deltas).mean(axis=1)

    # One-sided: fraction of permuted means >= observed mean
    p_value = float((perm_means >= observed_mean).sum() + 1) / (n_permutations + 1)

    return PermutationResult(
        observed_mean_diff=round(observed_mean, 6),
        p_value=round(p_value, 6),
        significant=p_value < _SIGNIFICANCE_THRESHOLD,
        n_permutations=n_permutations,
        effect_size=effect_size,
    )


def detect_test(
    base_records: dict[tuple[int, int], dict[str, Any]],
    cand_records: dict[tuple[int, int], dict[str, Any]],
) -> Literal["mcnemar", "sign", "permutation"]:
    """Auto-detect the appropriate statistical test based on data shape.

    - All rewards are 0.0 or 1.0 → mcnemar (N=1 binary)
    - Any non-binary rewards → permutation (N>1 averaged or continuous scores)
    """
    paired_keys = set(base_records) & set(cand_records)
    for key in paired_keys:
        b_reward = float(base_records[key].get("reward", 0))
        c_reward = float(cand_records[key].get("reward", 0))
        if b_reward not in (0.0, 1.0) or c_reward not in (0.0, 1.0):
            return "permutation"
    return "mcnemar"


def aggregate_repeats(
    records: dict[tuple[int, int], dict[str, Any]],
) -> dict[tuple[int, int], dict[str, Any]]:
    """Aggregate multiple repeats per problem_idx by mean reward.

    If a problem has repeats (0, 1, 2, ...), collapse to a single entry
    keyed by (problem_idx, 0) with the mean reward. Preserves metadata
    from the first repeat.
    """
    from collections import defaultdict

    by_problem: dict[int, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for (pid, rep), record in records.items():
        by_problem[pid].append((rep, record))

    aggregated: dict[tuple[int, int], dict[str, Any]] = {}
    for pid, entries in by_problem.items():
        if len(entries) == 1:
            # Single repeat — pass through unchanged
            rep, record = entries[0]
            aggregated[(pid, rep)] = record
        else:
            # Multiple repeats — average rewards
            rewards = [float(r.get("reward", 0)) for _, r in entries]
            mean_reward = sum(rewards) / len(rewards)
            # Use first repeat's metadata as the base
            _, base_record = sorted(entries, key=lambda x: x[0])[0]
            agg_record = dict(base_record)
            agg_record["reward"] = mean_reward
            agg_record["repeat"] = 0
            agg_record["_aggregated_from_repeats"] = len(entries)
            aggregated[(pid, 0)] = agg_record

    return aggregated


def mde_estimate(n_discordant: int) -> float:
    """Minimum detectable effect at 80% power for one-sided binomial."""
    if n_discordant <= 0:
        return 1.0
    return 2.8 / math.sqrt(max(1, n_discordant))


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
        return (f"All capabilities held. {n_reg} problem(s) flipped but within normal variation "
                f"for {n_paired} samples.")

    parts = []
    if held:
        parts.append(f"Safe for {', '.join(held)}.")

    cat_bd = summary.get("category_breakdown", {})
    for cat in broke:
        delta_pct = abs(category_deltas[cat]["delta"]) * 100
        cat_reg = cat_bd.get(cat, {}).get("regressions", 0)
        parts.append(f"{cat} regresses {delta_pct:.1f}% ({cat_reg} problems).")

    return " ".join(parts) if parts else None


def write_regression(report: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return path


# ── Private helpers ────────────────────────────────────────────────────


def _load_bundle(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Bundle not found: {p}")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in bundle {p}: {e}") from e
    if not isinstance(data, dict):
        raise TypeError(f"Bundle {p} is not a JSON object")
    return data


def _ci_overlap(a: dict, b: dict) -> bool:
    a_lo = a.get("ci_lower")
    a_hi = a.get("ci_upper")
    b_lo = b.get("ci_lower")
    b_hi = b.get("ci_upper")
    if any(v is None for v in (a_lo, a_hi, b_lo, b_hi)):
        return True
    return a_lo <= b_hi and b_lo <= a_hi


def _make_flip_entry(
    key: tuple[int, int],
    base_record: dict[str, Any],
    cand_record: dict[str, Any],
) -> FlipEntry:
    meta = base_record.get("metadata", {})
    category = meta.get("category") or base_record.get("scoring_details", {}).get("category")

    # Item #23: include truncated model response
    base_resp = base_record.get("model_response") or base_record.get("extracted_answer")
    cand_resp = cand_record.get("model_response") or cand_record.get("extracted_answer")
    if isinstance(base_resp, str) and len(base_resp) > 80:
        base_resp = base_resp[:77] + "..."
    if isinstance(cand_resp, str) and len(cand_resp) > 80:
        cand_resp = cand_resp[:77] + "..."

    return FlipEntry(
        problem_idx=key[0],
        repeat=key[1],
        expected_answer=base_record.get("expected_answer"),
        baseline_reward=float(base_record.get("reward", 0)),
        candidate_reward=float(cand_record.get("reward", 0)),
        category=category,
        baseline_response=base_resp if isinstance(base_resp, str) else None,
        candidate_response=cand_resp if isinstance(cand_resp, str) else None,
        scoring_details=base_record.get("scoring_details"),
    )


def _compute_verdict(
    report: RegressionReport,
    min_effect: float,
) -> tuple[str, list[str]]:
    """Compute PASS / WARN / BLOCK / INCONCLUSIVE verdict.

    Dispatches to the appropriate test result based on report.test_used.
    BLOCK requires p < alpha AND |effect| > min_effect.
    INCONCLUSIVE when test lacks power to detect min_effect.
    """
    s = report.flip_report.summary if report.flip_report else None
    if s is None:
        return "INCONCLUSIVE", ["no paired data available; cannot determine regression status"]

    # Extract p-value, significance, and effect size from whichever test was used
    p_value: float | None = None
    significant: bool | None = None
    effect_size: float | None = None
    test_label = report.test_used or "mcnemar"
    n_discordant = 0

    if report.test_used == "permutation" and report.permutation_result:
        pr = report.permutation_result
        p_value = pr.p_value
        significant = pr.significant
        effect_size = pr.effect_size
        # For permutation, all non-zero deltas are "discordant"
        n_discordant = s.n_regressions + s.n_improvements
    elif report.test_used == "sign" and report.sign_test_result:
        sr = report.sign_test_result
        p_value = sr.p_value
        significant = sr.significant
        # Effect size from flip summary
        effect_size = report.mcnemar.effect_size if report.mcnemar else None
        n_discordant = sr.n_positive + sr.n_negative
    elif report.mcnemar:
        m = report.mcnemar
        p_value = m.p_value
        significant = m.significant
        effect_size = m.effect_size
        n_discordant = m.n_discordant
    else:
        return "INCONCLUSIVE", ["no test result available; cannot determine regression status"]

    if p_value is None:
        return "INCONCLUSIVE", ["scipy not installed; statistical test skipped (install nemo-evaluator[stats])"]

    # Identical results
    if n_discordant == 0:
        return "PASS", ["no discordant pairs; baseline and candidate produced identical per-sample results"]

    # Power check: INCONCLUSIVE if test can't detect regressions within 2x the threshold
    mde = mde_estimate(n_discordant)
    underpowered = mde > min_effect * 2

    reasons: list[str] = []

    if significant and effect_size is not None and abs(effect_size) > min_effect:
        if s.n_regressions > s.n_improvements:
            reasons.append(f"{test_label} p={p_value:.4f}, effect={effect_size:.4f} > {min_effect}")
            return "BLOCK", reasons

    if significant:
        reasons.append(f"{test_label} p={p_value:.4f} (significant) but effect below practical threshold")
        return "WARN", reasons

    if underpowered and s.n_paired > 0:
        n_needed = max(1, int((2.8 / min_effect) ** 2))
        reasons.append(
            f"Test underpowered: {n_discordant} discordant pairs can detect "
            f"~{mde * 100:.1f}% regression at 80% power, but practical threshold is {min_effect * 100:.1f}%. "
            f"Need ~{n_needed} discordant pairs to detect {min_effect * 100:.1f}%. "
            f"Increase sample size, add repeats, or use a higher-N proxy benchmark."
        )
        return "INCONCLUSIVE", reasons

    reasons.append(f"{test_label} p={p_value:.4f} (not significant)")
    return "PASS", reasons
