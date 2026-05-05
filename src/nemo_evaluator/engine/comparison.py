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
"""Paired regression analysis: flip report, verdict logic, and comparison orchestration.

Public API
----------
- ``compare_runs``  -- compare two eval bundles (file paths)
- ``compare_results`` -- compare pre-loaded records (in-memory)
- ``build_flip_report`` -- 2x2 contingency table + per-sample flips
- ``load_paired_records`` -- load results.jsonl preserving pairing keys
- ``aggregate_repeats`` -- collapse repeats to mean reward per problem
- ``write_regression`` -- serialize report to JSON
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from nemo_evaluator.metrics.paired_tests import (
    POWER_80_FACTOR,
    SIGNIFICANCE_THRESHOLD,
    McNemarResult,
    PermutationResult,
    SignTestResult,
    detect_test,
    mcnemar_test,
    mde_estimate,
    permutation_test,
    sign_test,
)

logger = logging.getLogger(__name__)

_MIN_EFFECT_SIZE = 0.005  # 0.5% practical threshold
_REPORT_VERSION = 1


# ── Dataclasses (public API) ──────────────────────────────────────────


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
    alpha: float = SIGNIFICANCE_THRESHOLD,
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
        binary rewards -> McNemar, non-binary -> permutation test.
    """
    base = _load_bundle(baseline_path)
    cand = _load_bundle(candidate_path)

    report = RegressionReport(
        baseline={"run_id": base.get("run_id", "unknown"), "config": base.get("config", {})},
        candidate={"run_id": cand.get("run_id", "unknown"), "config": cand.get("config", {})},
        threshold=min_effect,
    )

    # Benchmark name validation
    b_name = base.get("benchmark", {}).get("name")
    c_name = cand.get("benchmark", {}).get("name")
    if b_name and c_name and b_name != c_name:
        report.warnings.append(f"Benchmark mismatch: baseline={b_name!r}, candidate={c_name!r}")

    # Score deltas
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

    # Paired analysis
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
            base_agg = aggregate_repeats(base_records)
            cand_agg = aggregate_repeats(cand_records)

            selected_test = test if test != "auto" else detect_test(base_agg, cand_agg)
            report.test_used = selected_test

            if selected_test == "mcnemar":
                report.test_reason = "all per-problem rewards are binary (0.0 or 1.0)"
                report.flip_report = build_flip_report(
                    base_agg,
                    cand_agg,
                    threshold=reward_threshold,
                )
                report.mcnemar = mcnemar_test(
                    report.flip_report.contingency, report.flip_report.summary.n_paired, alpha=alpha
                )
            else:
                report.test_reason = "per-problem rewards are continuous (N>1 repeats averaged or non-binary scores)"
                report.flip_report = build_flip_report(
                    base_agg,
                    cand_agg,
                    threshold=reward_threshold,
                )
                paired_keys = sorted(set(base_agg) & set(cand_agg))
                paired_deltas = [
                    float(base_agg[k].get("reward", 0)) - float(cand_agg[k].get("reward", 0)) for k in paired_keys
                ]
                if selected_test == "sign":
                    report.sign_test_result = sign_test(paired_deltas, alpha=alpha)
                else:  # permutation
                    report.permutation_result = permutation_test(paired_deltas, alpha=alpha)
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

    report.verdict, report.verdict_reasons = _compute_verdict(report, min_effect)

    return report.to_dict()


def compare_results(
    baseline_records: dict[tuple[int, int], dict[str, Any]],
    candidate_records: dict[tuple[int, int], dict[str, Any]],
    reward_threshold: float = 0.0,
    alpha: float = SIGNIFICANCE_THRESHOLD,
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
        report.mcnemar = mcnemar_test(flip.contingency, flip.summary.n_paired, alpha=alpha)
    else:
        report.test_reason = "per-problem rewards are continuous (N>1 repeats averaged or non-binary scores)"
        paired_keys = sorted(set(base_agg) & set(cand_agg))
        paired_deltas = [float(base_agg[k].get("reward", 0)) - float(cand_agg[k].get("reward", 0)) for k in paired_keys]
        if selected_test == "sign":
            report.sign_test_result = sign_test(paired_deltas, alpha=alpha)
        else:
            report.permutation_result = permutation_test(paired_deltas, alpha=alpha)

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

    all_cats = sorted(set(list(cat_regressions.keys()) + list(cat_improvements.keys())))
    cat_breakdown = (
        {
            cat: {"regressions": cat_regressions.get(cat, 0), "improvements": cat_improvements.get(cat, 0)}
            for cat in all_cats
        }
        if all_cats
        else None
    )

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
            rep, record = entries[0]
            aggregated[(pid, rep)] = record
        else:
            rewards = [float(r.get("reward", 0)) for _, r in entries]
            mean_reward = sum(rewards) / len(rewards)
            _, base_record = sorted(entries, key=lambda x: x[0])[0]
            agg_record = dict(base_record)
            agg_record["reward"] = mean_reward
            agg_record["repeat"] = 0
            agg_record["_aggregated_from_repeats"] = len(entries)
            aggregated[(pid, 0)] = agg_record

    return aggregated


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
        n_discordant = s.n_regressions + s.n_improvements
    elif report.test_used == "sign" and report.sign_test_result:
        sr = report.sign_test_result
        p_value = sr.p_value
        significant = sr.significant
        n_discordant = sr.n_positive + sr.n_negative
        effect_size = (s.n_regressions - s.n_improvements) / s.n_paired if s.n_paired > 0 else None
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

    if n_discordant == 0:
        return "PASS", ["no discordant pairs; baseline and candidate produced identical per-sample results"]

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
        n_needed = max(1, int((POWER_80_FACTOR / min_effect) ** 2))
        reasons.append(
            f"Test underpowered: {n_discordant} discordant pairs can detect "
            f"~{mde * 100:.1f}% regression at 80% power, but practical threshold is {min_effect * 100:.1f}%. "
            f"Need ~{n_needed} discordant pairs to detect {min_effect * 100:.1f}%. "
            f"Increase sample size, add repeats, or use a higher-N proxy benchmark."
        )
        return "INCONCLUSIVE", reasons

    reasons.append(f"{test_label} p={p_value:.4f} (not significant)")
    return "PASS", reasons
