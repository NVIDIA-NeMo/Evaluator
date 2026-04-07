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
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SIGNIFICANCE_THRESHOLD = 0.05


def compare_runs(baseline_path: str | Path, candidate_path: str | Path) -> dict[str, Any]:
    """Compare two eval run bundles. Returns regression report with deltas and significance."""
    base = _load_bundle(baseline_path)
    cand = _load_bundle(candidate_path)

    base_rewards = _load_rewards(Path(baseline_path).parent)
    cand_rewards = _load_rewards(Path(candidate_path).parent)

    report: dict[str, Any] = {
        "baseline": {"run_id": base.get("run_id", "unknown"), "config": base.get("config", {})},
        "candidate": {"run_id": cand.get("run_id", "unknown"), "config": cand.get("config", {})},
        "score_deltas": {},
        "runtime_deltas": {},
    }

    b_scores = base.get("benchmark", {}).get("scores", {})
    c_scores = cand.get("benchmark", {}).get("scores", {})

    for metric in set(list(b_scores.keys()) + list(c_scores.keys())):
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
        entry: dict[str, Any] = {
            "baseline": b_val,
            "candidate": c_val,
            "delta": round(delta, 4),
            "relative_pct": round(100 * delta / b_val, 2) if b_val != 0 else 0,
            "ci_overlap": _ci_overlap(bv, cv),
            "p_value": None,
            "significant": None,
        }

        p = _mann_whitney_p(base_rewards, cand_rewards)
        if p is not None:
            entry["p_value"] = round(p, 6)
            entry["significant"] = p < _SIGNIFICANCE_THRESHOLD

        report["score_deltas"][metric] = entry

    b_rt = b_scores.get("runtime", {})
    c_rt = c_scores.get("runtime", {})
    if isinstance(b_rt, dict) and isinstance(c_rt, dict):
        for k in ["total_tokens", "steps_per_second", "tokens_per_second"]:
            if k in b_rt and k in c_rt:
                try:
                    report["runtime_deltas"][k] = {
                        "baseline": b_rt[k],
                        "candidate": c_rt[k],
                        "delta": round(float(c_rt[k]) - float(b_rt[k]), 2),
                    }
                except (TypeError, ValueError):
                    pass

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
        report["category_deltas"] = cat_deltas

    return report


def _load_bundle(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Bundle not found: {p}")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in bundle {p}: {e}")
    if not isinstance(data, dict):
        raise ValueError(f"Bundle {p} is not a JSON object")
    return data


def _ci_overlap(a: dict, b: dict) -> bool:
    a_lo = a.get("ci_lower")
    a_hi = a.get("ci_upper")
    b_lo = b.get("ci_lower")
    b_hi = b.get("ci_upper")
    if any(v is None for v in (a_lo, a_hi, b_lo, b_hi)):
        return True  # Cannot determine, assume overlap
    return a_lo <= b_hi and b_lo <= a_hi


def _load_rewards(task_dir: Path) -> list[float]:
    """Load per-sample reward values from results.jsonl in the given directory."""
    results_path = task_dir / "results.jsonl"
    if not results_path.exists():
        return []
    rewards = []
    for line in results_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
            r = record.get("reward")
            if r is not None:
                rewards.append(float(r))
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
    return rewards


def _mann_whitney_p(base_rewards: list[float], cand_rewards: list[float]) -> float | None:
    """Compute Mann-Whitney U two-sided p-value. Returns None if scipy is unavailable or data insufficient."""
    if len(base_rewards) < 2 or len(cand_rewards) < 2:
        return None
    try:
        from scipy.stats import mannwhitneyu
    except ImportError:
        logger.debug("scipy not installed; skipping p-value computation (pip install nemo-evaluator[stats])")
        return None
    try:
        _, p = mannwhitneyu(base_rewards, cand_rewards, alternative="two-sided")
        return float(p)
    except ValueError:
        return None


def write_regression(report: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return path
