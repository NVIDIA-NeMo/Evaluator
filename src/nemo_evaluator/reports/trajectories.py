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
"""Trajectory coverage / audit report.

Post-processes ``trajectories.jsonl`` (per-trial trajectory native data) and
``model_traffic.jsonl`` (per-call wire captures, gchlebus's MR #117) to
produce a single ``trajectories_report.json`` describing:

  * dataset counts (trials / problems / repeats)
  * score reconciliation (mean_reward in trajectories vs reported in eval-*.json)
  * trajectory_native field coverage (which ATIF-v1.6 fields are present)
  * wire_captures summary (counts, per-session distribution, finish_reasons)
  * step <-> capture mismatches and token reconciliation
  * per-trial ATIF presence audit

Pure file-to-file. No adapter, no in-memory store, no eval-loop coupling.
"""

from __future__ import annotations

import json
import logging
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * pct
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    if lo == hi:
        return float(s[lo])
    return float(s[lo] + (s[hi] - s[lo]) * (k - lo))


def _stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "p10": None, "p90": None, "max": None}
    return {
        "mean": round(statistics.fmean(values), 4),
        "median": round(statistics.median(values), 4),
        "p10": _percentile(values, 0.10),
        "p90": _percentile(values, 0.90),
        "max": round(max(values), 4),
    }


def _agent_steps(row: dict[str, Any]) -> list[dict[str, Any]]:
    traj = row.get("trajectory") or []
    if not traj:
        return []
    steps = traj[0].get("steps") or []
    return [s for s in steps if isinstance(s, dict) and s.get("source") == "agent"]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("trajectories_report: skipping invalid JSON line in %s", path)
    return rows


def _load_eval_summary(bench_dir: Path) -> dict[str, Any] | None:
    """Pick the most recent ``eval-*.json`` bundle's score summary, if any."""
    bundles = sorted(bench_dir.glob("eval-*.json"))
    if not bundles:
        return None
    try:
        data = json.loads(bundles[-1].read_text(encoding="utf-8"))
    except Exception:
        return None
    bench = data.get("benchmark") or {}
    scores = bench.get("scores") or {}
    return scores.get("summary") or scores.get("mean_reward")


def _present_count(rows: list[dict[str, Any]], path: tuple[str, ...]) -> int:
    n = 0
    for r in rows:
        cur: Any = r
        for key in path:
            if not isinstance(cur, dict) or key not in cur:
                cur = None
                break
            cur = cur[key]
        if cur is not None and cur != "":
            n += 1
    return n


def _trial_key(row: dict[str, Any]) -> tuple[Any, Any]:
    return (row.get("problem_idx"), row.get("repeat"))


def _index_traffic_by_trial(
    traffic_rows: list[dict[str, Any]],
) -> dict[tuple[Any, Any], list[dict[str, Any]]]:
    bucket: dict[tuple[Any, Any], list[dict[str, Any]]] = {}
    for r in traffic_rows:
        key = (r.get("problem_idx"), r.get("repeat"))
        bucket.setdefault(key, []).append(r)
    return bucket


def _build_bench_report(
    bench_name: str,
    bench_dir: Path,
) -> dict[str, Any]:
    traj_rows = _read_jsonl(bench_dir / "trajectories.jsonl")
    traffic_rows = _read_jsonl(bench_dir / "model_traffic.jsonl")
    traffic_by_trial = _index_traffic_by_trial(traffic_rows)

    n_trials = len(traj_rows)
    n_problems = len({r.get("problem_idx") for r in traj_rows if r.get("problem_idx") is not None})
    repeats = (n_trials // n_problems) if n_problems else 0

    # ─── score reconciliation ──────────────────────────────────────────
    rewards = [r["reward"] for r in traj_rows if isinstance(r.get("reward"), (int, float))]
    traj_mean = round(sum(rewards) / len(rewards), 6) if rewards else None
    reported = _load_eval_summary(bench_dir)
    reported_mean = None
    if isinstance(reported, dict):
        reported_mean = reported.get("mean")
    failures = Counter()
    for r in traj_rows:
        sd = r.get("scoring_details") or {}
        cat = sd.get("error_category") or r.get("failure_category")
        if cat:
            failures[cat] += 1

    # ─── trajectory_native ────────────────────────────────────────────
    all_agent_steps: list[dict[str, Any]] = []
    per_trial_step_counts: list[int] = []
    tool_calls_total = 0
    reasoning_chars_total = 0
    for r in traj_rows:
        steps = _agent_steps(r)
        per_trial_step_counts.append(len(steps))
        all_agent_steps.extend(steps)
        for s in steps:
            tc = s.get("tool_calls") or []
            if isinstance(tc, list):
                tool_calls_total += len(tc)
            rc = s.get("reasoning_content")
            if isinstance(rc, str):
                reasoning_chars_total += len(rc)
    n_steps = len(all_agent_steps)

    def _field_present(field: str) -> str:
        n = sum(1 for s in all_agent_steps if s.get(field) is not None)
        return f"{n}/{n_steps}"

    def _field_nonempty(field: str) -> str:
        def _ok(v: Any) -> bool:
            if v is None:
                return False
            if isinstance(v, (str, list, dict)) and not v:
                return False
            return True
        n = sum(1 for s in all_agent_steps if _ok(s.get(field)))
        return f"{n}/{n_steps}"

    # ─── wire_captures (model_traffic.jsonl) ──────────────────────────
    total_wire = len(traffic_rows)
    captures_per_trial = [len(v) for v in traffic_by_trial.values()]
    finish_reason_hist = Counter()
    for r in traffic_rows:
        fr = r.get("finish_reason")
        if fr:
            finish_reason_hist[fr] += 1

    # ─── step <-> wire mismatch ───────────────────────────────────────
    delta_signs = {"captures>steps": 0, "captures<steps": 0, "equal": 0}
    for r in traj_rows:
        key = _trial_key(r)
        cap_n = len(traffic_by_trial.get(key, []))
        step_n = len(_agent_steps(r))
        if cap_n > step_n:
            delta_signs["captures>steps"] += 1
        elif cap_n < step_n:
            delta_signs["captures<steps"] += 1
        else:
            delta_signs["equal"] += 1

    # ─── token reconciliation ─────────────────────────────────────────
    def _step_tokens(s: dict[str, Any]) -> int:
        m = s.get("metrics") or {}
        return int(m.get("prompt_tokens") or 0) + int(m.get("completion_tokens") or 0)

    per_step_total = sum(_step_tokens(s) for s in all_agent_steps)

    def _wire_tokens(r: dict[str, Any]) -> int:
        usage = r.get("usage") or {}
        if not isinstance(usage, dict):
            return 0
        if usage.get("total_tokens") is not None:
            try:
                return int(usage["total_tokens"])
            except (TypeError, ValueError):
                return 0
        return int(usage.get("prompt_tokens") or 0) + int(usage.get("completion_tokens") or 0)

    wire_total = sum(_wire_tokens(r) for r in traffic_rows)

    ref_total = 0
    for r in traj_rows:
        model = r.get("model") or {}
        tokens = model.get("tokens") if isinstance(model, dict) else None
        total = tokens.get("total") if isinstance(tokens, dict) else None
        if isinstance(total, (int, float)):
            ref_total += int(total)

    # ─── ATIF per-trial presence ──────────────────────────────────────
    atif_paths = {
        "schema_version": ("trajectory", 0, "schema_version"),
        "session_id": ("trajectory", 0, "session_id"),
        "agent.name": ("trajectory", 0, "agent", "name"),
        "agent.version": ("trajectory", 0, "agent", "version"),
        "agent.model_name": ("trajectory", 0, "agent", "model_name"),
        "steps": ("trajectory", 0, "steps"),
        "final_metrics.total_prompt_tokens": ("trajectory", 0, "final_metrics", "total_prompt_tokens"),
        "final_metrics.total_completion_tokens": ("trajectory", 0, "final_metrics", "total_completion_tokens"),
        "final_metrics.total_steps": ("trajectory", 0, "final_metrics", "total_steps"),
    }
    atif_presence: dict[str, str] = {}
    for label, path in atif_paths.items():
        n = 0
        for r in traj_rows:
            cur: Any = r
            ok = True
            for key in path:
                if isinstance(cur, list):
                    if not isinstance(key, int) or key >= len(cur):
                        ok = False
                        break
                    cur = cur[key]
                elif isinstance(cur, dict):
                    if key not in cur:
                        ok = False
                        break
                    cur = cur[key]
                else:
                    ok = False
                    break
            if ok and cur is not None and cur != "":
                n += 1
        atif_presence[label] = f"{n}/{n_trials}"

    return {
        "bench": bench_name,
        "counts": {
            "n_trials": n_trials,
            "n_problems": n_problems,
            "repeats": repeats,
        },
        "score": {
            "mean_reward": traj_mean,
            "reported_mean": reported_mean,
            "trajectories_vs_reported_match": (
                None
                if traj_mean is None or reported_mean is None
                else abs(traj_mean - reported_mean) < 1e-6
            ),
            "failures": {
                "total": sum(failures.values()),
                "counts_by_category": dict(failures),
            },
        },
        "trajectory_native": {
            "agent_steps_total": n_steps,
            "agent_steps_per_trial": _stats([float(x) for x in per_trial_step_counts]),
            "tool_calls_total": tool_calls_total,
            "reasoning_chars_total": reasoning_chars_total,
            "fields_present_on_steps": {
                "step_id": _field_present("step_id"),
                "source": _field_present("source"),
                "message": _field_present("message"),
                "reasoning_content": _field_present("reasoning_content"),
                "tool_calls": _field_present("tool_calls"),
            },
            "fields_nonempty_on_steps": {
                "step_id": _field_nonempty("step_id"),
                "source": _field_nonempty("source"),
                "message": _field_nonempty("message"),
                "reasoning_content": _field_nonempty("reasoning_content"),
                "tool_calls": _field_nonempty("tool_calls"),
            },
        },
        "wire_captures": {
            "total_observed": total_wire,
            "captures_per_trial": _stats([float(x) for x in captures_per_trial]) if captures_per_trial else None,
            "finish_reasons": dict(finish_reason_hist),
        },
        "mismatches": {
            "agent_steps_vs_wire_calls": delta_signs,
            "tokens": {
                "ref_total": ref_total,
                "per_step_total": per_step_total,
                "wire_total": wire_total,
                "match_ref_vs_wire": (ref_total == wire_total) if ref_total or wire_total else None,
            },
        },
        "atif_per_trial_presence": atif_presence,
    }


def generate_trajectories_report(
    output_dir: Path,
    *,
    enrich: bool = False,
) -> Path | None:
    """Audit-mode report over an evaluation run dir.

    Walks each per-benchmark subdir under *output_dir*, reads
    ``trajectories.jsonl`` + ``model_traffic.jsonl`` if present, and writes
    ``<output_dir>/trajectories_report.json`` summarizing coverage and any
    mismatches between native trajectory data and wire captures.

    The *enrich* flag is reserved for the next iteration (splicing wire
    captures into trajectories.jsonl); currently it only triggers a warning.
    """
    bench_dirs = sorted(
        p
        for p in output_dir.iterdir()
        if p.is_dir() and (p / "trajectories.jsonl").is_file()
    )
    if not bench_dirs:
        logger.info("trajectories_report: no <bench>/trajectories.jsonl found under %s", output_dir)
        return None

    payload = {
        "schema_version": "trajectories_report-v0.1",
        "benchmarks": [_build_bench_report(d.name, d) for d in bench_dirs],
    }

    if enrich:
        logger.warning(
            "trajectories_report: enrich=True requested; "
            "enrichment writes are not implemented in this release (audit-only)."
        )

    out_path = output_dir / "trajectories_report.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("trajectories_report: wrote %s (%d benchmark(s))", out_path, len(bench_dirs))
    return out_path
