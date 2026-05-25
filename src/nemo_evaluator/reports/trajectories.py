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
"""Trajectory audit + optional enrichment.

Reads two files from a run directory:

  trajectories.jsonl   -- one row per trial (rewards, ATIF trajectory)
  model_traffic.jsonl  -- one row per upstream model call

Writes ``trajectories_report.json`` summarising counts, score reconciliation,
ATIF field presence, step/wire mismatches, duplicates, and token totals.
Each metric is reported as ``{value, from}`` so the calculation is explicit.

With ``enrich=True``, also writes ``trajectories_enriched.jsonl``: per-trial,
when the agent-step count equals the wire-call count, step metrics are
backfilled 1:1 from the matching wire call; otherwise all wire calls land in
``trajectory[0].extra.captured_model_calls`` (no partial splice).

The module is read-only over the eval pipeline -- runs offline against any
existing run directory, including archived bundles.
"""

from __future__ import annotations

import hashlib
import json
import logging
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _step_content_hash(step: dict[str, Any]) -> str:
    """Deterministic hash of an agent step's user-visible content.

    Two steps with identical message + reasoning_content + tool-call names
    + tool-call arguments hash to the same value — a strong duplicate
    signal when it repeats inside a single trial.
    """
    msg = step.get("message") or ""
    reasoning = step.get("reasoning_content") or ""
    tool_calls = step.get("tool_calls") or []
    tc_repr: list[Any] = []
    if isinstance(tool_calls, list):
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") or {}
            name = tc.get("function_name") or fn.get("name") or ""
            args = tc.get("arguments") if "arguments" in tc else fn.get("arguments")
            tc_repr.append((name, args))
    payload = json.dumps([msg, reasoning, tc_repr], sort_keys=True, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _wire_dedup_key(record: dict[str, Any]) -> tuple[Any, ...]:
    """Coarse duplicate-detection key for a single ``model_traffic.jsonl`` row.

    ``model_traffic.jsonl`` currently stores per-call response stats but not
    the request body, so we can't fingerprint the actual prompt yet. Two rows
    with identical (model, path, finish_reason, prompt_tokens, completion_tokens,
    latency_ms_rounded) on the same trial almost certainly indicate the same
    upstream call being captured twice.
    """
    usage = record.get("usage") or {}
    return (
        record.get("model"),
        record.get("path"),
        record.get("finish_reason"),
        usage.get("prompt_tokens"),
        usage.get("completion_tokens"),
        round(float(record.get("latency_ms") or 0), 1),
    )


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

    def _nested_present(*keys: str) -> str:
        n = 0
        for s in all_agent_steps:
            cur: Any = s
            ok = True
            for key in keys:
                if not isinstance(cur, dict) or cur.get(key) is None:
                    ok = False
                    break
                cur = cur[key]
            if ok:
                n += 1
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

    # ─── ATIF per-trial presence (full ATIF-v1.6 coverage) ────────────
    atif_paths = {
        "schema_version": ("trajectory", 0, "schema_version"),
        "session_id": ("trajectory", 0, "session_id"),
        "agent.name": ("trajectory", 0, "agent", "name"),
        "agent.version": ("trajectory", 0, "agent", "version"),
        "agent.model_name": ("trajectory", 0, "agent", "model_name"),
        "agent.parent_agent": ("trajectory", 0, "agent", "parent_agent"),
        "extra": ("trajectory", 0, "extra"),
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

    # ─── anomalies (the things you actually want to know) ────────────
    # Zero / None tokens per step + per trial
    steps_with_zero_or_none_tokens = 0
    trials_with_any_zero_step = 0
    trials_all_zero_token_steps = 0
    for r in traj_rows:
        steps = _agent_steps(r)
        if not steps:
            continue
        zero_in_trial = 0
        for s in steps:
            m = s.get("metrics") or {}
            pt = m.get("prompt_tokens")
            ct = m.get("completion_tokens")
            if (not pt) and (not ct):
                steps_with_zero_or_none_tokens += 1
                zero_in_trial += 1
        if zero_in_trial > 0:
            trials_with_any_zero_step += 1
        if zero_in_trial == len(steps):
            trials_all_zero_token_steps += 1

    # Duplicate agent steps inside a single trial (content hash)
    duplicate_steps_in_trial = 0
    trials_with_duplicate_steps = 0
    for r in traj_rows:
        hashes = Counter(_step_content_hash(s) for s in _agent_steps(r))
        per_trial_dups = sum(v - 1 for v in hashes.values() if v > 1)
        if per_trial_dups > 0:
            duplicate_steps_in_trial += per_trial_dups
            trials_with_duplicate_steps += 1

    # Duplicate wire calls inside a single trial (coarse fingerprint)
    duplicate_wire_calls_in_trial = 0
    trials_with_duplicate_wire_calls = 0
    for key, recs in traffic_by_trial.items():
        hashes = Counter(_wire_dedup_key(r) for r in recs)
        per_trial_dups = sum(v - 1 for v in hashes.values() if v > 1)
        if per_trial_dups > 0:
            duplicate_wire_calls_in_trial += per_trial_dups
            trials_with_duplicate_wire_calls += 1

    # Trials where step count != wire-call count (a different mismatch axis
    # than ref/wire token totals; signals lost or duplicated captures).
    mismatched_steps_vs_wire_trials = delta_signs["captures>steps"] + delta_signs["captures<steps"]

    # After removing duplicates on both sides, does the mismatch resolve?
    mismatched_steps_vs_wire_trials_after_dedup = 0
    for r in traj_rows:
        key = _trial_key(r)
        uniq_steps = len({_step_content_hash(s) for s in _agent_steps(r)})
        uniq_wire = len({_wire_dedup_key(w) for w in traffic_by_trial.get(key, [])})
        if uniq_steps != uniq_wire:
            mismatched_steps_vs_wire_trials_after_dedup += 1

    # Per-trial: sum of step.metrics.{prompt+completion}_tokens equal to
    # final_metrics.{total_prompt_tokens + total_completion_tokens}?
    per_step_vs_final_metrics_mismatch = 0
    for r in traj_rows:
        steps = _agent_steps(r)
        per_step = sum(_step_tokens(s) for s in steps)
        fm = ((r.get("trajectory") or [{}])[0]).get("final_metrics") or {}
        fm_total = int(fm.get("total_prompt_tokens") or 0) + int(fm.get("total_completion_tokens") or 0)
        if (per_step or fm_total) and per_step != fm_total:
            per_step_vs_final_metrics_mismatch += 1

    is_mean_reward_correct: bool | None = None
    if traj_mean is not None and reported_mean is not None:
        is_mean_reward_correct = abs(traj_mean - reported_mean) < 1e-6

    def vf(value: Any, formula: str) -> dict[str, Any]:
        """{value, from} pair where `from` documents the calculation."""
        return {"value": value, "from": formula}

    return {
        "bench": bench_name,
        "counts": {
            "n_trials": vf(n_trials, "trajectories.jsonl row count"),
            "n_problems": vf(n_problems, "distinct trajectories.jsonl row.problem_idx"),
            "repeats": vf(repeats, "n_trials // n_problems"),
        },
        "score": {
            "mean_reward": vf(traj_mean, "mean(row.reward) across trajectories.jsonl trials"),
            "reported_mean": vf(
                reported_mean,
                "latest <bench>/eval-*.json benchmark.scores.summary.mean",
            ),
            "is_mean_reward_correct": vf(
                is_mean_reward_correct,
                "|mean_reward - reported_mean| < 1e-6 (null if either side is missing)",
            ),
            "failures": vf(
                {"total": sum(failures.values()), "counts_by_category": dict(failures)},
                "row.scoring_details.error_category OR row.failure_category",
            ),
        },
        # The actionable summary: what's broken, in counts.
        "anomalies": {
            "steps_with_zero_or_none_tokens": vf(
                steps_with_zero_or_none_tokens,
                "agent steps where (metrics.prompt_tokens or 0) + (metrics.completion_tokens or 0) == 0",
            ),
            "trials_with_any_zero_token_step": vf(
                trials_with_any_zero_step,
                "trials with at least one zero-or-none-tokens agent step",
            ),
            "trials_with_all_zero_token_steps": vf(
                trials_all_zero_token_steps,
                "trials where every agent step is zero-or-none-tokens",
            ),
            "duplicate_steps_in_trial_total": vf(
                duplicate_steps_in_trial,
                "sum over trials of (#repeated step content_hashes - 1); "
                "content_hash = sha1(message + reasoning_content + tool_calls names+args)",
            ),
            "trials_with_duplicate_steps": vf(
                trials_with_duplicate_steps,
                "trials with at least one duplicate step content_hash",
            ),
            "duplicate_wire_calls_in_trial_total": vf(
                duplicate_wire_calls_in_trial,
                "sum over trials of (#repeated wire_dedup_keys - 1); "
                "wire_dedup_key = (model, path, finish_reason, prompt_tokens, "
                "completion_tokens, round(latency_ms,1))",
            ),
            "trials_with_duplicate_wire_calls": vf(
                trials_with_duplicate_wire_calls,
                "trials with at least one duplicate wire_dedup_key",
            ),
            "mismatched_steps_vs_wire_trials": vf(
                mismatched_steps_vs_wire_trials,
                "trials where len(agent_steps) != len(model_traffic rows for that trial)",
            ),
            "mismatched_steps_vs_wire_trials_after_dedup": vf(
                mismatched_steps_vs_wire_trials_after_dedup,
                "same as above but on unique sets: "
                "trials where len(set(step content_hashes)) != len(set(wire_dedup_keys)); "
                "non-zero here means the mismatch is NOT explained by duplicates",
            ),
            "trials_with_per_step_vs_final_metrics_token_mismatch": vf(
                per_step_vs_final_metrics_mismatch,
                "trials where sum(step.metrics.prompt+completion_tokens) != "
                "final_metrics.total_prompt+total_completion_tokens; "
                "non-zero means the trajectory's internal totals are inconsistent",
            ),
        },
        "trajectory_native": {
            "agent_steps_total": vf(
                n_steps,
                "len(row.trajectory[0].steps[source=='agent']) summed across trials",
            ),
            "agent_steps_per_trial": vf(
                _stats([float(x) for x in per_trial_step_counts]),
                "stats of agent-step counts per trial",
            ),
            "tool_calls_total": vf(
                tool_calls_total,
                "sum of len(step.tool_calls) over agent steps",
            ),
            "reasoning_chars_total": vf(
                reasoning_chars_total,
                "sum of len(step.reasoning_content) over agent steps",
            ),
            "fields_present_on_steps": vf(
                {
                    "step_id": _field_present("step_id"),
                    "source": _field_present("source"),
                    "timestamp": _field_present("timestamp"),
                    "model_name": _field_present("model_name"),
                    "status_code": _field_present("status_code"),
                    "message": _field_present("message"),
                    "reasoning_content": _field_present("reasoning_content"),
                    "tool_calls": _field_present("tool_calls"),
                    "metrics.prompt_tokens": _nested_present("metrics", "prompt_tokens"),
                    "metrics.completion_tokens": _nested_present("metrics", "completion_tokens"),
                    "metrics.total_tokens": _nested_present("metrics", "total_tokens"),
                    "metrics.extra.latency_ms": _nested_present("metrics", "extra", "latency_ms"),
                    "metrics.extra.finish_reason": _nested_present("metrics", "extra", "finish_reason"),
                    "metrics.extra.cached_tokens": _nested_present("metrics", "extra", "cached_tokens"),
                    "metrics.extra.reasoning_tokens": _nested_present("metrics", "extra", "reasoning_tokens"),
                },
                f"N/{n_steps} agent steps where the dotted-path field is non-None",
            ),
            "fields_nonempty_on_steps": vf(
                {
                    "step_id": _field_nonempty("step_id"),
                    "source": _field_nonempty("source"),
                    "message": _field_nonempty("message"),
                    "reasoning_content": _field_nonempty("reasoning_content"),
                    "tool_calls": _field_nonempty("tool_calls"),
                },
                f"N/{n_steps} agent steps where the field is non-empty (non-None and not '' / [] / {{}})",
            ),
        },
        "wire_captures": {
            "total_observed": vf(
                total_wire,
                "model_traffic.jsonl row count",
            ),
            "captures_per_trial": vf(
                _stats([float(x) for x in captures_per_trial]) if captures_per_trial else None,
                "stats of len(model_traffic rows) grouped by (problem_idx, repeat)",
            ),
            "finish_reasons": vf(
                dict(finish_reason_hist),
                "histogram of model_traffic.jsonl row.finish_reason",
            ),
        },
        "mismatches": {
            "agent_steps_vs_wire_calls": vf(
                delta_signs,
                "for each trial: sign(len(wire_rows) - len(agent_steps))",
            ),
            "tokens": vf(
                {
                    "ref_total": ref_total,
                    "per_step_total": per_step_total,
                    "wire_total": wire_total,
                    "match_ref_vs_wire": (ref_total == wire_total) if ref_total or wire_total else None,
                },
                "ref_total = sum(row.model.tokens.total); "
                "per_step_total = sum(step.metrics.prompt_tokens + step.metrics.completion_tokens); "
                "wire_total = sum(model_traffic row.usage.total_tokens, fallback to prompt+completion)",
            ),
        },
        "atif_per_trial_presence": vf(
            atif_presence,
            f"N/{n_trials} trajectories.jsonl rows where the dotted-path field is non-None and non-empty",
        ),
        "row_required_fields_presence": vf(
            {
                "problem_idx": f"{sum(1 for r in traj_rows if r.get('problem_idx') is not None)}/{n_trials}",
                "repeat": f"{sum(1 for r in traj_rows if r.get('repeat') is not None)}/{n_trials}",
                "reward": f"{sum(1 for r in traj_rows if isinstance(r.get('reward'), (int, float)))}/{n_trials}",
                "trajectory": f"{sum(1 for r in traj_rows if r.get('trajectory'))}/{n_trials}",
                "model": f"{sum(1 for r in traj_rows if r.get('model'))}/{n_trials}",
            },
            f"N/{n_trials} trajectories.jsonl rows where the top-level required field is present",
        ),
        "stashed_model_calls": vf(
            {
                "trials_with_stashed_calls": sum(
                    1
                    for r in traj_rows
                    if isinstance(((r.get("trajectory") or [{}])[0]).get("extra"), dict)
                    and ((r.get("trajectory") or [{}])[0]).get("extra", {}).get("captured_model_calls")
                ),
                "stashed_calls_total": sum(
                    len(((r.get("trajectory") or [{}])[0]).get("extra", {}).get("captured_model_calls") or [])
                    for r in traj_rows
                    if isinstance(((r.get("trajectory") or [{}])[0]).get("extra"), dict)
                ),
            },
            "trajectory[0].extra.captured_model_calls — wire calls that couldn't be spliced 1:1 "
            "into agent steps (e.g. solver failed before producing a matching step). "
            "Populated by the enrichment writer when trial step count != wire-call count.",
        ),
    }


def _enrich_bench(bench_dir: Path) -> dict[str, int]:
    """All-or-nothing per-trial enrichment from ``model_traffic.jsonl``.

    For each trial:

    * If ``len(agent_steps) == len(wire_calls)`` — splice 1:1 by order
      (backfill ``metrics.{prompt,completion}_tokens`` and
      ``metrics.extra.{latency_ms, finish_reason}``).
    * Otherwise — leave the steps alone and stash **all** wire calls
      under ``trajectory[0].extra.captured_model_calls``. We never
      mix-and-match when the counts don't line up: a partial splice
      would attribute the wrong call to the wrong step.

    Writes ``trajectories_enriched.jsonl`` next to the original.
    """
    traj_path = bench_dir / "trajectories.jsonl"
    traffic_path = bench_dir / "model_traffic.jsonl"
    enriched_path = bench_dir / "trajectories_enriched.jsonl"

    traj_rows = _read_jsonl(traj_path)
    traffic_rows = _read_jsonl(traffic_path)
    by_trial = _index_traffic_by_trial(traffic_rows)

    counts = {
        "trials_spliced": 0,
        "trials_stashed_unmatched": 0,
        "trials_no_wire_data": 0,
        "steps_backfilled_prompt_tokens": 0,
        "steps_backfilled_completion_tokens": 0,
        "steps_backfilled_latency_ms": 0,
        "steps_backfilled_finish_reason": 0,
        "rows_written": 0,
    }

    with enriched_path.open("w", encoding="utf-8") as fh:
        for r in traj_rows:
            steps = _agent_steps(r)
            wire = by_trial.get(_trial_key(r), [])
            if not wire:
                counts["trials_no_wire_data"] += 1
            elif len(steps) == len(wire):
                # 1:1 splice
                for s, w in zip(steps, wire):
                    metrics = s.setdefault("metrics", {})
                    usage = w.get("usage") or {}
                    if metrics.get("prompt_tokens") in (None, 0) and usage.get("prompt_tokens"):
                        metrics["prompt_tokens"] = usage["prompt_tokens"]
                        counts["steps_backfilled_prompt_tokens"] += 1
                    if metrics.get("completion_tokens") in (None, 0) and usage.get("completion_tokens"):
                        metrics["completion_tokens"] = usage["completion_tokens"]
                        counts["steps_backfilled_completion_tokens"] += 1
                    extra = metrics.setdefault("extra", {})
                    if extra.get("latency_ms") in (None, 0) and w.get("latency_ms"):
                        extra["latency_ms"] = w["latency_ms"]
                        counts["steps_backfilled_latency_ms"] += 1
                    if not extra.get("finish_reason") and w.get("finish_reason"):
                        extra["finish_reason"] = w["finish_reason"]
                        counts["steps_backfilled_finish_reason"] += 1
                counts["trials_spliced"] += 1
            else:
                # Count mismatch — stash ALL wire calls; don't touch steps.
                traj = (r.get("trajectory") or [{}])[0]
                traj.setdefault("extra", {})["captured_model_calls"] = wire
                counts["trials_stashed_unmatched"] += 1
            fh.write(json.dumps(r) + "\n")
            counts["rows_written"] += 1
    return counts


def generate_trajectories_report(
    output_dir: Path,
    *,
    enrich: bool = False,
) -> Path | None:
    """Audit (and optionally enrich) trajectories under *output_dir*.

    Walks each per-benchmark subdir, reads ``trajectories.jsonl`` +
    ``model_traffic.jsonl`` if present, and writes
    ``<output_dir>/trajectories_report.json``.

    With ``enrich=True``, also writes ``<bench>/trajectories_enriched.jsonl``
    per benchmark — agent-step ``metrics`` are backfilled from the matching
    ``model_traffic.jsonl`` row (1:1 by order within a trial), and any
    extra wire calls are stashed under ``trajectory[0].extra.captured_model_calls``.
    """
    bench_dirs = sorted(p for p in output_dir.iterdir() if p.is_dir() and (p / "trajectories.jsonl").is_file())
    if not bench_dirs:
        logger.info("trajectories_report: no <bench>/trajectories.jsonl found under %s", output_dir)
        return None

    benches: list[dict[str, Any]] = []
    for d in bench_dirs:
        bench_report = _build_bench_report(d.name, d)
        if enrich:
            counts = _enrich_bench(d)
            bench_report["enrichment"] = {
                "value": counts,
                "from": (
                    "trajectories_enriched.jsonl written next to trajectories.jsonl; "
                    "agent steps backfilled 1:1 with model_traffic.jsonl rows for the same trial; "
                    "extra wire calls stashed in trajectory[0].extra.captured_model_calls"
                ),
            }
        benches.append(bench_report)

    payload = {"schema_version": "trajectories_report-v0.1", "benchmarks": benches}

    out_path = output_dir / "trajectories_report.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info(
        "trajectories_report: wrote %s (%d benchmark(s)%s)",
        out_path,
        len(bench_dirs),
        ", enriched" if enrich else "",
    )
    return out_path
