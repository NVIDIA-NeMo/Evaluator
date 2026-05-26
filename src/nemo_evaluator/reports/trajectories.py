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

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _is_successful_wire(record: dict[str, Any]) -> bool:
    """Mirror ``model_traffic._is_success``: only 2xx/3xx responses count.

    Failed calls live in ``model_traffic.jsonl`` too (with ``status_code=None``
    or ``>=400``); we exclude them from both the silent-failure metric and
    the enrichment splice so retries/errors don't get attributed to a step.
    """
    sc = record.get("status_code")
    if sc is None:
        return False
    try:
        sc_int = int(sc)
    except (TypeError, ValueError):
        return False
    return 200 <= sc_int < 400


def _wire_dedup_key(record: dict[str, Any]) -> tuple[Any, ...]:
    """Duplicate-detection key for a single ``model_traffic.jsonl`` row.

    Prefers ``record.request_hash`` (sha1 prefix of the upstream request body,
    written by ``ModelTrafficStore.start_request``). Falls back to a coarse
    response-side fingerprint for older runs that don't carry ``request_hash``.
    """
    rh = record.get("request_hash")
    if rh:
        return ("rh", rh)
    usage = record.get("usage") or {}
    return (
        "resp",
        record.get("model"),
        record.get("path"),
        record.get("finish_reason"),
        usage.get("prompt_tokens"),
        usage.get("completion_tokens"),
        round(float(record.get("latency_ms") or 0), 1),
    )


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


def _trial_key(row: dict[str, Any]) -> tuple[Any, Any]:
    return (row.get("problem_idx"), row.get("repeat"))


def _present_at(items: list[dict[str, Any]], *keys: Any) -> str:
    """Render ``N/total`` -- how many items have a non-empty value at the given path.

    Keys may be strings (dict lookup) or ints (list index). A value counts as
    present when it is not None, not an empty string, and not an empty
    list/dict -- so ``reward: 0.0`` counts but ``trajectory: []`` doesn't.
    """
    n = 0
    for item in items:
        cur: Any = item
        ok = True
        for key in keys:
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
        if ok and cur is not None and cur != "" and not (isinstance(cur, (list, dict)) and not cur):
            n += 1
    return f"{n}/{len(items)}"


def _fraction(numer: int, denom: int) -> str:
    """Render ``N/total (P.P%)``; ``0/0 (0.0%)`` when the denominator is zero."""
    if denom <= 0:
        return "0/0 (0.0%)"
    pct = 100.0 * numer / denom
    return f"{numer}/{denom} ({pct:.1f}%)"


def _index_traffic_by_trial(
    traffic_rows: list[dict[str, Any]],
    *,
    only_successful: bool = True,
) -> dict[tuple[Any, Any], list[dict[str, Any]]]:
    bucket: dict[tuple[Any, Any], list[dict[str, Any]]] = {}
    for r in traffic_rows:
        if only_successful and not _is_successful_wire(r):
            continue
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
    # Repeats: count trials per problem; if uniform, that's the repeats value,
    # otherwise expose the histogram so a partial run (e.g. shard) is visible.
    per_problem_repeat_counts = Counter(r.get("problem_idx") for r in traj_rows if r.get("problem_idx") is not None)
    distinct_counts = set(per_problem_repeat_counts.values())
    repeats: Any = (
        next(iter(distinct_counts))
        if len(distinct_counts) == 1
        else (dict(Counter(per_problem_repeat_counts.values())) if distinct_counts else 0)
    )

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
    all_agent_steps: list[dict[str, Any]] = [s for r in traj_rows for s in _agent_steps(r)]
    n_steps = len(all_agent_steps)

    # ─── wire_captures (model_traffic.jsonl) ──────────────────────────
    total_wire = len(traffic_rows)
    successful_wire = sum(1 for r in traffic_rows if _is_successful_wire(r))
    finish_reason_hist = Counter()
    for r in traffic_rows:
        fr = r.get("finish_reason")
        if fr:
            finish_reason_hist[fr] += 1

    # ─── step <-> wire mismatch + silent-failure trials ───────────────
    delta_signs = {"captures>steps": 0, "captures<steps": 0, "equal": 0}
    trials_no_agent_steps = 0
    trials_no_wire_calls = 0
    trials_silent_either = 0
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
        no_steps = step_n == 0
        no_wire = cap_n == 0
        if no_steps:
            trials_no_agent_steps += 1
        if no_wire:
            trials_no_wire_calls += 1
        if no_steps or no_wire:
            trials_silent_either += 1

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
    atif_presence = {label: _present_at(traj_rows, *path) for label, path in atif_paths.items()}

    # Trials that have ≥1 duplicate wire call (request_hash-based, or fallback key).
    trials_with_duplicate_wire_calls = 0
    for recs in traffic_by_trial.values():
        hashes = Counter(_wire_dedup_key(r) for r in recs)
        if any(v > 1 for v in hashes.values()):
            trials_with_duplicate_wire_calls += 1

    # After removing duplicate wire calls, does the mismatch resolve?
    mismatched_steps_vs_wire_trials_after_wire_dedup = 0
    for r in traj_rows:
        key = _trial_key(r)
        step_n = len(_agent_steps(r))
        uniq_wire = len({_wire_dedup_key(w) for w in traffic_by_trial.get(key, [])})
        if step_n != uniq_wire:
            mismatched_steps_vs_wire_trials_after_wire_dedup += 1

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

    _STEP_FIELDS: tuple[tuple[str, ...], ...] = (
        ("step_id",),
        ("source",),
        ("timestamp",),
        ("model_name",),
        ("status_code",),
        ("message",),
        ("reasoning_content",),
        ("tool_calls",),
        ("metrics", "prompt_tokens"),
        ("metrics", "completion_tokens"),
        ("metrics", "total_tokens"),
        ("metrics", "extra", "latency_ms"),
        ("metrics", "extra", "finish_reason"),
        ("metrics", "extra", "cached_tokens"),
        ("metrics", "extra", "reasoning_tokens"),
    )
    step_field_coverage = {".".join(p): _present_at(all_agent_steps, *p) for p in _STEP_FIELDS}

    row_field_coverage = {
        "problem_idx": _present_at(traj_rows, "problem_idx"),
        "repeat": _present_at(traj_rows, "repeat"),
        "reward": _present_at(traj_rows, "reward"),
        "trajectory": _present_at(traj_rows, "trajectory"),
        "model": _present_at(traj_rows, "model"),
        **{f"trajectory[0].{k}": v for k, v in atif_presence.items()},
    }

    # Wire-call dedup (request_hash-based, falls back to heuristic)
    wire_keys_total = [_wire_dedup_key(r) for r in traffic_rows]
    wire_keys_unique = set(wire_keys_total)
    duplicates_total = total_wire - len(wire_keys_unique)

    out: dict[str, Any] = {
        "bench": bench_name,
        "_note": (
            "Health check over a run dir. Sections answer four questions, in order: "
            "(1) score -- did the eval score what the bundle reported? "
            "(2) tokens -- do the three independent token totals agree? "
            "(3) wire_calls -- any duplicates or unexpected counts on the wire? "
            "(4) field_coverage -- what fields are populated in trajectories.jsonl?"
        ),
        "counts": {
            "trials": n_trials,
            "problems": n_problems,
            "repeats": repeats,
        },
        # ── 1. Did the eval score match what's reported in the bundle? ──
        "score": {
            "_note": "mean_reward = mean(row.reward) across trials; reported_mean = bundle's eval-*.json summary.mean",
            "mean_reward": traj_mean,
            "reported_mean": reported_mean,
            "is_mean_reward_correct": is_mean_reward_correct,
            "failures_by_category": dict(failures),
        },
        # ── 2. Do per-step / trajectory-declared / wire totals agree? ──
        "tokens": {
            "_note": (
                "Three independent totals for prompt+completion tokens across all trials. "
                "When all three agree, the trajectory faithfully encoded what the wire saw. "
                "When per_step is 0 but wire isn't, the solver dropped per-step metrics "
                "(enable enrich=true to backfill them from model_traffic.jsonl)."
            ),
            "per_step_sum": per_step_total,
            "final_metrics_total": ref_total,
            "wire_total": wire_total,
            "per_step_matches_wire": (per_step_total == wire_total) if (per_step_total or wire_total) else None,
            "final_metrics_matches_wire": (ref_total == wire_total) if (ref_total or wire_total) else None,
            "trials_with_per_step_vs_final_metrics_mismatch": per_step_vs_final_metrics_mismatch,
        },
        # ── 3. Wire call summary + duplicates ──
        "wire_calls": {
            "_note": (
                "model_traffic.jsonl row stats. Step/wire alignment and the silent-failure "
                "metrics below count only successful (2xx/3xx) wire rows -- a 500 or aborted "
                "request is not a 'captured' model call. 'total' is raw (all statuses) so the "
                "failure volume stays visible. duplicates_total is via record.request_hash "
                "(sha1 prefix of the upstream body)."
            ),
            "total": total_wire,
            "successful": successful_wire,
            "unique": len(wire_keys_unique),
            "duplicates_total": duplicates_total,
            "trials_with_duplicates": trials_with_duplicate_wire_calls,
            "trials_with_more_wire_than_steps": delta_signs["captures>steps"],
            "trials_with_fewer_wire_than_steps": delta_signs["captures<steps"],
            "trials_with_step_wire_mismatch_after_dedup": mismatched_steps_vs_wire_trials_after_wire_dedup,
            "trials_with_no_agent_steps": _fraction(trials_no_agent_steps, n_trials),
            "trials_with_no_wire_calls": _fraction(trials_no_wire_calls, n_trials),
            "trials_silent_either_way": _fraction(trials_silent_either, n_trials),
            "finish_reasons": dict(finish_reason_hist),
        },
        # ── 4. Field coverage (presence audit) ──
        "field_coverage": {
            "_note": (
                "How many trials/steps actually have each ATIF-v1.6 field. "
                "Format 'N/total' means N populated out of total. "
                "Run with enrich=true to backfill metrics.* on steps from the wire layer."
            ),
            "per_trial": row_field_coverage,
            "per_agent_step": step_field_coverage,
            "agent_step_count": n_steps,
        },
    }
    return out


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
        "steps_backfilled_reasoning_content": 0,
        "steps_backfilled_message": 0,
        "steps_backfilled_tool_calls": 0,
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
                    # Opt-in capture fields: backfill iff present on the wire row
                    # and missing on the step.
                    if w.get("reasoning_content") and not s.get("reasoning_content"):
                        s["reasoning_content"] = w["reasoning_content"]
                        counts["steps_backfilled_reasoning_content"] += 1
                    if w.get("message_content") and not s.get("message"):
                        s["message"] = w["message_content"]
                        counts["steps_backfilled_message"] += 1
                    if w.get("tool_calls_full") and not s.get("tool_calls"):
                        s["tool_calls"] = w["tool_calls_full"]
                        counts["steps_backfilled_tool_calls"] += 1
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
            # After-enrichment coverage so the diff is right there in the report.
            enriched_path = d / "trajectories_enriched.jsonl"
            enriched_step_coverage: dict[str, str] = {}
            if enriched_path.is_file():
                enriched_steps = [s for r in _read_jsonl(enriched_path) for s in _agent_steps(r)]
                if enriched_steps:
                    enriched_step_coverage = {
                        ".".join(p): _present_at(enriched_steps, *p)
                        for p in (
                            ("metrics", "prompt_tokens"),
                            ("metrics", "completion_tokens"),
                            ("metrics", "extra", "latency_ms"),
                            ("metrics", "extra", "finish_reason"),
                        )
                    }
            bench_report["enrichment"] = {
                "_note": (
                    "trajectories_enriched.jsonl was written next to trajectories.jsonl. "
                    "For each trial: if #agent_steps == #wire_calls, splice 1:1; otherwise "
                    "stash all wire calls into trajectory[0].extra.captured_model_calls "
                    "(so failures still preserve every captured call)."
                ),
                "trials_spliced_1_to_1": counts["trials_spliced"],
                "trials_with_unmatched_calls_stashed": counts["trials_stashed_unmatched"],
                "trials_with_no_wire_data": counts["trials_no_wire_data"],
                "steps_backfilled": {
                    "prompt_tokens": counts["steps_backfilled_prompt_tokens"],
                    "completion_tokens": counts["steps_backfilled_completion_tokens"],
                    "latency_ms": counts["steps_backfilled_latency_ms"],
                    "finish_reason": counts["steps_backfilled_finish_reason"],
                    "reasoning_content": counts["steps_backfilled_reasoning_content"],
                    "message": counts["steps_backfilled_message"],
                    "tool_calls": counts["steps_backfilled_tool_calls"],
                },
                "step_field_coverage_after_enrichment": enriched_step_coverage,
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
