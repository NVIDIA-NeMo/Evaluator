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

Reads two files from each per-benchmark subdir of a run dir:

  trajectories.jsonl   -- one row per trial (rewards, ATIF trajectory)
  model_traffic.jsonl  -- one row per upstream model call

Writes ``trajectories_report.json`` with four sections per benchmark:

  counts          -- trial / problem / repeat shape
  score           -- mean(row.reward) vs the bundle's reported summary.mean
  tokens          -- per-step token sum vs wire-call token sum
  wire_calls      -- duplicates, step/wire alignment, silent-failure trials
                     (step/wire alignment uses only successful 2xx/3xx rows)
  field_coverage  -- ATIF / per-step fields that aren't 100% populated

With ``enrich=True``, also writes ``trajectories_enriched.jsonl``: per-trial,
when the agent-step count equals the wire-call count, step metrics are
backfilled 1:1 from the matching wire call; otherwise all wire calls land in
``trajectory[0].extra.captured_model_calls`` (no partial splice).

Read-only over the eval pipeline -- runs offline against any run directory.
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


def _wire_dedup_key(record: dict[str, Any], fallback_id: int) -> Any:
    """Dedup key based on ``request_hash`` (sha1 prefix of the upstream body).

    Rows without a ``request_hash`` (older logs) fall back to a per-row unique
    sentinel so they never collapse into each other.
    """
    return record.get("request_hash") or ("_no_hash", fallback_id)


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


def _count_present(items: list[dict[str, Any]], *keys: Any) -> int:
    """How many items have a non-empty value at the given path.

    Keys may be strings (dict lookup) or ints (list index). Counts when the
    value is not None, not "", and not an empty list/dict -- so ``reward: 0.0``
    counts but ``trajectory: []`` doesn't.
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
    return n


def _missing_fields(items: list[dict[str, Any]], paths: dict[str, tuple[Any, ...]]) -> list[dict[str, str]]:
    """Return entries where fewer than all items carry the field, as ``{field, presence}``.

    A fully-populated field is silent: only gaps show up in the report.
    """
    total = len(items)
    if not total:
        return []
    out: list[dict[str, str]] = []
    for label, path in paths.items():
        n = _count_present(items, *path)
        if n < total:
            out.append({"field": label, "presence": f"{n}/{total}"})
    return out


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


_TRIAL_FIELDS: dict[str, tuple[Any, ...]] = {
    "problem_idx": ("problem_idx",),
    "repeat": ("repeat",),
    "reward": ("reward",),
    "trajectory": ("trajectory",),
    "model": ("model",),
    "trajectory[0].schema_version": ("trajectory", 0, "schema_version"),
    "trajectory[0].session_id": ("trajectory", 0, "session_id"),
    "trajectory[0].agent.name": ("trajectory", 0, "agent", "name"),
    "trajectory[0].agent.version": ("trajectory", 0, "agent", "version"),
    "trajectory[0].agent.model_name": ("trajectory", 0, "agent", "model_name"),
    "trajectory[0].agent.parent_agent": ("trajectory", 0, "agent", "parent_agent"),
    "trajectory[0].extra": ("trajectory", 0, "extra"),
    "trajectory[0].steps": ("trajectory", 0, "steps"),
    "trajectory[0].final_metrics.total_prompt_tokens": ("trajectory", 0, "final_metrics", "total_prompt_tokens"),
    "trajectory[0].final_metrics.total_completion_tokens": (
        "trajectory",
        0,
        "final_metrics",
        "total_completion_tokens",
    ),
    "trajectory[0].final_metrics.total_steps": ("trajectory", 0, "final_metrics", "total_steps"),
}

_STEP_FIELDS: dict[str, tuple[Any, ...]] = {
    "step_id": ("step_id",),
    "source": ("source",),
    "timestamp": ("timestamp",),
    "model_name": ("model_name",),
    "status_code": ("status_code",),
    "message": ("message",),
    "reasoning_content": ("reasoning_content",),
    "tool_calls": ("tool_calls",),
    "metrics.prompt_tokens": ("metrics", "prompt_tokens"),
    "metrics.completion_tokens": ("metrics", "completion_tokens"),
    "metrics.total_tokens": ("metrics", "total_tokens"),
    "metrics.extra.latency_ms": ("metrics", "extra", "latency_ms"),
    "metrics.extra.finish_reason": ("metrics", "extra", "finish_reason"),
    "metrics.extra.cached_tokens": ("metrics", "extra", "cached_tokens"),
    "metrics.extra.reasoning_tokens": ("metrics", "extra", "reasoning_tokens"),
}


def _step_tokens(s: dict[str, Any]) -> int:
    m = s.get("metrics") or {}
    return int(m.get("prompt_tokens") or 0) + int(m.get("completion_tokens") or 0)


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


def _repeats_summary(traj_rows: list[dict[str, Any]]) -> Any:
    """Single int when every problem has the same trial count; otherwise a histogram."""
    counts = Counter(r.get("problem_idx") for r in traj_rows if r.get("problem_idx") is not None)
    distinct = set(counts.values())
    if not distinct:
        return 0
    if len(distinct) == 1:
        return next(iter(distinct))
    return dict(Counter(counts.values()))


def _build_bench_report(bench_name: str, bench_dir: Path) -> dict[str, Any]:
    traj_rows = _read_jsonl(bench_dir / "trajectories.jsonl")
    traffic_rows = _read_jsonl(bench_dir / "model_traffic.jsonl")
    traffic_by_trial = _index_traffic_by_trial(traffic_rows)

    n_trials = len(traj_rows)
    n_problems = len({r.get("problem_idx") for r in traj_rows if r.get("problem_idx") is not None})

    rewards = [r["reward"] for r in traj_rows if isinstance(r.get("reward"), (int, float))]
    traj_mean = round(sum(rewards) / len(rewards), 6) if rewards else None
    reported = _load_eval_summary(bench_dir)
    reported_mean = reported.get("mean") if isinstance(reported, dict) else None

    # Single pass over traj_rows: collect everything per-trial needs.
    all_agent_steps: list[dict[str, Any]] = []
    failures: Counter[str] = Counter()
    trials_more_wire = trials_fewer_wire = 0
    trials_no_steps = trials_no_wire = trials_silent = 0
    per_step_vs_fm_mismatch = 0
    for r in traj_rows:
        steps = _agent_steps(r)
        all_agent_steps.extend(steps)

        cat = (r.get("scoring_details") or {}).get("error_category") or r.get("failure_category")
        if cat:
            failures[cat] += 1

        step_n = len(steps)
        wire_n = len(traffic_by_trial.get(_trial_key(r), []))
        if wire_n > step_n:
            trials_more_wire += 1
        elif wire_n < step_n:
            trials_fewer_wire += 1
        if step_n == 0:
            trials_no_steps += 1
        if wire_n == 0:
            trials_no_wire += 1
        if step_n == 0 or wire_n == 0:
            trials_silent += 1

        per_step = sum(_step_tokens(s) for s in steps)
        fm = ((r.get("trajectory") or [{}])[0]).get("final_metrics") or {}
        fm_total = int(fm.get("total_prompt_tokens") or 0) + int(fm.get("total_completion_tokens") or 0)
        if (per_step or fm_total) and per_step != fm_total:
            per_step_vs_fm_mismatch += 1

    per_step_token_sum = sum(_step_tokens(s) for s in all_agent_steps)
    wire_token_sum = sum(_wire_tokens(r) for r in traffic_rows)

    # Wire dedup: how many trials had ≥1 dup, and the unique count overall.
    trials_with_dup_wire = 0
    for recs in traffic_by_trial.values():
        keys = Counter(_wire_dedup_key(r, i) for i, r in enumerate(recs))
        if any(v > 1 for v in keys.values()):
            trials_with_dup_wire += 1
    wire_unique = len({_wire_dedup_key(r, i) for i, r in enumerate(traffic_rows)})

    return {
        "bench": bench_name,
        "counts": {"trials": n_trials, "problems": n_problems, "repeats": _repeats_summary(traj_rows)},
        "score": {
            "mean_reward": traj_mean,
            "reported_mean": reported_mean,
            "failures_by_category": dict(failures),
        },
        "tokens": {
            "per_step_sum": per_step_token_sum,
            "wire_total": wire_token_sum,
            "trials_with_per_step_vs_final_metrics_mismatch": per_step_vs_fm_mismatch,
        },
        "wire_calls": {
            "total": len(traffic_rows),
            "successful": sum(1 for r in traffic_rows if _is_successful_wire(r)),
            "unique": wire_unique,
            "trials_with_duplicates": trials_with_dup_wire,
            "trials_with_more_wire_than_steps": trials_more_wire,
            "trials_with_fewer_wire_than_steps": trials_fewer_wire,
            "trials_with_no_agent_steps": _fraction(trials_no_steps, n_trials),
            "trials_with_no_wire_calls": _fraction(trials_no_wire, n_trials),
            "trials_silent_either_way": _fraction(trials_silent, n_trials),
        },
        "field_coverage": {
            "per_trial_missing": _missing_fields(traj_rows, _TRIAL_FIELDS),
            "per_agent_step_missing": _missing_fields(all_agent_steps, _STEP_FIELDS),
        },
    }


def _enrich_bench(bench_dir: Path) -> dict[str, int]:
    """All-or-nothing per-trial enrichment from ``model_traffic.jsonl``.

    ``wire_calls`` below is the list of **successful (2xx/3xx)** rows for the
    trial -- failed/retry rows live in the log too but are filtered out by
    ``_index_traffic_by_trial`` so they never get attributed to a step.

    For each trial:

    * If ``len(agent_steps) == len(successful_wire_calls)`` — splice 1:1
      by order (backfill ``metrics.{prompt,completion}_tokens`` and
      ``metrics.extra.{latency_ms, finish_reason}``).
    * Otherwise — leave the steps alone and stash **all** successful wire
      calls under ``trajectory[0].extra.captured_model_calls``. We never
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
        "steps_backfilled_timestamp": 0,
        "steps_backfilled_model_name": 0,
        "steps_backfilled_status_code": 0,
        "steps_backfilled_prompt_tokens": 0,
        "steps_backfilled_completion_tokens": 0,
        "steps_backfilled_total_tokens": 0,
        "steps_backfilled_cached_tokens": 0,
        "steps_backfilled_reasoning_tokens": 0,
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
                    # Top-level step fields populated directly from wire row.
                    if not s.get("timestamp") and w.get("timestamp"):
                        s["timestamp"] = w["timestamp"]
                        counts["steps_backfilled_timestamp"] += 1
                    if not s.get("model_name") and w.get("model"):
                        s["model_name"] = w["model"]
                        counts["steps_backfilled_model_name"] += 1
                    if s.get("status_code") is None and w.get("status_code") is not None:
                        s["status_code"] = w["status_code"]
                        counts["steps_backfilled_status_code"] += 1

                    metrics = s.setdefault("metrics", {})
                    usage = w.get("usage") or {}
                    if metrics.get("prompt_tokens") in (None, 0) and usage.get("prompt_tokens"):
                        metrics["prompt_tokens"] = usage["prompt_tokens"]
                        counts["steps_backfilled_prompt_tokens"] += 1
                    if metrics.get("completion_tokens") in (None, 0) and usage.get("completion_tokens"):
                        metrics["completion_tokens"] = usage["completion_tokens"]
                        counts["steps_backfilled_completion_tokens"] += 1
                    if metrics.get("total_tokens") in (None, 0) and usage.get("total_tokens"):
                        metrics["total_tokens"] = usage["total_tokens"]
                        counts["steps_backfilled_total_tokens"] += 1
                    extra = metrics.setdefault("extra", {})
                    if extra.get("cached_tokens") in (None, 0) and usage.get("cached_tokens"):
                        extra["cached_tokens"] = usage["cached_tokens"]
                        counts["steps_backfilled_cached_tokens"] += 1
                    if extra.get("reasoning_tokens") in (None, 0) and usage.get("reasoning_tokens"):
                        extra["reasoning_tokens"] = usage["reasoning_tokens"]
                        counts["steps_backfilled_reasoning_tokens"] += 1
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
            enriched_steps: list[dict[str, Any]] = []
            enriched_path = d / "trajectories_enriched.jsonl"
            if enriched_path.is_file():
                enriched_steps = [s for r in _read_jsonl(enriched_path) for s in _agent_steps(r)]
            bench_report["enrichment"] = {
                "trials_spliced_1_to_1": counts["trials_spliced"],
                "trials_with_unmatched_calls_stashed": counts["trials_stashed_unmatched"],
                "trials_with_no_wire_data": counts["trials_no_wire_data"],
                "steps_backfilled": {
                    "timestamp": counts["steps_backfilled_timestamp"],
                    "model_name": counts["steps_backfilled_model_name"],
                    "status_code": counts["steps_backfilled_status_code"],
                    "prompt_tokens": counts["steps_backfilled_prompt_tokens"],
                    "completion_tokens": counts["steps_backfilled_completion_tokens"],
                    "total_tokens": counts["steps_backfilled_total_tokens"],
                    "cached_tokens": counts["steps_backfilled_cached_tokens"],
                    "reasoning_tokens": counts["steps_backfilled_reasoning_tokens"],
                    "latency_ms": counts["steps_backfilled_latency_ms"],
                    "finish_reason": counts["steps_backfilled_finish_reason"],
                    "reasoning_content": counts["steps_backfilled_reasoning_content"],
                    "message": counts["steps_backfilled_message"],
                    "tool_calls": counts["steps_backfilled_tool_calls"],
                },
                "step_field_coverage_after_enrichment_missing": _missing_fields(
                    enriched_steps,
                    {
                        ".".join(p): p
                        for p in (
                            ("timestamp",),
                            ("model_name",),
                            ("status_code",),
                            ("metrics", "prompt_tokens"),
                            ("metrics", "completion_tokens"),
                            ("metrics", "total_tokens"),
                            ("metrics", "extra", "cached_tokens"),
                            ("metrics", "extra", "reasoning_tokens"),
                            ("metrics", "extra", "latency_ms"),
                            ("metrics", "extra", "finish_reason"),
                        )
                    },
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
