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

Writes ``trajectories_report.json`` with three sections per benchmark:

  trajectories    -- problem count, score, failures, Piotr-style quality
                     checks (zero-token turns, missed metrics, clean/dirty),
                     and field coverage
  tokens_stats    -- per-step vs wire vs final_metrics token comparison;
                     ``all_sources_match`` is True only when all three agree
                     and no trial is missing final_metrics token fields
  wire_calls      -- duplicates, step/wire alignment, non-200 calls (with
                     status breakdown and one example body per code), last-
                     call non-200, and silent-failure trials

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

from nemo_evaluator.observability.failure_classification import classify_model_failure
from nemo_evaluator.reports.schemas import AgentStep, TrajectoryRow, fields_as_coverage_paths

logger = logging.getLogger(__name__)

# Coverage-path dicts derived from schemas so the report stays in sync
# with the schema without manual duplication.
_TRIAL_FIELDS = fields_as_coverage_paths(TrajectoryRow)
_STEP_FIELDS = fields_as_coverage_paths(AgentStep)
_EXAMPLE_TEXT_CHARS = 500
_MAX_EXAMPLES_PER_ERROR_TYPE = 5
_MAX_RETRY_EXAMPLES = 5


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


def _wire_session_id(record: dict[str, Any]) -> str:
    session_id = str(record.get("session_id") or "").strip()
    if session_id:
        return session_id
    request_id = str(record.get("model_traffic_request_id") or "")
    if ":" in request_id:
        return request_id.split(":", 1)[0]
    return ""


def _wire_has_tool(record: dict[str, Any], name: str) -> bool:
    names = ((record.get("tool_calls") or {}).get("names") or {}) if isinstance(record.get("tool_calls"), dict) else {}
    if names.get(name):
        return True
    for tool_call in record.get("tool_calls_full") or []:
        if not isinstance(tool_call, dict):
            continue
        fn = tool_call.get("function") if isinstance(tool_call.get("function"), dict) else {}
        tool_name = tool_call.get("name") or tool_call.get("function_name") or fn.get("name")
        if tool_name == name:
            return True
    return False


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


def _truncate_text(value: Any, limit: int = _EXAMPLE_TEXT_CHARS) -> str:
    text = "" if value is None else str(value)
    text = " ".join(text.split())
    if len(text) > limit:
        return text[:limit] + f"...[truncated, {len(text) - limit} chars dropped]"
    return text


def _trajectory_doc(row: dict[str, Any]) -> dict[str, Any]:
    traj = row.get("trajectory") or []
    if not traj or not isinstance(traj[0], dict):
        return {}
    return traj[0]


def _final_metrics(row: dict[str, Any]) -> dict[str, Any]:
    return _trajectory_doc(row).get("final_metrics") or {}


def _failure_category(row: dict[str, Any]) -> str:
    return str((row.get("scoring_details") or {}).get("error_category") or row.get("failure_category") or "")


def _failure_error(row: dict[str, Any]) -> str:
    scoring_details = row.get("scoring_details") or {}
    return _truncate_text(scoring_details.get("error") or row.get("model_error") or row.get("error") or "")


def _completed_harbor_verification_without_error(row: dict[str, Any]) -> bool:
    scoring_details = row.get("scoring_details")
    if not isinstance(scoring_details, dict):
        return False
    if scoring_details.get("error") or scoring_details.get("error_category"):
        return False
    if row.get("model_error") or row.get("error"):
        return False
    if scoring_details.get("method") != "harbor":
        return False
    return "test_exit_code" in scoring_details or "test_summary" in scoring_details


def _wire_failure_text(record: dict[str, Any]) -> str:
    parts = [
        f"HTTP {record.get('status_code')}" if record.get("status_code") is not None else "",
        record.get("error_type") or "",
        record.get("error_code") or "",
        record.get("error_message") or "",
        record.get("error_body") or "",
    ]
    return " ".join(str(part) for part in parts if part)


def _wire_failure_category(record: dict[str, Any]) -> str:
    if not record or _is_successful_wire(record):
        return ""
    return classify_model_failure(_wire_failure_text(record), status_code=record.get("status_code")) or "model_error"


def _trial_wire_failure(row: dict[str, Any], all_wire_rows: list[dict[str, Any]]) -> tuple[str, str]:
    if not all_wire_rows:
        return "", ""
    last_wire = all_wire_rows[-1]
    category = _wire_failure_category(last_wire)
    if not category:
        return "", ""
    return category, _truncate_text(_wire_failure_text(last_wire))


def _trial_failure_category(row: dict[str, Any], all_wire_rows: list[dict[str, Any]]) -> str:
    category = _failure_category(row)
    wire_category, _ = _trial_wire_failure(row, all_wire_rows)
    if wire_category and category in {"", "empty_response", "model_error"}:
        return wire_category
    if category == "empty_response" and _completed_harbor_verification_without_error(row):
        return ""
    return category


def _trial_failure_error(row: dict[str, Any], all_wire_rows: list[dict[str, Any]]) -> str:
    error = _failure_error(row)
    _, wire_error = _trial_wire_failure(row, all_wire_rows)
    return error or wire_error


def _has_final_metric_tokens(row: dict[str, Any]) -> bool:
    fm = _final_metrics(row)
    return fm.get("total_prompt_tokens") is not None or fm.get("total_completion_tokens") is not None


def _wire_example(record: dict[str, Any], *, is_last_wire: bool) -> dict[str, Any]:
    example = {
        "problem_idx": record.get("problem_idx"),
        "repeat": record.get("repeat"),
        "status_code": record.get("status_code"),
        "error_type": record.get("error_type") or "",
        "error_code": record.get("error_code") or "",
        "error_message": _truncate_text(record.get("error_message")),
        "latency_ms": record.get("latency_ms"),
        "finish_reason": record.get("finish_reason") or "",
        "model": record.get("model") or "",
        "path": record.get("path") or "",
        "model_traffic_request_id": record.get("model_traffic_request_id") or "",
        "session_id": record.get("session_id") or "",
        "adapter_request_id": record.get("adapter_request_id") or "",
        "request_hash": record.get("request_hash") or "",
        "is_last_wire": is_last_wire,
    }
    if record.get("error_body"):
        example["error_body"] = _truncate_text(record.get("error_body"))
    return example


def _wire_error_type_key(example: dict[str, Any]) -> str:
    error_type = str(example.get("error_type") or "").strip()
    if error_type:
        return error_type
    error_code = str(example.get("error_code") or "").strip()
    if error_code:
        return error_code
    status_code = example.get("status_code")
    if status_code is not None and status_code != "":
        return f"http_{status_code}"
    return "unknown_error_type"


def _group_wire_examples_by_error_type(examples: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for example in examples:
        key = _wire_error_type_key(example)
        bucket = grouped.setdefault(key, {"count": 0, "examples": []})
        bucket["count"] += 1
        if len(bucket["examples"]) < _MAX_EXAMPLES_PER_ERROR_TYPE:
            bucket["examples"].append(example)
        else:
            bucket["omitted_examples"] = bucket.get("omitted_examples", 0) + 1
    return grouped


def _session_group_example(
    trial_key: tuple[Any, Any],
    sessions: dict[str, list[dict[str, Any]]],
    *,
    selected_session_id: str = "",
) -> dict[str, Any]:
    session_rows = []
    for session_id, rows in sessions.items():
        session_rows.append(
            {
                "session_id": session_id,
                "selected": session_id == selected_session_id,
                "wire_calls": len(rows),
                "successful_wire_calls": sum(1 for row in rows if _is_successful_wire(row)),
                "non_success_wire_calls": sum(1 for row in rows if not _is_successful_wire(row)),
                "finish_calls": sum(1 for row in rows if _wire_has_tool(row, "finish")),
                "wire_tokens": sum(_wire_tokens(row) for row in rows if _is_successful_wire(row)),
                "first_timestamp": rows[0].get("timestamp") or "",
                "last_timestamp": rows[-1].get("timestamp") or "",
                "first_model_traffic_request_id": rows[0].get("model_traffic_request_id") or "",
                "last_model_traffic_request_id": rows[-1].get("model_traffic_request_id") or "",
            }
        )
    return {
        "problem_idx": trial_key[0],
        "repeat": trial_key[1],
        "session_count": len(sessions),
        "selected_session_id": selected_session_id,
        "retry_session_count": max(0, len(sessions) - 1),
        "total_wire_calls": sum(len(rows) for rows in sessions.values()),
        "total_finish_calls": sum(row["finish_calls"] for row in session_rows),
        "retry_successful_wire_calls": sum(
            row["successful_wire_calls"] for row in session_rows if row["session_id"] != selected_session_id
        ),
        "retry_wire_tokens": sum(
            row["wire_tokens"] for row in session_rows if row["session_id"] != selected_session_id
        ),
        "sessions": session_rows,
    }


def _retry_summary(
    traffic_by_trial_all: dict[tuple[Any, Any], list[dict[str, Any]]],
    selected_sessions_by_trial: dict[tuple[Any, Any], str],
) -> dict[str, Any]:
    examples: list[dict[str, Any]] = []
    problems_with_retries = 0
    problems_with_multiple_finish_calls = 0
    retry_sessions = 0
    retry_successful_wire_calls = 0
    retry_wire_tokens = 0

    for trial_key, rows in traffic_by_trial_all.items():
        sessions: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            session_id = _wire_session_id(row)
            if not session_id:
                continue
            sessions.setdefault(session_id, []).append(row)

        if len(sessions) <= 1:
            continue

        problems_with_retries += 1
        total_finish_calls = sum(
            1 for session_rows in sessions.values() for row in session_rows if _wire_has_tool(row, "finish")
        )
        if total_finish_calls > 1:
            problems_with_multiple_finish_calls += 1

        selected_session_id = selected_sessions_by_trial.get(trial_key, "")
        if selected_session_id:
            extra_rows = [
                row
                for session_id, session_rows in sessions.items()
                if session_id != selected_session_id
                for row in session_rows
            ]
            retry_sessions += sum(1 for session_id in sessions if session_id != selected_session_id)
            retry_successful_wire_calls += sum(1 for row in extra_rows if _is_successful_wire(row))
            retry_wire_tokens += sum(_wire_tokens(row) for row in extra_rows if _is_successful_wire(row))

        if len(examples) < _MAX_RETRY_EXAMPLES:
            examples.append(_session_group_example(trial_key, sessions, selected_session_id=selected_session_id))

    return {
        "problems_with_retries": problems_with_retries,
        "problems_with_multiple_finish_calls": problems_with_multiple_finish_calls,
        "retry_sessions": retry_sessions,
        "retry_successful_wire_calls": retry_successful_wire_calls,
        "retry_wire_tokens": retry_wire_tokens,
        "retry_examples": examples,
    }


def _trial_failure_example(
    row: dict[str, Any],
    *,
    agent_step_count: int,
    successful_wire_count: int,
    all_wire_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    fm = _final_metrics(row)
    last_wire = all_wire_rows[-1] if all_wire_rows else {}
    return {
        "problem_idx": row.get("problem_idx"),
        "repeat": row.get("repeat"),
        "reward": row.get("reward"),
        "failure_category": _trial_failure_category(row, all_wire_rows),
        "error": _trial_failure_error(row, all_wire_rows),
        "agent_steps": agent_step_count,
        "successful_wire_calls": successful_wire_count,
        "all_wire_calls": len(all_wire_rows),
        "last_wire_status_code": last_wire.get("status_code"),
        "last_wire_error_type": last_wire.get("error_type") or "",
        "last_wire_failure_category": _wire_failure_category(last_wire),
        "last_wire_error_message": _truncate_text(last_wire.get("error_message")),
        "last_wire_error_body": _truncate_text(last_wire.get("error_body")),
        "missing_final_metrics_tokens": not _has_final_metric_tokens(row),
        "workspace_diff_preview_present": bool(fm.get("workspace_diff_preview")),
    }


def _is_timeout_failure(example: dict[str, Any]) -> bool:
    text = f"{example.get('failure_category', '')} {example.get('error', '')}".lower()
    return "timeout" in text or "run_timeout" in text


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


def _count_present(items: list[dict[str, Any]], *keys: Any) -> tuple[int, int]:
    """Return (present, empty) counts along *keys* path across *items*.

    - present: non-None, non-"", non-empty-collection value found at path
    - empty: path reachable but value is None / "" / [] / {}
    Key not found anywhere along the path → neither bucket (truly absent).
    """
    present = empty = 0
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
        if not ok:
            continue
        if cur is not None and cur != "" and not (isinstance(cur, (list, dict)) and not cur):
            present += 1
        else:
            empty += 1
    return present, empty


def _missing_fields(items: list[dict[str, Any]], paths: dict[str, tuple[Any, ...]]) -> list[dict[str, str]]:
    """Return entries where fewer than all items have a non-empty value at the field path.

    Each entry includes ``presence`` (non-empty count), and ``empty`` when some
    items have the key set to None / [] / {} rather than a real value — so callers
    can distinguish a capture gap (truly absent) from an intentionally-empty field
    (e.g. ``tool_calls: null`` on a text-only step).
    A fully-populated field is silent.
    """
    total = len(items)
    if not total:
        return []
    out: list[dict[str, str]] = []
    for label, path in paths.items():
        present, empty = _count_present(items, *path)
        if present < total:
            entry: dict[str, Any] = {"field": label, "presence": f"{present}/{total}"}
            if empty:
                entry["empty"] = empty
            out.append(entry)
    return out


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


def _last_wire_session_id(rows: list[dict[str, Any]]) -> str:
    for row in reversed(rows):
        session_id = _wire_session_id(row)
        if session_id:
            return session_id
    return ""


def _select_wire_session_id(_row: dict[str, Any], wire_rows: list[dict[str, Any]]) -> str:
    """Select the wire session that represents a trajectory row.

    Model traffic can contain abandoned/replayed sessions for the same
    ``(problem_idx, repeat)``. The saved trajectory corresponds to the final
    attempt, so token/step alignment uses the last session in file order.
    Rows with no session ids are left unfiltered by returning an empty string.
    """
    return _last_wire_session_id(wire_rows)


def _selected_sessions_by_trial(
    traj_rows: list[dict[str, Any]],
    traffic_by_trial_all: dict[tuple[Any, Any], list[dict[str, Any]]],
) -> dict[tuple[Any, Any], str]:
    selected: dict[tuple[Any, Any], str] = {}
    for row in traj_rows:
        key = _trial_key(row)
        selected[key] = _select_wire_session_id(row, traffic_by_trial_all.get(key, []))
    return selected


def _filter_traffic_to_selected_sessions(
    traffic_by_trial_all: dict[tuple[Any, Any], list[dict[str, Any]]],
    selected_sessions_by_trial: dict[tuple[Any, Any], str],
    *,
    only_successful: bool = True,
) -> dict[tuple[Any, Any], list[dict[str, Any]]]:
    selected_rows: dict[tuple[Any, Any], list[dict[str, Any]]] = {}
    for key, rows in traffic_by_trial_all.items():
        if key not in selected_sessions_by_trial:
            continue
        selected_session_id = selected_sessions_by_trial[key]
        if selected_session_id:
            rows = [row for row in rows if _wire_session_id(row) == selected_session_id]
        if only_successful:
            rows = [row for row in rows if _is_successful_wire(row)]
        selected_rows[key] = rows
    return selected_rows


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


def _zero_token_turn_count(agent_steps: list[dict[str, Any]]) -> int:
    """Count turns where both input and completion tokens are zero.

    Uses delta prompt_tokens between consecutive steps (Piotr-style), since
    prompt_tokens may be cumulative in some solvers. When delta is negative,
    falls back to the absolute prompt_tokens for that step.
    """
    previous_prompt: int | None = None
    zero_count = 0
    for step in agent_steps:
        m = step.get("metrics") or {}
        prompt_tokens = int(m.get("prompt_tokens") or 0)
        completion_tokens = int(m.get("completion_tokens") or 0)
        if previous_prompt is None:
            input_tokens = prompt_tokens
        else:
            input_tokens = prompt_tokens - previous_prompt
            if input_tokens < 0:
                input_tokens = prompt_tokens
        previous_prompt = prompt_tokens
        if input_tokens == 0 and completion_tokens == 0:
            zero_count += 1
    return zero_count


def _wire_error_summary(wire_rows: list[dict[str, Any]]) -> tuple[dict[str, int], dict[str, dict[str, Any]]]:
    """Aggregate non-200 wire calls into a count-by-status dict and one example per code."""
    by_status: Counter[str] = Counter()
    examples: dict[str, dict[str, Any]] = {}
    for r in wire_rows:
        if not _is_successful_wire(r):
            code = str(r.get("status_code", "unknown"))
            by_status[code] += 1
            if code not in examples:
                examples[code] = {
                    k: r[k]
                    for k in (
                        "status_code",
                        "error_type",
                        "error_code",
                        "error_message",
                        "error_body",
                        "latency_ms",
                        "model",
                        "path",
                        "model_traffic_request_id",
                    )
                    if r.get(k) is not None
                }
    return dict(by_status), examples


def _piotr_quality(agent_steps: list[dict[str, Any]]) -> dict[str, int]:
    """Piotr-style trajectory quality checks over a list of agent steps.

    Returns a dict with zero-token-turn counts, missed-metrics counts, and
    clean/dirty/fully-zero problem counts (all treated as a single problem).
    For multi-problem batches use ``_piotr_quality_aggregate`` instead.
    """
    ztt = _zero_token_turn_count(agent_steps)
    missed = sum(1 for s in agent_steps if not s.get("metrics"))
    has_no_steps = not agent_steps
    has_zero = ztt > 0
    return {
        "zero_token_turns": ztt,
        "missed_metrics": missed,
        "has_no_agent_steps": has_no_steps,
        "has_zero_token_turns": has_zero,
        "is_fully_zero": bool(agent_steps) and ztt == len(agent_steps),
        "is_clean": not has_no_steps and not has_zero,
    }


def _piotr_quality_aggregate(traj_rows: list[dict[str, Any]]) -> dict[str, int]:
    """Aggregate Piotr quality checks across all trials in a benchmark."""
    total_zero_token_turns = 0
    total_missed_metrics = 0
    problems_with_zero_token_turns = 0
    problems_with_missed_metrics = 0
    clean_problems = 0
    dirty_problems = 0
    fully_zero_problems = 0

    for r in traj_rows:
        steps = _agent_steps(r)
        q = _piotr_quality(steps)
        total_zero_token_turns += q["zero_token_turns"]
        total_missed_metrics += q["missed_metrics"]
        if q["has_zero_token_turns"]:
            problems_with_zero_token_turns += 1
        if q["missed_metrics"] > 0:
            problems_with_missed_metrics += 1
        if q["is_fully_zero"]:
            fully_zero_problems += 1
        if q["is_clean"]:
            clean_problems += 1
        else:
            dirty_problems += 1

    return {
        "clean_problems": clean_problems,
        "dirty_problems": dirty_problems,
        "fully_zero_problems": fully_zero_problems,
        "problems_with_zero_token_turns": problems_with_zero_token_turns,
        "problems_with_missed_metrics": problems_with_missed_metrics,
        "total_zero_token_turns": total_zero_token_turns,
        "total_missed_metrics": total_missed_metrics,
    }


def _build_bench_report(bench_name: str, bench_dir: Path) -> dict[str, Any]:
    traj_rows = _read_jsonl(bench_dir / "trajectories.jsonl")
    traffic_rows = _read_jsonl(bench_dir / "model_traffic.jsonl")
    traffic_by_trial_all_raw = _index_traffic_by_trial(traffic_rows, only_successful=False)
    selected_sessions = _selected_sessions_by_trial(traj_rows, traffic_by_trial_all_raw)
    traffic_by_trial = _filter_traffic_to_selected_sessions(traffic_by_trial_all_raw, selected_sessions)
    traffic_by_trial_all = _filter_traffic_to_selected_sessions(
        traffic_by_trial_all_raw,
        selected_sessions,
        only_successful=False,
    )

    n_problems = len({r.get("problem_idx") for r in traj_rows if r.get("problem_idx") is not None})

    rewards = [r["reward"] for r in traj_rows if isinstance(r.get("reward"), (int, float))]
    traj_mean = round(sum(rewards) / len(rewards), 6) if rewards else None
    reported = _load_eval_summary(bench_dir)
    reported_mean = reported.get("mean") if isinstance(reported, dict) else None

    # Single pass over traj_rows: collect everything per-trial needs.
    all_agent_steps: list[dict[str, Any]] = []
    failures: Counter[str] = Counter()
    problems_more_wire = problems_fewer_wire = 0
    problems_no_steps = problems_no_wire = problems_silent = 0
    per_step_vs_fm_mismatch = 0
    wire_vs_fm_mismatch = 0
    problems_missing_fm_tokens = 0
    final_metrics_token_sum = 0
    problems_with_last_wire_non_200 = 0
    failure_examples: list[dict[str, Any]] = []
    timeout_examples: list[dict[str, Any]] = []
    no_agent_step_examples: list[dict[str, Any]] = []

    for r in traj_rows:
        steps = _agent_steps(r)
        all_agent_steps.extend(steps)

        trial_wire_rows = traffic_by_trial.get(_trial_key(r), [])
        all_trial_wire = traffic_by_trial_all.get(_trial_key(r), [])
        cat = _trial_failure_category(r, all_trial_wire)
        if cat:
            failures[cat] += 1

        step_n = len(steps)
        wire_n = len(trial_wire_rows)
        if wire_n > step_n:
            problems_more_wire += 1
        elif wire_n < step_n:
            problems_fewer_wire += 1
        if step_n == 0:
            problems_no_steps += 1
        if wire_n == 0:
            problems_no_wire += 1
        if step_n == 0 or wire_n == 0:
            problems_silent += 1

        fm = _final_metrics(r)
        fm_prompt = fm.get("total_prompt_tokens")
        fm_completion = fm.get("total_completion_tokens")
        fm_has_tokens = fm_prompt is not None or fm_completion is not None
        fm_total = int(fm_prompt or 0) + int(fm_completion or 0)

        if not fm_has_tokens:
            problems_missing_fm_tokens += 1
        else:
            final_metrics_token_sum += fm_total

        per_step = sum(_step_tokens(s) for s in steps)
        if per_step > 0 and per_step != fm_total:
            per_step_vs_fm_mismatch += 1

        wire_trial_total = sum(_wire_tokens(w) for w in trial_wire_rows)
        if wire_trial_total > 0 and wire_trial_total != fm_total:
            wire_vs_fm_mismatch += 1

        # Last wire call: check if the chronologically last call ended with non-200
        if all_trial_wire and not _is_successful_wire(all_trial_wire[-1]):
            problems_with_last_wire_non_200 += 1

        failure_signal = bool(
            cat
            or _failure_error(r)
            or step_n == 0
            or not fm_has_tokens
            or any(not _is_successful_wire(w) for w in all_trial_wire)
        )
        if failure_signal:
            example = _trial_failure_example(
                r,
                agent_step_count=step_n,
                successful_wire_count=wire_n,
                all_wire_rows=all_trial_wire,
            )
            failure_examples.append(example)
            if example["agent_steps"] == 0:
                no_agent_step_examples.append(example)
            if _is_timeout_failure(example):
                timeout_examples.append(example)

    per_step_token_sum = sum(_step_tokens(s) for s in all_agent_steps)
    selected_all_traffic_rows = [row for rows in traffic_by_trial_all.values() for row in rows]
    selected_successful_traffic_rows = [row for rows in traffic_by_trial.values() for row in rows]
    wire_token_sum = sum(_wire_tokens(r) for r in selected_successful_traffic_rows)
    wire_token_sum_all_sessions = sum(_wire_tokens(r) for r in traffic_rows if _is_successful_wire(r))

    # Wire dedup: how many trials had ≥1 dup, and the unique count overall.
    problems_with_dup_wire = 0
    for recs in traffic_by_trial_all_raw.values():
        keys = Counter(_wire_dedup_key(r, i) for i, r in enumerate(recs))
        if any(v > 1 for v in keys.values()):
            problems_with_dup_wire += 1
    wire_unique = len({_wire_dedup_key(r, i) for i, r in enumerate(traffic_rows)})

    non_200_by_status, non_200_examples = _wire_error_summary(selected_all_traffic_rows)
    non_200_by_status_all_sessions, non_200_examples_all_sessions = _wire_error_summary(traffic_rows)
    last_wire_by_trial = {key: rows[-1] for key, rows in traffic_by_trial_all_raw.items() if rows}
    non_success_examples = [
        _wire_example(r, is_last_wire=last_wire_by_trial.get(_trial_key(r)) is r)
        for r in traffic_rows
        if not _is_successful_wire(r)
    ]
    non_success_by_error_type = _group_wire_examples_by_error_type(non_success_examples)
    last_non_success_by_error_type = _group_wire_examples_by_error_type(
        [e for e in non_success_examples if e["is_last_wire"]]
    )
    retry_summary = _retry_summary(traffic_by_trial_all_raw, selected_sessions)

    all_sources_match = per_step_vs_fm_mismatch == 0 and wire_vs_fm_mismatch == 0 and problems_missing_fm_tokens == 0

    report: dict[str, Any] = {
        "bench": bench_name,
        "trajectories": {
            "problems": n_problems,
            "repeats": _repeats_summary(traj_rows),
            "mean_reward": traj_mean,
            "reported_mean": reported_mean,
            "failures_by_category": dict(failures),
            "failure_examples": failure_examples,
            "timeout_examples": timeout_examples,
            "no_agent_step_examples": no_agent_step_examples,
            "quality": _piotr_quality_aggregate(traj_rows),
            "field_coverage": {
                "per_trial_missing": _missing_fields(traj_rows, _TRIAL_FIELDS),
                "per_agent_step_missing": _missing_fields(all_agent_steps, _STEP_FIELDS),
            },
        },
        "tokens_stats": {
            "per_step_sum": per_step_token_sum,
            "wire_total": wire_token_sum,
            "wire_total_for_trajectory_comparison": wire_token_sum,
            "wire_total_all_sessions": wire_token_sum_all_sessions,
            "wire_total_earlier_retry_sessions": retry_summary["retry_wire_tokens"],
            "final_metrics_total": final_metrics_token_sum,
            "problems_with_missing_final_metrics_tokens": problems_missing_fm_tokens,
            "problems_with_per_step_vs_final_metrics_mismatch": per_step_vs_fm_mismatch,
            "problems_with_wire_vs_final_metrics_mismatch": wire_vs_fm_mismatch,
            "all_sources_match": all_sources_match,
        },
        "wire_calls": {
            "total": len(selected_all_traffic_rows),
            "successful": len(selected_successful_traffic_rows),
            "non_200": sum(1 for r in selected_all_traffic_rows if not _is_successful_wire(r)),
            "total_for_trajectory_comparison": len(selected_all_traffic_rows),
            "successful_for_trajectory_comparison": len(selected_successful_traffic_rows),
            "total_all_sessions": len(traffic_rows),
            "successful_all_sessions": sum(1 for r in traffic_rows if _is_successful_wire(r)),
            "non_200_all_sessions": sum(1 for r in traffic_rows if not _is_successful_wire(r)),
            "non_200_by_status": non_200_by_status,
            "non_200_examples": non_200_examples,
            "non_200_by_status_all_sessions": non_200_by_status_all_sessions,
            "non_200_examples_all_sessions": non_200_examples_all_sessions,
            "non_success_examples": non_success_by_error_type,
            "last_non_success_examples": last_non_success_by_error_type,
            "unique": wire_unique,
            "problems_with_duplicates": problems_with_dup_wire,
            "selected_session_policy": "last_wire_session",
            "problems_with_retries": retry_summary["problems_with_retries"],
            "retry_sessions": retry_summary["retry_sessions"],
            "problems_with_multiple_finish_calls": retry_summary["problems_with_multiple_finish_calls"],
            "retry_successful_wire_calls": retry_summary["retry_successful_wire_calls"],
            "retry_wire_tokens": retry_summary["retry_wire_tokens"],
            "retry_examples": retry_summary["retry_examples"],
            "problems_with_more_wire_than_steps": problems_more_wire,
            "problems_with_fewer_wire_than_steps": problems_fewer_wire,
            "problems_with_no_agent_steps": problems_no_steps,
            "problems_with_no_wire_calls": problems_no_wire,
            "problems_silent_either_way": problems_silent,
            "problems_with_last_wire_non_200": problems_with_last_wire_non_200,
        },
    }
    return report


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
    all_by_trial = _index_traffic_by_trial(traffic_rows, only_successful=False)
    selected_sessions = _selected_sessions_by_trial(traj_rows, all_by_trial)
    by_trial = _filter_traffic_to_selected_sessions(all_by_trial, selected_sessions)
    by_trial_all = _filter_traffic_to_selected_sessions(all_by_trial, selected_sessions, only_successful=False)

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
        "rows_reclassified_from_wire_failure": 0,
        "rows_cleared_empty_response_after_verification": 0,
        "rows_written": 0,
    }

    def derive_missing_step_total_tokens(steps: list[dict[str, Any]]) -> None:
        for step in steps:
            metrics = step.get("metrics")
            if not isinstance(metrics, dict) or metrics.get("total_tokens") not in (None, 0):
                continue
            prompt_tokens = metrics.get("prompt_tokens") or 0
            completion_tokens = metrics.get("completion_tokens") or 0
            if not prompt_tokens and not completion_tokens:
                continue
            try:
                metrics["total_tokens"] = int(prompt_tokens) + int(completion_tokens)
            except (TypeError, ValueError):
                continue
            counts["steps_backfilled_total_tokens"] += 1

    with enriched_path.open("w", encoding="utf-8") as fh:
        for r in traj_rows:
            steps = _agent_steps(r)
            wire = by_trial.get(_trial_key(r), [])
            all_wire = by_trial_all.get(_trial_key(r), [])
            category = _trial_failure_category(r, all_wire)
            original_category = _failure_category(r)
            if category != original_category:
                if category:
                    r["failure_category"] = category
                    counts["rows_reclassified_from_wire_failure"] += 1
                elif original_category == "empty_response" and _completed_harbor_verification_without_error(r):
                    r.pop("failure_category", None)
                    counts["rows_cleared_empty_response_after_verification"] += 1
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
            derive_missing_step_total_tokens(steps)
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
            enriched_path = d / "trajectories_enriched.jsonl"
            enriched_rows = _read_jsonl(enriched_path) if enriched_path.is_file() else []
            enriched_steps = [s for r in enriched_rows for s in _agent_steps(r)]
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
                "rows_reclassified_from_wire_failure": counts["rows_reclassified_from_wire_failure"],
                "rows_cleared_empty_response_after_verification": counts[
                    "rows_cleared_empty_response_after_verification"
                ],
                "quality": _piotr_quality_aggregate(enriched_rows),
                "per_step_sum_after_enrichment": sum(_step_tokens(s) for s in enriched_steps),
                "step_field_coverage_after_enrichment_missing": _missing_fields(
                    enriched_steps,
                    {
                        ".".join(str(p) for p in path): path
                        for path in (
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
