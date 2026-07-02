# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

from nemo_evaluator.engine.step_log import StepLog
from nemo_evaluator.observability.model_traffic import ModelTrafficStore

_USAGE_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens", "reasoning_tokens", "cached_tokens")
_PROVIDER = "provider_reported"


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _traffic_id(record: dict[str, Any]) -> str:
    rid = str(record.get("request_id") or "")
    sid = record.get("session_id")
    return f"{sid}:{rid}" if sid else rid


def _timestamp(value: Any) -> str:
    try:
        ts = float(value)
    except (TypeError, ValueError):
        ts = time.time()
    return datetime.fromtimestamp(ts, timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def drain_model_traffic_session(
    model_traffic_store: ModelTrafficStore | None,
    session_id: str | None,
) -> list[dict[str, Any]]:
    """Return and clear traffic records for one Adapter Session."""
    if model_traffic_store is None or not session_id:
        return []
    return model_traffic_store.drain_session(session_id)


async def append_model_traffic_records(
    model_traffic_log: StepLog | None,
    records: list[dict[str, Any]],
    *,
    benchmark: str,
    problem_idx: int,
    repeat: int,
) -> None:
    """Append captured model traffic as attempt-scoped step-log rows."""
    if model_traffic_log is None or not records:
        return
    for traffic_record in format_model_traffic_log_records(
        records,
        benchmark=benchmark,
        problem_idx=problem_idx,
        repeat=repeat,
    ):
        await model_traffic_log.append(traffic_record)


def format_model_traffic_log_records(
    records: list[dict[str, Any]],
    *,
    benchmark: str,
    problem_idx: int,
    repeat: int,
) -> list[dict[str, Any]]:
    """Convert drained traffic records into model_traffic.jsonl rows."""
    rows: list[dict[str, Any]] = []
    for record in records:
        usage = record.get("usage") or {}
        row = {
            "model_traffic_request_id": _traffic_id(record),
            "timestamp": _timestamp(record.get("timestamp")),
            "benchmark": benchmark,
            "problem_idx": problem_idx,
            "repeat": repeat,
            "session_id": record.get("session_id"),
            "adapter_request_id": record.get("request_id"),
            "request_hash": record.get("request_hash") or "",
            "service": record.get("service"),
            "method": record.get("method"),
            "path": record.get("path"),
            "status_code": record.get("status_code"),
            "latency_ms": record.get("latency_ms"),
            "model": record.get("model") or "",
            "finish_reason": record.get("finish_reason") or "",
            "usage": {key: _int(usage.get(key)) for key in _USAGE_KEYS if key in usage},
            "tool_calls": record.get("tool_calls") or {"count": 0, "names": {}},
        }
        if row["usage"]:
            row["token_provenance"] = _PROVIDER
        for key in ("error_type", "error_message", "error_body", "error_code"):
            if record.get(key):
                row[key] = record[key]
        # Opt-in capture fields from ModelTrafficStore.finish_response: only
        # forwarded when present in the in-memory record (controlled by the
        # service's proxy.model_traffic.capture_{tool_calls,reasoning,messages}).
        for key in ("tool_calls_full", "reasoning_content", "message_content"):
            if key in record:
                row[key] = record[key]
        rows.append(row)
    return rows
