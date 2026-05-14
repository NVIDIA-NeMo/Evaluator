# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import threading
import time
import uuid
from collections import Counter
from typing import TYPE_CHECKING, Any, Iterator

if TYPE_CHECKING:
    from nemo_evaluator.adapters.types import AdapterRequest, AdapterResponse

_REGISTRY: dict[str, "ModelTrafficStore"] = {}
_REGISTRY_LOCK = threading.Lock()
_USAGE_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens", "reasoning_tokens", "cached_tokens")
_PROVIDER = "provider_reported"


def register_store(store: "ModelTrafficStore") -> str:
    with _REGISTRY_LOCK:
        _REGISTRY[store.store_id] = store
    return store.store_id


def unregister_store(store_id: str) -> None:
    with _REGISTRY_LOCK:
        _REGISTRY.pop(store_id, None)


def get_store(store_id: str) -> "ModelTrafficStore":
    with _REGISTRY_LOCK:
        store = _REGISTRY.get(store_id)
    if store is None:
        raise ValueError(f"Unknown model traffic store id: {store_id}")
    return store


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _token_value(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _first_token(data: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = _token_value(data.get(key))
        if value is not None:
            return value
    return None


def _usage(raw: Any) -> dict[str, int]:
    if not isinstance(raw, dict):
        return {}
    prompt = _first_token(raw, "prompt_tokens", "input_tokens", "inputTokens")
    completion = _first_token(raw, "completion_tokens", "output_tokens", "outputTokens", "content_tokens")
    total = _first_token(raw, "total_tokens", "totalTokens")
    reasoning = _first_token(raw, "reasoning_tokens")
    cached = _first_token(raw, "cached_tokens")

    for details_key in ("completion_tokens_details", "output_tokens_details"):
        details = raw.get(details_key)
        if reasoning is None and isinstance(details, dict):
            reasoning = _first_token(details, "reasoning_tokens")
    for details_key in ("prompt_tokens_details", "input_tokens_details"):
        details = raw.get(details_key)
        if cached is None and isinstance(details, dict):
            cached = _first_token(details, "cached_tokens")
    if total is None and prompt is not None and completion is not None:
        total = prompt + completion

    out: dict[str, int] = {}
    for key, value in {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
        "reasoning_tokens": reasoning,
        "cached_tokens": cached,
    }.items():
        if value is not None:
            out[key] = value
    return out


def _tool_stats(names: list[str]) -> dict[str, Any]:
    counts = Counter(name for name in names if name)
    return {"count": len(names), "names": dict(sorted(counts.items()))}


def _tool_names_from_message(message: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for raw in message.get("tool_calls") or []:
        if not isinstance(raw, dict):
            continue
        function = raw.get("function") if isinstance(raw.get("function"), dict) else {}
        names.append(str(function.get("name") or raw.get("name") or ""))
    return names


def _summary_from_json(body: dict[str, Any]) -> dict[str, Any]:
    tool_names: list[str] = []
    finish_reason = ""
    for choice in body.get("choices") or []:
        if not isinstance(choice, dict):
            continue
        finish_reason = str(choice.get("finish_reason") or finish_reason)
        message = choice.get("message") or choice.get("delta") or {}
        if isinstance(message, dict):
            tool_names.extend(_tool_names_from_message(message))
    return {
        "model": str(body.get("model") or ""),
        "finish_reason": finish_reason,
        "usage": _usage(body.get("usage")),
        "tool_calls": _tool_stats(tool_names),
    }


def _iter_sse(body: Any) -> Iterator[dict[str, Any]]:
    text = body.decode("utf-8", errors="replace") if isinstance(body, bytes) else body
    if not isinstance(text, str):
        return

    data_lines: list[str] = []

    def load_payload(payload: str) -> dict[str, Any] | None:
        payload = payload.strip()
        if not payload or payload == "[DONE]":
            return None
        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            return None
        return chunk if isinstance(chunk, dict) else None

    for raw_line in text.splitlines():
        line = raw_line.rstrip("\r")
        if not line:
            chunk = load_payload("\n".join(data_lines))
            data_lines = []
            if chunk is not None:
                yield chunk
            continue
        if not line.startswith("data:"):
            continue
        payload = line[5:]
        if payload.startswith(" "):
            payload = payload[1:]
        data_lines.append(payload)

    chunk = load_payload("\n".join(data_lines))
    if chunk is not None:
        yield chunk


def _summary_from_sse(body: Any) -> dict[str, Any]:
    model = ""
    finish_reason = ""
    usage: dict[str, int] = {}
    stream_tools: dict[tuple[int, int], str] = {}
    for chunk in _iter_sse(body):
        model = str(chunk.get("model") or model)
        if isinstance(chunk.get("usage"), dict):
            usage = _usage(chunk["usage"])
        for choice in chunk.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            finish_reason = str(choice.get("finish_reason") or finish_reason)
            delta = choice.get("delta") or choice.get("message") or {}
            if not isinstance(delta, dict):
                continue
            for raw in delta.get("tool_calls") or []:
                if not isinstance(raw, dict):
                    continue
                idx = (_int(choice.get("index")), _int(raw.get("index")))
                function = raw.get("function") if isinstance(raw.get("function"), dict) else {}
                stream_tools[idx] = f"{stream_tools.get(idx, '')}{function.get('name') or raw.get('name') or ''}"
    return {
        "model": model,
        "finish_reason": finish_reason,
        "usage": usage,
        "tool_calls": _tool_stats(list(stream_tools.values())),
    }


def _summary(body: Any) -> dict[str, Any]:
    return _summary_from_json(body) if isinstance(body, dict) else _summary_from_sse(body)


def _is_success(record: dict[str, Any]) -> bool:
    return 200 <= _int(record.get("status_code")) < 400


class ModelTrafficStore:
    def __init__(self, *, service_name: str | None = None) -> None:
        self.store_id = uuid.uuid4().hex
        self.service_name = service_name
        self._lock = threading.Lock()
        self._pending: dict[str, dict[str, Any]] = {}
        self._records_by_session: dict[str, list[dict[str, Any]]] = {}

    def close(self) -> None:
        unregister_store(self.store_id)

    def start_request(self, req: AdapterRequest) -> None:
        body = req.ctx.extra.get("upstream_request_body") or req.body
        record = {
            "request_id": req.ctx.request_id,
            "session_id": req.ctx.extra.get("session_id"),
            "timestamp": time.time(),
            "method": req.method,
            "path": req.path,
            "service": self.service_name,
            "model": body.get("model") if isinstance(body, dict) else "",
        }
        with self._lock:
            self._pending[req.ctx.request_id] = record

    def finish_response(self, resp: AdapterResponse) -> None:
        with self._lock:
            record = self._pending.pop(resp.ctx.request_id, None)
        if record is None:
            record = {
                "request_id": resp.ctx.request_id,
                "session_id": resp.ctx.extra.get("session_id"),
                "timestamp": time.time(),
            }

        summary = _summary(resp.body)
        success = 200 <= resp.status_code < 400
        record.update(
            {
                "status_code": resp.status_code,
                "latency_ms": resp.latency_ms,
                "model": summary["model"] or record.get("model") or "",
                "finish_reason": summary["finish_reason"] if success else "",
                "usage": summary["usage"] if success else {},
                "tool_calls": summary["tool_calls"] if success else {"count": 0, "names": {}},
            }
        )
        session_id = record.get("session_id")
        if session_id:
            with self._lock:
                self._records_by_session.setdefault(str(session_id), []).append(record)

    def finish_error(self, req: AdapterRequest, *, latency_ms: float, error_type: str) -> None:
        with self._lock:
            record = self._pending.pop(req.ctx.request_id, None)
        if record is None:
            record = {
                "request_id": req.ctx.request_id,
                "session_id": req.ctx.extra.get("session_id"),
                "timestamp": time.time(),
                "method": req.method,
                "path": req.path,
                "service": self.service_name,
                "model": "",
            }

        record.update(
            {
                "status_code": None,
                "latency_ms": latency_ms,
                "finish_reason": "",
                "usage": {},
                "tool_calls": {"count": 0, "names": {}},
                "error_type": error_type,
            }
        )
        session_id = record.get("session_id")
        if session_id:
            with self._lock:
                self._records_by_session.setdefault(str(session_id), []).append(record)

    def drain_session(self, session_id: str | None) -> list[dict[str, Any]]:
        if not session_id:
            return []
        with self._lock:
            return self._records_by_session.pop(session_id, [])


def _successful(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [record for record in records if _is_success(record)]


def _aggregate_usage(records: list[dict[str, Any]]) -> dict[str, Any]:
    usage: dict[str, int] = {}
    for record in _successful(records):
        for key, value in (record.get("usage") or {}).items():
            if key in _USAGE_KEYS:
                usage[key] = usage.get(key, 0) + _int(value)
    return {**usage, "token_provenance": _PROVIDER} if usage else {}


def _aggregate_tool_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = 0
    names: Counter[str] = Counter()
    for record in _successful(records):
        stats = record.get("tool_calls") or {}
        total += _int(stats.get("count"))
        names.update(stats.get("names") or {})
    return {"count": total, "names": dict(sorted(names.items()))}


def aggregate_model_traffic_stats(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not records:
        return None
    return {
        "usage": _aggregate_usage(records),
        "tool_calls": _aggregate_tool_stats(records),
        "model_calls": len(records),
    }
