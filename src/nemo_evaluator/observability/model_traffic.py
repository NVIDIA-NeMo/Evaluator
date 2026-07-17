# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from collections import Counter
from typing import TYPE_CHECKING, Any, Iterator


def _request_hash(body: Any) -> str | None:
    """Deterministic content hash of an upstream request body.

    Returns 16 hex chars (a sha1 prefix) suitable for grouping duplicate
    calls. ``None`` when *body* isn't a dict (e.g. raw bytes / unset).
    """
    if not isinstance(body, dict):
        return None
    try:
        payload = json.dumps(body, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return None
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


if TYPE_CHECKING:
    from nemo_evaluator.adapters.types import AdapterRequest, AdapterResponse

_REGISTRY: dict[str, "ModelTrafficStore"] = {}
_REGISTRY_LOCK = threading.Lock()
_USAGE_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens", "reasoning_tokens", "cached_tokens")
_PROVIDER = "provider_reported"
_ERROR_SUMMARY_CHARS = 4000


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


def _truncate(value: Any, limit: int) -> Any:
    if isinstance(value, str) and limit > 0 and len(value) > limit:
        return value[:limit] + f"...[truncated, {len(value) - limit} chars dropped]"
    return value


def _capture_request_body_value(body: Any, max_content_chars: int) -> Any | None:
    if not isinstance(body, dict):
        return None
    try:
        serialized = json.dumps(body, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return None
    if max_content_chars > 0 and len(serialized) > max_content_chars:
        return _truncate(serialized, max_content_chars)
    return body


def _first_str(data: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = data.get(key)
        if value is not None and value != "":
            return str(value)
    return ""


def _error_summary(body: Any, max_content_chars: int) -> dict[str, Any]:
    """Return a compact, non-request error summary for non-success responses."""
    limit = _ERROR_SUMMARY_CHARS if max_content_chars <= 0 else min(max_content_chars, _ERROR_SUMMARY_CHARS)
    out: dict[str, Any] = {}

    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            message = _first_str(error, "message", "detail", "type", "code")
            if message:
                out["error_message"] = _truncate(message, limit)
            if error.get("type") is not None:
                out["error_type"] = str(error["type"])
            if error.get("code") is not None:
                out["error_code"] = str(error["code"])
        elif error is not None:
            out["error_message"] = _truncate(str(error), limit)

        if "error_message" not in out:
            message = _first_str(body, "message", "detail", "error_message")
            if message:
                out["error_message"] = _truncate(message, limit)
        try:
            out["error_body"] = _truncate(json.dumps(body, sort_keys=True, default=str), limit)
        except (TypeError, ValueError):
            pass
        return out

    if isinstance(body, bytes):
        text = body.decode("utf-8", errors="replace")
    else:
        text = str(body) if body is not None else ""
    if text:
        out["error_message"] = _truncate(text, limit)
        out["error_body"] = _truncate(text, limit)
    return out


def _full_tool_calls_from_message(message: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for raw in message.get("tool_calls") or []:
        if not isinstance(raw, dict):
            continue
        function = raw.get("function") if isinstance(raw.get("function"), dict) else {}
        args = function.get("arguments") if "arguments" in function else raw.get("arguments")
        if isinstance(args, str):
            args = _truncate(args, limit)
        elif isinstance(args, (dict, list)):
            try:
                args = _truncate(json.dumps(args), limit)
            except (TypeError, ValueError):
                args = None
        out.append(
            {
                "id": raw.get("id") or "",
                "name": function.get("name") or raw.get("name") or "",
                "arguments": args,
            }
        )
    return out


def _summary_from_json(
    body: dict[str, Any],
    *,
    capture_tool_calls: bool = True,
    capture_reasoning: bool = True,
    capture_messages: bool = True,
    max_content_chars: int = 100_000,
) -> dict[str, Any]:
    tool_names: list[str] = []
    finish_reason = ""
    full_tool_calls: list[dict[str, Any]] = []
    reasoning_parts: list[str] = []
    message_parts: list[str] = []
    for choice in body.get("choices") or []:
        if not isinstance(choice, dict):
            continue
        finish_reason = str(choice.get("finish_reason") or finish_reason)
        message = choice.get("message") or choice.get("delta") or {}
        if isinstance(message, dict):
            tool_names.extend(_tool_names_from_message(message))
            if capture_tool_calls:
                full_tool_calls.extend(_full_tool_calls_from_message(message, max_content_chars))
            if capture_reasoning:
                rc = message.get("reasoning_content")
                if isinstance(rc, str):
                    reasoning_parts.append(rc)
            if capture_messages:
                ct = message.get("content")
                if isinstance(ct, str):
                    message_parts.append(ct)
    out: dict[str, Any] = {
        "model": str(body.get("model") or ""),
        "finish_reason": finish_reason,
        "usage": _usage(body.get("usage")),
        "tool_calls": _tool_stats(tool_names),
    }
    if capture_tool_calls:
        out["tool_calls_full"] = full_tool_calls
    if capture_reasoning:
        out["reasoning_content"] = _truncate("".join(reasoning_parts), max_content_chars)
    if capture_messages:
        out["message_content"] = _truncate("".join(message_parts), max_content_chars)
    return out


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


def _summary_from_sse(
    body: Any,
    *,
    capture_tool_calls: bool = True,
    capture_reasoning: bool = True,
    capture_messages: bool = True,
    max_content_chars: int = 100_000,
) -> dict[str, Any]:
    model = ""
    finish_reason = ""
    usage: dict[str, int] = {}
    stream_tools: dict[tuple[int, int], str] = {}
    # Per (choice, tool) index buffers for the full-capture path.
    tool_full: dict[tuple[int, int], dict[str, Any]] = {}
    reasoning_parts: list[str] = []
    message_parts: list[str] = []
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
                name_chunk = function.get("name") or raw.get("name") or ""
                stream_tools[idx] = f"{stream_tools.get(idx, '')}{name_chunk}"
                if capture_tool_calls:
                    entry = tool_full.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                    if raw.get("id"):
                        entry["id"] = raw["id"]
                    if name_chunk:
                        entry["name"] = entry["name"] + name_chunk
                    args_chunk = function.get("arguments") if "arguments" in function else raw.get("arguments")
                    if isinstance(args_chunk, str):
                        entry["arguments"] = entry["arguments"] + args_chunk
            if capture_reasoning:
                rc = delta.get("reasoning_content")
                if isinstance(rc, str):
                    reasoning_parts.append(rc)
            if capture_messages:
                ct = delta.get("content")
                if isinstance(ct, str):
                    message_parts.append(ct)
    out: dict[str, Any] = {
        "model": model,
        "finish_reason": finish_reason,
        "usage": usage,
        "tool_calls": _tool_stats(list(stream_tools.values())),
    }
    if capture_tool_calls:
        out["tool_calls_full"] = [
            {
                "id": v["id"],
                "name": v["name"],
                "arguments": _truncate(v["arguments"], max_content_chars) or None,
            }
            for v in tool_full.values()
        ]
    if capture_reasoning:
        out["reasoning_content"] = _truncate("".join(reasoning_parts), max_content_chars)
    if capture_messages:
        out["message_content"] = _truncate("".join(message_parts), max_content_chars)
    return out


def _summary(body: Any, **opts: Any) -> dict[str, Any]:
    return _summary_from_json(body, **opts) if isinstance(body, dict) else _summary_from_sse(body, **opts)


def _is_success(record: dict[str, Any]) -> bool:
    return 200 <= _int(record.get("status_code")) < 400


class ModelTrafficStore:
    def __init__(
        self,
        *,
        service_name: str | None = None,
        capture_tool_calls: bool = True,
        capture_reasoning: bool = True,
        capture_messages: bool = True,
        capture_request_body: bool = False,
        max_content_chars: int = 0,
    ) -> None:
        self.store_id = uuid.uuid4().hex
        self.service_name = service_name
        self._lock = threading.Lock()
        self._pending: dict[str, dict[str, Any]] = {}
        self._records_by_session: dict[str, list[dict[str, Any]]] = {}
        self._capture_request_body = capture_request_body
        self._capture_opts = {
            "capture_tool_calls": capture_tool_calls,
            "capture_reasoning": capture_reasoning,
            "capture_messages": capture_messages,
            "max_content_chars": max_content_chars,
        }

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
            # Content-derived id of the upstream request body. Lets downstream
            # tooling (e.g. trajectories audit report) detect duplicate calls
            # exactly, instead of fingerprinting on response stats.
            "request_hash": _request_hash(body),
        }
        if self._capture_request_body:
            captured = _capture_request_body_value(body, self._capture_opts["max_content_chars"])
            if captured is not None:
                record["request_body"] = captured
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

        summary = _summary(resp.body, **self._capture_opts)
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
        if not success:
            record.update(_error_summary(resp.body, self._capture_opts["max_content_chars"]))
        # Persist opt-in capture fields when present (and the call succeeded).
        if success:
            for key in ("tool_calls_full", "reasoning_content", "message_content"):
                if key in summary:
                    record[key] = summary[key]
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
