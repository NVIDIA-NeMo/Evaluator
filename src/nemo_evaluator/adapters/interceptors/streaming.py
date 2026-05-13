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
from typing import Any

from nemo_evaluator.adapters.types import AdapterResponse, ResponseInterceptor


def _normalize_message_content(body: dict[str, Any]) -> None:
    choices = body.get("choices")
    if not isinstance(choices, list):
        return

    for choice in choices:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if isinstance(message, dict) and "content" in message and message["content"] is None:
            message["content"] = ""


class Interceptor(ResponseInterceptor):
    async def intercept_response(self, resp: AdapterResponse) -> AdapterResponse:
        if isinstance(resp.body, dict):
            _normalize_message_content(resp.body)
            return resp

        content_type = _header_value(resp.headers, "content-type").lower()
        if "text/event-stream" not in content_type:
            return resp

        text = _body_to_text(resp.body)
        if text is None:
            return resp

        body = _coalesce_sse(text)
        if body is None:
            return resp

        headers = {k: v for k, v in resp.headers.items() if k.lower() != "content-type"}
        headers["Content-Type"] = "application/json"
        status_code = _status_code_for_body(resp.status_code, body)
        return AdapterResponse(
            status_code=status_code,
            headers=headers,
            body=body,
            latency_ms=resp.latency_ms,
            ctx=resp.ctx,
        )


def _header_value(headers: dict[str, str], name: str) -> str:
    for key, value in headers.items():
        if key.lower() == name:
            return value
    return ""


def _body_to_text(body: dict[str, Any] | bytes) -> str | None:
    if isinstance(body, bytes):
        return body.decode(errors="replace")
    return None


def _status_code_for_body(status_code: int, body: dict[str, Any]) -> int:
    if not (200 <= status_code < 400) or "error" not in body:
        return status_code

    error = body.get("error")
    if isinstance(error, dict):
        error_status = error.get("status") or error.get("status_code")
        if isinstance(error_status, int) and 400 <= error_status < 600:
            return error_status
    return 502


def _coalesce_sse(text: str) -> dict[str, Any] | None:
    first_chunk: dict[str, Any] = {}
    last_usage: dict[str, Any] = {}
    contents: dict[int, str] = {}
    reasoning_contents: dict[int, str] = {}
    roles: dict[int, str] = {}
    finish_reasons: dict[int, Any] = {}
    tool_calls_map: dict[int, dict[int, dict[str, Any]]] = {}

    for line in text.splitlines():
        line = line.strip()
        if not line or not line.startswith("data:"):
            continue
        data = line[len("data:") :].strip()
        if data == "[DONE]":
            break

        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue
        if not isinstance(chunk, dict):
            continue

        if "error" in chunk:
            return chunk

        if not first_chunk:
            first_chunk = chunk
        if isinstance(chunk.get("usage"), dict):
            last_usage = chunk["usage"]
        _merge_chunk(chunk, contents, reasoning_contents, roles, finish_reasons, tool_calls_map)

    if not contents and not reasoning_contents and not tool_calls_map and not roles and not finish_reasons:
        return None

    body: dict[str, Any] = {
        "id": first_chunk.get("id"),
        "object": "chat.completion",
        "created": first_chunk.get("created"),
        "model": first_chunk.get("model"),
        "choices": _build_choices(contents, reasoning_contents, roles, finish_reasons, tool_calls_map),
    }
    if last_usage:
        body["usage"] = last_usage

    _normalize_message_content(body)
    return body


def _merge_chunk(
    chunk: dict[str, Any],
    contents: dict[int, str],
    reasoning_contents: dict[int, str],
    roles: dict[int, str],
    finish_reasons: dict[int, Any],
    tool_calls_map: dict[int, dict[int, dict[str, Any]]],
) -> None:
    for choice in chunk.get("choices", []) or []:
        if not isinstance(choice, dict):
            continue
        idx = int(choice.get("index", 0) or 0)
        # TODO: Preserve /v1/completions streaming chunks (`choices[].text`) as completion responses.
        delta = choice.get("delta") or {}
        if not isinstance(delta, dict):
            delta = {}

        if isinstance(delta.get("role"), str):
            roles.setdefault(idx, delta["role"])
        if isinstance(delta.get("content"), str) and delta["content"]:
            contents[idx] = contents.get(idx, "") + delta["content"]

        reasoning_content = delta.get("reasoning_content") or delta.get("reasoning")
        if isinstance(reasoning_content, str) and reasoning_content:
            reasoning_contents[idx] = reasoning_contents.get(idx, "") + reasoning_content

        tool_call_deltas = delta.get("tool_calls") or []
        if isinstance(tool_call_deltas, list):
            _merge_tool_calls(idx, tool_call_deltas, tool_calls_map)

        if choice.get("finish_reason") is not None:
            finish_reasons[idx] = choice["finish_reason"]


def _merge_tool_calls(
    choice_idx: int,
    tool_call_deltas: list[Any],
    tool_calls_map: dict[int, dict[int, dict[str, Any]]],
) -> None:
    for tool_call_delta in tool_call_deltas:
        if not isinstance(tool_call_delta, dict):
            continue
        tool_call_idx = int(tool_call_delta.get("index", 0) or 0)
        tool_call_by_idx = tool_calls_map.setdefault(choice_idx, {})
        if tool_call_idx not in tool_call_by_idx:
            tool_call_by_idx[tool_call_idx] = {
                "id": tool_call_delta.get("id") or "",
                "type": tool_call_delta.get("type") or "function",
                "function": {"name": "", "arguments": ""},
            }

        tool_call = tool_call_by_idx[tool_call_idx]
        if tool_call_delta.get("id"):
            tool_call["id"] = tool_call_delta["id"]
        if tool_call_delta.get("type"):
            tool_call["type"] = tool_call_delta["type"]

        function_delta = tool_call_delta.get("function") or {}
        if not isinstance(function_delta, dict):
            continue
        if isinstance(function_delta.get("name"), str):
            tool_call["function"]["name"] += function_delta["name"]
        if isinstance(function_delta.get("arguments"), str):
            tool_call["function"]["arguments"] += function_delta["arguments"]


def _build_choices(
    contents: dict[int, str],
    reasoning_contents: dict[int, str],
    roles: dict[int, str],
    finish_reasons: dict[int, Any],
    tool_calls_map: dict[int, dict[int, dict[str, Any]]],
) -> list[dict[str, Any]]:
    choices: list[dict[str, Any]] = []
    output_indices = sorted(
        set(contents) | set(reasoning_contents) | set(roles) | set(finish_reasons) | set(tool_calls_map)
    )
    for idx in output_indices:
        message: dict[str, Any] = {
            "role": roles.get(idx, "assistant"),
            "content": contents.get(idx),
        }
        if idx in reasoning_contents:
            message["reasoning_content"] = reasoning_contents[idx]
        if idx in tool_calls_map:
            message["tool_calls"] = [
                tool_calls_map[idx][tool_call_idx] for tool_call_idx in sorted(tool_calls_map[idx])
            ]

        choices.append(
            {
                "index": idx,
                "message": message,
                "finish_reason": finish_reasons.get(idx),
            }
        )
    return choices
