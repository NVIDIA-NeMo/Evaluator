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

import http.server
import json
import threading
import urllib.request
from typing import Any

import pytest

from nemo_evaluator.adapters.interceptors.streaming import Interceptor
from nemo_evaluator.adapters.proxy import start_adapter_proxy
from nemo_evaluator.adapters.registry import InterceptorRegistry
from nemo_evaluator.adapters.types import AdapterResponse, InterceptorContext


def _sse_body(chunks: list[dict[str, Any]]) -> bytes:
    lines = b"".join(b"data: " + json.dumps(chunk).encode() + b"\n\n" for chunk in chunks)
    return lines + b"data: [DONE]\n\n"


def _resp(body: dict[str, Any] | bytes, content_type: str = "text/event-stream") -> AdapterResponse:
    return AdapterResponse(
        status_code=200,
        headers={"Content-Type": content_type},
        body=body,
        ctx=InterceptorContext(),
    )


def _post_json(url: str, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer local-test"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        return resp.status, json.loads(resp.read())


@pytest.fixture
def local_server():
    servers = []

    def start(handler_cls: type[http.server.BaseHTTPRequestHandler]) -> str:
        server = http.server.HTTPServer(("127.0.0.1", 0), handler_cls)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return f"http://127.0.0.1:{server.server_address[1]}"

    yield start

    for server in servers:
        server.shutdown()


class _QuietHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass


def test_registry_exposes_streaming():
    assert InterceptorRegistry.resolve_class("streaming") is Interceptor
    assert "streaming" in InterceptorRegistry.available()


async def test_coalesces_streaming_response():
    resp = _resp(
        _sse_body(
            [
                {
                    "id": "cmpl-test",
                    "created": 1,
                    "model": "test-model",
                    "choices": [{"index": 0, "delta": {"role": "assistant", "content": "hello "}}],
                },
                {
                    "id": "cmpl-test",
                    "created": 1,
                    "model": "test-model",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": "world", "reasoning_content": "because"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
                },
            ]
        )
    )

    result = await Interceptor().intercept_response(resp)

    assert result.headers["Content-Type"] == "application/json"
    assert result.body["id"] == "cmpl-test"
    assert result.body["object"] == "chat.completion"
    assert result.body["choices"][0]["message"]["content"] == "hello world"
    assert result.body["choices"][0]["message"]["reasoning_content"] == "because"
    assert result.body["choices"][0]["finish_reason"] == "stop"
    assert result.body["usage"]["total_tokens"] == 5


async def test_merges_tool_call_deltas():
    resp = _resp(
        _sse_body(
            [
                {
                    "id": "cmpl-tools",
                    "created": 1,
                    "model": "test-model",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "role": "assistant",
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {"name": "search", "arguments": '{"q"'},
                                    }
                                ],
                            },
                        }
                    ],
                },
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"tool_calls": [{"index": 0, "function": {"arguments": ':"x"}'}}]},
                            "finish_reason": "tool_calls",
                        }
                    ],
                },
            ]
        )
    )

    result = await Interceptor().intercept_response(resp)

    tool_call = result.body["choices"][0]["message"]["tool_calls"][0]
    assert tool_call["id"] == "call_1"
    assert tool_call["type"] == "function"
    assert tool_call["function"] == {"name": "search", "arguments": '{"q":"x"}'}
    assert result.body["choices"][0]["finish_reason"] == "tool_calls"


async def test_streamed_error_event_overrides_partial_delta():
    error = {"message": "upstream failed", "type": "server_error"}
    resp = _resp(
        _sse_body(
            [
                {
                    "id": "cmpl-error",
                    "created": 1,
                    "model": "test-model",
                    "choices": [{"index": 0, "delta": {"role": "assistant", "content": "partial"}}],
                },
                {"error": error},
            ]
        )
    )

    result = await Interceptor().intercept_response(resp)

    assert result.status_code == 502
    assert result.headers["Content-Type"] == "application/json"
    assert result.body == {"error": error}


async def test_replaces_null_message_content_in_json_response():
    resp = _resp(
        {"choices": [{"index": 0, "message": {"role": "assistant", "content": None}}]},
        content_type="application/json",
    )

    result = await Interceptor().intercept_response(resp)

    assert result.body["choices"][0]["message"]["content"] == ""


async def test_replaces_missing_stream_content_with_empty_string():
    resp = _resp(
        _sse_body(
            [
                {
                    "id": "cmpl-empty",
                    "created": 1,
                    "model": "test-model",
                    "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": "stop"}],
                }
            ]
        )
    )

    result = await Interceptor().intercept_response(resp)

    assert result.body["choices"][0]["message"]["content"] == ""


async def test_non_sse_bytes_are_left_unchanged():
    resp = _resp(b"not json", content_type="text/plain")

    result = await Interceptor().intercept_response(resp)

    assert result is resp
    assert result.body == b"not json"


def test_start_adapter_proxy_coalesces_endpoint_streaming_response(local_server):
    class Handler(_QuietHandler):
        seen_body: dict[str, Any] | None = None

        def do_POST(self):
            Handler.seen_body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            body = _sse_body(
                [
                    {
                        "id": "cmpl-proxy",
                        "created": 1,
                        "model": "test-model",
                        "choices": [{"index": 0, "delta": {"role": "assistant", "content": "done"}}],
                    }
                ]
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    base = local_server(Handler)
    handle = start_adapter_proxy(
        upstream_url=base,
        model_id="test-model",
        port=0,
        interceptor_specs=[
            {"name": "payload_modifier", "config": {"params_to_add": {"stream": True}}},
            {"name": "streaming", "config": {}},
        ],
    )
    try:
        status, body = _post_json(f"{handle.url}/v1/chat/completions", {"messages": []})
    finally:
        handle.stop()

    assert status == 200
    assert Handler.seen_body is not None
    assert Handler.seen_body["stream"] is True
    assert body["id"] == "cmpl-proxy"
    assert body["choices"][0]["message"]["content"] == "done"
