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
from pathlib import Path

from nemo_evaluator.adapters.proxy import start_adapter_proxy
from nemo_evaluator.adapters.types import AdapterRequest, AdapterResponse, InterceptorContext
from nemo_evaluator.engine.eval_loop import run_evaluation
from nemo_evaluator.engine.model_client import ModelClient
from nemo_evaluator.engine.model_traffic_log import format_model_traffic_log_records
from nemo_evaluator.engine.step_log import INFERENCE_LOG, MODEL_TRAFFIC_LOG
from nemo_evaluator.environments.base import EvalEnvironment, SeedResult, VerifyResult
from nemo_evaluator.observability.model_traffic import (
    ModelTrafficStore,
    register_store,
)
from nemo_evaluator.solvers.base import SolveResult


class _OneProblemEnv(EvalEnvironment):
    name = "model_traffic_env"

    async def dataset_size(self) -> int:
        return 1

    async def seed(self, idx: int) -> SeedResult:
        return SeedResult(prompt="solve with a tool", expected_answer="answer", metadata={})

    async def verify(self, response: str, expected: str, sandbox=None, **meta) -> VerifyResult:
        return VerifyResult(reward=1.0 if response == expected else 0.0)


class _ModelClientSolver:
    def __init__(self, base_url: str) -> None:
        self._client = ModelClient(base_url=base_url, model="solver-model")

    async def solve(self, task: SeedResult) -> SolveResult:
        response = await self._client.chat(task.prompt)
        return SolveResult(response=response.content, model_response=response, trajectory=[])

    async def close(self) -> None:
        await self._client.close()


def _start_upstream() -> tuple[http.server.HTTPServer, str]:
    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            self.rfile.read(length)
            payload = json.dumps(
                {
                    "id": "chatcmpl-model-traffic",
                    "object": "chat.completion",
                    "model": "solver-model",
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "answer",
                                "reasoning_content": "I need to inspect the workspace.",
                                "tool_calls": [
                                    {
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {
                                            "name": "bash",
                                            "arguments": '{"command":"ls"}',
                                        },
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 5,
                        "completion_tokens": 4,
                        "total_tokens": 9,
                        "prompt_tokens_details": {"cached_tokens": 3},
                        "completion_tokens_details": {"reasoning_tokens": 2},
                    },
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def _start_traffic_proxy(upstream_url: str):
    store = ModelTrafficStore(service_name="solver")
    register_store(store)
    handle = start_adapter_proxy(
        upstream_url=upstream_url,
        model_id="solver-model",
        port=0,
        model_traffic_store_id=store.store_id,
    )
    return handle, store


def _jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_format_log_record_forwards_opt_in_capture_fields() -> None:
    """When the store captured tool_calls_full / reasoning_content / message_content,
    they must land on the model_traffic.jsonl row (not be silently dropped)."""
    ctx = InterceptorContext()
    ctx.extra["session_id"] = "cap-session"
    store = ModelTrafficStore(
        service_name="solver",
        capture_tool_calls=True,
        capture_reasoning=True,
        capture_messages=True,
    )
    store.start_request(
        AdapterRequest(method="POST", path="/chat/completions", headers={}, body={"model": "m"}, ctx=ctx)
    )
    store.finish_response(
        AdapterResponse(
            status_code=200,
            headers={"content-type": "application/json"},
            latency_ms=10.0,
            ctx=ctx,
            body={
                "model": "m",
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "the answer is 42",
                        "reasoning_content": "let me think",
                        "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "w", "arguments": "{}"}}],
                    },
                    "finish_reason": "tool_calls",
                }],
                "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
            },
        )
    )
    rows = format_model_traffic_log_records(
        store.drain_session("cap-session"), benchmark="b", problem_idx=0, repeat=0
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["tool_calls"] == {"count": 1, "names": {"w": 1}}
    assert "tool_calls_full" in row
    assert row["tool_calls_full"][0]["name"] == "w"
    assert row["reasoning_content"] == "let me think"
    assert row["message_content"] == "the answer is 42"


def test_model_traffic_parses_streaming_sse_usage_and_tool_calls() -> None:
    ctx = InterceptorContext()
    ctx.extra["session_id"] = "stream-session"
    store = ModelTrafficStore(service_name="solver")
    store.start_request(
        AdapterRequest(
            method="POST",
            path="/chat/completions",
            headers={},
            body={"model": "stream-model", "stream": True},
            ctx=ctx,
        )
    )

    body = b"""
data: {"model":"stream-model",
data: "choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"name":"ba"}}]}}]}

data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"name":"sh"}}]}},{"index":1,"delta":{"tool_calls":[{"index":0,"function":{"name":"python"}}]}}]}

data: {"choices":[{"index":0,"finish_reason":"tool_calls"}],
data: "usage":{"prompt_tokens":7,"completion_tokens":5,"total_tokens":12,
data: "completion_tokens_details":{"reasoning_tokens":3}}}

data: [DONE]
"""
    store.finish_response(
        AdapterResponse(
            status_code=200,
            headers={"content-type": "text/event-stream"},
            body=body,
            latency_ms=12.5,
            ctx=ctx,
        )
    )

    rows = format_model_traffic_log_records(
        store.drain_session("stream-session"),
        benchmark="model_traffic_env",
        problem_idx=0,
        repeat=0,
    )

    assert len(rows) == 1
    assert rows[0]["model"] == "stream-model"
    assert rows[0]["finish_reason"] == "tool_calls"
    assert rows[0]["usage"] == {
        "prompt_tokens": 7,
        "completion_tokens": 5,
        "total_tokens": 12,
        "reasoning_tokens": 3,
    }
    assert rows[0]["tool_calls"] == {"count": 2, "names": {"bash": 1, "python": 1}}
    assert rows[0]["token_provenance"] == "provider_reported"


async def test_model_traffic_writes_compact_log_and_inference_stats(tmp_path: Path) -> None:
    upstream, upstream_url = _start_upstream()
    handle, store = _start_traffic_proxy(upstream_url)
    try:
        await run_evaluation(
            _OneProblemEnv(),
            _ModelClientSolver(handle.url),
            max_concurrent=1,
            config={"benchmark": "model_traffic_env", "model": "solver-model"},
            model_url=handle.url,
            model_traffic_store=store,
            step_log_dir=tmp_path,
        )

        traffic_rows = _jsonl(tmp_path / MODEL_TRAFFIC_LOG)
        assert len(traffic_rows) == 1
        traffic = traffic_rows[0]
        assert traffic["timestamp"].endswith("Z")
        assert traffic["benchmark"] == "model_traffic_env"
        assert traffic["problem_idx"] == 0
        assert traffic["repeat"] == 0
        assert traffic["session_id"]
        assert traffic["adapter_request_id"]
        assert traffic["model_traffic_request_id"] == f"{traffic['session_id']}:{traffic['adapter_request_id']}"
        assert traffic["service"] == "solver"
        assert traffic["status_code"] == 200
        assert traffic["path"] == "/chat/completions"
        assert traffic["usage"] == {
            "prompt_tokens": 5,
            "completion_tokens": 4,
            "total_tokens": 9,
            "reasoning_tokens": 2,
            "cached_tokens": 3,
        }
        assert traffic["tool_calls"] == {"count": 1, "names": {"bash": 1}}
        assert traffic["token_provenance"] == "provider_reported"
        assert "request" not in traffic
        assert "response" not in traffic

        inference_rows = [row for row in _jsonl(tmp_path / INFERENCE_LOG) if row.get("_type") != "meta"]
        assert len(inference_rows) == 1
        assert inference_rows[0]["reasoning_tokens"] == 2
        assert inference_rows[0]["model_stats"] == {
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 4,
                "total_tokens": 9,
                "reasoning_tokens": 2,
                "cached_tokens": 3,
                "token_provenance": "provider_reported",
            },
            "tool_calls": {"count": 1, "names": {"bash": 1}},
            "model_calls": 1,
        }

    finally:
        await handle.async_stop()
        store.close()
        upstream.shutdown()


# ─── opt-in capture flags (this MR) ─────────────────────────────────


def test_summary_from_json_capture_flags_off_keeps_default_shape() -> None:
    """Default behavior: stats only — no tool_calls_full / reasoning_content / message_content."""
    from nemo_evaluator.observability.model_traffic import _summary_from_json

    body = {
        "model": "qwen",
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "content": "hello",
                    "reasoning_content": "think",
                    "tool_calls": [{"id": "t1", "function": {"name": "search", "arguments": '{"q":"x"}'}}],
                },
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    out = _summary_from_json(body)
    assert "tool_calls_full" not in out
    assert "reasoning_content" not in out
    assert "message_content" not in out
    assert out["tool_calls"]["count"] == 1  # stats still there


def test_summary_from_json_capture_flags_on_populate_fields() -> None:
    from nemo_evaluator.observability.model_traffic import _summary_from_json

    body = {
        "model": "qwen",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "content": "calling search",
                    "reasoning_content": "I need to look this up",
                    "tool_calls": [
                        {"id": "t1", "function": {"name": "search", "arguments": '{"q":"x"}'}},
                        {"id": "t2", "function": {"name": "fetch", "arguments": '{"u":"y"}'}},
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    out = _summary_from_json(body, capture_tool_calls=True, capture_reasoning=True, capture_messages=True)
    assert out["reasoning_content"] == "I need to look this up"
    assert out["message_content"] == "calling search"
    assert len(out["tool_calls_full"]) == 2
    assert out["tool_calls_full"][0] == {"id": "t1", "name": "search", "arguments": '{"q":"x"}'}


def test_summary_truncates_long_content() -> None:
    from nemo_evaluator.observability.model_traffic import _summary_from_json

    long_text = "x" * 1000
    body = {
        "model": "qwen",
        "choices": [{"finish_reason": "stop", "message": {"content": long_text, "reasoning_content": long_text}}],
    }
    out = _summary_from_json(body, capture_reasoning=True, capture_messages=True, max_content_chars=100)
    assert out["message_content"].startswith("x" * 100)
    assert "truncated" in out["message_content"]
    assert out["reasoning_content"].startswith("x" * 100)
