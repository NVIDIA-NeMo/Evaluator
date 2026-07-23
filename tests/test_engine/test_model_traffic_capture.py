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

import gc
import http.server
import json
import stat
import threading
from pathlib import Path

import pytest

from nemo_evaluator.adapters.proxy import start_adapter_proxy
from nemo_evaluator.adapters.types import AdapterRequest, AdapterResponse, InterceptorContext
from nemo_evaluator.engine.eval_loop import run_evaluation
from nemo_evaluator.engine.model_client import ModelClient
from nemo_evaluator.engine.model_traffic_log import append_model_traffic_records, format_model_traffic_log_records
from nemo_evaluator.engine.step_log import INFERENCE_LOG, MODEL_TRAFFIC_LOG, StepLog
from nemo_evaluator.environments.base import EvalEnvironment, SeedResult, VerifyResult
from nemo_evaluator.observability.model_traffic import (
    ModelTrafficStore,
    aggregate_model_traffic_stats,
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


def _format_drained(
    store: ModelTrafficStore,
    session_id: str,
    *,
    benchmark: str = "b",
    problem_idx: int = 0,
    repeat: int = 0,
) -> dict:
    rows = list(
        format_model_traffic_log_records(
            store.drain_session(session_id),
            benchmark=benchmark,
            problem_idx=problem_idx,
            repeat=repeat,
        )
    )
    assert len(rows) == 1
    return rows[0]


def _capture_response_body(
    *,
    content: str = "the answer is 42",
    reasoning: str = "let me think",
    tool_calls: list[dict] | None = None,
) -> dict:
    return {
        "model": "m",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": content,
                    "reasoning_content": reasoning,
                    "tool_calls": tool_calls
                    or [{"id": "c1", "type": "function", "function": {"name": "w", "arguments": "{}"}}],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
    }


async def test_model_traffic_spools_and_flushes_session(tmp_path: Path) -> None:
    session_id = "flush-session"
    spool_dir = tmp_path / "spool"
    store = ModelTrafficStore(service_name="solver", capture_request_body=True, spool_dir=spool_dir)
    log = StepLog(tmp_path / MODEL_TRAFFIC_LOG)
    log.open(truncate=True)
    try:
        for index, status_code in enumerate((200, 400)):
            request_body = {"model": "m", "messages": [{"role": "user", "content": f"turn {index}"}]}
            ctx = InterceptorContext()
            ctx.extra["session_id"] = session_id
            ctx.extra["upstream_request_body"] = request_body
            store.start_request(
                AdapterRequest(
                    method="POST",
                    path="/chat/completions",
                    headers={},
                    body=request_body,
                    ctx=ctx,
                )
            )
            store.finish_response(
                AdapterResponse(
                    status_code=status_code,
                    headers={"content-type": "application/json"},
                    latency_ms=10.0,
                    ctx=ctx,
                    body=_capture_response_body()
                    if status_code == 200
                    else {"error": {"message": "bad request", "type": "invalid_request_error"}},
                )
            )

        assert store._pending == {}
        assert not hasattr(store, "_records_by_session")
        assert stat.S_IMODE(store._spool_dir.stat().st_mode) == 0o700
        spool_files = list(spool_dir.rglob("*.jsonl"))
        assert len(spool_files) == 1
        assert stat.S_IMODE(spool_files[0].stat().st_mode) == 0o600
        model_traffic = store.drain_session(session_id)
        await append_model_traffic_records(
            log,
            model_traffic,
            benchmark="b",
            problem_idx=12,
            repeat=3,
        )
        stats = aggregate_model_traffic_stats(model_traffic)
    finally:
        log.close()
        store.close()

    rows = _jsonl(tmp_path / MODEL_TRAFFIC_LOG)
    assert len(rows) == 2
    assert [(row["status_code"], row["problem_idx"], row["repeat"]) for row in rows] == [(200, 12, 3), (400, 12, 3)]
    assert all(row["request_body"] for row in rows)
    assert rows[1]["usage"] == {}
    assert not list(spool_dir.rglob("*.drain"))
    assert not [p for p in spool_dir.rglob("*") if p.is_file()]
    assert stats == {
        "usage": {
            "prompt_tokens": 3,
            "completion_tokens": 4,
            "total_tokens": 7,
            "token_provenance": "provider_reported",
        },
        "tool_calls": {"count": 1, "names": {"w": 1}},
        "model_calls": 2,
    }


def test_dropped_drained_session_unlinks_spool_without_iterating(tmp_path: Path) -> None:
    session_id = "discard-session"
    spool_dir = tmp_path / "spool"
    store = ModelTrafficStore(service_name="solver", spool_dir=spool_dir)
    try:
        ctx = InterceptorContext()
        ctx.extra["session_id"] = session_id
        store.start_request(
            AdapterRequest(
                method="POST",
                path="/chat/completions",
                headers={},
                body={"model": "m", "messages": [{"role": "user", "content": "turn"}]},
                ctx=ctx,
            )
        )
        store.finish_response(
            AdapterResponse(
                status_code=200,
                headers={"content-type": "application/json"},
                latency_ms=10.0,
                ctx=ctx,
                body=_capture_response_body(),
            )
        )

        # The eval-loop error paths drain the session and drop the return value
        # without iterating it, so only __del__ -> discard() unlinks the spool file.
        drained = store.drain_session(session_id)
        drain_path = drained._path
        assert drain_path is not None and drain_path.exists()

        del drained
        gc.collect()

        assert not drain_path.exists()
        assert not [p for p in spool_dir.rglob("*") if p.is_file()]
    finally:
        store.close()


@pytest.mark.parametrize(
    "capture_kwargs,expect_response_fields",
    [
        pytest.param({}, True, id="default-capture"),
        pytest.param(
            {"capture_tool_calls": False, "capture_reasoning": False, "capture_messages": False},
            False,
            id="capture-disabled",
        ),
    ],
)
def test_format_log_record_response_capture_fields(capture_kwargs: dict, expect_response_fields: bool) -> None:
    ctx = InterceptorContext()
    ctx.extra["session_id"] = "cap-session"
    store = ModelTrafficStore(service_name="solver", **capture_kwargs)
    store.start_request(
        AdapterRequest(method="POST", path="/chat/completions", headers={}, body={"model": "m"}, ctx=ctx)
    )
    store.finish_response(
        AdapterResponse(
            status_code=200,
            headers={"content-type": "application/json"},
            latency_ms=10.0,
            ctx=ctx,
            body=_capture_response_body(),
        )
    )
    row = _format_drained(store, "cap-session")
    assert row["tool_calls"] == {"count": 1, "names": {"w": 1}}
    assert row["request_hash"]
    if expect_response_fields:
        assert row["tool_calls_full"] == [{"id": "c1", "name": "w", "arguments": "{}"}]
        assert row["reasoning_content"] == "let me think"
        assert row["message_content"] == "the answer is 42"
    else:
        assert "tool_calls_full" not in row
        assert "reasoning_content" not in row
        assert "message_content" not in row


@pytest.mark.parametrize(
    "capture_kwargs,expect_request_fields",
    [
        pytest.param({}, False, id="request-body-default-off"),
        pytest.param({"capture_request_body": True}, True, id="request-body-enabled"),
    ],
)
def test_format_log_record_request_body_capture_fields(
    capture_kwargs: dict,
    expect_request_fields: bool,
) -> None:
    ctx = InterceptorContext()
    ctx.extra["session_id"] = "req-body-session"
    request_body = {
        "model": "m",
        "messages": [{"role": "user", "content": "solve with a tool"}],
    }
    ctx.extra["upstream_request_body"] = request_body
    store = ModelTrafficStore(service_name="solver", **capture_kwargs)
    store.start_request(
        AdapterRequest(method="POST", path="/chat/completions", headers={}, body={"model": "m"}, ctx=ctx)
    )
    store.finish_response(
        AdapterResponse(
            status_code=200,
            headers={"content-type": "application/json"},
            latency_ms=10.0,
            ctx=ctx,
            body=_capture_response_body(),
        )
    )
    row = _format_drained(store, "req-body-session")
    assert row["request_hash"]
    if expect_request_fields:
        assert row["request_body"] == json.dumps(request_body, sort_keys=True, default=str)
    else:
        assert "request_body" not in row


def test_format_log_record_request_body_truncation() -> None:
    ctx = InterceptorContext()
    ctx.extra["session_id"] = "req-body-trunc-session"
    request_body = {"model": "m", "messages": [{"role": "user", "content": "x" * 200}]}
    ctx.extra["upstream_request_body"] = request_body
    store = ModelTrafficStore(service_name="solver", capture_request_body=True, max_content_chars=50)
    store.start_request(
        AdapterRequest(method="POST", path="/chat/completions", headers={}, body={"model": "m"}, ctx=ctx)
    )
    store.finish_response(
        AdapterResponse(
            status_code=200,
            headers={"content-type": "application/json"},
            latency_ms=10.0,
            ctx=ctx,
            body=_capture_response_body(),
        )
    )
    row = _format_drained(store, "req-body-trunc-session")
    assert isinstance(row["request_body"], str)
    assert "...[truncated," in row["request_body"]


def test_format_log_record_request_body_on_error() -> None:
    ctx = InterceptorContext()
    ctx.extra["session_id"] = "req-body-error-session"
    request_body = {"model": "m", "messages": [{"role": "user", "content": "hello"}]}
    ctx.extra["upstream_request_body"] = request_body
    store = ModelTrafficStore(service_name="solver", capture_request_body=True)
    req = AdapterRequest(method="POST", path="/chat/completions", headers={}, body={"model": "m"}, ctx=ctx)
    store.start_request(req)
    store.finish_error(req=req, latency_ms=1.0, error_type="timeout")

    row = _format_drained(store, "req-body-error-session")
    assert row["request_body"] == json.dumps(request_body, sort_keys=True, default=str)
    assert row["error_type"] == "timeout"


def test_non_success_response_forwards_compact_error_summary() -> None:
    ctx = InterceptorContext()
    ctx.extra["session_id"] = "bad-request-session"
    store = ModelTrafficStore(service_name="solver")
    store.start_request(
        AdapterRequest(method="POST", path="/chat/completions", headers={}, body={"model": "m"}, ctx=ctx)
    )
    store.finish_response(
        AdapterResponse(
            status_code=400,
            headers={"content-type": "application/json"},
            latency_ms=5.0,
            ctx=ctx,
            body={
                "error": {
                    "message": "OpenAIException - Expecting ',' delimiter: line 1 column 237 (char 236)",
                    "type": "invalid_request_error",
                    "code": "bad_tool_json",
                }
            },
        )
    )

    row = _format_drained(store, "bad-request-session")
    assert row["status_code"] == 400
    assert row["usage"] == {}
    assert "token_provenance" not in row
    assert row["error_message"] == "OpenAIException - Expecting ',' delimiter: line 1 column 237 (char 236)"
    assert row["error_type"] == "invalid_request_error"
    assert row["error_code"] == "bad_tool_json"
    assert row["request_hash"]
    assert "bad_tool_json" in row["error_body"]


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

    row = _format_drained(
        store,
        "stream-session",
        benchmark="model_traffic_env",
        problem_idx=0,
        repeat=0,
    )

    assert row["model"] == "stream-model"
    assert row["finish_reason"] == "tool_calls"
    assert row["usage"] == {
        "prompt_tokens": 7,
        "completion_tokens": 5,
        "total_tokens": 12,
        "reasoning_tokens": 3,
    }
    assert row["tool_calls"] == {"count": 2, "names": {"bash": 1, "python": 1}}
    assert row["token_provenance"] == "provider_reported"


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
        assert traffic["request_hash"]
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
        assert traffic["message_content"] == "answer"
        assert traffic["reasoning_content"] == "I need to inspect the workspace."
        assert traffic["tool_calls_full"] == [{"id": "call_1", "name": "bash", "arguments": '{"command":"ls"}'}]
        assert traffic["token_provenance"] == "provider_reported"
        assert "request_body" not in traffic
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


# ─── response capture flags ──────────────────────────────────────────


@pytest.mark.parametrize(
    "capture_kwargs,expect_response_fields",
    [
        pytest.param({}, True, id="default-capture"),
        pytest.param(
            {"capture_tool_calls": False, "capture_reasoning": False, "capture_messages": False},
            False,
            id="capture-disabled",
        ),
        pytest.param(
            {"capture_tool_calls": True, "capture_reasoning": True, "capture_messages": True},
            True,
            id="capture-explicit",
        ),
    ],
)
def test_summary_from_json_response_capture_fields(capture_kwargs: dict, expect_response_fields: bool) -> None:
    from nemo_evaluator.observability.model_traffic import _summary_from_json

    body = _capture_response_body(
        content="calling search",
        reasoning="I need to look this up",
        tool_calls=[
            {"id": "t1", "function": {"name": "search", "arguments": '{"q":"x"}'}},
            {"id": "t2", "function": {"name": "fetch", "arguments": '{"u":"y"}'}},
        ],
    )
    body["model"] = "qwen"
    body["usage"] = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    out = _summary_from_json(body, **capture_kwargs)
    assert out["tool_calls"] == {"count": 2, "names": {"search": 1, "fetch": 1}}
    if expect_response_fields:
        assert out["reasoning_content"] == "I need to look this up"
        assert out["message_content"] == "calling search"
        assert out["tool_calls_full"] == [
            {"id": "t1", "name": "search", "arguments": '{"q":"x"}'},
            {"id": "t2", "name": "fetch", "arguments": '{"u":"y"}'},
        ]
    else:
        assert "tool_calls_full" not in out
        assert "reasoning_content" not in out
        assert "message_content" not in out


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
