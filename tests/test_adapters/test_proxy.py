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
"""Tests for the adapter proxy HTTP layer.

- Unit tests use Starlette's TestClient (in-process, no real port).
- Integration tests start a real uvicorn proxy via ``start_adapter_proxy``
  and hit it with ``urllib`` over the loopback.
"""

from __future__ import annotations

import http.server
import json
import threading
import urllib.error
import urllib.request

import pytest
from starlette.testclient import TestClient

from nemo_evaluator.adapters.pipeline import AdapterPipeline
from nemo_evaluator.adapters.proxy import _build_app, start_adapter_proxy
from nemo_evaluator.adapters.types import (
    AdapterRequest,
    AdapterResponse,
    RequestInterceptor,
    RequestToResponseInterceptor,
)

# ==================================================================
# Helpers
# ==================================================================


class _EchoEndpoint(RequestToResponseInterceptor):
    """Returns the incoming request body and path as the response."""

    async def intercept_request(self, req: AdapterRequest) -> AdapterResponse:
        return AdapterResponse(
            status_code=200,
            headers={"x-upstream": "true", "transfer-encoding": "chunked"},
            body={"echo": req.body, "path": req.path},
            ctx=req.ctx,
        )


class _BoomEndpoint(RequestToResponseInterceptor):
    """Always raises to simulate pipeline failure."""

    async def intercept_request(self, req: AdapterRequest) -> AdapterResponse:
        raise RuntimeError("upstream exploded")


class _HeaderCapture(RequestInterceptor):
    """Captures the last request for assertions."""

    last_req: AdapterRequest | None = None

    async def intercept_request(self, req: AdapterRequest) -> AdapterRequest:
        _HeaderCapture.last_req = req
        return req


def _test_client(model_id: str = "test-model", pipeline: AdapterPipeline | None = None):
    pipeline = pipeline or AdapterPipeline([_EchoEndpoint()])
    return TestClient(_build_app(pipeline, model_id), raise_server_exceptions=False)


def _http_post(url: str, body: dict, timeout: float = 5) -> tuple[int, dict]:
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def _http_get(url: str, timeout: float = 5) -> tuple[int, dict]:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.status, json.loads(resp.read())


# ==================================================================
# Fixtures
# ==================================================================


@pytest.fixture()
def echo_upstream():
    """Tiny HTTP server that echoes POST bodies back as JSON."""

    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            payload = json.dumps(
                {
                    "id": "echo-1",
                    "object": "chat.completion",
                    "choices": [{"message": {"role": "assistant", "content": "echo"}}],
                    "echo_path": self.path,
                    "echo_body": body,
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


@pytest.fixture()
def proxy(echo_upstream):
    """Start a real adapter proxy pointed at the echo upstream."""
    handle = start_adapter_proxy(upstream_url=echo_upstream, model_id="test-model", port=0)
    yield handle
    handle.stop()


# ==================================================================
# Unit tests (TestClient — no real port)
# ==================================================================


class TestUnit:
    def test_health(self):
        resp = _test_client().get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    @pytest.mark.parametrize(
        "path",
        [
            "/v1/chat/completions",
            "/chat/completions",
            "/v1/completions",
            "/v1/embeddings",
            "/v1/models",
            "/custom/endpoint",
        ],
    )
    def test_catch_all_post(self, path: str):
        resp = _test_client().post(path, json={"messages": [{"role": "user", "content": "hi"}]})
        assert resp.status_code == 200
        assert resp.json()["path"] == path

    def test_session_path_stripped(self):
        resp = _test_client().post(
            "/s/abc123def456/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 200
        assert resp.json()["path"] == "/v1/chat/completions"

    def test_session_id_in_context(self):
        """The session_id extracted from the path is visible to interceptors."""
        _HeaderCapture.last_req = None
        client = _test_client(pipeline=AdapterPipeline([_HeaderCapture(), _EchoEndpoint()]))
        client.post("/s/deadbeef1234/v1/chat/completions", json={"messages": []})
        assert _HeaderCapture.last_req is not None
        assert _HeaderCapture.last_req.ctx.extra.get("session_id") == "deadbeef1234"

    def test_no_session_for_plain_path(self):
        """Requests without /s/ prefix work normally with no session ID."""
        _HeaderCapture.last_req = None
        client = _test_client(pipeline=AdapterPipeline([_HeaderCapture(), _EchoEndpoint()]))
        client.post("/v1/chat/completions", json={"messages": []})
        assert _HeaderCapture.last_req is not None
        assert "session_id" not in _HeaderCapture.last_req.ctx.extra

    def test_model_id_injected(self):
        resp = _test_client(model_id="my-model").post("/v1/chat/completions", json={"messages": []})
        assert resp.json()["echo"]["model"] == "my-model"

    def test_model_id_not_overridden(self):
        resp = _test_client(model_id="my-model").post(
            "/v1/chat/completions",
            json={"model": "caller-model", "messages": []},
        )
        assert resp.json()["echo"]["model"] == "caller-model"

    def test_no_model_id_when_empty(self):
        resp = _test_client(model_id="").post("/v1/chat/completions", json={"messages": []})
        assert "model" not in resp.json()["echo"]

    def test_invalid_json_returns_400(self):
        resp = _test_client().post(
            "/v1/chat/completions",
            content=b"not json",
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 400

    def test_pipeline_error_returns_500(self):
        client = _test_client(pipeline=AdapterPipeline([_BoomEndpoint()]))
        resp = client.post("/v1/chat/completions", json={"messages": []})
        assert resp.status_code == 500

    def test_hop_by_hop_stripped(self):
        resp = _test_client().post("/v1/chat/completions", json={"messages": []})
        assert "transfer-encoding" not in {k.lower() for k in resp.headers}

    def test_custom_headers_forwarded(self):
        resp = _test_client().post("/v1/chat/completions", json={"messages": []})
        assert resp.headers.get("x-upstream") == "true"

    def test_get_returns_405(self):
        assert _test_client().get("/v1/chat/completions").status_code == 405

    def test_request_headers_reach_pipeline(self):
        _HeaderCapture.last_req = None
        client = _test_client(pipeline=AdapterPipeline([_HeaderCapture(), _EchoEndpoint()]))
        client.post("/v1/chat/completions", json={"messages": []}, headers={"x-custom": "hello"})
        assert _HeaderCapture.last_req is not None
        assert _HeaderCapture.last_req.headers["x-custom"] == "hello"


# ==================================================================
# Integration tests (real uvicorn + real HTTP)
# ==================================================================


class TestIntegration:
    def test_health(self, proxy):
        status, body = _http_get(f"{proxy.url}/health")
        assert status == 200 and body == {"status": "ok"}

    @pytest.mark.parametrize(
        "path",
        [
            "/v1/chat/completions",
            "/chat/completions",
            "/v1/completions",
            "/v1/embeddings",
        ],
    )
    def test_post_paths(self, proxy, path):
        status, body = _http_post(f"{proxy.url}{path}", {"messages": []})
        assert status == 200
        assert body["echo_path"] == path

    def test_model_injected(self, proxy):
        _, body = _http_post(f"{proxy.url}/v1/chat/completions", {"messages": []})
        assert body["echo_body"]["model"] == "test-model"

    def test_interceptors(self, echo_upstream):
        handle = start_adapter_proxy(
            upstream_url=echo_upstream,
            model_id="m",
            port=0,
            interceptor_specs=[{"name": "drop_params", "config": {"params": ["temperature"]}}],
        )
        try:
            _, body = _http_post(
                f"{handle.url}/v1/chat/completions",
                {"messages": [], "temperature": 0.7, "max_tokens": 100},
            )
            assert "temperature" not in body["echo_body"]
            assert body["echo_body"]["max_tokens"] == 100
        finally:
            handle.stop()

    def test_upstream_url_with_endpoint_suffix(self, echo_upstream):
        """upstream_url may include /v1/chat/completions — endpoint must strip it."""
        handle = start_adapter_proxy(
            upstream_url=f"{echo_upstream}/v1/chat/completions",
            model_id="m",
            port=0,
        )
        try:
            status, body = _http_post(
                f"{handle.url}/chat/completions",
                {"messages": []},
            )
            assert status == 200
            assert body["echo_path"] == "/v1/chat/completions"
        finally:
            handle.stop()

    def test_session_path_stripped_before_upstream(self, echo_upstream):
        """Session prefix /s/<id> is stripped so the upstream sees the clean path."""
        handle = start_adapter_proxy(
            upstream_url=echo_upstream,
            model_id="m",
            port=0,
        )
        try:
            status, body = _http_post(
                f"{handle.url}/s/deadbeef1234/v1/chat/completions",
                {"messages": [{"role": "user", "content": "hi"}]},
            )
            assert status == 200
            assert body["echo_path"] == "/v1/chat/completions"
        finally:
            handle.stop()

    def test_stop_closes_port(self, echo_upstream):
        handle = start_adapter_proxy(upstream_url=echo_upstream, model_id="m", port=0)
        url = handle.url
        handle.stop()
        with pytest.raises((urllib.error.URLError, ConnectionRefusedError, OSError)):
            _http_get(f"{url}/health")
