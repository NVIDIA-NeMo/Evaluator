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

import asyncio
import logging
import socket
import threading
import time
from dataclasses import dataclass
from typing import Any

import uvicorn
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from nemo_evaluator.adapters.pipeline import AdapterPipeline
from nemo_evaluator.adapters.registry import InterceptorRegistry
from nemo_evaluator.adapters.types import AdapterRequest, InterceptorContext

logger = logging.getLogger(__name__)

_HOP_BY_HOP_HEADERS = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
        "content-length",
        "content-encoding",
    }
)


async def _proxy_handler(request: Request) -> Response:
    """Catch-all handler that forwards any POST request through the pipeline.

    Matches the original Evaluator proxy behavior: all paths are forwarded
    to upstream (chat/completions, completions, embeddings, models, etc.).
    """
    pipeline: AdapterPipeline = request.app.state.pipeline
    model_id: str = request.app.state.model_id

    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            {"error": {"message": "Invalid JSON body", "type": "invalid_request_error"}},
            status_code=400,
        )

    if model_id:
        body.setdefault("model", model_id)

    adapter_req = AdapterRequest(
        method=request.method,
        path=request.url.path,
        headers=dict(request.headers),
        body=body,
        ctx=InterceptorContext(),
    )

    try:
        resp = await pipeline.process(adapter_req)
    except Exception:
        logger.exception("Pipeline error")
        return JSONResponse(
            {"error": {"message": "Internal adapter proxy error", "type": "server_error"}},
            status_code=500,
        )

    fwd_headers = {k: v for k, v in (resp.headers or {}).items() if k.lower() not in _HOP_BY_HOP_HEADERS}

    if isinstance(resp.body, bytes):
        return Response(content=resp.body, status_code=resp.status_code, headers=fwd_headers)
    return JSONResponse(content=resp.body, status_code=resp.status_code, headers=fwd_headers)


async def _health(request: Request) -> Response:
    return JSONResponse({"status": "ok"})


class _ProxyApp:
    """Minimal ASGI app — bypasses Starlette routing entirely.

    GET  /health  → health check
    POST *        → proxy handler (forwarded to pipeline)
    everything else → 405
    """

    def __init__(self, pipeline: AdapterPipeline, model_id: str) -> None:
        self.state = type("State", (), {"pipeline": pipeline, "model_id": model_id})()

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] == "lifespan":
            while True:
                msg = await receive()
                if msg["type"] == "lifespan.startup":
                    await send({"type": "lifespan.startup.complete"})
                elif msg["type"] == "lifespan.shutdown":
                    await send({"type": "lifespan.shutdown.complete"})
                    return
            return

        if scope["type"] != "http":
            return

        scope["app"] = self
        request = Request(scope, receive, send)

        if request.method == "GET" and request.url.path == "/health":
            response = await _health(request)
        elif request.method == "POST":
            response = await _proxy_handler(request)
        else:
            response = JSONResponse({"error": "Method not allowed"}, status_code=405)

        await response(scope, receive, send)


def _build_app(pipeline: AdapterPipeline, model_id: str) -> _ProxyApp:
    return _ProxyApp(pipeline, model_id)


def _find_free_port(preferred: int) -> int:
    for port in [preferred, 0]:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", port))
                return s.getsockname()[1]
        except OSError:
            if port != 0:
                continue
            raise


@dataclass
class ProxyHandle:
    url: str
    port: int
    _server: uvicorn.Server
    _thread: threading.Thread
    _endpoint_interceptor: Any

    def stop(self) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=10)
        if self._endpoint_interceptor is None:
            return
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None
        if running_loop is not None:
            running_loop.create_task(self._endpoint_interceptor.close())
        else:
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(self._endpoint_interceptor.close())
            finally:
                loop.close()


def start_adapter_proxy(
    *,
    upstream_url: str,
    model_id: str,
    api_key: str | None = None,
    port: int = 4000,
    interceptor_specs: list[dict] | None = None,
    extra_body: dict | None = None,
    request_timeout: float = 120,
    max_retries: int = 0,
    retry_on_status: list[int] | None = None,
    max_concurrent_upstream: int = 64,
    verbose: bool = False,
) -> ProxyHandle:
    from nemo_evaluator.adapters.types import Stage

    endpoint_config: dict[str, Any] = {
        "upstream_url": upstream_url,
        "api_key": api_key,
        "extra_body": extra_body,
        "request_timeout": request_timeout,
        "max_retries": max_retries,
        "retry_on_status": retry_on_status,
        "max_concurrent": max_concurrent_upstream,
    }
    endpoint_spec = {"name": "endpoint", "config": endpoint_config}

    request_side: list[dict[str, Any]] = []
    response_side: list[dict[str, Any]] = []
    for spec in interceptor_specs or []:
        cls = InterceptorRegistry.resolve_class(spec["name"])
        stage = getattr(cls, "stage", Stage.REQUEST)
        if stage == Stage.RESPONSE:
            response_side.append(spec)
        else:
            request_side.append(spec)

    chain_specs = request_side + [endpoint_spec] + response_side
    pipeline = AdapterPipeline.from_config(chain_specs)

    endpoint_idx = len(request_side)
    endpoint_interceptor = pipeline._chain[endpoint_idx]

    actual_port = _find_free_port(port)
    app = _build_app(pipeline, model_id)

    log_level = "info" if verbose else "warning"
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=actual_port,
        log_level=log_level,
        access_log=verbose,
    )
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    url = f"http://127.0.0.1:{actual_port}"
    _wait_for_health(url, timeout=10)
    logger.info("Adapter proxy ready at %s (port=%d)", url, actual_port)

    return ProxyHandle(
        url=url,
        port=actual_port,
        _server=server,
        _thread=thread,
        _endpoint_interceptor=endpoint_interceptor,
    )


def _wait_for_health(url: str, timeout: float = 10) -> None:
    import urllib.request

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            resp = urllib.request.urlopen(f"{url}/health", timeout=2)
            if resp.status == 200:
                return
        except Exception:
            pass
        time.sleep(0.1)
    raise RuntimeError(f"Adapter proxy at {url} did not become healthy in {timeout}s")
