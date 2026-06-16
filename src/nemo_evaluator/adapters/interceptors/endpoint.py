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
import json
import logging
import time
from typing import Any

import aiohttp

from nemo_evaluator.adapters.types import (
    AdapterRequest,
    AdapterResponse,
    RequestToResponseInterceptor,
)
from nemo_evaluator.completions_guard import enforce_text_completions_body
from nemo_evaluator.observability.model_traffic import get_store

logger = logging.getLogger(__name__)

# Transport-level headers that are managed by aiohttp / the runtime; user
# config must not override them. Authorization is intentionally not in this
# set so callers can override auth for inference-gateway style deployments.
_HOP_BY_HOP_HEADERS: frozenset[str] = frozenset({"host", "content-length", "connection", "transfer-encoding"})


def _redact_request_headers_for_ctx(headers: dict[str, str]) -> dict[str, str]:
    """Return a shallow copy of *headers* with Authorization headers removed.

    Used when stashing the outbound request headers into the interceptor
    context so downstream observers (e.g. the http_pairs_dump interceptor)
    never see the upstream API key. Header names are compared case-insensitively;
    any key containing the substring "authorization" is dropped. Other headers
    pass through unchanged.
    """
    return {k: v for k, v in headers.items() if "authorization" not in k.lower()}


class Interceptor(RequestToResponseInterceptor):
    def __init__(
        self,
        *,
        upstream_url: str,
        api_key: str | None = None,
        extra_body: dict[str, Any] | None = None,
        extra_headers: dict[str, str] | None = None,
        request_timeout: float = 120,
        max_retries: int = 0,
        retry_on_status: list[int] | None = None,
        max_concurrent: int = 64,
        model_traffic_store_id: str | None = None,
    ) -> None:
        clean = upstream_url.rstrip("/")
        for suffix in ("/chat/completions", "/completions", "/embeddings"):
            if clean.endswith(suffix):
                clean = clean[: -len(suffix)]
                break
        self._upstream_url = clean
        self._api_key = api_key
        self._extra_body = extra_body or {}
        self._extra_headers: dict[str, str] = {}
        for name, value in (extra_headers or {}).items():
            if name.lower() in _HOP_BY_HOP_HEADERS:
                logger.warning("Dropping hop-by-hop header from extra_headers: %s", name)
                continue
            self._extra_headers[name] = value
        self._timeout = aiohttp.ClientTimeout(total=request_timeout)
        self._max_retries = max_retries
        self._retry_on_status = set(retry_on_status or [429, 502, 503, 504])
        self._max_concurrent = max_concurrent
        self._model_traffic_store = get_store(model_traffic_store_id) if model_traffic_store_id else None
        self._session: aiohttp.ClientSession | None = None
        self._lock = asyncio.Lock()
        logger.info(
            "Endpoint interceptor initialized: upstream=%s request_timeout=%s max_retries=%d extra_body_keys=%s extra_headers=%s",
            self._upstream_url,
            self._timeout.total,
            self._max_retries,
            sorted(self._extra_body),
            sorted(self._extra_headers),
        )

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            async with self._lock:
                if self._session is None or self._session.closed:
                    connector = aiohttp.TCPConnector(
                        limit=self._max_concurrent,
                        keepalive_timeout=30,
                    )
                    self._session = aiohttp.ClientSession(
                        connector=connector,
                        timeout=self._timeout,
                    )
        return self._session

    @staticmethod
    def _normalize_content(body: dict[str, Any]) -> None:
        for choice in body.get("choices", []):
            msg = choice.get("message") or choice.get("delta") or {}
            if "content" in msg and msg["content"] is None:
                msg["content"] = ""

    @staticmethod
    def _request_stream_usage_chunks(path: str, body: dict[str, Any]) -> None:
        if body.get("stream") is not True:
            return
        if not path.rstrip("/").endswith("/chat/completions"):
            return

        stream_options = body.get("stream_options")
        if stream_options is None:
            body["stream_options"] = {"include_usage": True}
        elif isinstance(stream_options, dict) and "include_usage" not in stream_options:
            body["stream_options"] = {**stream_options, "include_usage": True}

    def _capture_response(self, resp: AdapterResponse) -> AdapterResponse:
        if self._model_traffic_store is not None:
            self._model_traffic_store.finish_response(resp)
        return resp

    def _capture_error(self, req: AdapterRequest, *, latency_ms: float, error_type: str) -> None:
        if self._model_traffic_store is not None:
            self._model_traffic_store.finish_error(req, latency_ms=latency_ms, error_type=error_type)

    async def intercept_request(
        self,
        req: AdapterRequest,
    ) -> AdapterRequest | AdapterResponse:
        session = await self._get_session()
        url = f"{self._upstream_url}{req.path}"

        body = {**req.body, **self._extra_body}
        enforce_text_completions_body(req.path, body)
        # Capture the effective upstream request, including endpoint-level
        # body overrides, before sending it.
        self._request_stream_usage_chunks(req.path, body)
        req.ctx.extra["upstream_request_body"] = body
        if self._model_traffic_store is not None:
            self._model_traffic_store.start_request(req)
        headers = {k: v for k, v in req.headers.items() if k.lower() not in _HOP_BY_HOP_HEADERS}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        headers.setdefault("Content-Type", "application/json")
        if self._extra_headers:
            # Override case-insensitively so user-set Authorization wins.
            override_lc = {k.lower() for k in self._extra_headers}
            headers = {k: v for k, v in headers.items() if k.lower() not in override_lc}
            headers.update(self._extra_headers)
        req.ctx.extra["upstream_request_headers"] = _redact_request_headers_for_ctx(headers)

        attempt = 0
        while True:
            t0 = time.perf_counter()
            try:
                async with session.post(
                    url,
                    data=json.dumps(body),
                    headers=headers,
                ) as resp:
                    raw = await resp.read()
                    latency = (time.perf_counter() - t0) * 1000
                    resp_headers = dict(resp.headers)
                    status = resp.status

                    if status in self._retry_on_status and attempt < self._max_retries:
                        retry_after = resp.headers.get("Retry-After")
                        delay = float(retry_after) if retry_after else min(2**attempt, 60)
                        logger.warning(
                            "endpoint: %s returned %d, retry %d/%d in %.1fs",
                            url,
                            status,
                            attempt + 1,
                            self._max_retries,
                            delay,
                        )
                        attempt += 1
                        await asyncio.sleep(delay)
                        continue

                    try:
                        parsed = json.loads(raw)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        parsed = raw

                    if isinstance(parsed, dict):
                        self._normalize_content(parsed)

                    return self._capture_response(
                        AdapterResponse(
                            status_code=status,
                            headers=resp_headers,
                            body=parsed,
                            latency_ms=latency,
                            ctx=req.ctx,
                        )
                    )

            except asyncio.TimeoutError as exc:
                latency = (time.perf_counter() - t0) * 1000
                if attempt < self._max_retries:
                    delay = min(2**attempt, 60)
                    logger.warning(
                        "endpoint: %s timed out (exc_class=%s exc_msg=%s t_in_flight_s=%.1f), retry %d/%d in %.1fs",
                        url,
                        type(exc).__name__,
                        exc,
                        latency / 1000,
                        attempt + 1,
                        self._max_retries,
                        delay,
                    )
                    attempt += 1
                    await asyncio.sleep(delay)
                    continue
                logger.error(
                    "endpoint: %s timed out after %d attempts (exc_class=%s exc_msg=%s t_in_flight_s=%.1f)",
                    url,
                    attempt + 1,
                    type(exc).__name__,
                    exc,
                    latency / 1000,
                )
                return self._capture_response(
                    AdapterResponse(
                        status_code=504,
                        headers={},
                        body={
                            "error": {"message": f"Upstream timed out after {self._timeout.total}s", "type": "timeout"}
                        },
                        latency_ms=latency,
                        ctx=req.ctx,
                    )
                )

            except aiohttp.ClientError as exc:
                latency = (time.perf_counter() - t0) * 1000
                if attempt < self._max_retries:
                    delay = min(2**attempt, 60)
                    logger.warning(
                        "endpoint: %s failed (exc_class=%s exc_msg=%s t_in_flight_s=%.1f), retry %d/%d in %.1fs",
                        url,
                        type(exc).__name__,
                        exc,
                        latency / 1000,
                        attempt + 1,
                        self._max_retries,
                        delay,
                    )
                    attempt += 1
                    await asyncio.sleep(delay)
                    continue
                logger.error(
                    "endpoint: %s failed after %d attempts (exc_class=%s exc_msg=%s t_in_flight_s=%.1f)",
                    url,
                    attempt + 1,
                    type(exc).__name__,
                    exc,
                    latency / 1000,
                )
                self._capture_error(req, latency_ms=latency, error_type=type(exc).__name__)
                raise

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
