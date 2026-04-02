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

logger = logging.getLogger(__name__)


class Interceptor(RequestToResponseInterceptor):
    def __init__(
        self,
        *,
        upstream_url: str,
        api_key: str | None = None,
        extra_body: dict[str, Any] | None = None,
        request_timeout: float = 120,
        max_retries: int = 0,
        retry_on_status: list[int] | None = None,
        max_concurrent: int = 64,
    ) -> None:
        clean = upstream_url.rstrip("/")
        for suffix in ("/chat/completions", "/completions", "/embeddings"):
            if clean.endswith(suffix):
                clean = clean[: -len(suffix)]
                break
        self._upstream_url = clean
        self._api_key = api_key
        self._extra_body = extra_body or {}
        self._timeout = aiohttp.ClientTimeout(total=request_timeout)
        self._max_retries = max_retries
        self._retry_on_status = set(retry_on_status or [429, 502, 503, 504])
        self._max_concurrent = max_concurrent
        self._session: aiohttp.ClientSession | None = None
        self._lock = asyncio.Lock()

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

    async def intercept_request(
        self,
        req: AdapterRequest,
    ) -> AdapterRequest | AdapterResponse:
        session = await self._get_session()
        url = f"{self._upstream_url}{req.path}"

        body = {**req.body, **self._extra_body}
        headers = {
            k: v for k, v in req.headers.items() if k.lower() not in ("host", "content-length", "transfer-encoding")
        }
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        headers.setdefault("Content-Type", "application/json")

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

                    return AdapterResponse(
                        status_code=status,
                        headers=resp_headers,
                        body=parsed,
                        latency_ms=latency,
                        ctx=req.ctx,
                    )

            except asyncio.TimeoutError:
                latency = (time.perf_counter() - t0) * 1000
                if attempt < self._max_retries:
                    delay = min(2**attempt, 60)
                    logger.warning(
                        "endpoint: %s timed out, retry %d/%d in %.1fs",
                        url,
                        attempt + 1,
                        self._max_retries,
                        delay,
                    )
                    attempt += 1
                    await asyncio.sleep(delay)
                    continue
                logger.error("endpoint: %s timed out after %d attempts", url, attempt + 1)
                return AdapterResponse(
                    status_code=504,
                    headers={},
                    body={"error": {"message": f"Upstream timed out after {self._timeout.total}s", "type": "timeout"}},
                    latency_ms=latency,
                    ctx=req.ctx,
                )

            except aiohttp.ClientError as exc:
                latency = (time.perf_counter() - t0) * 1000
                if attempt < self._max_retries:
                    delay = min(2**attempt, 60)
                    logger.warning(
                        "endpoint: %s failed (%s), retry %d/%d in %.1fs",
                        url,
                        exc,
                        attempt + 1,
                        self._max_retries,
                        delay,
                    )
                    attempt += 1
                    await asyncio.sleep(delay)
                    continue
                raise

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
