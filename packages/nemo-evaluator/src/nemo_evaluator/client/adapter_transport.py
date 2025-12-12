# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Custom httpx transport that processes requests through adapter pipeline in client mode."""

import asyncio
import json
import time
from typing import Any, Dict
from urllib.parse import urlparse

import httpx

from nemo_evaluator.adapters.adapter_config import AdapterConfig, InterceptorConfig
from nemo_evaluator.adapters.pipeline import AdapterPipeline
from nemo_evaluator.adapters.types import (
    AdapterGlobalContext,
    AdapterRequest,
    AdapterRequestContext,
    AdapterResponse,
    FatalErrorException,
)
from nemo_evaluator.logging import bind_request_id, get_logger

logger = get_logger(__name__)


def create_async_adapter_http_client(
    endpoint_url: str,
    adapter_config: AdapterConfig | None = None,
    output_dir: str = "./nemo_eval_output",
    base_transport: httpx.AsyncBaseTransport | None = None,
    is_base_url: bool = False,
    model_name: str | None = None,
) -> tuple[httpx.AsyncClient, "AsyncAdapterTransport"]:
    """Create an async httpx client with adapter transport for client-mode evaluation.

    Args:
        endpoint_url: The endpoint URL (required)
        adapter_config: Adapter configuration. If None, creates default with endpoint interceptor.
        output_dir: Directory for output files
        base_transport: Optional base async transport to wrap
        is_base_url: If True, client appends paths. If False, use URL as-is.
        model_name: Optional model name for logging context

    Returns:
        Tuple of (httpx.AsyncClient, AsyncAdapterTransport)
    """
    if adapter_config is None:
        adapter_config = AdapterConfig(
            mode="client",
            interceptors=[InterceptorConfig(name="endpoint", enabled=True, config={})],
            post_eval_hooks=[],
        )

    adapter_transport = AsyncAdapterTransport(
        endpoint_url=endpoint_url,
        adapter_config=adapter_config,
        output_dir=output_dir,
        base_transport=base_transport,
        is_base_url=is_base_url,
        model_name=model_name,
    )

    adapter_http_client = httpx.AsyncClient(transport=adapter_transport)

    enabled_interceptors = [ic.name for ic in adapter_config.interceptors if ic.enabled]
    logger.info(
        "Created async adapter HTTP client",
        interceptors=enabled_interceptors,
        is_base_url=is_base_url,
    )

    return adapter_http_client, adapter_transport


class IterableHeaders:
    """Wrapper to make httpx.Headers iterable as (key, value) tuples."""

    def __init__(self, httpx_headers):
        self._headers = httpx_headers

    def __iter__(self):
        return iter(self._headers.items())

    def __getitem__(self, key):
        return self._headers[key]

    def get(self, key, default=None):
        return self._headers.get(key, default)

    def items(self):
        return self._headers.items()

    def keys(self):
        return self._headers.keys()

    def values(self):
        return self._headers.values()

    def __len__(self):
        return len(self._headers)


class HttpxRequestWrapper:
    """Wrapper to make httpx.Request compatible with flask.Request interface."""

    def __init__(self, httpx_request: httpx.Request, override_url: str | None = None):
        self._request = httpx_request
        self._json_cache = None
        self._override_url = override_url
        self._headers_override = None

        if override_url:
            parsed = urlparse(override_url)
            modified_headers_dict = dict(self._request.headers)
            modified_headers_dict["host"] = parsed.netloc
            self._headers_override = httpx.Headers(modified_headers_dict)

    @property
    def method(self) -> str:
        return self._request.method

    @property
    def url(self) -> str:
        return self._override_url if self._override_url else str(self._request.url)

    @property
    def headers(self):
        headers_to_use = (
            self._headers_override if self._headers_override else self._request.headers
        )
        return IterableHeaders(headers_to_use)

    @property
    def cookies(self):
        return None

    @property
    def json(self) -> Dict[str, Any] | None:
        if self._json_cache is not None:
            return self._json_cache

        content_type = self.headers.get("content-type", "")
        if "application/json" in content_type and self._request.content:
            try:
                self._json_cache = json.loads(self._request.content.decode("utf-8"))
                return self._json_cache
            except (json.JSONDecodeError, UnicodeDecodeError):
                return None
        return None

    def get_json(
        self, force: bool = False, silent: bool = False
    ) -> Dict[str, Any] | None:
        return self.json

    @property
    def path(self) -> str:
        """Return the path component of the URL."""
        return self._request.url.path

    def get_data(self, as_text: bool = False) -> bytes | str:
        """Get request body data, compatible with flask.Request.get_data()."""
        if self._request.content:
            if as_text:
                return self._request.content.decode("utf-8", errors="ignore")
            return self._request.content
        return b"" if not as_text else ""


class RequestsResponseWrapper:
    """Wrapper to make httpx.Response compatible with requests.Response interface."""

    def __init__(self, httpx_response: httpx.Response):
        self._response = httpx_response

    @property
    def status_code(self) -> int:
        return self._response.status_code

    @property
    def reason(self) -> str:
        return self._response.reason_phrase

    @property
    def headers(self):
        return self._response.headers

    @property
    def content(self) -> bytes:
        return self._response.content

    @property
    def text(self) -> str:
        return self._response.text

    def json(self) -> Dict[str, Any]:
        return self._response.json()


class AsyncAdapterTransport(httpx.AsyncBaseTransport):
    """Async httpx transport that processes requests through adapter interceptor pipeline."""

    def __init__(
        self,
        endpoint_url: str,
        adapter_config: AdapterConfig,
        output_dir: str,
        base_transport: httpx.AsyncBaseTransport | None = None,
        is_base_url: bool = False,
        model_name: str | None = None,
    ):
        self.adapter_config = adapter_config
        self.output_dir = output_dir
        self.base_transport = base_transport or httpx.AsyncHTTPTransport()
        self.endpoint_url = endpoint_url
        self.is_base_url = is_base_url
        self.model_name = model_name
        self.pipeline = AdapterPipeline(adapter_config, output_dir, model_name)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        from nemo_evaluator.logging import bind_model_name

        request_id = bind_request_id()

        # Bind the model name to the logging context if available
        if self.model_name:
            bind_model_name(self.model_name)

        request_logger = get_logger()

        original_url = str(request.url)
        request_url = original_url if self.is_base_url else self.endpoint_url

        request_logger.info(
            "Request started",
            method=request.method,
            url=request_url,
        )

        response = await self._process_request_through_pipeline(
            request, request_url, request_id, request_logger
        )

        return response

    async def _process_request_through_pipeline(
        self,
        request: httpx.Request,
        request_url: str,
        request_id: str,
        request_logger,
    ) -> httpx.Response:
        try:
            global_context = AdapterGlobalContext(
                output_dir=self.output_dir,
                url=request_url,
                model_name=self.model_name,
            )

            wrapped_request = HttpxRequestWrapper(request, override_url=request_url)
            adapter_request = AdapterRequest(
                r=wrapped_request,  # type: ignore
                rctx=AdapterRequestContext(request_id=request_id),
            )

            # Run synchronous pipeline in thread pool
            current_request, adapter_response = await asyncio.to_thread(
                self.pipeline.process_request, adapter_request, global_context
            )

            if adapter_response is None:
                modified_httpx_request = self._adapter_request_to_httpx(
                    current_request, request, request_url
                )

                start_time = time.time()
                httpx_response = await self.base_transport.handle_async_request(
                    modified_httpx_request
                )
                latency_ms = round((time.time() - start_time) * 1000, 2)

                await httpx_response.aread()

                wrapped_response = RequestsResponseWrapper(httpx_response)
                adapter_response = AdapterResponse(
                    r=wrapped_response,  # type: ignore
                    rctx=current_request.rctx,
                    latency_ms=latency_ms,
                )

            current_response = await asyncio.to_thread(
                self.pipeline.process_response, adapter_response, global_context
            )

            return self._adapter_response_to_httpx(current_response, request)

        except FatalErrorException as e:
            request_logger.error(f"Fatal error in adapter pipeline: {e}")
            raise
        except Exception as e:
            request_logger.error(f"Error processing request through adapters: {e}")
            raise

    def _adapter_request_to_httpx(
        self,
        adapter_request: AdapterRequest,
        original_request: httpx.Request,
        normalized_url: str | None = None,
    ) -> httpx.Request:
        json_data = adapter_request.r.get_json()
        content = (
            json.dumps(json_data).encode("utf-8")
            if json_data
            else original_request.content
        )
        url = normalized_url if normalized_url else str(original_request.url)

        return httpx.Request(
            method=adapter_request.r.method,
            url=url,
            headers=dict(adapter_request.r.headers),
            content=content,
        )

    def _adapter_response_to_httpx(
        self, adapter_response: AdapterResponse, original_request: httpx.Request
    ) -> httpx.Response:
        return httpx.Response(
            status_code=adapter_response.r.status_code,
            headers=dict(adapter_response.r.headers),
            content=adapter_response.r.content,
            request=original_request,
        )

    def run_post_eval_hooks(self) -> None:
        self.pipeline.run_post_eval_hooks(url="")

    async def aclose(self) -> None:
        try:
            await asyncio.to_thread(self.run_post_eval_hooks)
        finally:
            await self.base_transport.aclose()
