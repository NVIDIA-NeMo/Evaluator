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

import json
import time
from typing import Any, Dict

import httpx

from nemo_evaluator.adapters.adapter_config import AdapterConfig
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


def create_adapter_http_client(
    adapter_config: AdapterConfig,
    output_dir: str = "./nemo_eval_output",
    endpoint_url: str | None = None,
    base_transport: httpx.BaseTransport | None = None,
) -> tuple[httpx.Client, "AdapterTransport"]:
    """Factory function to create an httpx client with adapter transport.

    This is a shared helper that can be used by all client types.

    Args:
        adapter_config: Adapter configuration with interceptors
        output_dir: Directory for output files
        endpoint_url: The endpoint URL - can be base URL (e.g., http://host/v1)
                     or full endpoint (e.g., http://host/v1/chat/completions)
        base_transport: Optional base transport to wrap

    Returns:
        Tuple of (httpx.Client with adapter transport, AdapterTransport instance)
    """
    # Validate adapter config has at least one enabled component
    enabled_interceptors = [ic for ic in adapter_config.interceptors if ic.enabled]
    enabled_hooks = [hook for hook in adapter_config.post_eval_hooks if hook.enabled]

    if not enabled_interceptors and not enabled_hooks:
        logger.warning("Creating adapter client with no enabled interceptors or hooks")

    # Create adapter transport
    adapter_transport = AdapterTransport(
        adapter_config=adapter_config,
        output_dir=output_dir,
        base_transport=base_transport,
        endpoint_url=endpoint_url,
    )

    # Create httpx client with adapter transport
    adapter_http_client = httpx.Client(transport=adapter_transport)

    logger.info(
        "Created adapter HTTP client",
        interceptors=[ic.name for ic in enabled_interceptors],
        hooks=[hook.name for hook in enabled_hooks],
    )

    return adapter_http_client, adapter_transport


class HttpxRequestWrapper:
    """Wrapper to make httpx.Request look like flask.Request for interceptors."""

    def __init__(self, httpx_request: httpx.Request):
        self._request = httpx_request
        self._json_cache = None

    @property
    def method(self) -> str:
        return self._request.method

    @property
    def url(self) -> str:
        return str(self._request.url)

    @property
    def headers(self):
        return self._request.headers

    @property
    def cookies(self):
        return self._request.headers.get("cookie", "")

    @property
    def json(self) -> Dict[str, Any] | None:
        """Get JSON body from request."""
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
        """Flask-compatible get_json method."""
        return self.json


class RequestsResponseWrapper:
    """Wrapper to make httpx.Response look like requests.Response for interceptors."""

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


class AdapterTransport(httpx.BaseTransport):
    """Custom httpx transport that processes requests through adapter interceptor pipeline.

    Automatically detects and adapts to both URL modes:
    - Base URL mode: base_url="http://host/v1" → client appends paths like /chat/completions
    - Passthrough mode: base_url="http://custom/endpoint" → use this exact URL

    Detection is done at runtime: first request tries the constructed URL, and if it fails
    with 404/405, automatically retries with the user's endpoint_url directly.
    """

    def __init__(
        self,
        adapter_config: AdapterConfig,
        output_dir: str,
        base_transport: httpx.BaseTransport | None = None,
        endpoint_url: str | None = None,
    ):
        """Initialize adapter transport.

        Args:
            adapter_config: Adapter configuration with interceptors
            output_dir: Directory for output files
            base_transport: Underlying transport to use for actual HTTP requests.
                          If None, uses default HTTPTransport.
            endpoint_url: The endpoint URL provided by the user. If the first request fails
                         with 404/405, we'll retry using this URL directly (passthrough mode).
        """
        self.adapter_config = adapter_config
        self.output_dir = output_dir
        self.base_transport = base_transport or httpx.HTTPTransport()
        self.endpoint_url = endpoint_url

        # Runtime detection of passthrough mode
        # None = not yet detected, True = passthrough mode, False = base URL mode
        self._detected_mode: bool | None = None

        # Initialize the shared adapter pipeline
        self.pipeline = AdapterPipeline(adapter_config, output_dir)

        logger.info(
            "AdapterTransport initialized",
            interceptors=[type(i).__name__ for i in self.pipeline.interceptor_chain],
            post_eval_hooks=[type(h).__name__ for h in self.pipeline.post_eval_hooks],
        )

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        """Process request through adapter pipeline with adaptive URL mode detection.

        Args:
            request: The httpx request to process

        Returns:
            httpx.Response after processing through interceptor chain
        """
        # Generate unique request ID for this request
        request_id = bind_request_id()
        request_logger = get_logger()

        original_url = str(request.url)

        # Determine URL to use based on detected mode
        if self._detected_mode is True:
            # Passthrough mode detected - use endpoint_url directly
            request_url = self.endpoint_url if self.endpoint_url else original_url
        elif self._detected_mode is False:
            # Base URL mode detected - use constructed URL
            request_url = original_url
        else:
            # Mode not yet detected - will try constructed URL first
            request_url = original_url

        request_logger.debug(
            "Processing request through adapter pipeline",
            method=request.method,
            url=request_url,
            detected_mode="passthrough"
            if self._detected_mode is True
            else "base_url"
            if self._detected_mode is False
            else "detecting",
        )

        try:
            # Try making the request with constructed URL
            response = self._process_request_through_pipeline(
                request, request_url, request_id, request_logger
            )

            # If mode not yet detected and request succeeded, we're in base URL mode
            if self._detected_mode is None:
                self._detected_mode = False
                logger.info("Detected base URL mode (first request succeeded)")

            return response

        except Exception as e:
            # If this is the first request and we got a client error, try passthrough mode
            if self._detected_mode is None and self._should_retry_with_endpoint_url(e):
                if self.endpoint_url and self.endpoint_url != original_url:
                    logger.info(
                        "First request failed with client error, retrying with endpoint_url directly",
                        constructed_url=original_url,
                        endpoint_url=self.endpoint_url,
                        error=str(e),
                    )

                    try:
                        # Retry with user's endpoint_url directly
                        response = self._process_request_through_pipeline(
                            request, self.endpoint_url, request_id, request_logger
                        )

                        # Success! Remember passthrough mode for future requests
                        self._detected_mode = True
                        logger.info(
                            "Detected passthrough mode (retry succeeded with endpoint_url)"
                        )

                        return response
                    except Exception as retry_error:
                        logger.error(
                            f"Retry with endpoint_url also failed: {retry_error}"
                        )
                        # Both attempts failed - raise the retry error
                        raise

            # Re-raise the original exception
            raise

    def _should_retry_with_endpoint_url(self, error: Exception) -> bool:
        """Check if error indicates we should retry with endpoint_url directly.

        Args:
            error: The exception that occurred

        Returns:
            True if we should retry with endpoint_url
        """
        # Check if it's an httpx response with client error status
        if hasattr(error, "response") and hasattr(error.response, "status_code"):
            status_code = error.response.status_code
            # 404 Not Found, 405 Method Not Allowed, 501 Not Implemented
            return status_code in (404, 405, 501)

        # Also check for FatalErrorException or other exceptions that might wrap response errors
        error_str = str(error).lower()
        return any(
            indicator in error_str
            for indicator in ["404", "405", "501", "not found", "method not allowed"]
        )

    def _process_request_through_pipeline(
        self,
        request: httpx.Request,
        request_url: str,
        request_id: str,
        request_logger,
    ) -> httpx.Response:
        """Process a request through the adapter pipeline.

        Args:
            request: The original httpx request
            request_url: The URL to use (potentially normalized)
            request_id: Unique request ID
            request_logger: Logger for this request

        Returns:
            httpx.Response after processing
        """
        try:
            # Create global context
            global_context = AdapterGlobalContext(
                output_dir=self.output_dir,
                url=request_url,
            )

            # Wrap httpx request to look like flask request
            wrapped_request = HttpxRequestWrapper(request)

            # Create adapter request
            adapter_request = AdapterRequest(
                r=wrapped_request,  # type: ignore
                rctx=AdapterRequestContext(request_id=request_id),
            )

            # Process through request interceptors using shared pipeline
            current_request, adapter_response = self.pipeline.process_request(
                adapter_request, global_context
            )

            # If no interceptor returned a response, make the actual HTTP call
            if adapter_response is None:
                # Reconstruct httpx request with potentially modified data and URL
                modified_httpx_request = self._adapter_request_to_httpx(
                    current_request, request, request_url
                )

                # Make actual HTTP call using base transport
                start_time = time.time()
                httpx_response = self.base_transport.handle_request(
                    modified_httpx_request
                )
                latency_ms = round((time.time() - start_time) * 1000, 2)

                # Wrap response for adapter pipeline
                wrapped_response = RequestsResponseWrapper(httpx_response)
                adapter_response = AdapterResponse(
                    r=wrapped_response,  # type: ignore
                    rctx=current_request.rctx,
                    latency_ms=latency_ms,
                )

            # Process through response interceptors using shared pipeline
            current_response = self.pipeline.process_response(
                adapter_response, global_context
            )

            request_logger.debug(
                "Request processing completed",
                status_code=current_response.r.status_code,
            )

            # Convert adapter response back to httpx response
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
        """Convert AdapterRequest back to httpx.Request.

        Args:
            adapter_request: The adapter request (potentially modified)
            original_request: The original httpx request
            normalized_url: Optional normalized URL to use (for passthrough mode)

        Returns:
            A new httpx.Request with potentially modified data
        """
        # Get JSON body if available
        json_data = adapter_request.r.get_json()

        # Prepare content
        if json_data is not None:
            content = json.dumps(json_data).encode("utf-8")
        else:
            content = original_request.content

        # Use normalized URL if provided, otherwise keep original
        url = normalized_url if normalized_url else str(original_request.url)

        # Create new request with potentially modified data
        return httpx.Request(
            method=adapter_request.r.method,
            url=url,
            headers=dict(adapter_request.r.headers),
            content=content,
        )

    def _adapter_response_to_httpx(
        self, adapter_response: AdapterResponse, original_request: httpx.Request
    ) -> httpx.Response:
        """Convert AdapterResponse to httpx.Response.

        Args:
            adapter_response: The adapter response
            original_request: The original request (needed for httpx.Response constructor)

        Returns:
            An httpx.Response
        """
        return httpx.Response(
            status_code=adapter_response.r.status_code,
            headers=dict(adapter_response.r.headers),
            content=adapter_response.r.content,
            request=original_request,
        )

    def run_post_eval_hooks(self) -> None:
        """Run all configured post-evaluation hooks."""
        self.pipeline.run_post_eval_hooks(url="")

    def close(self) -> None:
        """Close the transport and run post-eval hooks."""
        try:
            self.run_post_eval_hooks()
        finally:
            self.base_transport.close()
