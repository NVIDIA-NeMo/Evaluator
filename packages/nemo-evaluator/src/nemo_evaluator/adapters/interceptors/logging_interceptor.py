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

"""Logging interceptors with registry support."""

import threading
from typing import Optional, final

from pydantic import Field

from nemo_evaluator.adapters.decorators import register_for_adapter
from nemo_evaluator.adapters.types import (
    AdapterGlobalContext,
    AdapterRequest,
    AdapterResponse,
    RequestInterceptor,
    ResponseInterceptor,
)
from nemo_evaluator.logging import BaseLoggingParams, get_logger


def _get_safe_headers(headers: dict[str, str]) -> dict[str, str]:
    """Create a copy of headers with authorization redacted."""
    safe_headers = dict(headers)
    for header in safe_headers:
        if header.lower() == "authorization":
            safe_headers[header] = "[REDACTED]"
    return safe_headers


@register_for_adapter(
    name="request_logging",
    description="Logs incoming requests",
)
@final
class RequestLoggingInterceptor(RequestInterceptor):
    """Logs incoming requests."""

    class Params(BaseLoggingParams):
        """Configuration parameters for request logging."""

        log_request_body: bool = Field(
            default=True, description="Whether to log request body"
        )
        log_request_headers: bool = Field(
            default=True, description="Whether to log request headers"
        )
        max_requests: Optional[int] = Field(
            default=2,
            description="Maximum number of requests to log (None for unlimited)",
        )

    log_request_body: bool
    log_request_headers: bool
    max_requests: Optional[int]
    _request_count: int

    def __init__(self, params: Params):
        """
        Initialize the request logging interceptor.

        Args:
            params: Configuration parameters
        """
        self.log_request_body = params.log_request_body
        self.log_request_headers = params.log_request_headers
        self.max_requests = params.max_requests
        self._lock = threading.Lock()
        self._request_count = 0

        # Get logger for this interceptor with interceptor context
        self.logger = get_logger(self.__class__.__name__)

        self.logger.info(
            "Request logging interceptor initialized",
            log_request_body=self.log_request_body,
            log_request_headers=self.log_request_headers,
        )

    @final
    def intercept_request(
        self, ar: AdapterRequest, context: AdapterGlobalContext
    ) -> AdapterRequest:
        """Log the incoming request."""

        # NOTE(agronskiy): to reduce lock contention, stop locking after the
        # count is reached.
        with self._lock:
            if (
                self.max_requests is not None
                and self._request_count >= self.max_requests
            ):
                self.logger.debug(
                    "Request logging limit reached, skipping log",
                    max_requests=self.max_requests,
                    current_count=self._request_count,
                )
                return ar

            self._request_count += 1

        log_data = {
            "method": ar.r.method,
            "url": ar.r.url,
            "path": ar.r.path,
        }

        if self.log_request_headers:
            log_data["headers"] = _get_safe_headers(dict(ar.r.headers))

        if self.log_request_body:
            try:
                log_data["body"] = ar.r.get_json()
            except Exception:
                log_data["body"] = ar.r.get_data().decode("utf-8", errors="ignore")

        # Use standard logging (request_id is automatically included from context)
        self.logger.info("Incoming request", **log_data)

        return ar


@register_for_adapter(
    name="response_logging",
    description="Logs responses",
)
@final
class ResponseLoggingInterceptor(ResponseInterceptor):
    """Logs responses."""

    class Params(BaseLoggingParams):
        """Configuration parameters for response logging."""

        log_response_body: bool = Field(
            default=True, description="Whether to log response body"
        )
        log_response_headers: bool = Field(
            default=True, description="Whether to log response headers"
        )
        max_responses: Optional[int] = Field(
            default=None,
            description="Maximum number of responses to log (None for unlimited)",
        )

    log_response_body: bool
    log_response_headers: bool
    max_responses: Optional[int]
    _response_count: int

    def __init__(self, params: Params):
        """
        Initialize the response logging interceptor.

        Args:
            params: Configuration parameters
        """
        self.log_response_body = params.log_response_body
        self.log_response_headers = params.log_response_headers
        self.max_responses = params.max_responses
        self._response_count = 0
        self._lock = threading.Lock()

        # Get logger for this interceptor with interceptor context
        self.logger = get_logger(self.__class__.__name__)

        self.logger.info(
            "Response logging interceptor initialized",
            log_response_body=self.log_response_body,
            log_response_headers=self.log_response_headers,
            max_responses=self.max_responses,
        )

    @final
    def intercept_response(
        self, resp: AdapterResponse, context: AdapterGlobalContext
    ) -> AdapterResponse:
        """Log the outgoing response."""
        # Check if we should log this response based on max_responses limit
        # NOTE(agronskiy): to reduce lock contention, stop locking after the
        # count is reached.
        with self._lock:
            if (
                self.max_responses is not None
                and self._response_count >= self.max_responses
            ):
                self.logger.debug(
                    "Response logging limit reached, skipping log",
                    max_responses=self.max_responses,
                    current_count=self._response_count,
                )
                return resp

            self._response_count += 1

        log_data = {
            "status_code": resp.r.status_code,
            "reason": resp.r.reason,
        }

        if self.log_response_headers:
            log_data["headers"] = dict(resp.r.headers)

        if self.log_response_body:
            try:
                log_data["body"] = resp.r.json()
            except Exception:
                log_data["body"] = resp.r.text

        # Use standard logging (request_id is automatically included from context)
        self.logger.info("Outgoing response", **log_data)

        return resp
