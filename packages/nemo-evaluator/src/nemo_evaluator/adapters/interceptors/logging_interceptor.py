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

from typing import Optional, final

import structlog
from nemo_evaluator.adapters.decorators import register_for_adapter
from nemo_evaluator.adapters.types import (
    AdapterGlobalContext,
    AdapterRequest,
    AdapterResponse,
    RequestInterceptor,
    ResponseInterceptor,
)
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


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

    class Params(BaseModel):
        """Configuration parameters for request logging."""

        log_level: str = Field(
            default="INFO", description="Log level for request logging"
        )
        log_request_body: bool = Field(
            default=True, description="Whether to log request body"
        )
        log_request_headers: bool = Field(
            default=True, description="Whether to log request headers"
        )

    log_level: str
    log_request_body: bool
    log_request_headers: bool

    def __init__(self, params: Params):
        """
        Initialize the request logging interceptor.

        Args:
            params: Configuration parameters
        """
        self.log_level = params.log_level
        self.log_request_body = params.log_request_body
        self.log_request_headers = params.log_request_headers

    @final
    def intercept_request(
        self, ar: AdapterRequest, context: AdapterGlobalContext
    ) -> AdapterRequest:
        """Log the incoming request."""
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

        getattr(logger, self.log_level.lower())(
            "Incoming request",
            **log_data,
        )

        return ar


@register_for_adapter(
    name="response_logging",
    description="Logs outgoing responses",
)
@final
class ResponseLoggingInterceptor(ResponseInterceptor):
    """Logs outgoing responses."""

    class Params(BaseModel):
        """Configuration parameters for response logging."""

        log_level: str = Field(
            default="INFO", description="Log level for response logging"
        )
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

    log_level: str
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
        self.log_level = params.log_level
        self.log_response_body = params.log_response_body
        self.log_response_headers = params.log_response_headers
        self.max_responses = params.max_responses
        self._response_count = 0

    @final
    def intercept_response(
        self, resp: AdapterResponse, context: AdapterGlobalContext
    ) -> AdapterResponse:
        """Log the outgoing response."""
        # Check if we should log this response based on max_responses limit
        if (
            self.max_responses is not None
            and self._response_count >= self.max_responses
        ):
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

        getattr(logger, self.log_level.lower())(
            "Outgoing response",
            **log_data,
        )

        return resp
