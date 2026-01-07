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

"""Endpoint interceptor that makes actual requests to the upstream API."""

import json
import time
from typing import final

import requests

from nemo_evaluator.adapters.decorators import register_for_adapter
from nemo_evaluator.adapters.types import (
    AdapterGlobalContext,
    AdapterRequest,
    AdapterResponse,
    RequestToResponseInterceptor,
)
from nemo_evaluator.logging import BaseLoggingParams, get_logger


@register_for_adapter(
    name="endpoint",
    description="Makes the actual request to the upstream API",
)
@final
class EndpointInterceptor(RequestToResponseInterceptor):
    """Required interceptor that handles the actual API communication. This interceptor must be present in every configuration as it performs the final request to the target API endpoint.
    Important: This interceptor should always be placed after the last request interceptor and before the first response interceptor."""

    class Params(BaseLoggingParams):
        """Configuration parameters for endpoint interceptor."""

        pass

    def __init__(self, params: Params):
        """
        Initialize the endpoint interceptor.

        Args:
            params: Configuration parameters
        """
        # Get logger for this interceptor with interceptor context
        self.logger = get_logger(self.__class__.__name__)

        self.logger.info("Endpoint interceptor initialized")

    def intercept_request(
        self, ar: AdapterRequest, context: AdapterGlobalContext
    ) -> AdapterResponse:
        """Make the actual request to the upstream API.

        Args:
            ar: The adapter request
            context: Global context containing server-level configuration

        Returns:
            AdapterResponse with the response from the upstream API
        """
        self.logger.debug(
            "Making request to upstream API",
            method=ar.r.method,
            url=context.url,
            headers_count=len(ar.r.headers),
            has_json=ar.r.json is not None,
        )

        # Record start time for latency tracking
        start_time = time.time()

        # This is a final interceptor, we'll need the flask_request and api
        raw_response = requests.request(
            method=ar.r.method,
            url=context.url,
            headers={k: v for k, v in ar.r.headers if k.lower() != "host"},
            json=ar.r.json,
            cookies=ar.r.cookies,
            allow_redirects=False,
        )

        # replace choices[xx].message.content=None with empty string
        if raw_response.content is not None:
            try:
                response_json = json.loads(raw_response.content)
                if (
                    "choices" in response_json
                    and isinstance(response_json["choices"], list)
                    and len(response_json["choices"]) > 0
                ):
                    for i, choice in enumerate(response_json["choices"]):
                        if (
                            "message" in choice
                            and "content" in choice["message"]
                            and choice["message"]["content"] is None
                        ):
                            self.logger.warning(
                                f"choices[{i}].message.content is None, replacing with empty string"
                            )
                            choice["message"]["content"] = ""
                raw_response._content = json.dumps(response_json).encode("utf-8")
            except (json.JSONDecodeError, TypeError, KeyError) as e:
                # If JSON parsing fails or unexpected structure, leave response unchanged
                self.logger.debug(
                    "Could not parse response as JSON, leaving unchanged", error=str(e)
                )

        resp = AdapterResponse(
            r=raw_response,
            rctx=ar.rctx,
            latency_ms=round(
                (time.time() - start_time) * 1000, 2
            ),  # Convert to milliseconds with 2 decimal places
        )

        self.logger.debug(
            "Upstream API request completed",
            status_code=resp.r.status_code,
            reason=resp.r.reason,
            response_headers_count=len(resp.r.headers),
            response_content_length=len(resp.r.content) if resp.r.content else 0,
            latency_ms=resp.latency_ms,
        )

        return resp
