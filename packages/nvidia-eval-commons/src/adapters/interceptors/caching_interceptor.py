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

"""Caching interceptor with registry support."""

import hashlib
import json
import threading
from typing import Any, final

import requests
import requests.structures
from nvidia_eval_commons.adapters.caching.diskcaching import Cache
from nvidia_eval_commons.adapters.decorators import register_for_adapter
from nvidia_eval_commons.adapters.types import (
    AdapterGlobalContext,
    AdapterRequest,
    AdapterResponse,
    RequestToResponseInterceptor,
    ResponseInterceptor,
)
from pydantic import BaseModel, Field


@register_for_adapter(
    name="caching",
    description="Caches requests and responses with disk storage",
)
@final
class CachingInterceptor(RequestToResponseInterceptor, ResponseInterceptor):
    """Caching interceptor is special in the sense that it intercepts both requests and responses."""

    class Params(BaseModel):
        """Configuration parameters for caching."""

        cache_dir: str = Field(..., description="Directory to store cache files")
        reuse_cached_responses: bool = Field(
            default=True, description="Whether to reuse cached responses"
        )
        save_requests: bool = Field(
            default=False, description="Whether to save requests to cache"
        )
        save_responses: bool = Field(
            default=True, description="Whether to save responses to cache"
        )
        max_saved_requests: int | None = Field(
            default=None, description="Maximum number of requests to save"
        )
        max_saved_responses: int | None = Field(
            default=None, description="Maximum number of responses to save"
        )

    responses_cache: Cache
    requests_cache: Cache
    headers_cache: Cache

    def __init__(self, params: Params):
        """
        Initialize the caching interceptor.

        Args:
            params: Configuration parameters
        """
        self.responses_cache = Cache(directory=f"{params.cache_dir}/responses")
        self.requests_cache = Cache(directory=f"{params.cache_dir}/requests")
        self.headers_cache = Cache(directory=f"{params.cache_dir}/headers")
        self.reuse_cached_responses = params.reuse_cached_responses
        self.save_requests = params.save_requests
        self.save_responses = params.save_responses or params.reuse_cached_responses
        self.max_saved_requests = params.max_saved_requests
        self.max_saved_responses = params.max_saved_responses
        self._cached_requests_count = 0
        self._cached_responses_count = 0
        self._count_lock = threading.Lock()

    @staticmethod
    def _generate_cache_key(data: Any) -> str:
        """
        Generate a hash for the request data to be used as the cache key.

        Args:
            data: Data to be hashed

        Returns:
            str: Hash of the data
        """
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_str.encode("utf-8")).hexdigest()

    def _get_from_cache(self, cache_key: str) -> tuple[Any, Any] | None:
        """
        Attempt to retrieve content and headers from cache.

        Args:
            cache_key (str): Cache key to lookup

        Returns:
            Optional[tuple[Any, Any]]: Tuple of (content, headers) if found, None if not
        """
        try:
            cached_content = self.responses_cache[cache_key]
            cached_headers = self.headers_cache[cache_key]
            return cached_content, cached_headers
        except KeyError:
            return None

    def _save_to_cache(self, cache_key: str, content: Any, headers: Any) -> None:
        """
        Save content and headers to cache.

        Args:
            cache_key (str): Cache key to store under
            content: Content to cache
            headers: Headers to cache
        """
        # Check if we've reached the max responses limit
        if self.max_saved_responses is not None:
            with self._count_lock:
                if self._cached_responses_count >= self.max_saved_responses:
                    return
                self._cached_responses_count += 1

        # Save content to cache
        self.responses_cache[cache_key] = content

        # NOTE: headers are `CaseInsensitiveDict()` which is not serializable
        # by `Cache` class. If this is the case, transform to a normal dict.
        if isinstance(headers, requests.structures.CaseInsensitiveDict):
            cached_headers = dict(headers)
        else:
            cached_headers = headers
        self.headers_cache[cache_key] = cached_headers

    @final
    def intercept_request(
        self, req: AdapterRequest, context: AdapterGlobalContext
    ) -> AdapterRequest | AdapterResponse:
        """Shall return request if no cache hit, and response if it is.
        Args:
            ar (AdapterRequest): The adapter request to intercept
            context (AdapterGlobalContext): Global context containing server-level configuration
        """
        request_data = req.r.get_json()

        # Check cache. Create cache key that will be used everywhere (also if no cache hit)
        req.rctx.cache_key = self._generate_cache_key(request_data)

        # Cache request if needed and within limit
        if self.save_requests:
            with self._count_lock:
                if (
                    self.max_saved_requests is None
                    or self._cached_requests_count < self.max_saved_requests
                ):
                    self.requests_cache[req.rctx.cache_key] = request_data
                    self._cached_requests_count += 1

        # Only check cache if response reuse is enabled
        if self.reuse_cached_responses:
            cached_result = self._get_from_cache(req.rctx.cache_key)
            if cached_result:
                content, headers = cached_result

                requests_response = requests.Response()
                requests_response._content = content
                requests_response.status_code = 200
                requests_response.reason = "OK"
                requests_response.headers = requests.utils.CaseInsensitiveDict(headers)
                requests_response.request = request_data

                # Make downstream know
                req.rctx.cache_hit = True

                return AdapterResponse(r=requests_response, rctx=req.rctx)

        return req

    @final
    def intercept_response(
        self, resp: AdapterResponse, context: AdapterGlobalContext
    ) -> AdapterResponse:
        """Cache the response if caching is enabled and response is successful."""
        # first, if caching was used, we do nothing
        if resp.rctx.cache_hit:
            return resp

        if resp.r.status_code == 200 and self.save_responses:
            # Save both content and headers to cache
            try:
                assert resp.rctx.cache_key, "cache key is unset, this is a bug"
                self._save_to_cache(
                    cache_key=resp.rctx.cache_key,
                    content=resp.r.content,
                    headers=resp.r.headers,
                )
            except Exception as e:
                print(f"Warning: Could not cache response: {e}")

        # And just propagate
        return resp
