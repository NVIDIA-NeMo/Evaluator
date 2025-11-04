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

import difflib
import hashlib
import json
import os
import pickle
import threading
from typing import Any, final

import requests
import requests.structures
from pydantic import Field
from rapidfuzz import process, fuzz

from nemo_evaluator.adapters.caching.diskcaching import Cache
from nemo_evaluator.adapters.decorators import register_for_adapter
from nemo_evaluator.adapters.types import (
    AdapterGlobalContext,
    AdapterRequest,
    AdapterResponse,
    RequestToResponseInterceptor,
    ResponseInterceptor,
    PostEvalHook,
    PreEvalHook
)
from nemo_evaluator.logging import BaseLoggingParams, get_logger


class CacheMissInTestModeError(Exception):
    """Raised when a cache miss occurs in test mode."""
    
    def __init__(self, message: str, request_data: dict, most_similar_request: dict | None = None, similarity_score: float | None = None, diff: str | None = None):
        super().__init__(message)
        self.request_data = request_data
        self.most_similar_request = most_similar_request
        self.similarity_score = similarity_score
        self.diff = diff


@register_for_adapter(
    name="caching",
    description="Caches requests and responses with disk storage",
)
@final
class CachingInterceptor(RequestToResponseInterceptor, ResponseInterceptor, PreEvalHook, PostEvalHook):
    """Caching interceptor is special in the sense that it intercepts both requests and responses."""

    class Params(BaseLoggingParams):
        """Configuration parameters for caching."""

        prefill_from_export: str = Field(
            default=None,
            description="Path to an exported .cache file (single pickle file) to prefill the cache from. If provided, the cache will be populated from the selected export file at evaluation start. This is different from the local cache_dir which uses 3 separate directories.",
        )
        cache_dir: str = Field(
            default="/tmp", description="Directory to store local cache files (3 separate directories: requests, responses, headers)"
        )
        reuse_cached_responses: bool = Field(
            default=False,
            description="Whether to reuse cached responses. If True, this overrides save_responses (sets it to True) and max_saved_responses (sets it to None)",
        )
        save_requests: bool = Field(
            default=False, description="Whether to save requests to cache"
        )
        save_responses: bool = Field(
            default=True,
            description="Whether to save responses to cache. Note: This is automatically set to True if reuse_cached_responses is True",
        )
        max_saved_requests: int | None = Field(
            default=None, description="Maximum number of requests to save"
        )
        max_saved_responses: int | None = Field(
            default=None,
            description="Maximum number of responses to cache. Note: This is automatically set to None if reuse_cached_responses is True",
        )
        export_cache: bool = Field(
            default=False,
            description="Whether to export the cache to a single .cache file (pickle format). If True, the final cache will be exported to a binary cache file in the output directory. This creates a portable single-file cache that can be used with prefill_from_export.",
        )
        test_mode: bool = Field(
            default=False,
            description="Enable test mode which fails on first cache miss and shows diff with most similar cached request. Useful for debugging request changes when using cached responses.",
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
        # Initializing caches is now done in the pre_eval_hook method as this might be compute-heavy
        self.cache_dir = params.cache_dir
        self.reuse_cached_responses = params.reuse_cached_responses
        self.save_requests = params.save_requests
        self.prefill_from_export = params.prefill_from_export
        self.export_cache = params.export_cache
        self.test_mode = params.test_mode

        # If reuse_cached_responses is True, override save_responses and max_saved_responses
        if params.reuse_cached_responses:
            self.save_responses = True
            self.max_saved_responses = None
        else:
            self.save_responses = params.save_responses
            self.max_saved_responses = params.max_saved_responses

        self.max_saved_requests = params.max_saved_requests

        # Counters for cache management
        self._cached_requests_count = 0
        self._cached_responses_count = 0

        # Thread safety
        self._count_lock = threading.Lock()

        # Get logger for this interceptor with interceptor context
        self.logger = get_logger(self.__class__.__name__)

        self.logger.info(
            "Caching interceptor initialized",
            cache_dir=params.cache_dir,
            reuse_cached_responses=self.reuse_cached_responses,
            save_requests=self.save_requests,
            save_responses=self.save_responses,
            max_saved_requests=self.max_saved_requests,
            max_saved_responses=self.max_saved_responses,
            prefill_from_export=self.prefill_from_export,
            export_cache=self.export_cache,
            test_mode=self.test_mode,
        )

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
    
    def _find_most_similar_cached_request(self, request_data: dict) -> tuple[dict | None, float]:
        """
        Find the most similar cached request using rapidfuzz.
        
        Args:
            request_data: The request data to compare
            
        Returns:
            Tuple of (most_similar_request_data, similarity_score)
        """
        # Build collection of cached requests
        cached_requests = {}
        try:
            for key in self.requests_cache.iterkeys():
                try:
                    cached_data = self.requests_cache[key]
                    cached_str = json.dumps(cached_data, sort_keys=True, indent=2)
                    cached_requests[cached_str] = cached_data
                except Exception:
                    continue
        except Exception:
            return None, 0.0
        
        if not cached_requests:
            return None, 0.0
        
        # Convert current request to string
        request_str = json.dumps(request_data, sort_keys=True, indent=2)
        
        # Use rapidfuzz process.extractOne for efficient best match finding
        result = process.extractOne(
            request_str,
            cached_requests.keys(),
            scorer=fuzz.ratio
        )
        
        if result is None:
            return None, 0.0
        
        best_match_str, best_score, _ = result
        best_cached_data = cached_requests[best_match_str]
        
        return best_cached_data, best_score
    
    def _generate_diff(self, request_data: dict, cached_data: dict) -> str:
        """
        Generate a human-readable diff between two requests.
        
        Args:
            request_data: The current request data
            cached_data: The cached request data to compare against
            
        Returns:
            A unified diff string
        """
        request_str = json.dumps(request_data, sort_keys=True, indent=2)
        cached_str = json.dumps(cached_data, sort_keys=True, indent=2)
        
        diff = difflib.unified_diff(
            cached_str.splitlines(keepends=True),
            request_str.splitlines(keepends=True),
            fromfile='cached_request',
            tofile='current_request',
            lineterm=''
        )
        
        return ''.join(diff)

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
            self.logger.debug("Cache hit", cache_key=cache_key[:8] + "...")
            return cached_content, cached_headers
        except KeyError:
            self.logger.debug("Cache miss", cache_key=cache_key[:8] + "...")
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
                    self.logger.warning(
                        "Maximum cached responses limit reached",
                        max_saved_responses=self.max_saved_responses,
                    )
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

        self.logger.debug(
            "Saved response to cache",
            cache_key=cache_key[:8] + "...",
            content_size=len(content) if hasattr(content, "__len__") else "unknown",
        )

    def _export_cache_to_dict(self) -> dict[str, Any]:
        """Export cache contents to a dictionary structure.
        
        Returns:
            Dictionary with requests, responses, and headers caches
        """
        cache_data = {
            "requests": {},
            "responses": {},
            "headers": {},
        }
        
        # Export requests cache
        for key in self.requests_cache.iterkeys():
            try:
                value = self.requests_cache[key]
                cache_data["requests"][key] = value
            except (KeyError, Exception) as e:
                self.logger.warning(f"Failed to export request key {key}: {e}")
        
        # Export responses cache (binary data handled natively by pickle)
        for key in self.responses_cache.iterkeys():
            try:
                value = self.responses_cache[key]
                cache_data["responses"][key] = value
            except (KeyError, Exception) as e:
                self.logger.warning(f"Failed to export response key {key}: {e}")
        
        # Export headers cache
        for key in self.headers_cache.iterkeys():
            try:
                value = self.headers_cache[key]
                cache_data["headers"][key] = value
            except (KeyError, Exception) as e:
                self.logger.warning(f"Failed to export header key {key}: {e}")
        
        return cache_data
    
    def _load_cache_from_dict(self, cache_data: dict[str, Any]) -> None:
        """Load cache contents from a dictionary structure.
        
        Args:
            cache_data: Dictionary with requests, responses, and headers caches
        """
        # Load requests cache
        if "requests" in cache_data:
            for key, value in cache_data["requests"].items():
                try:
                    self.requests_cache[key] = value
                    self._cached_requests_count += 1
                except Exception as e:
                    self.logger.warning(f"Failed to import request key {key}: {e}")
        
        # Load responses cache (binary data handled natively by pickle)
        if "responses" in cache_data:
            for key, value in cache_data["responses"].items():
                try:
                    self.responses_cache[key] = value
                    self._cached_responses_count += 1
                except Exception as e:
                    self.logger.warning(f"Failed to import response key {key}: {e}")
        
        # Load headers cache
        if "headers" in cache_data:
            for key, value in cache_data["headers"].items():
                try:
                    self.headers_cache[key] = value
                except Exception as e:
                    self.logger.warning(f"Failed to import header key {key}: {e}")

    @final
    def intercept_request(
        self, req: AdapterRequest, context: AdapterGlobalContext
    ) -> AdapterRequest | AdapterResponse:
        """Shall return request if no cache hit, and response if it is.
        Args:
            req (AdapterRequest): The adapter request to intercept
            context (AdapterGlobalContext): Global context containing server-level configuration
        """
        request_data = req.r.get_json()

        # Check cache. Create cache key that will be used everywhere (also if no cache hit)
        req.rctx.cache_key = self._generate_cache_key(request_data)

        self.logger.debug(
            "Processing request for caching",
            cache_key=req.rctx.cache_key[:8] + "...",
            request_data_keys=(
                list(request_data.keys())
                if isinstance(request_data, dict)
                else "unknown"
            ),
        )

        # Cache request if needed and within limit
        if self.save_requests:
            with self._count_lock:
                if (
                    self.max_saved_requests is None
                    or self._cached_requests_count < self.max_saved_requests
                ):
                    self.requests_cache[req.rctx.cache_key] = request_data
                    self._cached_requests_count += 1
                    self.logger.debug(
                        "Saved request to cache",
                        cache_key=req.rctx.cache_key[:8] + "...",
                    )
                else:
                    self.logger.warning(
                        "Maximum cached requests limit reached",
                        max_saved_requests=self.max_saved_requests,
                    )

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

                self.logger.info(
                    "Returning cached response",
                    cache_key=req.rctx.cache_key[:8] + "...",
                    status_code=200,
                )

                return AdapterResponse(r=requests_response, rctx=req.rctx)
            
            # Cache miss - handle test mode
            if self.test_mode:
                self.logger.error(
                    "Cache miss in test mode",
                    cache_key=req.rctx.cache_key[:8] + "...",
                )
                
                # Find most similar request
                most_similar, similarity = self._find_most_similar_cached_request(request_data)
                
                error_message = f"Cache miss in test mode for request with cache_key={req.rctx.cache_key[:16]}..."
                
                if most_similar and similarity > 0:
                    diff = self._generate_diff(request_data, most_similar)
                    error_message += f"\n\nMost similar cached request (similarity: {similarity:.2f}%):\n{diff}"
                    
                    self.logger.error(
                        "Most similar cached request found",
                        similarity=f"{similarity:.2f}%",
                        diff=diff[:500] + "..." if len(diff) > 500 else diff,
                    )
                    
                    raise CacheMissInTestModeError(
                        error_message,
                        request_data=request_data,
                        most_similar_request=most_similar,
                        similarity_score=similarity,
                        diff=diff,
                    )
                else:
                    error_message += "\n\nNo similar cached requests found."
                    self.logger.error("No similar cached requests found")
                    
                    raise CacheMissInTestModeError(
                        error_message,
                        request_data=request_data,
                    )

        self.logger.debug(
            "No cache hit, proceeding with request",
            cache_key=req.rctx.cache_key[:8] + "...",
        )
        return req

    @final
    def intercept_response(
        self, resp: AdapterResponse, context: AdapterGlobalContext
    ) -> AdapterResponse:
        """Cache the response if caching is enabled and response is successful."""

        # first, if caching was used, we do nothing
        if resp.rctx.cache_hit:
            self.logger.debug(
                "Response was from cache, skipping caching",
                cache_key=(
                    resp.rctx.cache_key[:8] + "..."
                    if hasattr(resp.rctx, "cache_key")
                    else "unknown"
                ),
            )
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
                self.logger.info(
                    "Cached successful response",
                    cache_key=resp.rctx.cache_key[:8] + "...",
                )
            except Exception as e:
                self.logger.error(
                    "Could not cache response",
                    error=str(e),
                    cache_key=(
                        resp.rctx.cache_key[:8] + "..."
                        if hasattr(resp.rctx, "cache_key")
                        else "unknown"
                    ),
                )
        else:
            self.logger.debug(
                "Response not cached",
                status_code=resp.r.status_code,
                save_responses=self.save_responses,
            )

        # And just propagate
        return resp


    @final
    def pre_eval_hook(self, context: AdapterGlobalContext) -> None:
        """Initialize caches and optionally load from exported cache file.
        
        Args:
            context: Global adapter context
        """
        # Initialize the cache directories (local 3-directory format)
        self.responses_cache = Cache(directory=f"{self.cache_dir}/responses")
        self.requests_cache = Cache(directory=f"{self.cache_dir}/requests")
        self.headers_cache = Cache(directory=f"{self.cache_dir}/headers")

        self.logger.info(
            "Cache directories initialized",
            responses_cache=f"{self.cache_dir}/responses",
            requests_cache=f"{self.cache_dir}/requests",
            headers_cache=f"{self.cache_dir}/headers",
        )

        # If prefill_from_export is provided, load cache from exported file
        if self.prefill_from_export:
            if not os.path.exists(self.prefill_from_export):
                self.logger.error(
                    "Prefill export cache file not found",
                    prefill_from_export=self.prefill_from_export,
                )
                raise FileNotFoundError(f"Export cache file not found: {self.prefill_from_export}")
            
            try:
                self.logger.info(
                    "Loading cache from exported binary file",
                    prefill_from_export=self.prefill_from_export,
                )
                with open(self.prefill_from_export, "rb") as f:
                    cache_data = pickle.load(f)
                
                self._load_cache_from_dict(cache_data)
                
                self.logger.info(
                    "Cache loaded successfully from export",
                    requests_count=self._cached_requests_count,
                    responses_count=self._cached_responses_count,
                )
            except Exception as e:
                self.logger.error(
                    "Failed to load cache from exported binary file",
                    prefill_from_export=self.prefill_from_export,
                    error=str(e),
                )
                raise

    @final
    def post_eval_hook(self, context: AdapterGlobalContext) -> None:
        """Optionally export cache to binary cache file.
        
        Args:
            context: Global adapter context
        """
        if self.export_cache:
            try:
                # Generate output filename in cache directory
                output_file = os.path.join(self.cache_dir, "cache_export.cache")
                
                self.logger.info(
                    "Exporting cache to binary file",
                    output_file=output_file,
                )
                
                cache_data = self._export_cache_to_dict()
                
                # Write to file using pickle
                with open(output_file, "wb") as f:
                    pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
                
                # Log statistics
                requests_count = len(cache_data["requests"])
                responses_count = len(cache_data["responses"])
                headers_count = len(cache_data["headers"])
                
                self.logger.info(
                    "Cache exported successfully",
                    output_file=output_file,
                    requests_count=requests_count,
                    responses_count=responses_count,
                    headers_count=headers_count,
                )
            except Exception as e:
                self.logger.error(
                    "Failed to export cache to binary file",
                    error=str(e),
                )

