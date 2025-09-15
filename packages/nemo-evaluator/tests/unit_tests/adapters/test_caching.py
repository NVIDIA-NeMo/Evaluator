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

import json
from unittest.mock import MagicMock, patch

import pytest
import requests
from flask import Request

# Import to register interceptors
from nemo_evaluator.adapters.interceptors import CachingInterceptor
from nemo_evaluator.adapters.types import (
    AdapterGlobalContext,
    AdapterRequest,
    AdapterRequestContext,
    AdapterResponse,
)
from requests.utils import CaseInsensitiveDict


@pytest.fixture
def mock_requests():
    with patch("requests.request") as mock_req:
        yield mock_req


@pytest.fixture
def create_response():
    """Fixture to create parameterized response"""

    def _create_response(status_code, content, headers):
        response = requests.Response()
        response.status_code = status_code
        response._content = content
        response.headers = CaseInsensitiveDict(headers)
        response.raw = MagicMock()
        response.raw.headers = headers
        return response

    return _create_response


@pytest.fixture
def mock_context():
    """Create a mock global context for testing."""
    return AdapterGlobalContext(
        output_dir="/tmp/test_output",
        url="http://localhost:3000",
    )


@pytest.mark.parametrize(
    "data1,data2",
    [
        (
            {"prompt": "test", "parameters": {"temp": 0.7}},
            {"parameters": {"temp": 0.7}, "prompt": "test"},
        ),
        (
            {
                "prompt": "test",
                "parameters": {"temp": 0.7, "options": {"a": 1, "b": 2}},
            },
            {
                "parameters": {"options": {"b": 2, "a": 1}, "temp": 0.7},
                "prompt": "test",
            },
        ),
    ],
)
def test_generate_cache_key(data1, data2):
    # Given: Two differently ordered but equivalent data structures

    # When: Cache keys are generated for both
    key1 = CachingInterceptor._generate_cache_key(data1)
    key2 = CachingInterceptor._generate_cache_key(data2)

    # Then: The cache keys should be identical
    assert key1 == key2


@pytest.mark.parametrize(
    "test_data,cached_content,cached_headers",
    [
        (
            {"prompt": "test prompt", "parameters": {"temperature": 0.7}},
            b'{"result": "cached result"}',
            {"Content-Type": "application/json"},
        ),
        (
            {
                "prompt": "complex prompt",
                "parameters": {"temperature": 0.9, "max_tokens": 100},
            },
            b'{"result": "complex cached result"}',
            {"Content-Type": "application/json", "X-Custom-Header": "test-value"},
        ),
    ],
)
def test_cache_hit(
    tmp_path, mock_requests, mock_context, test_data, cached_content, cached_headers
):
    interceptor = CachingInterceptor(
        params=CachingInterceptor.Params(
            cache_dir=str(tmp_path), reuse_cached_responses=True
        )
    )

    # Given: A cached response exists for a specific request
    cache_key = interceptor._generate_cache_key(test_data)
    interceptor.responses_cache[cache_key] = cached_content
    interceptor.headers_cache[cache_key] = cached_headers

    # When: A request is made with the same data
    request = Request.from_values(
        method="POST",
        headers={"Content-Type": "application/json"},
        data=json.dumps(test_data),
    )
    response = interceptor.intercept_request(
        req=AdapterRequest(r=request, rctx=AdapterRequestContext()),
        context=mock_context,
    )

    # Then: The cached response should be returned without making an API call
    assert isinstance(response, AdapterResponse)
    mock_requests.assert_not_called()
    assert response.r.status_code == 200
    assert response.rctx.cache_hit
    assert response.r.content == cached_content
    assert all(response.r.headers.get(k) == v for k, v in cached_headers.items())


@pytest.mark.parametrize(
    "test_data,expected_headers",
    [
        (
            {"prompt": "test prompt", "parameters": {"temperature": 0.7}},
            {"Content-Type": "application/json"},
        ),
        (
            {
                "prompt": "complex prompt",
                "parameters": {"temperature": 0.9, "max_tokens": 100},
            },
            {"Content-Type": "application/json", "X-Custom-Header": "test-value"},
        ),
    ],
)
def test_cache_miss_and_store(
    tmp_path, mock_requests, create_response, mock_context, test_data, expected_headers
):
    interceptor = CachingInterceptor(
        params=CachingInterceptor.Params(cache_dir=str(tmp_path))
    )

    # When: A request is made with that data
    request = Request.from_values(
        method="POST",
        headers={"Content-Type": "application/json"},
        data=json.dumps(test_data),
    )
    adapter_request = AdapterRequest(r=request, rctx=AdapterRequestContext())
    result = interceptor.intercept_request(req=adapter_request, context=mock_context)

    # Then: the interceptor does not intercept, but rather propagates as a request
    assert isinstance(result, AdapterRequest)
    assert result.rctx.cache_key

    # First, check that the error response is not cached
    error_response = AdapterResponse(
        r=create_response(404, b'{"error": "bad request"}', expected_headers),
        rctx=result.rctx,
    )
    interceptor.intercept_response(resp=error_response, context=mock_context)

    # We assert that the errored response did not result in caching
    assert result.rctx.cache_key not in interceptor.responses_cache
    assert result.rctx.cache_key not in interceptor.headers_cache

    # Now we model successful response
    success_response = AdapterResponse(
        r=create_response(200, b'{"result": "success"}', expected_headers),
        rctx=result.rctx,
    )
    interceptor.intercept_response(resp=success_response, context=mock_context)

    # Verify the successful response was cached
    assert (
        interceptor.responses_cache[result.rctx.cache_key] == b'{"result": "success"}'
    )
    assert interceptor.headers_cache[result.rctx.cache_key] == expected_headers


def test_request_caching(tmp_path, mock_context):
    interceptor = CachingInterceptor(
        params=CachingInterceptor.Params(
            cache_dir=str(tmp_path), save_requests=True, max_saved_requests=2
        )
    )

    test_data = {"prompt": "test prompt", "parameters": {"temperature": 0.7}}

    # Make first request
    request = Request.from_values(
        method="POST",
        headers={"Content-Type": "application/json"},
        data=json.dumps(test_data),
    )
    adapter_request = AdapterRequest(r=request, rctx=AdapterRequestContext())
    result = interceptor.intercept_request(req=adapter_request, context=mock_context)

    # Verify request was cached
    assert result.rctx.cache_key in interceptor.requests_cache
    assert interceptor.requests_cache[result.rctx.cache_key] == test_data

    # Make second request with different data
    test_data2 = {"prompt": "another prompt", "parameters": {"temperature": 0.8}}
    request2 = Request.from_values(
        method="POST",
        headers={"Content-Type": "application/json"},
        data=json.dumps(test_data2),
    )
    adapter_request2 = AdapterRequest(r=request2, rctx=AdapterRequestContext())
    result2 = interceptor.intercept_request(req=adapter_request2, context=mock_context)

    # Verify second request was cached
    assert result2.rctx.cache_key in interceptor.requests_cache
    assert interceptor.requests_cache[result2.rctx.cache_key] == test_data2

    # Make third request - should not be cached due to max_saved_requests=2
    test_data3 = {"prompt": "third prompt", "parameters": {"temperature": 0.9}}
    request3 = Request.from_values(
        method="POST",
        headers={"Content-Type": "application/json"},
        data=json.dumps(test_data3),
    )
    adapter_request3 = AdapterRequest(r=request3, rctx=AdapterRequestContext())
    result3 = interceptor.intercept_request(req=adapter_request3, context=mock_context)

    # Verify third request was not cached
    assert result3.rctx.cache_key not in interceptor.requests_cache


def test_response_caching_with_limit(tmp_path, create_response, mock_context):
    """Test that max_saved_responses limit works when reuse_cached_responses=False"""
    interceptor = CachingInterceptor(
        params=CachingInterceptor.Params(
            cache_dir=str(tmp_path),
            save_responses=True,
            max_saved_responses=2,
            reuse_cached_responses=False,  # Explicitly set to False to test limit
        )
    )

    test_data = {"prompt": "test prompt", "parameters": {"temperature": 0.7}}

    # Make request
    request = Request.from_values(
        method="POST",
        headers={"Content-Type": "application/json"},
        data=json.dumps(test_data),
    )
    adapter_request = AdapterRequest(r=request, rctx=AdapterRequestContext())
    result = interceptor.intercept_request(req=adapter_request, context=mock_context)

    # Create response
    response = create_response(
        200, b'{"result": "success"}', {"Content-Type": "application/json"}
    )
    adapter_response = AdapterResponse(r=response, rctx=result.rctx)

    # Intercept response
    interceptor.intercept_response(resp=adapter_response, context=mock_context)

    # Verify response was cached
    assert result.rctx.cache_key in interceptor.responses_cache
    assert (
        interceptor.responses_cache[result.rctx.cache_key] == b'{"result": "success"}'
    )

    # Make second request with different data
    test_data2 = {"prompt": "another prompt", "parameters": {"temperature": 0.8}}
    request2 = Request.from_values(
        method="POST",
        headers={"Content-Type": "application/json"},
        data=json.dumps(test_data2),
    )
    adapter_request2 = AdapterRequest(r=request2, rctx=AdapterRequestContext())
    result2 = interceptor.intercept_request(req=adapter_request2, context=mock_context)

    # Create second response
    response2 = create_response(
        200, b'{"result": "success2"}', {"Content-Type": "application/json"}
    )
    adapter_response2 = AdapterResponse(r=response2, rctx=result2.rctx)

    # Intercept second response
    interceptor.intercept_response(resp=adapter_response2, context=mock_context)

    # Verify second response was cached
    assert result2.rctx.cache_key in interceptor.responses_cache
    assert (
        interceptor.responses_cache[result2.rctx.cache_key] == b'{"result": "success2"}'
    )

    # Make third request - should NOT be cached due to max_saved_responses=2
    test_data3 = {"prompt": "third prompt", "parameters": {"temperature": 0.9}}
    request3 = Request.from_values(
        method="POST",
        headers={"Content-Type": "application/json"},
        data=json.dumps(test_data3),
    )
    adapter_request3 = AdapterRequest(r=request3, rctx=AdapterRequestContext())
    result3 = interceptor.intercept_request(req=adapter_request3, context=mock_context)

    # Create third response
    response3 = create_response(
        200, b'{"result": "success3"}', {"Content-Type": "application/json"}
    )
    adapter_response3 = AdapterResponse(r=response3, rctx=result3.rctx)

    # Intercept third response
    interceptor.intercept_response(resp=adapter_response3, context=mock_context)

    # Verify third response was NOT cached (limit should be enforced)
    assert result3.rctx.cache_key not in interceptor.responses_cache


def test_response_caching_with_reuse_enabled(tmp_path, create_response, mock_context):
    """Test that max_saved_responses limit is overridden when reuse_cached_responses=True"""
    interceptor = CachingInterceptor(
        params=CachingInterceptor.Params(
            cache_dir=str(tmp_path),
            save_responses=True,
            max_saved_responses=2,
            reuse_cached_responses=True,  # Explicitly set to True to override limit
        )
    )

    test_data = {"prompt": "test prompt", "parameters": {"temperature": 0.7}}

    # Make request
    request = Request.from_values(
        method="POST",
        headers={"Content-Type": "application/json"},
        data=json.dumps(test_data),
    )
    adapter_request = AdapterRequest(r=request, rctx=AdapterRequestContext())
    result = interceptor.intercept_request(req=adapter_request, context=mock_context)

    # Create response
    response = create_response(
        200, b'{"result": "success"}', {"Content-Type": "application/json"}
    )
    adapter_response = AdapterResponse(r=response, rctx=result.rctx)

    # Intercept response
    interceptor.intercept_response(resp=adapter_response, context=mock_context)

    # Verify response was cached
    assert result.rctx.cache_key in interceptor.responses_cache
    assert (
        interceptor.responses_cache[result.rctx.cache_key] == b'{"result": "success"}'
    )

    # Make second request with different data
    test_data2 = {"prompt": "another prompt", "parameters": {"temperature": 0.8}}
    request2 = Request.from_values(
        method="POST",
        headers={"Content-Type": "application/json"},
        data=json.dumps(test_data2),
    )
    adapter_request2 = AdapterRequest(r=request2, rctx=AdapterRequestContext())
    result2 = interceptor.intercept_request(req=adapter_request2, context=mock_context)

    # Create second response
    response2 = create_response(
        200, b'{"result": "success2"}', {"Content-Type": "application/json"}
    )
    adapter_response2 = AdapterResponse(r=response2, rctx=result2.rctx)

    # Intercept second response
    interceptor.intercept_response(resp=adapter_response2, context=mock_context)

    # Verify second response was cached
    assert result2.rctx.cache_key in interceptor.responses_cache
    assert (
        interceptor.responses_cache[result2.rctx.cache_key] == b'{"result": "success2"}'
    )

    # Make third request - SHOULD be cached since reuse_cached_responses=True overrides max_saved_responses
    test_data3 = {"prompt": "third prompt", "parameters": {"temperature": 0.9}}
    request3 = Request.from_values(
        method="POST",
        headers={"Content-Type": "application/json"},
        data=json.dumps(test_data3),
    )
    adapter_request3 = AdapterRequest(r=request3, rctx=AdapterRequestContext())
    result3 = interceptor.intercept_request(req=adapter_request3, context=mock_context)

    # Create third response
    response3 = create_response(
        200, b'{"result": "success3"}', {"Content-Type": "application/json"}
    )
    adapter_response3 = AdapterResponse(r=response3, rctx=result3.rctx)

    # Intercept third response
    interceptor.intercept_response(resp=adapter_response3, context=mock_context)

    # Verify third response WAS cached (limit should be overridden)
    assert result3.rctx.cache_key in interceptor.responses_cache
    assert (
        interceptor.responses_cache[result3.rctx.cache_key] == b'{"result": "success3"}'
    )
