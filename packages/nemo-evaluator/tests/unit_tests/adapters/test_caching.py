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
import pickle
from unittest.mock import MagicMock, patch

import pytest
import requests
from flask import Request
from requests.utils import CaseInsensitiveDict

# Import to register interceptors
from nemo_evaluator.adapters.interceptors import CachingInterceptor
from nemo_evaluator.adapters.types import (
    AdapterGlobalContext,
    AdapterRequest,
    AdapterRequestContext,
    AdapterResponse,
)


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
    
    # Initialize caches
    interceptor.pre_eval_hook(mock_context)

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
    
    # Initialize caches
    interceptor.pre_eval_hook(mock_context)

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
    
    # Initialize caches
    interceptor.pre_eval_hook(mock_context)

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
    
    # Initialize caches
    interceptor.pre_eval_hook(mock_context)

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
    
    # Initialize caches
    interceptor.pre_eval_hook(mock_context)

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


def test_export_cache_to_binary(tmp_path, create_response, mock_context):
    """Test exporting cache to binary .cache file"""
    # Create interceptor with export enabled
    interceptor = CachingInterceptor(
        params=CachingInterceptor.Params(
            cache_dir=str(tmp_path / "cache"),
            save_requests=True,
            save_responses=True,
            export_cache=True,
        )
    )
    
    # Initialize caches
    interceptor.pre_eval_hook(mock_context)
    
    # Create and cache some test data
    test_data1 = {"prompt": "test prompt 1", "parameters": {"temperature": 0.7}}
    test_data2 = {"prompt": "test prompt 2", "parameters": {"temperature": 0.8}}
    
    # First request/response
    request1 = Request.from_values(
        method="POST",
        headers={"Content-Type": "application/json"},
        data=json.dumps(test_data1),
    )
    adapter_request1 = AdapterRequest(r=request1, rctx=AdapterRequestContext())
    result1 = interceptor.intercept_request(req=adapter_request1, context=mock_context)
    
    response1 = create_response(
        200, b'{"result": "success1"}', {"Content-Type": "application/json"}
    )
    adapter_response1 = AdapterResponse(r=response1, rctx=result1.rctx)
    interceptor.intercept_response(resp=adapter_response1, context=mock_context)
    
    # Second request/response
    request2 = Request.from_values(
        method="POST",
        headers={"Content-Type": "application/json"},
        data=json.dumps(test_data2),
    )
    adapter_request2 = AdapterRequest(r=request2, rctx=AdapterRequestContext())
    result2 = interceptor.intercept_request(req=adapter_request2, context=mock_context)
    
    response2 = create_response(
        200, b'{"result": "success2"}', {"Content-Type": "application/json"}
    )
    adapter_response2 = AdapterResponse(r=response2, rctx=result2.rctx)
    interceptor.intercept_response(resp=adapter_response2, context=mock_context)
    
    # Export cache
    interceptor.post_eval_hook(mock_context)
    
    # Verify .cache file was created in cache directory
    cache_file = tmp_path / "cache" / "cache_export.cache"
    assert cache_file.exists()
    
    # Load and verify binary content
    with open(cache_file, "rb") as f:
        cache_data = pickle.load(f)
    
    assert "requests" in cache_data
    assert "responses" in cache_data
    assert "headers" in cache_data
    
    # Verify we have the expected number of entries
    assert len(cache_data["requests"]) == 2
    assert len(cache_data["responses"]) == 2
    assert len(cache_data["headers"]) == 2
    
    # Verify binary data is preserved natively
    for key, value in cache_data["responses"].items():
        assert isinstance(value, bytes)


def test_import_cache_from_binary(tmp_path, mock_context):
    """Test importing cache from binary .cache file"""
    # Create a cache binary file
    cache_data = {
        "requests": {
            "key1": {"prompt": "test1", "parameters": {"temperature": 0.7}},
            "key2": {"prompt": "test2", "parameters": {"temperature": 0.8}},
        },
        "responses": {
            "key1": b'{"result": "success1"}',
            "key2": b'{"result": "success2"}',
        },
        "headers": {
            "key1": {"Content-Type": "application/json"},
            "key2": {"Content-Type": "application/json", "X-Custom": "value"},
        },
    }
    
    cache_file = tmp_path / "cache_input.cache"
    with open(cache_file, "wb") as f:
        pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Create interceptor with prefill enabled
    interceptor = CachingInterceptor(
        params=CachingInterceptor.Params(
            cache_dir=str(tmp_path / "cache"),
            prefill_from_export=str(cache_file),
            reuse_cached_responses=True,
        )
    )
    
    # Initialize caches (this should load from binary file)
    interceptor.pre_eval_hook(mock_context)
    
    # Verify caches were populated
    assert interceptor._cached_requests_count == 2
    assert interceptor._cached_responses_count == 2
    
    # Verify we can retrieve the cached data
    assert "key1" in interceptor.requests_cache
    assert "key2" in interceptor.requests_cache
    assert interceptor.requests_cache["key1"] == {"prompt": "test1", "parameters": {"temperature": 0.7}}
    
    # Verify binary responses were loaded correctly
    assert "key1" in interceptor.responses_cache
    assert interceptor.responses_cache["key1"] == b'{"result": "success1"}'
    
    # Verify headers
    assert "key1" in interceptor.headers_cache
    assert interceptor.headers_cache["key1"] == {"Content-Type": "application/json"}


def test_export_and_import_roundtrip(tmp_path, create_response, mock_context):
    """Test that exporting and importing cache preserves data"""
    # Create first interceptor and populate it
    interceptor1 = CachingInterceptor(
        params=CachingInterceptor.Params(
            cache_dir=str(tmp_path / "cache1"),
            save_requests=True,
            save_responses=True,
            export_cache=True,
        )
    )
    
    interceptor1.pre_eval_hook(mock_context)
    
    # Add test data with binary content
    test_data = {"prompt": "roundtrip test", "parameters": {"temperature": 0.9}}
    request = Request.from_values(
        method="POST",
        headers={"Content-Type": "application/json"},
        data=json.dumps(test_data),
    )
    adapter_request = AdapterRequest(r=request, rctx=AdapterRequestContext())
    result = interceptor1.intercept_request(req=adapter_request, context=mock_context)
    
    response = create_response(
        200, b'{"result": "roundtrip success"}', {"Content-Type": "application/json", "X-Test": "value"}
    )
    adapter_response = AdapterResponse(r=response, rctx=result.rctx)
    interceptor1.intercept_response(resp=adapter_response, context=mock_context)
    
    # Export cache
    interceptor1.post_eval_hook(mock_context)
    
    # Cache file should be in the cache directory
    cache_file = tmp_path / "cache1" / "cache_export.cache"
    assert cache_file.exists()
    
    # Create second interceptor and import the exported cache
    interceptor2 = CachingInterceptor(
        params=CachingInterceptor.Params(
            cache_dir=str(tmp_path / "cache2"),
            prefill_from_export=str(cache_file),
            reuse_cached_responses=True,
        )
    )
    
    interceptor2.pre_eval_hook(mock_context)
    
    # Verify the data is identical
    cache_key = result.rctx.cache_key
    
    assert cache_key in interceptor2.requests_cache
    assert interceptor2.requests_cache[cache_key] == test_data
    
    assert cache_key in interceptor2.responses_cache
    assert interceptor2.responses_cache[cache_key] == b'{"result": "roundtrip success"}'
    
    assert cache_key in interceptor2.headers_cache
    assert interceptor2.headers_cache[cache_key] == {"Content-Type": "application/json", "X-Test": "value"}


def test_prefill_from_nonexistent_cache_raises_error(tmp_path, mock_context):
    """Test that providing a nonexistent cache file raises an error"""
    interceptor = CachingInterceptor(
        params=CachingInterceptor.Params(
            cache_dir=str(tmp_path / "cache"),
            prefill_from_export=str(tmp_path / "nonexistent.cache"),
        )
    )
    
    # Should raise FileNotFoundError
    with pytest.raises(FileNotFoundError):
        interceptor.pre_eval_hook(mock_context)


def test_export_without_flag_does_not_create_file(tmp_path, create_response, mock_context):
    """Test that export_cache=False doesn't create a file"""
    interceptor = CachingInterceptor(
        params=CachingInterceptor.Params(
            cache_dir=str(tmp_path / "cache"),
            save_responses=True,
            export_cache=False,  # Explicitly disabled
        )
    )
    
    interceptor.pre_eval_hook(mock_context)
    
    # Add some data
    test_data = {"prompt": "test"}
    request = Request.from_values(
        method="POST",
        headers={"Content-Type": "application/json"},
        data=json.dumps(test_data),
    )
    adapter_request = AdapterRequest(r=request, rctx=AdapterRequestContext())
    result = interceptor.intercept_request(req=adapter_request, context=mock_context)
    
    response = create_response(200, b'{"result": "success"}', {"Content-Type": "application/json"})
    adapter_response = AdapterResponse(r=response, rctx=result.rctx)
    interceptor.intercept_response(resp=adapter_response, context=mock_context)
    
    # Call post_eval_hook
    interceptor.post_eval_hook(mock_context)
    
    # Verify no cache file was created in cache directory
    cache_file = tmp_path / "cache" / "cache_export.cache"
    assert not cache_file.exists()


def test_test_mode_cache_miss_with_similar_request(tmp_path, mock_context):
    """Test that test_mode raises error on cache miss and shows diff with similar request"""
    from nemo_evaluator.adapters.interceptors.caching_interceptor import CacheMissInTestModeError
    
    # Generate the correct cache key for the cached data
    cached_data = {"prompt": "Hello world", "parameters": {"temperature": 0.7}}
    cache_key = CachingInterceptor._generate_cache_key(cached_data)
    
    # Create cache with the correctly generated key
    cache_data = {
        "requests": {
            cache_key: cached_data,
        },
        "responses": {
            cache_key: b'{"result": "success1"}',
        },
        "headers": {
            cache_key: {"Content-Type": "application/json"},
        },
    }
    
    cache_file = tmp_path / "cache_input.cache"
    with open(cache_file, "wb") as f:
        pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Create interceptor with test_mode enabled
    interceptor = CachingInterceptor(
        params=CachingInterceptor.Params(
            cache_dir=str(tmp_path / "cache"),
            prefill_from_export=str(cache_file),
            reuse_cached_responses=True,
            test_mode=True,  # Enable test mode
        )
    )
    
    interceptor.pre_eval_hook(mock_context)
    
    # Make a similar but different request
    test_data = {"prompt": "Hello world!", "parameters": {"temperature": 0.8}}  # Note: exclamation mark and different temp
    request = Request.from_values(
        method="POST",
        headers={"Content-Type": "application/json"},
        data=json.dumps(test_data),
    )
    adapter_request = AdapterRequest(r=request, rctx=AdapterRequestContext())
    
    # Should raise CacheMissInTestModeError
    with pytest.raises(CacheMissInTestModeError) as exc_info:
        interceptor.intercept_request(req=adapter_request, context=mock_context)
    
    # Verify exception details
    error = exc_info.value
    assert error.request_data == test_data
    assert error.most_similar_request is not None
    assert error.similarity_score >= 98.0
    assert error.diff is not None and len(error.diff) > 0
    assert "temperature" in error.diff or "prompt" in error.diff


def test_test_mode_cache_miss_without_similar_request(tmp_path, mock_context):
    """Test that test_mode raises error on cache miss without similar request"""
    from nemo_evaluator.adapters.interceptors.caching_interceptor import CacheMissInTestModeError
    
    # Create interceptor with test_mode enabled but no cached data
    interceptor = CachingInterceptor(
        params=CachingInterceptor.Params(
            cache_dir=str(tmp_path / "cache"),
            reuse_cached_responses=True,
            test_mode=True,  # Enable test mode
        )
    )
    
    interceptor.pre_eval_hook(mock_context)
    
    # Make a request
    test_data = {"prompt": "Hello world", "parameters": {"temperature": 0.7}}
    request = Request.from_values(
        method="POST",
        headers={"Content-Type": "application/json"},
        data=json.dumps(test_data),
    )
    adapter_request = AdapterRequest(r=request, rctx=AdapterRequestContext())
    
    # Should raise CacheMissInTestModeError
    with pytest.raises(CacheMissInTestModeError) as exc_info:
        interceptor.intercept_request(req=adapter_request, context=mock_context)
    
    # Verify exception details
    error = exc_info.value
    assert error.request_data == test_data
    assert error.most_similar_request is None
    assert error.similarity_score is None
    assert error.diff is None
    assert "No similar cached requests found" in str(error)


def test_test_mode_cache_hit_works_normally(tmp_path, mock_context):
    """Test that test_mode works normally when there's a cache hit"""
    from nemo_evaluator.adapters.interceptors.caching_interceptor import CacheMissInTestModeError
    
    # Generate the correct cache key for the test data
    test_data = {"prompt": "Hello world", "parameters": {"temperature": 0.7}}
    cache_key = CachingInterceptor._generate_cache_key(test_data)
    
    # Create cache with the correctly generated key
    cache_data = {
        "requests": {
            cache_key: test_data,
        },
        "responses": {
            cache_key: b'{"result": "success1"}',
        },
        "headers": {
            cache_key: {"Content-Type": "application/json"},
        },
    }
    
    cache_file = tmp_path / "cache_input.cache"
    with open(cache_file, "wb") as f:
        pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Create interceptor with test_mode enabled
    interceptor = CachingInterceptor(
        params=CachingInterceptor.Params(
            cache_dir=str(tmp_path / "cache"),
            prefill_from_export=str(cache_file),
            reuse_cached_responses=True,
            save_requests=True,  # Enable request saving so we can check cache key
            test_mode=True,  # Enable test mode
        )
    )
    
    interceptor.pre_eval_hook(mock_context)
    
    # Make the exact same request that's in cache
    request = Request.from_values(
        method="POST",
        headers={"Content-Type": "application/json"},
        data=json.dumps(test_data),
    )
    adapter_request = AdapterRequest(r=request, rctx=AdapterRequestContext())
    
    # Should NOT raise error - should return cached response
    result = interceptor.intercept_request(req=adapter_request, context=mock_context)
    
    # Verify it returned a response (cache hit)
    assert isinstance(result, AdapterResponse)
    assert result.rctx.cache_hit is True
    assert result.r.status_code == 200


def test_test_mode_disabled_allows_cache_miss(tmp_path, mock_context):
    """Test that disabling test_mode allows cache misses without error"""
    # Generate the correct cache key for the cached data
    cached_data = {"prompt": "Hello world", "parameters": {"temperature": 0.7}}
    cache_key = CachingInterceptor._generate_cache_key(cached_data)
    
    # Create cache with the correctly generated key
    cache_data = {
        "requests": {
            cache_key: cached_data,
        },
        "responses": {
            cache_key: b'{"result": "success1"}',
        },
        "headers": {
            cache_key: {"Content-Type": "application/json"},
        },
    }
    
    cache_file = tmp_path / "cache_input.cache"
    with open(cache_file, "wb") as f:
        pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Create interceptor with test_mode DISABLED
    interceptor = CachingInterceptor(
        params=CachingInterceptor.Params(
            cache_dir=str(tmp_path / "cache"),
            prefill_from_export=str(cache_file),
            reuse_cached_responses=True,
            test_mode=False,  # Disable test mode
        )
    )
    
    interceptor.pre_eval_hook(mock_context)
    
    # Make a different request (will be cache miss)
    test_data = {"prompt": "Different prompt", "parameters": {"temperature": 0.9}}
    request = Request.from_values(
        method="POST",
        headers={"Content-Type": "application/json"},
        data=json.dumps(test_data),
    )
    adapter_request = AdapterRequest(r=request, rctx=AdapterRequestContext())
    
    # Should NOT raise error - should return request to be processed
    result = interceptor.intercept_request(req=adapter_request, context=mock_context)
    
    # Verify it returned a request (cache miss but no error)
    assert isinstance(result, AdapterRequest)
    assert result.rctx.cache_key is not None
