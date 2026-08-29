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
import pytest

from nemo_evaluator.adapters.cache.disk_cache import DiskCache
from nemo_evaluator.adapters.interceptors.caching import Interceptor
from nemo_evaluator.adapters.types import AdapterRequest, AdapterResponse, InterceptorContext

_REQUEST_PATH = "/chat/completions"


def _compute_key(body):
    return DiskCache.cache_key(body, request_path=_REQUEST_PATH)


def test_golden_key_simple():
    body = {"model": "gpt-4", "messages": [{"role": "user", "content": "hello"}], "temperature": 0.7}
    key = _compute_key(body)
    assert key == _compute_key(body)
    assert key == "a645af96c78b89c2ff7ecacf9766427df863e5bcc7a6422a57c2d3d790244200"


def test_golden_key_with_tools():
    body = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [{"type": "function", "function": {"name": "get_weather"}}],
    }
    key = _compute_key(body)
    assert key == _compute_key(body)
    assert key == "53411c91ed09778aca705d15f5f85372812fe755a0e033dcae49676d9996a5be"


def test_key_changes_with_stream_mode():
    base = {"model": "gpt-4", "messages": [{"role": "user", "content": "hello"}]}
    with_stream = {**base, "stream": True}
    assert _compute_key(base) != _compute_key(with_stream)


def test_key_changes_with_temperature():
    body_a = {"model": "gpt-4", "messages": [{"role": "user", "content": "hello"}], "temperature": 0.7}
    body_b = {"model": "gpt-4", "messages": [{"role": "user", "content": "hello"}], "temperature": 0.9}
    assert _compute_key(body_a) != _compute_key(body_b)


@pytest.mark.parametrize(
    ("field", "value_a", "value_b"),
    [
        ("stop", ["END"], ["STOP"]),
        ("response_format", {"type": "json_object"}, {"type": "text"}),
        ("tool_choice", "auto", "none"),
        ("parallel_tool_calls", True, False),
        ("frequency_penalty", 0.0, 1.0),
        ("presence_penalty", 0.0, 1.0),
        ("logprobs", True, False),
    ],
)
def test_key_changes_with_response_affecting_fields(field, value_a, value_b):
    base = {"model": "gpt-4", "messages": [{"role": "user", "content": "hello"}]}
    assert _compute_key({**base, field: value_a}) != _compute_key({**base, field: value_b})


def test_key_changes_with_request_path():
    body = {"model": "gpt-4", "messages": [{"role": "user", "content": "hello"}]}
    chat_key = DiskCache.cache_key(body, request_path="/v1/chat/completions")
    responses_key = DiskCache.cache_key(body, request_path="/v1/responses")
    assert chat_key != responses_key


def test_key_is_stable_across_mapping_order():
    body_a = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "hello"}],
        "response_format": {"type": "json_schema", "json_schema": {"name": "answer"}},
    }
    body_b = {
        "response_format": {"json_schema": {"name": "answer"}, "type": "json_schema"},
        "messages": [{"content": "hello", "role": "user"}],
        "model": "gpt-4",
    }
    assert _compute_key(body_a) == _compute_key(body_b)


def test_key_differs_with_session_prefix():
    """Session prefix ensures repeats of the same problem never share cache entries."""
    body = {"model": "gpt-4", "messages": [{"role": "user", "content": "hello"}]}
    k_none = DiskCache.cache_key(body, request_path=_REQUEST_PATH)
    k_a = DiskCache.cache_key(body, request_path=_REQUEST_PATH, session_prefix="repeat-0")
    k_b = DiskCache.cache_key(body, request_path=_REQUEST_PATH, session_prefix="repeat-1")
    assert k_none != k_a
    assert k_a != k_b
    assert k_none == DiskCache.cache_key(body, request_path=_REQUEST_PATH, session_prefix="")


async def test_response_affecting_field_does_not_reuse_cached_response(tmp_path):
    interceptor = Interceptor(cache_dir=str(tmp_path))
    base = {"model": "test", "messages": [{"role": "user", "content": "cached"}]}
    json_request = AdapterRequest(
        method="POST",
        path="/chat/completions",
        headers={},
        body={**base, "response_format": {"type": "json_object"}},
        ctx=InterceptorContext(),
    )
    assert isinstance(await interceptor.intercept_request(json_request), AdapterRequest)
    await interceptor.intercept_response(
        AdapterResponse(
            status_code=200,
            headers={},
            body={"choices": [{"message": {"content": "{}"}}]},
            ctx=json_request.ctx,
        )
    )

    text_request = AdapterRequest(
        method="POST",
        path="/chat/completions",
        headers={},
        body={**base, "response_format": {"type": "text"}},
        ctx=InterceptorContext(),
    )
    assert isinstance(await interceptor.intercept_request(text_request), AdapterRequest)


async def test_stream_request_does_not_reuse_non_stream_response(tmp_path):
    interceptor = Interceptor(cache_dir=str(tmp_path))
    base = {"model": "test", "messages": [{"role": "user", "content": "cached"}]}
    non_stream_request = AdapterRequest(
        method="POST",
        path="/chat/completions",
        headers={},
        body={**base, "stream": False},
        ctx=InterceptorContext(),
    )
    assert isinstance(await interceptor.intercept_request(non_stream_request), AdapterRequest)
    await interceptor.intercept_response(
        AdapterResponse(
            status_code=200,
            headers={"content-type": "application/json"},
            body={"choices": [{"message": {"content": "cached response"}}]},
            ctx=non_stream_request.ctx,
        )
    )

    stream_request = AdapterRequest(
        method="POST",
        path="/chat/completions",
        headers={},
        body={**base, "stream": True},
        ctx=InterceptorContext(),
    )
    result = await interceptor.intercept_request(stream_request)

    assert isinstance(result, AdapterRequest)
    assert not stream_request.ctx.extra.get("cache_hit")
