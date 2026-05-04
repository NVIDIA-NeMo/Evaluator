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
"""Tests for interceptors not covered by test_interceptors.py."""

from __future__ import annotations


from nemo_evaluator.adapters.types import AdapterRequest, AdapterResponse, InterceptorContext


class TestEndpointInterceptorURLStripping:
    def _make(self, url, **kw):
        from nemo_evaluator.adapters.interceptors.endpoint import Interceptor

        return Interceptor(upstream_url=url, **kw)

    def test_strips_chat_completions(self):
        i = self._make("http://localhost:8000/v1/chat/completions")
        assert i._upstream_url == "http://localhost:8000/v1"

    def test_strips_completions(self):
        i = self._make("http://localhost:8000/v1/completions")
        assert i._upstream_url == "http://localhost:8000/v1"

    def test_strips_embeddings(self):
        i = self._make("http://localhost:8000/v1/embeddings")
        assert i._upstream_url == "http://localhost:8000/v1"

    def test_no_strip_when_no_suffix(self):
        i = self._make("http://localhost:8000/v1")
        assert i._upstream_url == "http://localhost:8000/v1"

    def test_trailing_slash_stripped(self):
        i = self._make("http://localhost:8000/v1/")
        assert i._upstream_url == "http://localhost:8000/v1"

    def test_full_api_url(self):
        i = self._make("https://integrate.api.nvidia.com/v1/chat/completions")
        assert i._upstream_url == "https://integrate.api.nvidia.com/v1"

    def test_bare_url_unchanged(self):
        i = self._make("http://vllm:8000")
        assert i._upstream_url == "http://vllm:8000"

    def test_api_key_stored(self):
        i = self._make("http://x", api_key="sk-123")
        assert i._api_key == "sk-123"

    def test_extra_body_merged(self):
        i = self._make("http://x", extra_body={"skip_special_tokens": False})
        assert i._extra_body == {"skip_special_tokens": False}

    def test_retry_on_status_default(self):
        i = self._make("http://x")
        assert 429 in i._retry_on_status
        assert 503 in i._retry_on_status


class TestCachingInterceptor:
    async def test_cache_miss_passes_through(self, tmp_path):
        from nemo_evaluator.adapters.interceptors.caching import Interceptor

        i = Interceptor(cache_dir=str(tmp_path))
        ctx = InterceptorContext()
        req = AdapterRequest(
            method="POST",
            path="/chat/completions",
            headers={},
            body={"messages": [{"role": "user", "content": "hi"}]},
            ctx=ctx,
        )
        result = await i.intercept_request(req)
        assert isinstance(result, AdapterRequest)

    async def test_bypass_mode(self, tmp_path):
        from nemo_evaluator.adapters.interceptors.caching import Interceptor

        i = Interceptor(cache_dir=str(tmp_path), bypass=True)
        ctx = InterceptorContext()
        req = AdapterRequest(
            method="POST",
            path="/v1/chat/completions",
            headers={},
            body={"messages": []},
            ctx=ctx,
        )
        result = await i.intercept_request(req)
        assert isinstance(result, AdapterRequest)

    async def test_cache_hit_returns_response(self, tmp_path):
        from nemo_evaluator.adapters.interceptors.caching import Interceptor

        i = Interceptor(cache_dir=str(tmp_path))
        ctx = InterceptorContext()
        body = {"messages": [{"role": "user", "content": "cached"}]}
        req = AdapterRequest(method="POST", path="/chat/completions", headers={}, body=body, ctx=ctx)

        result1 = await i.intercept_request(req)
        assert isinstance(result1, AdapterRequest)

        resp = AdapterResponse(
            status_code=200,
            headers={},
            body={"choices": [{"message": {"content": "answer"}}]},
            ctx=ctx,
        )
        await i.intercept_response(resp)

        req2 = AdapterRequest(method="POST", path="/chat/completions", headers={}, body=body, ctx=InterceptorContext())
        result2 = await i.intercept_request(req2)
        assert isinstance(result2, AdapterResponse)
        assert result2.status_code == 200

    async def test_cache_isolated_by_session(self, tmp_path):
        """Repeats with different session IDs never share cache entries."""
        from nemo_evaluator.adapters.interceptors.caching import Interceptor

        i = Interceptor(cache_dir=str(tmp_path))
        body = {"messages": [{"role": "user", "content": "hello"}]}

        ctx_a = InterceptorContext()
        ctx_a.extra["session_id"] = "session_aaa"
        req_a = AdapterRequest(method="POST", path="/chat/completions", headers={}, body=body, ctx=ctx_a)
        result_a = await i.intercept_request(req_a)
        assert isinstance(result_a, AdapterRequest)

        resp_a = AdapterResponse(
            status_code=200, headers={}, body={"choices": [{"message": {"content": "answer-a"}}]}, ctx=ctx_a
        )
        await i.intercept_response(resp_a)

        ctx_b = InterceptorContext()
        ctx_b.extra["session_id"] = "session_bbb"
        req_b = AdapterRequest(method="POST", path="/chat/completions", headers={}, body=body, ctx=ctx_b)
        result_b = await i.intercept_request(req_b)
        assert isinstance(result_b, AdapterRequest), "must be a cache miss, not a hit"
