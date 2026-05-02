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

    def test_extra_headers_stored(self):
        i = self._make(
            "http://x",
            extra_headers={
                "X-NMP-Principal-Id": "service:evaluator",
                "X-Inference-Priority": "batch",
            },
        )
        assert i._extra_headers == {
            "X-NMP-Principal-Id": "service:evaluator",
            "X-Inference-Priority": "batch",
        }

    def test_extra_headers_drops_hop_by_hop(self, caplog):
        import logging

        caplog.set_level(logging.WARNING)
        i = self._make(
            "http://x",
            extra_headers={
                "Host": "evil.example.com",
                "Content-Length": "9999",
                "Connection": "close",
                "Transfer-Encoding": "chunked",
                "X-Inference-Priority": "batch",
            },
        )
        assert i._extra_headers == {"X-Inference-Priority": "batch"}
        assert any("hop-by-hop" in r.message.lower() for r in caplog.records)

    def test_retry_on_status_default(self):
        i = self._make("http://x")
        assert 429 in i._retry_on_status
        assert 503 in i._retry_on_status


class TestEndpointInterceptorExceptionLogging:
    """Verify exception class + message are logged on every error path.

    Without these fields, debugging an upstream disconnect requires
    cross-correlating multiple logs to guess which aiohttp.ClientError
    subclass fired (ServerDisconnectedError, ClientPayloadError, etc).
    """

    def _make(self, **kw):
        from nemo_evaluator.adapters.interceptors.endpoint import Interceptor

        return Interceptor(upstream_url="http://upstream:8000/v1", **kw)

    def _make_req(self):
        return AdapterRequest(
            method="POST",
            path="/chat/completions",
            headers={},
            body={"messages": [{"role": "user", "content": "hi"}]},
            ctx=InterceptorContext(),
        )

    class _RaisingSession:
        """Minimal stand-in for aiohttp.ClientSession that raises on POST.

        On each POST call, pops the next exception from `to_raise` and raises
        it. Used to drive the interceptor's error-path branches.
        """

        def __init__(self, to_raise):
            self._to_raise = list(to_raise)
            self.closed = False

        def post(self, *args, **kwargs):
            exc = self._to_raise.pop(0)
            raise exc

        async def close(self):
            self.closed = True

    async def _run(self, interceptor, exceptions):
        """Stub the interceptor's session and run intercept_request once."""

        async def _stubbed_get_session():
            return self._RaisingSession(exceptions)

        interceptor._get_session = _stubbed_get_session  # type: ignore[assignment]
        return await interceptor.intercept_request(self._make_req())

    async def test_client_error_final_failure_logs_exc_class_and_message(self, caplog):
        """When max_retries=0 and an aiohttp.ClientError fires, the exception
        bubbles up — but a log line MUST identify the exception class and
        message before re-raising. Otherwise the proxy log shows nothing.
        """
        import logging

        import aiohttp
        import pytest

        caplog.set_level(logging.ERROR, logger="nemo_evaluator.adapters.interceptors.endpoint")

        ic = self._make(max_retries=0)
        with pytest.raises(aiohttp.ServerDisconnectedError):
            await self._run(ic, [aiohttp.ServerDisconnectedError("upstream closed")])

        records = [r for r in caplog.records if "endpoint" in r.message.lower()]
        assert records, "expected at least one endpoint error log line on final ClientError failure"
        msg = " ".join(r.getMessage() for r in records)
        assert "ServerDisconnectedError" in msg, f"exc class missing from log; got: {msg!r}"
        assert "upstream closed" in msg, f"exc message missing from log; got: {msg!r}"

    async def test_client_error_retry_log_includes_exc_class(self, caplog):
        """On a retry-then-succeed path, the retry-warning log must include
        the exception class so we can categorize the error (e.g.
        ClientPayloadError vs ServerDisconnectedError).
        """
        import logging

        import aiohttp

        caplog.set_level(logging.WARNING, logger="nemo_evaluator.adapters.interceptors.endpoint")

        ic = self._make(max_retries=1)
        # Force the second attempt to raise too — the test only inspects the
        # retry-warning log, not the final outcome.
        exceptions = [
            aiohttp.ClientPayloadError("malformed chunk"),
            aiohttp.ClientPayloadError("malformed chunk"),
        ]
        try:
            await self._run(ic, exceptions)
        except aiohttp.ClientError:
            pass

        retry_records = [r for r in caplog.records if r.levelno == logging.WARNING and "retry" in r.message.lower()]
        assert retry_records, "expected at least one retry-warning log line"
        msg = " ".join(r.getMessage() for r in retry_records)
        assert "ClientPayloadError" in msg, f"exc class missing from retry log; got: {msg!r}"


class TestProxyConfigExtraHeaders:
    def test_needs_proxy_with_extra_headers(self):
        from nemo_evaluator.config.services import ProxyConfig

        cfg = ProxyConfig(extra_headers={"X-Inference-Priority": "batch"})
        assert cfg.needs_proxy is True

    def test_needs_proxy_default_empty(self):
        from nemo_evaluator.config.services import ProxyConfig

        assert ProxyConfig().needs_proxy is False

    def test_extra_headers_default_factory(self):
        from nemo_evaluator.config.services import ProxyConfig

        # Distinct instances must not share the same dict.
        a = ProxyConfig()
        b = ProxyConfig()
        a.extra_headers["X-Test"] = "1"
        assert "X-Test" not in b.extra_headers


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
