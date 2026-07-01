# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the finish_reason reconciliation interceptor."""

from __future__ import annotations

import logging
from typing import Any

from nemo_evaluator.adapters.interceptors.finish_reason import Interceptor
from nemo_evaluator.adapters.types import AdapterRequest, AdapterResponse, InterceptorContext


def _make_pair(
    request_body: dict[str, Any],
    response_body: dict[str, Any],
) -> tuple[AdapterRequest, AdapterResponse]:
    ctx = InterceptorContext()
    req = AdapterRequest(method="POST", path="/chat/completions", headers={}, body=request_body, ctx=ctx)
    resp = AdapterResponse(status_code=200, headers={}, body=response_body, ctx=ctx)
    return req, resp


def _chat_response(
    *,
    finish_reason: str,
    content: str,
    completion_tokens: int | None = None,
    reasoning_tokens: int | None = None,
) -> dict[str, Any]:
    usage = {}
    if completion_tokens is not None:
        usage["completion_tokens"] = completion_tokens
    if reasoning_tokens is not None:
        usage["completion_tokens_details"] = {"reasoning_tokens": reasoning_tokens}
    return {
        "choices": [{"finish_reason": finish_reason, "message": {"content": content}}],
        "usage": usage,
    }


class TestFinishReasonRelabel:
    async def test_relabels_stop_to_length_on_cap_hit(self):
        i = Interceptor()
        req, resp = _make_pair(
            {"max_tokens": 64000},
            _chat_response(finish_reason="stop", content="", completion_tokens=64000),
        )
        await i.intercept_request(req)
        out = await i.intercept_response(resp)
        assert out.body["choices"][0]["finish_reason"] == "length"

    async def test_honors_max_completion_tokens_key(self):
        i = Interceptor()
        req, resp = _make_pair(
            {"max_completion_tokens": 100},
            _chat_response(finish_reason="stop", content="", completion_tokens=100),
        )
        await i.intercept_request(req)
        out = await i.intercept_response(resp)
        assert out.body["choices"][0]["finish_reason"] == "length"

    async def test_detects_truncation_via_reasoning_tokens(self):
        i = Interceptor()
        req, resp = _make_pair(
            {"max_tokens": 100},
            _chat_response(finish_reason="stop", content="", completion_tokens=0, reasoning_tokens=100),
        )
        await i.intercept_request(req)
        out = await i.intercept_response(resp)
        assert out.body["choices"][0]["finish_reason"] == "length"

    async def test_does_not_relabel_multi_choice_from_aggregate_usage(self):
        i = Interceptor()
        req, resp = _make_pair(
            {"max_tokens": 100},
            {
                "choices": [
                    {"finish_reason": "stop", "message": {"content": "first"}},
                    {"finish_reason": "stop", "message": {"content": "second"}},
                ],
                "usage": {"completion_tokens": 100},
            },
        )
        await i.intercept_request(req)
        out = await i.intercept_response(resp)
        assert [choice["finish_reason"] for choice in out.body["choices"]] == ["stop", "stop"]

    async def test_relabels_multi_choice_using_choice_usage(self):
        i = Interceptor()
        req, resp = _make_pair(
            {"max_tokens": 100},
            {
                "choices": [
                    {"finish_reason": "stop", "message": {"content": ""}, "usage": {"completion_tokens": 100}},
                    {"finish_reason": "stop", "message": {"content": "second"}, "usage": {"completion_tokens": 50}},
                ],
                "usage": {"completion_tokens": 150},
            },
        )
        await i.intercept_request(req)
        out = await i.intercept_response(resp)
        assert [choice["finish_reason"] for choice in out.body["choices"]] == ["length", "stop"]

    async def test_keeps_stop_when_under_cap(self):
        i = Interceptor()
        req, resp = _make_pair(
            {"max_tokens": 100},
            _chat_response(finish_reason="stop", content="42", completion_tokens=50),
        )
        await i.intercept_request(req)
        out = await i.intercept_response(resp)
        assert out.body["choices"][0]["finish_reason"] == "stop"

    async def test_no_relabel_without_request_cap(self):
        i = Interceptor()
        req, resp = _make_pair(
            {},
            _chat_response(finish_reason="stop", content="", completion_tokens=64000),
        )
        await i.intercept_request(req)
        out = await i.intercept_response(resp)
        assert out.body["choices"][0]["finish_reason"] == "stop"

    async def test_does_not_touch_non_stop_finish_reason(self):
        i = Interceptor()
        req, resp = _make_pair(
            {"max_tokens": 100},
            _chat_response(finish_reason="content_filter", content="", completion_tokens=100),
        )
        await i.intercept_request(req)
        out = await i.intercept_response(resp)
        assert out.body["choices"][0]["finish_reason"] == "content_filter"

    async def test_ignores_non_dict_body(self):
        i = Interceptor()
        ctx = InterceptorContext()
        resp = AdapterResponse(status_code=200, headers={}, body=b"raw-bytes", ctx=ctx)
        out = await i.intercept_response(resp)
        assert out.body == b"raw-bytes"


class TestEmptyGenerationWarning:
    async def test_warns_on_empty_generation(self, caplog):
        caplog.set_level(logging.WARNING, logger="nemo_evaluator.adapters.interceptors.finish_reason")
        i = Interceptor()
        req, resp = _make_pair(
            {"max_tokens": 100},
            _chat_response(finish_reason="stop", content="   ", completion_tokens=100),
        )
        await i.intercept_request(req)
        await i.intercept_response(resp)
        assert any("empty generation" in r.getMessage() for r in caplog.records)

    async def test_no_empty_warning_with_content(self, caplog):
        caplog.set_level(logging.WARNING, logger="nemo_evaluator.adapters.interceptors.finish_reason")
        i = Interceptor()
        req, resp = _make_pair(
            {"max_tokens": 100},
            _chat_response(finish_reason="stop", content="real answer", completion_tokens=10),
        )
        await i.intercept_request(req)
        await i.intercept_response(resp)
        assert not any("empty generation" in r.getMessage() for r in caplog.records)

    async def test_tool_calls_are_not_empty(self, caplog):
        caplog.set_level(logging.WARNING, logger="nemo_evaluator.adapters.interceptors.finish_reason")
        i = Interceptor()
        resp_body = {
            "choices": [{"finish_reason": "tool_calls", "message": {"content": "", "tool_calls": [{"id": "c1"}]}}],
            "usage": {"completion_tokens": 10},
        }
        req, resp = _make_pair({"max_tokens": 100}, resp_body)
        await i.intercept_request(req)
        await i.intercept_response(resp)
        assert not any("empty generation" in r.getMessage() for r in caplog.records)

    async def test_warns_on_empty_structured_content(self, caplog):
        caplog.set_level(logging.WARNING, logger="nemo_evaluator.adapters.interceptors.finish_reason")
        i = Interceptor()
        resp_body = {
            "choices": [{"finish_reason": "stop", "message": {"content": [{"type": "text", "text": "   "}]}}],
            "usage": {"completion_tokens": 10},
        }
        req, resp = _make_pair({"max_tokens": 100}, resp_body)
        await i.intercept_request(req)
        await i.intercept_response(resp)
        assert any("empty generation" in r.getMessage() for r in caplog.records)

    async def test_no_warning_with_structured_content_text(self, caplog):
        caplog.set_level(logging.WARNING, logger="nemo_evaluator.adapters.interceptors.finish_reason")
        i = Interceptor()
        resp_body = {
            "choices": [{"finish_reason": "stop", "message": {"content": [{"type": "text", "text": "real answer"}]}}],
            "usage": {"completion_tokens": 10},
        }
        req, resp = _make_pair({"max_tokens": 100}, resp_body)
        await i.intercept_request(req)
        await i.intercept_response(resp)
        assert not any("empty generation" in r.getMessage() for r in caplog.records)
