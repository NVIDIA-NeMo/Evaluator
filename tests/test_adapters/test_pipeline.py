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

from nemo_evaluator.adapters.pipeline import AdapterPipeline
from nemo_evaluator.adapters.types import (
    AdapterRequest,
    AdapterResponse,
    InterceptorContext,
    RequestInterceptor,
    RequestToResponseInterceptor,
    ResponseInterceptor,
)


def _req(**body_overrides):
    body = {"model": "test", "messages": [{"role": "user", "content": "hi"}]}
    body.update(body_overrides)
    return AdapterRequest(
        method="POST",
        path="/v1/chat/completions",
        headers={"content-type": "application/json"},
        body=body,
        ctx=InterceptorContext(),
    )


class AddHeaderInterceptor(RequestInterceptor):
    async def intercept_request(self, req: AdapterRequest) -> AdapterRequest:
        req.headers["x-test"] = "1"
        return req


class FixedEndpoint(RequestToResponseInterceptor):
    def __init__(self):
        self.last_request = None

    async def intercept_request(self, req: AdapterRequest) -> AdapterRequest | AdapterResponse:
        self.last_request = req
        return AdapterResponse(
            status_code=200,
            headers={},
            body={"result": "ok"},
            ctx=req.ctx,
        )


class AppendBodyInterceptor(ResponseInterceptor):
    async def intercept_response(self, resp: AdapterResponse) -> AdapterResponse:
        if isinstance(resp.body, dict):
            resp.body["appended"] = True
        return resp


class TrackingResponseInterceptor(ResponseInterceptor):
    def __init__(self):
        self.called = False

    async def intercept_response(self, resp: AdapterResponse) -> AdapterResponse:
        self.called = True
        return resp


class CacheHitInterceptor(RequestToResponseInterceptor):
    async def intercept_request(self, req: AdapterRequest) -> AdapterRequest | AdapterResponse:
        return AdapterResponse(
            status_code=200,
            headers={},
            body={"cached": True},
            ctx=req.ctx,
        )


class FailingRequestInterceptor(RequestInterceptor):
    def __init__(self, *, best_effort=False):
        self.best_effort = best_effort

    async def intercept_request(self, req: AdapterRequest) -> AdapterRequest:
        raise RuntimeError("boom")


def test_stage_order_validation():
    with pytest.raises(ValueError, match="Invalid interceptor order"):
        AdapterPipeline([AppendBodyInterceptor(), AddHeaderInterceptor()])


async def test_request_then_endpoint_then_response():
    endpoint = FixedEndpoint()
    pipeline = AdapterPipeline(
        [
            AddHeaderInterceptor(),
            endpoint,
        ]
    )
    resp = await pipeline.process(_req())
    assert resp.status_code == 200
    assert resp.body["result"] == "ok"
    assert endpoint.last_request.headers["x-test"] == "1"


async def test_short_circuit_skips_endpoint():
    """When a RequestToResponseInterceptor short-circuits, later request-side
    interceptors (like the endpoint) are skipped, but response interceptors
    still run so they can inspect the short-circuited response."""
    endpoint = FixedEndpoint()
    tracker = TrackingResponseInterceptor()
    pipeline = AdapterPipeline(
        [
            CacheHitInterceptor(),
            endpoint,
            tracker,
        ]
    )
    resp = await pipeline.process(_req())
    assert resp.body["cached"] is True
    assert endpoint.last_request is None  # endpoint was skipped
    assert tracker.called is True  # response interceptors still fire


async def test_best_effort_continues_on_error():
    pipeline = AdapterPipeline(
        [
            FailingRequestInterceptor(best_effort=True),
            FixedEndpoint(),
        ]
    )
    resp = await pipeline.process(_req())
    assert resp.status_code == 200


async def test_non_best_effort_raises():
    pipeline = AdapterPipeline(
        [
            FailingRequestInterceptor(best_effort=False),
            FixedEndpoint(),
        ]
    )
    with pytest.raises(RuntimeError, match="boom"):
        await pipeline.process(_req())


async def test_response_interceptors_run_after_endpoint():
    """Response interceptors placed after the endpoint in the chain must run."""
    appender = AppendBodyInterceptor()
    pipeline = AdapterPipeline(
        [
            AddHeaderInterceptor(),
            FixedEndpoint(),
            appender,
        ]
    )
    resp = await pipeline.process(_req())
    assert resp.body.get("appended") is True


async def test_multiple_response_interceptors_run_in_reverse():
    """Multiple response interceptors run in reverse chain order."""
    order: list[str] = []

    class First(ResponseInterceptor):
        async def intercept_response(self, resp):
            order.append("first")
            return resp

    class Second(ResponseInterceptor):
        async def intercept_response(self, resp):
            order.append("second")
            return resp

    pipeline = AdapterPipeline(
        [
            FixedEndpoint(),
            First(),
            Second(),
        ]
    )
    await pipeline.process(_req())
    assert order == ["second", "first"]


async def test_system_message_replace_deduplicates():
    """Replace strategy should produce exactly one system message even with multiple originals."""
    from nemo_evaluator.adapters.registry import InterceptorRegistry

    ic = InterceptorRegistry.create(
        "system_message",
        {
            "system_message": "New",
            "strategy": "replace",
        },
    )
    req = _req()
    req.body["messages"] = [
        {"role": "system", "content": "Old1"},
        {"role": "system", "content": "Old2"},
        {"role": "user", "content": "hi"},
    ]
    result = await ic.intercept_request(req)
    sys_msgs = [m for m in result.body["messages"] if m["role"] == "system"]
    assert len(sys_msgs) == 1
    assert sys_msgs[0]["content"] == "New"
