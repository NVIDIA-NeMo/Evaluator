# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Tests for the http_pairs_dump interceptor."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nemo_evaluator.adapters.interceptors.http_pairs_dump import Interceptor
from nemo_evaluator.adapters.registry import InterceptorRegistry
from nemo_evaluator.adapters.types import AdapterResponse, InterceptorContext


def _make_response(status: int, *, request_id: str) -> AdapterResponse:
    ctx = InterceptorContext(request_id=request_id)
    # Simulate what the endpoint interceptor stashes before the upstream call.
    # Note: endpoint.py already strips Authorization before stashing, so the
    # interceptor receives a pre-sanitized header dict.
    ctx.extra["upstream_request_headers"] = {"Content-Type": "application/json"}
    ctx.extra["upstream_request_body"] = {"model": "x", "messages": []}
    return AdapterResponse(
        status_code=status,
        headers={"content-type": "application/json"},
        body={"choices": [{"message": {"content": "ok"}}]},
        latency_ms=10.0,
        ctx=ctx,
    )


@pytest.mark.asyncio
async def test_http_pairs_dump(tmp_path: Path) -> None:
    # Wiring: registry resolves the name to our class.
    assert InterceptorRegistry.resolve_class("http_pairs_dump") is Interceptor

    # One scenario that exercises everything:
    #   - first-N rule keeps r0 (success) and r1 (failure) inside the cap
    #   - r1 (failure inside first-N) recorded exactly once (no later duplicate)
    #   - r2 after N, success -> dropped
    #   - r3 after N, failure -> kept by the !=200 rule
    # metrics aggregate every status seen, not just kept ones.
    interceptor = Interceptor(dump_path=str(tmp_path / "dump.json"), first_n=2)
    await interceptor.intercept_response(_make_response(200, request_id="r0"))
    await interceptor.intercept_response(_make_response(504, request_id="r1"))
    await interceptor.intercept_response(_make_response(200, request_id="r2"))
    await interceptor.intercept_response(_make_response(400, request_id="r3"))
    # Writes happen on a background worker; wait for the queue to drain
    # before reading the file.
    interceptor._snapshot_queue.join()

    data = json.loads((tmp_path / "dump.json").read_text())
    assert [p["request_id"] for p in data["pairs"]] == ["r0", "r1", "r3"]
    assert data["metrics"] == {"200": 2, "504": 1, "400": 1}

    # Each pair has both sides intact (faithful pass-through of what the
    # endpoint stashed in ctx). Authorization stripping is the endpoint's
    # responsibility and is tested separately in test_endpoint_interceptor.
    first = data["pairs"][0]
    assert first["request"]["headers"] == {"Content-Type": "application/json"}
    assert first["request"]["body"] == {"model": "x", "messages": []}
    assert first["response"]["status"] == 200
    assert first["response"]["headers"] == {"content-type": "application/json"}
    assert first["response"]["latency_ms"] == 10.0
