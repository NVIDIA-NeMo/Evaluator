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
"""Tests for the Prometheus /metrics sidecar scraper (FEP-831)."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest
from aiohttp import web

from nemo_evaluator.observability import scrape_metrics

_VLLM_LIKE_BODY = (
    "# HELP vllm:num_requests_running Number of requests running\n"
    "# TYPE vllm:num_requests_running gauge\n"
    'vllm:num_requests_running{model="m"} 12.0\n'
    "# HELP vllm:gpu_cache_usage_perc GPU KV cache usage\n"
    "# TYPE vllm:gpu_cache_usage_perc gauge\n"
    "vllm:gpu_cache_usage_perc 0.18\n"
)


async def _start_stub(handler):
    """Start an aiohttp app on a random port with a single GET /metrics handler.

    Returns (base_url, runner) — caller is responsible for awaiting
    ``runner.cleanup()``. Routes can't be added once the app starts, so
    the handler is registered up front.
    """
    app = web.Application()
    app.router.add_get("/metrics", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return f"http://127.0.0.1:{port}", runner


@pytest.mark.asyncio
async def test_writes_records_when_endpoint_is_prometheus(tmp_path: Path) -> None:
    async def _handler(_req):
        return web.Response(text=_VLLM_LIKE_BODY, content_type="text/plain")

    base, runner = await _start_stub(_handler)
    try:
        out = tmp_path / "engine_metrics.jsonl"
        stop = asyncio.Event()

        async def _stopper():
            await asyncio.sleep(0.25)
            stop.set()

        stopper = asyncio.create_task(_stopper())
        rc = await scrape_metrics._run(f"{base}/metrics", out, interval_s=0.05, stop_event=stop)
        stopper.cancel()

        assert rc == 0
        assert out.exists()
        records = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
        assert len(records) >= 2
        for r in records:
            assert r["url"] == f"{base}/metrics"
            assert r["text"] is not None
            assert "vllm:num_requests_running" in r["text"]
            assert r["exception"] is None
            assert r["scrape_latency_s"] >= 0.0
            assert r["t_sample"]
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_skips_when_endpoint_returns_404(tmp_path: Path) -> None:
    async def _handler(_req):
        return web.Response(status=404, text="not found")

    base, runner = await _start_stub(_handler)
    try:
        out = tmp_path / "engine_metrics.jsonl"
        stop = asyncio.Event()
        rc = await scrape_metrics._run(f"{base}/metrics", out, interval_s=0.05, stop_event=stop)
        assert rc == 0
        assert not out.exists()
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_skips_when_body_is_not_prometheus_shaped(tmp_path: Path) -> None:
    async def _handler(_req):
        return web.Response(text='{"hello": "world"}', content_type="application/json")

    base, runner = await _start_stub(_handler)
    try:
        out = tmp_path / "engine_metrics.jsonl"
        stop = asyncio.Event()
        rc = await scrape_metrics._run(f"{base}/metrics", out, interval_s=0.05, stop_event=stop)
        assert rc == 0
        assert not out.exists()
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_skips_when_endpoint_unreachable(tmp_path: Path) -> None:
    out = tmp_path / "engine_metrics.jsonl"
    stop = asyncio.Event()
    rc = await scrape_metrics._run(
        "http://127.0.0.1:1/metrics",
        out,
        interval_s=0.05,
        stop_event=stop,
    )
    assert rc == 0
    assert not out.exists()


@pytest.mark.asyncio
async def test_loop_keeps_writing_across_mixed_responses(tmp_path: Path) -> None:
    """First poll (probe) succeeds; mid-loop the server returns 500. The
    loop must keep writing records — a 500 response is still successful at
    the HTTP-client layer (``_poll_once`` writes the response body verbatim
    in ``text``), so this does NOT trigger ``exception != null``. Verifies
    the loop doesn't crash on mixed responses."""
    state = {"calls": 0}

    async def _handler(_req):
        state["calls"] += 1
        if state["calls"] <= 2:
            return web.Response(text=_VLLM_LIKE_BODY, content_type="text/plain")
        return web.Response(status=500, text="boom")

    base, runner = await _start_stub(_handler)
    try:
        out = tmp_path / "engine_metrics.jsonl"
        stop = asyncio.Event()

        async def _stopper():
            await asyncio.sleep(0.30)
            stop.set()

        stopper = asyncio.create_task(_stopper())
        rc = await scrape_metrics._run(f"{base}/metrics", out, interval_s=0.05, stop_event=stop)
        stopper.cancel()
        assert rc == 0
        records = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
        assert any("vllm:num_requests_running" in (r["text"] or "") for r in records)
        assert any(r["text"] == "boom" for r in records)
        assert len(records) >= 3
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_records_exception_when_server_disconnects(tmp_path: Path) -> None:
    """When the server is killed mid-loop, ``_poll_once`` raises an
    ``aiohttp.ClientConnectorError``-class exception which gets written
    as ``exception != null, text=null``. Loop continues until stop_event
    fires."""

    async def _handler(_req):
        return web.Response(text=_VLLM_LIKE_BODY, content_type="text/plain")

    base, runner = await _start_stub(_handler)
    out = tmp_path / "engine_metrics.jsonl"
    stop = asyncio.Event()

    async def _kill_then_stop():
        await asyncio.sleep(0.10)
        await runner.cleanup()
        await asyncio.sleep(0.20)
        stop.set()

    killer = asyncio.create_task(_kill_then_stop())
    rc = await scrape_metrics._run(f"{base}/metrics", out, interval_s=0.05, stop_event=stop)
    killer.cancel()
    assert rc == 0
    records = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    assert any(r["exception"] is not None and r["text"] is None for r in records), (
        f"expected at least one error record after server killed; got: {records}"
    )


@pytest.mark.asyncio
async def test_stop_event_cancels_loop_promptly(tmp_path: Path) -> None:
    """``stop_event.set()`` must wake the loop within a tick of the current
    interval, even when the interval is long."""

    async def _handler(_req):
        return web.Response(text=_VLLM_LIKE_BODY, content_type="text/plain")

    base, runner = await _start_stub(_handler)
    try:
        out = tmp_path / "engine_metrics.jsonl"
        stop = asyncio.Event()

        async def _stopper():
            await asyncio.sleep(0.10)
            stop.set()

        t0 = time.monotonic()
        stopper = asyncio.create_task(_stopper())
        rc = await scrape_metrics._run(f"{base}/metrics", out, interval_s=10.0, stop_event=stop)
        stopper.cancel()
        elapsed = time.monotonic() - t0
        assert rc == 0
        assert elapsed < 1.0
    finally:
        await runner.cleanup()


def test_looks_prometheus_helper() -> None:
    assert scrape_metrics._looks_prometheus("# HELP foo bar\n# TYPE foo gauge\nfoo 1.0\n")
    assert scrape_metrics._looks_prometheus("   # TYPE foo gauge\nfoo 1.0\n")
    assert not scrape_metrics._looks_prometheus('{"hello": "world"}')
    assert not scrape_metrics._looks_prometheus("<html><body>not metrics</body></html>")
    assert not scrape_metrics._looks_prometheus("")


def test_disabled_env_var_short_circuits(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("NEL_TRACING_METRICS_DISABLED", "1")
    out = tmp_path / "engine_metrics.jsonl"
    rc = scrape_metrics.main(["--url", "http://nope/metrics", "--out", str(out), "--interval", "0.05"])
    assert rc == 0
    assert not out.exists()
