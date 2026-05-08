# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sidecar scraper for Prometheus ``/metrics`` endpoints.

Hits a configured URL on a fixed interval and appends each response — raw
text plus wall-clock timestamp and fetch latency — to a JSONL file. Same
pattern as ``nvidia-smi --loop`` or ``dcgm-dmon``: a tool-agnostic poll
loop that captures engine state alongside an eval run for post-hoc
analysis.

Auto-skips when the endpoint isn't Prometheus-shaped (e.g. an OpenAI-only
API server with no ``/metrics``), so the spawn is safe across all NEL
service types — services without a Prometheus endpoint produce no file
and no error.

Records have shape::

    {"t_sample": "2026-05-04T19:23:14.123+00:00",
     "scrape_latency_s": 0.043,
     "url": "http://localhost:8000/metrics",
     "text": "<full /metrics body>",
     "exception": null}

On transient failures, ``text=null`` and ``exception`` carries the error
class + message. The gap stays visible in the trace.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)

_PROMETHEUS_PREFIXES = ("# HELP", "# TYPE")
_DETECT_TIMEOUT_S = 5.0


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _looks_prometheus(text: str) -> bool:
    return text.lstrip().startswith(_PROMETHEUS_PREFIXES)


async def _detect(session: aiohttp.ClientSession, url: str) -> bool:
    """Probe the URL once; return True iff body looks like Prometheus text."""
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=_DETECT_TIMEOUT_S)) as resp:
            if resp.status != 200:
                logger.info("scrape_metrics: %s returned status=%d, skipping", url, resp.status)
                return False
            body = await resp.text()
            if not _looks_prometheus(body):
                logger.info("scrape_metrics: %s body is not Prometheus-shaped, skipping", url)
                return False
            return True
    except Exception as exc:
        logger.info("scrape_metrics: probe of %s failed (%s: %s), skipping", url, type(exc).__name__, exc)
        return False


async def _poll_once(session: aiohttp.ClientSession, url: str) -> dict[str, Any]:
    t0 = time.monotonic()
    record: dict[str, Any] = {
        "t_sample": _now_iso(),
        "url": url,
        "scrape_latency_s": 0.0,
        "text": None,
        "exception": None,
    }
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=_DETECT_TIMEOUT_S)) as resp:
            record["text"] = await resp.text()
    except Exception as exc:  # noqa: BLE001 — sidecar must never crash the eval; capture and continue
        record["exception"] = f"{type(exc).__name__}: {exc}"
    record["scrape_latency_s"] = time.monotonic() - t0
    return record


async def _run(url: str, out_path: Path, interval_s: float, stop_event: asyncio.Event) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    async with aiohttp.ClientSession() as session:
        if not await _detect(session, url):
            return 0
        with out_path.open("a") as fp:
            while not stop_event.is_set():
                record = await _poll_once(session, url)
                fp.write(json.dumps(record, separators=(",", ":")) + "\n")
                fp.flush()
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(stop_event.wait(), timeout=interval_s)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="nel.scrape_metrics", description=__doc__.split("\n", 1)[0])
    parser.add_argument("--url", required=True, help="Full Prometheus /metrics URL to scrape")
    parser.add_argument("--out", required=True, help="Output JSONL path")
    parser.add_argument("--interval", type=float, default=10.0, help="Seconds between samples (default: 10)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if os.getenv("NEL_TRACING_METRICS_DISABLED"):
        logger.info("scrape_metrics: NEL_TRACING_METRICS_DISABLED set, exiting")
        return 0

    loop = asyncio.new_event_loop()
    stop_event = asyncio.Event()

    def _on_signal(*_: Any) -> None:
        loop.call_soon_threadsafe(stop_event.set)

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, _on_signal)

    try:
        return loop.run_until_complete(_run(args.url, Path(args.out), args.interval, stop_event))
    finally:
        loop.close()


if __name__ == "__main__":
    sys.exit(main())
