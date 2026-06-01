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
"""HTTP request/response pair dump with always-failed capture and status counts.

Writes a single JSON file with two top-level keys:

    {
        "metrics": {"200": 137, "504": 2, "400": 1, ...},
        "pairs":   [
            {
                "request_id": "...",
                "request":  {"headers": {...}, "body": {...}},
                "response": {"status": 200, "headers": {...}, "body": {...}, "latency_ms": 12.3},
            },
            ...
        ]
    }

A pair is kept if it is one of the first ``first_n`` seen *or* its response
status is not 200. Each pair is recorded at most once.

Request headers and body are read from ``resp.ctx.extra``, where the
``endpoint`` interceptor stashes them (``upstream_request_headers`` and
``upstream_request_body``) before the upstream call.
"""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import threading
from collections import Counter
from pathlib import Path
from typing import Any

from nemo_evaluator.adapters.types import AdapterResponse, ResponseInterceptor

logger = logging.getLogger(__name__)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    return repr(obj)


class Interceptor(ResponseInterceptor):
    """Dump HTTP req/res pairs + status-code histogram to one JSON file.

    The pair selection rule, evaluated exactly once per response:

        keep = (total_seen <= first_n) or (status != 200)

    so a failure occurring inside the first ``first_n`` is recorded once,
    not duplicated later.

    Snapshots are processed by a single dedicated daemon worker thread.
    The worker coalesces: each loop iteration drains every snapshot currently
    in the queue and only writes the most recent one, since each snapshot
    cumulatively includes the state of all earlier ones. This bounds memory
    even if disk writes lag behind incoming responses, and guarantees an
    older snapshot can never overwrite a newer one.
    """

    best_effort: bool = True

    def __init__(
        self,
        *,
        dump_path: str,
        first_n: int = 50,
    ) -> None:
        if first_n < 0:
            raise ValueError("first_n must be non-negative")
        self._dump_path = Path(dump_path)
        self._first_n = first_n

        self._lock = asyncio.Lock()
        self._status_counts: Counter[str] = Counter()
        self._pairs: list[dict[str, Any]] = []
        self._total_seen = 0

        # FIFO queue + single worker thread = strict creation-order writes.
        self._snapshot_queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self._worker = threading.Thread(
            target=self._snapshot_worker,
            name="http_pairs_dump",
            daemon=True,
        )
        self._worker.start()

        logger.info(
            "http_pairs_dump initialized: path=%s first_n=%d",
            self._dump_path,
            self._first_n,
        )

    async def intercept_response(self, resp: AdapterResponse) -> AdapterResponse:
        async with self._lock:
            self._total_seen += 1
            self._status_counts[str(resp.status_code)] += 1

            keep = (self._total_seen <= self._first_n) or (resp.status_code != 200)
            if keep:
                pair: dict[str, Any] = {
                    "request_id": resp.ctx.request_id,
                    "request": {
                        "headers": resp.ctx.extra.get("upstream_request_headers"),
                        "body": resp.ctx.extra.get("upstream_request_body"),
                    },
                    "response": {
                        "status": resp.status_code,
                        "headers": resp.headers,
                        "body": resp.body,
                        "latency_ms": resp.latency_ms,
                    },
                }
                session_id = resp.ctx.extra.get("session_id")
                if session_id is not None:
                    pair["session_id"] = session_id
                self._pairs.append(pair)

            # Snapshot under the async lock, enqueue, then return immediately.
            # The worker drains the queue in order — never a stale overwrite.
            snapshot = {"metrics": dict(self._status_counts), "pairs": list(self._pairs)}

        self._snapshot_queue.put(snapshot)
        return resp

    def _snapshot_worker(self) -> None:
        while True:
            snapshot = self._snapshot_queue.get()
            # Coalesce: drain any newer queued snapshots and discard them.
            # Each one cumulatively includes earlier state, so writing only
            # the latest is equivalent for the file content while bounding
            # memory and disk traffic. Every drained item still needs
            # task_done() to honor the queue's join() contract.
            while True:
                try:
                    snapshot = self._snapshot_queue.get_nowait()
                except queue.Empty:
                    break
                else:
                    self._snapshot_queue.task_done()
            try:
                self._write_snapshot(snapshot)
            finally:
                self._snapshot_queue.task_done()

    def _write_snapshot(self, snapshot: dict[str, Any]) -> None:
        """Atomic-replace write of the snapshot. Only the worker calls this.

        Catches OSError (filesystem) and (TypeError, ValueError) (json.dump
        rejecting an unserializable value or hitting a circular reference)
        so the worker thread can't die mid-run. ``best_effort=True`` swallows
        the error after logging; ``False`` re-raises so callers can surface it.
        """
        try:
            self._dump_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._dump_path.with_suffix(self._dump_path.suffix + ".tmp")
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(snapshot, f, ensure_ascii=False, default=_json_default)
            tmp.replace(self._dump_path)
        except (OSError, TypeError, ValueError) as exc:
            logger.warning(
                "http_pairs_dump: failed to write %s: %s",
                self._dump_path,
                exc,
            )
            if not self.best_effort:
                raise
