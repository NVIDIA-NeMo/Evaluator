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
"""Append-only JSONL step logs for incremental persistence and crash-safe resume.

Two log files per benchmark:
- inference_log.jsonl: written after solver.solve(), before env.verify()
- verified_log.jsonl:  written after env.verify() completes

On resume, the eval loop loads both logs to determine which (problem_idx, repeat)
pairs can be skipped or need only partial re-execution.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

INFERENCE_LOG = "inference_log.jsonl"
VERIFIED_LOG = "verified_log.jsonl"
META_TYPE = "meta"
MAX_TRAJECTORY_BYTES = 5 * 1024 * 1024


def config_hash(config: dict[str, Any]) -> str:
    """Deterministic hash of inference-affecting config fields."""
    keys = ["model", "base_url", "temperature", "max_tokens", "top_p", "system_prompt", "model_url"]
    subset = {k: config.get(k) for k in keys if config.get(k) is not None}
    raw = json.dumps(subset, sort_keys=True)
    return f"sha256:{hashlib.sha256(raw.encode()).hexdigest()}"


class StepLog:
    """Async-safe append-only JSONL log with fsync-on-write.

    Each instance owns an asyncio.Lock so concurrent _run_step() tasks can
    safely call ``await log.append(record)`` without external coordination.

    ``fsync_interval=1`` (default) ensures every write is immediately visible
    on network filesystems (e.g. Lustre).  Set higher to trade visibility for
    throughput on local disks.
    """

    def __init__(self, path: Path, *, fsync_interval: int = 1) -> None:
        self._path = path
        self._fsync_interval = fsync_interval
        self._lock = asyncio.Lock()
        self._fd: Any = None
        self._append_count = 0

    def open(self, *, truncate: bool = False) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        mode = "w" if truncate else "a"
        self._fd = open(self._path, mode, encoding="utf-8")
        self._append_count = 0

    def load(self) -> dict[tuple[int, int], dict[str, Any]]:
        """Read existing log lines. Skips the meta header and corrupt trailing lines.

        Returns dict keyed by (problem_idx, repeat). Last-write-wins for dupes.
        """
        cache: dict[tuple[int, int], dict[str, Any]] = {}
        if not self._path.exists():
            return cache

        with open(self._path, encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("step_log: skipping corrupt line %d in %s", lineno, self._path.name)
                    continue
                if record.get("_type") == META_TYPE:
                    continue
                key = (record.get("problem_idx", -1), record.get("repeat", -1))
                if key[0] < 0:
                    continue
                cache[key] = record
        return cache

    def load_meta(self) -> dict[str, Any] | None:
        """Read just the meta header line (first line with _type=meta)."""
        if not self._path.exists():
            return None
        with open(self._path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    return None
                if record.get("_type") == META_TYPE:
                    return record
                return None
        return None

    def compact(self, cache: dict[tuple[int, int], dict[str, Any]], meta: dict[str, Any] | None = None) -> None:
        """Rewrite the file deduped from the loaded cache.

        Prevents unbounded growth from repeated crash/resume cycles.
        """
        tmp = self._path.with_suffix(".compact.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            if meta:
                f.write(json.dumps(meta, default=str) + "\n")
            for _key in sorted(cache):
                f.write(json.dumps(cache[_key], default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(self._path)

    async def append(self, record: dict[str, Any]) -> None:
        """Append one JSON line. Async-safe via internal lock."""
        async with self._lock:
            if self._fd is None:
                raise RuntimeError(f"StepLog not open: {self._path}")

            if "trajectory" in record and record["trajectory"]:
                traj_str = json.dumps(record["trajectory"], default=str)
                if len(traj_str) > MAX_TRAJECTORY_BYTES:
                    record = {**record, "trajectory": None, "_truncated": True}

            self._fd.write(json.dumps(record, default=str) + "\n")
            self._fd.flush()
            self._append_count += 1

            if self._fsync_interval > 0 and self._append_count % self._fsync_interval == 0:
                os.fsync(self._fd.fileno())

    def write_meta(self, meta: dict[str, Any]) -> None:
        """Write the metadata header as the first line. Call before any appends."""
        if self._fd is None:
            raise RuntimeError(f"StepLog not open: {self._path}")
        meta_line = {**meta, "_type": META_TYPE}
        self._fd.write(json.dumps(meta_line, default=str) + "\n")
        self._fd.flush()
        os.fsync(self._fd.fileno())

    def close(self) -> None:
        if self._fd is not None:
            try:
                self._fd.flush()
                os.fsync(self._fd.fileno())
            except OSError:
                pass
            self._fd.close()
            self._fd = None
