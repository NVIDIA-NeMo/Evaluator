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
"""Checkpoint and resume for multi-benchmark evaluation runs."""

from __future__ import annotations

import fcntl
import json
import logging
import re
from pathlib import Path
from typing import Any

from nemo_evaluator.engine.step_log import INFERENCE_LOG, VERIFIED_LOG

logger = logging.getLogger(__name__)

CHECKPOINT_FILE = "checkpoint.json"


def _safe_name(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", s)


def _count_data_lines(path: Path) -> int:
    """Count non-empty, non-meta lines in a JSONL file."""
    if not path.exists():
        return 0
    count = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("_type") == "meta":
                continue
            count += 1
    return count


class CheckpointManager:
    def __init__(self, output_dir: str | Path) -> None:
        self.root = Path(output_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self._path = self.root / CHECKPOINT_FILE
        self._state: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        if self._path.exists():
            try:
                with open(self._path, encoding="utf-8") as f:
                    fcntl.flock(f, fcntl.LOCK_SH)
                    try:
                        return json.load(f)
                    finally:
                        fcntl.flock(f, fcntl.LOCK_UN)
            except (json.JSONDecodeError, OSError, ValueError) as e:
                logger.warning("Corrupt checkpoint, starting fresh: %s", e)
        return {"completed_benchmarks": {}, "failed_benchmarks": {}}

    def _save(self) -> None:
        tmp = self._path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                json.dump(self._state, f, indent=2, default=str)
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
        tmp.replace(self._path)

    def is_completed(self, benchmark: str) -> bool:
        return benchmark in self._state["completed_benchmarks"]

    def get_completed_result(self, benchmark: str) -> dict[str, Any] | None:
        return self._state["completed_benchmarks"].get(benchmark)

    def mark_completed(self, benchmark: str, bundle_path: str) -> None:
        self._state["completed_benchmarks"][benchmark] = {"bundle_path": bundle_path}
        self._state["failed_benchmarks"].pop(benchmark, None)
        self._save()

    def mark_failed(self, benchmark: str, error: str) -> None:
        self._state["failed_benchmarks"][benchmark] = {"error": error}
        self._save()

    def has_partial_progress(self, benchmark: str) -> bool:
        """Check if step log files exist with actual data for this benchmark."""
        bench_dir = self.root / _safe_name(benchmark)
        if not bench_dir.is_dir():
            return False
        inf_path = bench_dir / INFERENCE_LOG
        ver_path = bench_dir / VERIFIED_LOG
        return inf_path.exists() or ver_path.exists()

    def get_progress(self, benchmark: str) -> dict[str, int] | None:
        """Derive step-level progress from log files on disk."""
        bench_dir = self.root / _safe_name(benchmark)
        if not bench_dir.is_dir():
            return None
        inf_path = bench_dir / INFERENCE_LOG
        ver_path = bench_dir / VERIFIED_LOG
        if not inf_path.exists() and not ver_path.exists():
            return None
        return {
            "inferred": _count_data_lines(inf_path),
            "verified": _count_data_lines(ver_path),
        }

    def pending_benchmarks(self, all_benchmarks: list[str]) -> list[str]:
        done = set(self._state["completed_benchmarks"])
        return [b for b in all_benchmarks if b not in done]

    @property
    def summary(self) -> dict[str, int]:
        return {
            "completed": len(self._state["completed_benchmarks"]),
            "failed": len(self._state["failed_benchmarks"]),
        }

    def clear(self) -> None:
        self._state = {"completed_benchmarks": {}, "failed_benchmarks": {}}
        if self._path.exists():
            self._path.unlink()
