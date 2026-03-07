"""Checkpoint and resume for multi-benchmark evaluation runs."""
from __future__ import annotations

import fcntl
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CHECKPOINT_FILE = "checkpoint.json"


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
