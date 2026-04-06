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
"""Filesystem response cache keyed by request hash."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_MAX_ENTRIES = 50_000
DEFAULT_MAX_SIZE_MB = 2048


class ResponseCache:
    def __init__(
        self, cache_dir: str | Path, *, max_entries: int = DEFAULT_MAX_ENTRIES, max_size_mb: int = DEFAULT_MAX_SIZE_MB
    ) -> None:
        self.root = Path(cache_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_entries = max_entries
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self._hits = 0
        self._misses = 0

    @staticmethod
    def _key(
        model: str,
        prompt: str,
        system: str | None,
        temperature: float,
        max_tokens: int | None,
        top_p: float | None = None,
        seed: int | None = None,
        stop: list[str] | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        messages: list[dict] | None = None,
    ) -> str:
        if messages:
            content = json.dumps(messages, sort_keys=True, default=str)
        else:
            content = prompt
        parts = [
            model,
            content,
            system or "",
            str(temperature),
            str(max_tokens),
            str(top_p) if top_p is not None else "",
            str(seed) if seed is not None else "",
            json.dumps(sorted(stop)) if stop else "",
            str(frequency_penalty) if frequency_penalty is not None else "",
            str(presence_penalty) if presence_penalty is not None else "",
        ]
        return hashlib.sha256("|".join(parts).encode()).hexdigest()

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(
        self,
        model: str,
        prompt: str,
        system: str | None,
        temperature: float,
        max_tokens: int | None,
        *,
        top_p: float | None = None,
        seed: int | None = None,
        stop: list[str] | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        messages: list[dict] | None = None,
    ) -> dict[str, Any] | None:
        key = self._key(
            model,
            prompt,
            system,
            temperature,
            max_tokens,
            top_p,
            seed,
            stop,
            frequency_penalty,
            presence_penalty,
            messages,
        )
        p = self._path(key)
        if p.exists():
            self._hits += 1
            try:
                os.utime(p)
            except OSError:
                pass
            return json.loads(p.read_text(encoding="utf-8"))
        self._misses += 1
        return None

    def put(
        self,
        model: str,
        prompt: str,
        system: str | None,
        temperature: float,
        max_tokens: int | None,
        response: dict[str, Any],
        *,
        top_p: float | None = None,
        seed: int | None = None,
        stop: list[str] | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        messages: list[dict] | None = None,
    ) -> None:
        if temperature > 0:
            return
        key = self._key(
            model,
            prompt,
            system,
            temperature,
            max_tokens,
            top_p,
            seed,
            stop,
            frequency_penalty,
            presence_penalty,
            messages,
        )
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(response, default=str), encoding="utf-8")
        self._maybe_evict()

    def _maybe_evict(self) -> None:
        """LRU eviction when cache exceeds size or entry limits."""
        try:
            entries = list(self.root.rglob("*.json"))
        except OSError:
            return

        if len(entries) <= self.max_entries:
            total = sum(f.stat().st_size for f in entries)
            if total <= self.max_size_bytes:
                return

        entries.sort(key=lambda f: f.stat().st_mtime)
        removed = 0
        while len(entries) - removed > self.max_entries * 0.8:
            entries[removed].unlink(missing_ok=True)
            removed += 1

        if removed:
            logger.info("Cache eviction: removed %d stale entries", removed)

    @property
    def stats(self) -> dict[str, int]:
        return {"hits": self._hits, "misses": self._misses}
