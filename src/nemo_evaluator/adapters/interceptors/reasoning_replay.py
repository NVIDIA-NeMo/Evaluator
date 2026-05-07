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
"""Cross-turn ``reasoning_content`` recovery.

Caches ``reasoning_content`` from upstream responses (keyed by
``tool_call.id``, with a content-hash fallback for text-only turns) and
re-injects it into matching assistant messages on the next outbound
request.  Works around agent SDKs (OpenHands, pi-ai) that drop
``reasoning_content`` from chat-completions assistant messages on
replay, which would otherwise lose the model's chain-of-thought across
turns.

Cache entries are scoped by ``ctx.extra["session_id"]``.

The wrap tag in ``think_tags`` / ``both`` modes defaults to
``<think>…</think>`` (matches Nemotron-Super, DeepSeek-R1, QwQ) and is
configurable via ``tag_open`` / ``tag_close`` for chat templates that
expect a different marker (e.g. ``<thought>…</thought>``).

Composition:
* The ``reasoning`` normalizer (if also enabled) always runs before this
  interceptor on the response side regardless of YAML order, because the
  pipeline partitions interceptors by ``stage``.  Pair them when the
  upstream may emit ``<think>`` tags inside ``content`` instead of a
  native ``reasoning_content`` field.
* When ``caching`` is also enabled, list it **before** ``reasoning_replay``
  in the chain so disk cache keys hash the pre-injection request body.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections import OrderedDict
from typing import Any

from nemo_evaluator.adapters.types import (
    AdapterRequest,
    AdapterResponse,
    RequestInterceptor,
    ResponseInterceptor,
)

logger = logging.getLogger(__name__)

_VALID_MODES = ("think_tags", "native", "both")
_HASH_PREFIX_LEN = 256


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


def _content_key(content: Any) -> str | None:
    s = _content_text(content)[:_HASH_PREFIX_LEN]
    if not s.strip():
        return None
    return "h:" + hashlib.sha256(s.encode()).hexdigest()[:16]


def _message_keys(msg: dict, scope: str) -> list[str]:
    """Cache keys for an assistant message: one per tool_call id, or one
    content-hash key if no usable tool ids are present.
    """
    keys: list[str] = []
    for tc in msg.get("tool_calls") or []:
        if not isinstance(tc, dict):
            continue
        tc_id = tc.get("id")
        if isinstance(tc_id, str) and tc_id:
            keys.append(f"{scope}|c:{tc_id}")
    if not keys:
        ck = _content_key(msg.get("content"))
        if ck:
            keys.append(f"{scope}|{ck}")
    return keys


def _starts_with(content: Any, prefix: str) -> bool:
    return _content_text(content).lstrip().startswith(prefix)


def _wrap(content: Any, reasoning: str, tag_open: str, tag_close: str) -> Any:
    tag = f"{tag_open}{reasoning}{tag_close}\n"
    if isinstance(content, list):
        return [{"type": "text", "text": tag}, *content]
    if isinstance(content, str):
        return tag + content
    return tag


class Interceptor(RequestInterceptor, ResponseInterceptor):
    best_effort = True

    def __init__(
        self,
        *,
        mode: str = "think_tags",
        tag_open: str = "<think>",
        tag_close: str = "</think>",
        cache_max_entries: int = 10_000,
        log_misses: bool = False,
    ) -> None:
        if mode not in _VALID_MODES:
            raise ValueError(f"reasoning_replay mode must be one of {_VALID_MODES}, got {mode!r}")
        if not tag_open or not tag_close:
            raise ValueError("reasoning_replay tag_open and tag_close must be non-empty")
        self._mode = mode
        self._tag_open = tag_open
        self._tag_close = tag_close
        self._max = max(cache_max_entries, 1)
        self._log_misses = log_misses
        self._cache: OrderedDict[str, str] = OrderedDict()
        self._lock = asyncio.Lock()

    async def intercept_response(self, resp: AdapterResponse) -> AdapterResponse:
        body = resp.body
        if not isinstance(body, dict):
            return resp
        scope = resp.ctx.extra.get("session_id") or ""
        async with self._lock:
            for ch in body.get("choices") or []:
                if not isinstance(ch, dict):
                    continue
                msg = ch.get("message")
                if not isinstance(msg, dict):
                    continue
                rc = msg.get("reasoning_content")
                if not isinstance(rc, str) or not rc.strip():
                    continue
                for key in _message_keys(msg, scope):
                    self._store(key, rc)
        return resp

    async def intercept_request(self, req: AdapterRequest) -> AdapterRequest:
        body = req.body
        if not isinstance(body, dict):
            return req
        messages = body.get("messages")
        if not isinstance(messages, list):
            return req
        scope = req.ctx.extra.get("session_id") or ""
        hits = 0
        misses = 0
        async with self._lock:
            for msg in messages:
                if not isinstance(msg, dict) or msg.get("role") != "assistant":
                    continue
                native = msg.get("reasoning_content")
                native_set = isinstance(native, str) and bool(native.strip())
                if self._mode == "native" and native_set:
                    continue
                cached: str | None = None
                for key in _message_keys(msg, scope):
                    cached = self._get(key)
                    if cached:
                        break
                if cached is None:
                    if self._log_misses and (msg.get("tool_calls") or _content_text(msg.get("content"))):
                        misses += 1
                    continue
                if self._apply(msg, cached):
                    hits += 1
        if hits or misses:
            req.ctx.extra["reasoning_replay_hits"] = hits
            req.ctx.extra["reasoning_replay_misses"] = misses
            logger.debug("reasoning_replay hits=%d misses=%d mode=%s", hits, misses, self._mode)
        return req

    def _store(self, key: str, value: str) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        while len(self._cache) > self._max:
            self._cache.popitem(last=False)

    def _get(self, key: str) -> str | None:
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def _apply(self, msg: dict, reasoning: str) -> bool:
        changed = False
        native = msg.get("reasoning_content")
        native_set = isinstance(native, str) and bool(native.strip())
        if self._mode in ("native", "both") and not native_set:
            msg["reasoning_content"] = reasoning
            changed = True
        if self._mode in ("think_tags", "both") and not _starts_with(msg.get("content"), self._tag_open):
            msg["content"] = _wrap(msg.get("content"), reasoning, self._tag_open, self._tag_close)
            changed = True
        return changed
