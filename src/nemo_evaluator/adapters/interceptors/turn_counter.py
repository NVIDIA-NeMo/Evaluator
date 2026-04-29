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
from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from nemo_evaluator.adapters.types import AdapterRequest, RequestInterceptor

logger = logging.getLogger(__name__)

_WARN_THRESHOLD = 0.80
_URGENT_THRESHOLD = 0.95
_STALE_SESSION_SEC = 900.0
_GC_INTERVAL_SEC = 300.0


class InjectionPosition(str, Enum):
    SYSTEM_MESSAGE = "system_message"
    USER_MESSAGE = "user_message"


class InjectionTrigger(str, Enum):
    THRESHOLD = "threshold"
    PERIODIC = "periodic"


class _Severity(str, Enum):
    URGENT = "urgent"
    WARN = "warn"
    NON_ACTIONABLE = "non_actionable"


_REMINDER_TEMPLATE = "ENVIRONMENT REMINDER: You have {remaining} turns left to complete the task."


def _session_key_from_body(body: dict[str, Any]) -> str:
    """Fallback: derive a session key from the first non-system message."""
    messages = body.get("messages") or []
    for msg in messages:
        if msg.get("role") == "system":
            continue
        content = msg.get("content", "")
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        if content:
            return hashlib.sha256(content.encode()).hexdigest()[:8]
    return "unknown"


@dataclass
class _Session:
    count: int = 0
    last_time: float = field(default_factory=time.monotonic)


class Interceptor(RequestInterceptor):
    """Turn counter with configurable reminder injection.

    Two orthogonal axes:

    * ``position`` — where the reminder lands in the request payload
      (``system_message`` appends a new system message; ``user_message``
      appends to the last user message's content).
    * ``trigger`` — when the reminder fires
      (``threshold`` at 80% / 95% of ``max_turns``; ``periodic`` every
      ``interval`` turns).

    All four combinations are valid. Defaults reproduce the prior
    threshold-based system-message behavior.
    """

    def __init__(
        self,
        *,
        every: int = 1,
        max_turns: int | None = None,
        position: str | InjectionPosition = InjectionPosition.SYSTEM_MESSAGE,
        trigger: str | InjectionTrigger = InjectionTrigger.THRESHOLD,
        interval: int = 1,
    ) -> None:
        if interval < 1:
            raise ValueError(f"interval must be >= 1, got {interval}")
        self._every = max(every, 1)
        self._max = max_turns
        self._position = InjectionPosition(position)
        self._trigger = InjectionTrigger(trigger)
        self._interval = interval
        self._sessions: dict[str, _Session] = {}
        self._lock = asyncio.Lock()
        self._last_gc = time.monotonic()

        if max_turns is None:
            logger.warning(
                "turn_counter: max_turns is unset; injection will be silently disabled (position=%s trigger=%s).",
                self._position.value,
                self._trigger.value,
            )

    async def intercept_request(self, req: AdapterRequest) -> AdapterRequest:
        key = req.ctx.extra.get("session_id")
        if not key:
            key = _session_key_from_body(req.body)
            logger.warning("no session_id in context — falling back to body-hash key %s", key)

        async with self._lock:
            now = time.monotonic()
            if now - self._last_gc > _GC_INTERVAL_SEC:
                self._gc(now)
                self._last_gc = now

            sess = self._sessions.setdefault(key, _Session())
            sess.count += 1
            n = sess.count
            dt = now - sess.last_time if sess.count > 1 else 0.0
            sess.last_time = now
            active = len(self._sessions)

        if n % self._every == 0 or n == 1:
            cap = f"/{self._max}" if self._max else ""
            elapsed = f" (+{dt:.1f}s)" if dt > 0 else ""
            logger.info(
                "task %s turn %d%s%s (%d active)",
                key,
                n,
                cap,
                elapsed,
                active,
            )

        if self._max is None:
            return req

        if n > self._max:
            logger.warning(
                "task %s REJECTED turn %d (max_turns=%d exceeded)",
                key,
                n,
                self._max,
            )
            from nemo_evaluator.errors import GracefulError

            raise GracefulError(
                f"Turn budget exhausted: {n}/{self._max} turns used. "
                f"The evaluation framework has terminated this agent session."
            )

        remaining = self._max - n
        messages = req.body.get("messages")
        if not isinstance(messages, list):
            return req

        if self._trigger is InjectionTrigger.THRESHOLD:
            severity = self._threshold_severity(n)
            if severity is _Severity.NON_ACTIONABLE:
                return req
            body = self._threshold_message_body(n, remaining, severity)
            notice = f"[SYSTEM] {body}" if self._position is InjectionPosition.SYSTEM_MESSAGE else body
        else:
            if n % self._interval != 0:
                return req
            notice = _REMINDER_TEMPLATE.format(remaining=remaining)

        if self._position is InjectionPosition.SYSTEM_MESSAGE:
            messages.append({"role": "system", "content": notice})
        else:
            self._append_to_last_user_message(messages, notice)
        return req

    def _threshold_severity(self, n: int) -> _Severity:
        ratio = n / self._max
        if ratio >= _URGENT_THRESHOLD:
            return _Severity.URGENT
        if ratio >= _WARN_THRESHOLD:
            return _Severity.WARN
        return _Severity.NON_ACTIONABLE

    def _threshold_message_body(self, n: int, remaining: int, severity: _Severity) -> str:
        if severity is _Severity.URGENT:
            return (
                f"URGENT: Turn {n}/{self._max} — only {remaining} turn(s) left. "
                f"You MUST provide your final answer NOW. Do not start new work."
            )
        return (
            f"Turn {n}/{self._max} — {remaining} turns remaining. "
            f"Begin wrapping up: finish current work and prepare your final answer."
        )

    def _append_to_last_user_message(self, messages: list, notice: str) -> None:
        # OpenAI chat completions accept either a string or a list of content
        # blocks for ``content`` (multimodal: text + image_url, audio, etc.).
        # The reminder must sit at the very end of the user payload. For a
        # string we concatenate with a blank-line separator; for a list, if
        # the final block is text we extend it (so the result doesn't end
        # in two adjacent text blocks), otherwise we append a new text
        # block at the tail.
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content")
                if isinstance(content, list):
                    tail = content[-1] if content else None
                    if isinstance(tail, dict) and tail.get("type") == "text":
                        existing = tail.get("text", "")
                        tail["text"] = f"{existing}\n\n{notice}" if existing else notice
                    else:
                        content.append({"type": "text", "text": notice})
                elif isinstance(content, str):
                    msg["content"] = f"{content}\n\n{notice}" if content else notice
                else:
                    logger.warning(
                        "turn_counter: unexpected user content type %s (%r); replacing with notice.",
                        type(content).__name__,
                        content,
                    )
                    msg["content"] = notice
                return
        logger.debug("turn_counter: no user message found in %d-message payload; notice not injected.", len(messages))

    def _gc(self, now: float) -> None:
        """Remove sessions idle longer than ``_STALE_SESSION_SEC``."""
        stale = [k for k, s in self._sessions.items() if now - s.last_time > _STALE_SESSION_SEC]
        for k in stale:
            del self._sessions[k]
        if stale:
            logger.debug("turn_counter GC: removed %d stale sessions, %d remaining", len(stale), len(self._sessions))
