"""Interceptor that counts and logs LLM turns per task as a live progress indicator.

Each concurrent agent task gets its own counter, keyed by a hash of the
first user/system message (which contains the unique task prompt).

Configuration (via YAML)::

    interceptors:
      - turn_counter              # log every turn
      - turn_counter:
          every: 5                # log every 5th turn
          max_turns: 100          # show "turn N/100"
"""
from __future__ import annotations

import hashlib
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from litellm.integrations.custom_logger import CustomLogger

logger = logging.getLogger(__name__)


def _session_key(data: dict) -> str:
    """Derive a short key that uniquely identifies a task/conversation.

    Uses the content of the first message (system or user prompt) which
    contains the per-task problem statement — unique across concurrent jobs.
    """
    messages = data.get("messages") or []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            content = "".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        if content:
            return hashlib.sha256(content.encode()).hexdigest()[:8]
    return "unknown"


@dataclass
class _Session:
    count: int = 0
    last_time: float = field(default_factory=time.monotonic)


class Interceptor(CustomLogger):
    """Counts LLM requests per task and logs progress."""

    def __init__(
        self,
        *,
        every: int = 1,
        max_turns: int | None = None,
    ) -> None:
        super().__init__()
        self._every = max(every, 1)
        self._max = max_turns
        self._sessions: dict[str, _Session] = {}
        self._lock = threading.Lock()

    async def async_pre_call_hook(  # type: ignore[override]
        self,
        user_api_key_dict: Any,
        cache: Any,
        data: dict,
        call_type: str,
    ) -> dict | None:
        key = _session_key(data)

        with self._lock:
            sess = self._sessions.setdefault(key, _Session())
            sess.count += 1
            n = sess.count
            now = time.monotonic()
            dt = now - sess.last_time if sess.count > 1 else 0.0
            sess.last_time = now
            active = len(self._sessions)

        if n % self._every == 0 or n == 1:
            model = data.get("model", "?")
            cap = f"/{self._max}" if self._max else ""
            elapsed = f" (+{dt:.1f}s)" if dt > 0 else ""
            logger.info(
                "task %s turn %d%s%s [%s] (%d active)",
                key, n, cap, elapsed, model, active,
            )

        return None
