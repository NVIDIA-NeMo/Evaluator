# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator
from urllib.parse import urlsplit, urlunsplit


@dataclass(frozen=True)
class AttemptContext:
    adapter_session_id: str | None = None


_current_attempt: ContextVar[AttemptContext | None] = ContextVar("nel_attempt_context", default=None)


@contextmanager
def attempt_context(ctx: AttemptContext | None) -> Iterator[None]:
    token = _current_attempt.set(ctx)
    try:
        yield
    finally:
        _current_attempt.reset(token)


def url_for_current_adapter_session(url: str) -> str:
    ctx = _current_attempt.get()
    session_id = ctx.adapter_session_id if ctx else None
    if not session_id:
        return url
    parts = urlsplit(url)
    if not parts.scheme or not parts.netloc or parts.path.startswith("/s/"):
        return url
    path = parts.path or ""
    return urlunsplit((parts.scheme, parts.netloc, f"/s/{session_id}{path}", parts.query, parts.fragment))
