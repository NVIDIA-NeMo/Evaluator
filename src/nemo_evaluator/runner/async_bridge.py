"""Bridge for calling async code from synchronous contexts.

External harnesses (lm-eval, simple-evals) call our model through sync
callbacks. Our ModelClient is async. This module provides a safe bridge
that avoids the pitfalls of asyncio.run() inside an already-running loop.

A single background event loop is spun up once and reused for all calls.
"""
from __future__ import annotations

import asyncio
import threading
from typing import Any, Coroutine, TypeVar

T = TypeVar("T")

_lock = threading.Lock()
_loop: asyncio.AbstractEventLoop | None = None
_thread: threading.Thread | None = None


def _get_loop() -> asyncio.AbstractEventLoop:
    global _loop, _thread
    with _lock:
        if _loop is None or _loop.is_closed():
            _loop = asyncio.new_event_loop()
            _thread = threading.Thread(target=_loop.run_forever, daemon=True, name="nel-async-bridge")
            _thread.start()
    return _loop


def run_async(coro: Coroutine[Any, Any, T]) -> T:
    """Run an async coroutine from synchronous code, safely.

    If no event loop is running: uses asyncio.run() directly (simplest path).
    If an event loop IS running (Jupyter, FastAPI, etc): dispatches to a
    dedicated background loop via run_coroutine_threadsafe, avoiding nested
    event loops entirely.
    """
    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        running = None

    if running is None:
        return asyncio.run(coro)

    bg = _get_loop()
    future = asyncio.run_coroutine_threadsafe(coro, bg)
    return future.result()
