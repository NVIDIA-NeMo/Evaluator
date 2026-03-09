"""Opt-in telemetry for usage analytics.

Controlled by:
  - NEMO_EVALUATOR_TELEMETRY_LEVEL env var (off/minimal/default)
  - Persistent config: telemetry.level
  - Persistent config: telemetry.endpoint
"""
from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from enum import IntEnum
from queue import Empty, Queue
from typing import Any

logger = logging.getLogger(__name__)


class TelemetryLevel(IntEnum):
    OFF = 0
    MINIMAL = 1
    DEFAULT = 2


@dataclass
class TelemetryEvent:
    event_type: str
    timestamp: float = field(default_factory=time.time)
    task: str = ""
    framework: str = ""
    model: str = ""
    duration_s: float = 0.0
    status: str = ""
    samples: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


class TelemetryHandler:
    """Background telemetry sender. Events are queued and flushed asynchronously."""

    def __init__(self) -> None:
        self._level = _resolve_level()
        self._endpoint = _resolve_endpoint()
        self._queue: Queue[TelemetryEvent] = Queue(maxsize=1000)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

        if self._level > TelemetryLevel.OFF and self._endpoint:
            self._thread = threading.Thread(target=self._flush_loop, daemon=True)
            self._thread.start()

    @property
    def level(self) -> TelemetryLevel:
        return self._level

    def emit(self, event: TelemetryEvent) -> None:
        if self._level == TelemetryLevel.OFF:
            return

        if self._level == TelemetryLevel.MINIMAL:
            from dataclasses import replace
            event = replace(event, model="", extra={})

        try:
            self._queue.put_nowait(event)
        except Exception:
            pass

    def shutdown(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5.0)

    def _flush_loop(self) -> None:
        import httpx

        while not self._stop.is_set():
            batch: list[dict] = []
            try:
                while len(batch) < 50:
                    ev = self._queue.get(timeout=5.0)
                    batch.append(asdict(ev))
            except Empty:
                pass

            if batch and self._endpoint:
                try:
                    httpx.post(
                        self._endpoint,
                        json={"events": batch},
                        timeout=10.0,
                    )
                except Exception as e:
                    logger.debug("Telemetry flush failed: %s", e)


_handler: TelemetryHandler | None = None


def get_telemetry() -> TelemetryHandler:
    global _handler
    if _handler is None:
        _handler = TelemetryHandler()
    return _handler


def emit(event_type: str, **kwargs: Any) -> None:
    get_telemetry().emit(TelemetryEvent(event_type=event_type, **kwargs))


def _resolve_level() -> TelemetryLevel:
    env_val = os.environ.get("NEMO_EVALUATOR_TELEMETRY_LEVEL", "").lower()
    if env_val == "off":
        return TelemetryLevel.OFF
    if env_val == "minimal":
        return TelemetryLevel.MINIMAL
    if env_val == "default":
        return TelemetryLevel.DEFAULT

    try:
        from nemo_evaluator.cli.config_cmd import get_persistent_defaults
        cfg = get_persistent_defaults()
        level_str = cfg.get("telemetry", {}).get("level", "off")
        return TelemetryLevel[level_str.upper()]
    except Exception:
        return TelemetryLevel.OFF


def _resolve_endpoint() -> str | None:
    env_val = os.environ.get("NEMO_EVALUATOR_TELEMETRY_ENDPOINT")
    if env_val:
        return env_val

    try:
        from nemo_evaluator.cli.config_cmd import get_persistent_defaults
        cfg = get_persistent_defaults()
        return cfg.get("telemetry", {}).get("endpoint")
    except Exception:
        return None
