# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
Telemetry handler for NeMo Evaluator.

Environment variables:
- NEMO_EVALUATOR_TELEMETRY_ENABLED: Whether telemetry is enabled (default: true).
- NEMO_EVALUATOR_TELEMETRY_SESSION_ID: Session ID for correlating events across components.
- NEMO_EVALUATOR_TELEMETRY_ENDPOINT: The endpoint to send the telemetry events to.
"""

from __future__ import annotations

import asyncio
import os
import platform
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, ClassVar

from pydantic import BaseModel, Field

from nemo_evaluator.logging import get_logger

logger = get_logger(__name__)

# Environment variable for session ID propagation
SESSION_ID_ENV_VAR = "NEMO_EVALUATOR_TELEMETRY_SESSION_ID"

# Telemetry configuration
CLIENT_ID = "14399890258381784"
# Maps to "eventSysVer" in the payload — identifies the telemetry client version.
NEMO_TELEMETRY_VERSION = "nemo-telemetry/1.0"
MAX_RETRIES = 3
NEMO_TELEMETRY_ENDPOINT = os.getenv(
    "NEMO_EVALUATOR_TELEMETRY_ENDPOINT", "https://events.telemetry.data.nvidia.com/v1.1/events/json"
)
CPU_ARCHITECTURE = platform.uname().machine

# Track if notification has been shown this session
_notification_shown = False


def _get_httpx():
    """Lazy import httpx to minimize startup latency."""
    import httpx

    return httpx


def is_telemetry_enabled() -> bool:
    """Check if telemetry is enabled via environment variable."""
    return os.getenv("NEMO_EVALUATOR_TELEMETRY_ENABLED", "true").lower() in ("1", "true", "yes")


def generate_session_id() -> str:
    """Generate a new UUID session ID."""
    return str(uuid.uuid4())


def get_session_id() -> str:
    """Get session ID from environment or generate a new one."""
    return os.getenv(SESSION_ID_ENV_VAR) or generate_session_id()


def show_telemetry_notification() -> None:
    """Show telemetry notification to the user (only once per session)."""
    global _notification_shown
    if not _notification_shown:
        logger.info("Telemetry enabled. Set NEMO_EVALUATOR_TELEMETRY_ENABLED=false to disable.")
        _notification_shown = True


class TaskStatusEnum(str, Enum):
    """Status of an evaluation task. Event is collected anonymously."""

    STARTED = "started"
    SUCCESS = "success"
    FAILURE = "failure"


class TelemetryEvent(BaseModel):
    """Base class for telemetry events."""

    _event_name: ClassVar[str]  # Subclasses must define this
    # Maps to "eventSchemaVer" in the payload — versioning for the event data structure.
    _schema_version: ClassVar[str] = "1.0"

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "_event_name" not in cls.__dict__:
            raise TypeError(f"{cls.__name__} must define '_event_name' class variable")


class EvaluationTaskEvent(TelemetryEvent):
    """
    Telemetry event for evaluation task execution.

    All fields are collected anonymously for understanding usage patterns.
    """

    _event_name: ClassVar[str] = "EvaluationTaskEvent"

    task: str = Field(
        ...,
        description="The task/benchmark name. Event is collected anonymously.",
    )
    framework_name: str = Field(
        ...,
        alias="frameworkName",
        description="The evaluation framework name (lm-eval, helm, etc.). Event is collected anonymously.",
    )
    model: str = Field(
        ...,
        description="The model name being evaluated. Event is collected anonymously.",
    )
    execution_duration_seconds: float = Field(
        default=0.0,
        alias="executionDurationSeconds",
        description="Duration of the evaluation in seconds. Event is collected anonymously.",
    )
    status: TaskStatusEnum = Field(
        ...,
        description="The status of the task (started/success/failure). Event is collected anonymously.",
    )

    model_config = {"populate_by_name": True}


@dataclass
class QueuedEvent:
    """Container for queued telemetry events with metadata."""

    event: TelemetryEvent
    timestamp: datetime
    retry_count: int = 0


def _get_iso_timestamp(dt: datetime | None = None) -> str:
    """Format datetime as ISO timestamp string."""
    if dt is None:
        dt = datetime.now(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def build_payload(
    events: list[QueuedEvent], *, source_client_version: str, session_id: str = "undefined"
) -> dict[str, Any]:
    """Build the telemetry payload for sending to the endpoint."""
    return {
        "browserType": "undefined",  # do not change
        "clientId": CLIENT_ID,
        "clientType": "Native",  # do not change
        "clientVariant": "Release",  # do not change
        "clientVer": source_client_version,
        "cpuArchitecture": CPU_ARCHITECTURE,
        "deviceGdprBehOptIn": "None",  # do not change
        "deviceGdprFuncOptIn": "None",  # do not change
        "deviceGdprTechOptIn": "None",  # do not change
        "deviceId": "undefined",  # do not change
        "deviceMake": "undefined",  # do not change
        "deviceModel": "undefined",  # do not change
        "deviceOS": "undefined",  # do not change
        "deviceOSVersion": "undefined",  # do not change
        "deviceType": "undefined",  # do not change
        "eventProtocol": "1.6",  # do not change
        "eventSchemaVer": events[0].event._schema_version,
        "eventSysVer": NEMO_TELEMETRY_VERSION,
        "externalUserId": "undefined",  # do not change
        "gdprBehOptIn": "None",  # do not change
        "gdprFuncOptIn": "None",  # do not change
        "gdprTechOptIn": "None",  # do not change
        "idpId": "undefined",  # do not change
        "integrationId": "undefined",  # do not change
        "productName": "undefined",  # do not change
        "productVersion": "undefined",  # do not change
        "sentTs": _get_iso_timestamp(),
        "sessionId": session_id,
        "userId": "undefined",  # do not change
        "events": [
            {
                "ts": _get_iso_timestamp(queued.timestamp),
                "parameters": queued.event.model_dump(by_alias=True),
                "name": queued.event._event_name,
            }
            for queued in events
        ],
    }


class TelemetryHandler:
    """
    Handles telemetry event batching, flushing, and retry logic.

    Args:
        flush_interval_seconds: The interval in seconds to flush the events.
        max_queue_size: The maximum number of events to queue before flushing.
        max_retries: The maximum number of times to retry sending an event.
        source_client_version: The version of the source client (package version).
        session_id: An optional session ID to associate with the events.
    """

    def __init__(
        self,
        flush_interval_seconds: float = 120.0,
        max_queue_size: int = 50,
        max_retries: int = MAX_RETRIES,
        source_client_version: str = "undefined",
        session_id: str = "undefined",
    ):
        self._flush_interval = flush_interval_seconds
        self._max_queue_size = max_queue_size
        self._max_retries = max_retries
        self._events: list[QueuedEvent] = []
        self._dlq: list[QueuedEvent] = []  # Dead letter queue for retry
        self._flush_signal = asyncio.Event()
        self._timer_task: asyncio.Task | None = None
        self._running = False
        self._source_client_version = source_client_version
        self._session_id = session_id

    async def _astart(self) -> None:
        """Start the telemetry handler asynchronously."""
        if self._running:
            return
        self._running = True
        self._timer_task = asyncio.create_task(self._timer_loop())

    async def _astop(self) -> None:
        """Stop the telemetry handler asynchronously."""
        self._running = False
        self._flush_signal.set()
        if self._timer_task:
            self._timer_task.cancel()
            try:
                await self._timer_task
            except asyncio.CancelledError:
                pass
            self._timer_task = None
        await self._flush_events()

    def start(self) -> None:
        """Start the telemetry handler synchronously."""
        logger.debug(
            "Telemetry handler starting",
            endpoint=NEMO_TELEMETRY_ENDPOINT, session_id=self._session_id,
            client_id=CLIENT_ID, version=self._source_client_version,
        )
        self._run_sync(self._astart())

    def stop(self) -> None:
        """Stop the telemetry handler synchronously."""
        logger.debug(
            "Telemetry handler stopping",
            pending=len(self._events), dlq=len(self._dlq),
        )
        self._run_sync(self._astop())

    def flush(self) -> None:
        """Trigger a flush synchronously."""
        self._flush_signal.set()

    def enqueue(self, event: TelemetryEvent) -> None:
        """Add an event to the queue for sending."""
        if not is_telemetry_enabled():
            logger.debug("Telemetry disabled, skipping event", event_type=type(event).__name__)
            return
        if not isinstance(event, TelemetryEvent):
            logger.debug("Ignoring non-TelemetryEvent object", event_type=type(event).__name__)
            return
        queued = QueuedEvent(event=event, timestamp=datetime.now(timezone.utc))
        self._events.append(queued)
        logger.debug("Enqueued telemetry event", event_name=event._event_name, queue_size=len(self._events))
        if len(self._events) >= self._max_queue_size:
            self._flush_signal.set()

    def _run_sync(self, coro: Any) -> Any:
        """Run a coroutine synchronously."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        else:
            return asyncio.run(coro)

    def __enter__(self) -> TelemetryHandler:
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()

    async def _timer_loop(self) -> None:
        """Background loop that flushes events periodically."""
        while self._running:
            try:
                await asyncio.wait_for(
                    self._flush_signal.wait(),
                    timeout=self._flush_interval,
                )
            except asyncio.TimeoutError:
                pass
            self._flush_signal.clear()
            await self._flush_events()

    async def _flush_events(self) -> None:
        """Flush all pending events."""
        dlq_events, self._dlq = self._dlq, []
        new_events, self._events = self._events, []
        events_to_send = dlq_events + new_events
        if events_to_send:
            await self._send_events(events_to_send)

    async def _send_events(self, events: list[QueuedEvent]) -> None:
        """Send events to the telemetry endpoint."""
        logger.debug(
            "Sending telemetry events",
            count=len(events), endpoint=NEMO_TELEMETRY_ENDPOINT, session_id=self._session_id,
        )
        try:
            httpx = _get_httpx()
            async with httpx.AsyncClient() as client:
                await self._send_events_with_client(client, events)
        except Exception as e:
            logger.debug("Telemetry send failed (events moved to DLQ)", error=str(e))
            self._add_to_dlq(events)

    async def _send_events_with_client(self, client: Any, events: list[QueuedEvent]) -> None:
        """Send events using the provided HTTP client."""
        if not events:
            return

        payload = build_payload(
            events, source_client_version=self._source_client_version, session_id=self._session_id
        )
        try:
            response = await client.post(NEMO_TELEMETRY_ENDPOINT, json=payload)
            # 2xx, 400, 422 are all considered complete (no retry)
            # 400/422 indicate bad payload which retrying won't fix
            if response.is_success:
                logger.debug("Telemetry events sent successfully", status=response.status_code)
                return
            if response.status_code in (400, 422):
                logger.debug(
                    "Telemetry endpoint rejected payload",
                    status=response.status_code, response=response.text[:200],
                )
                return
            # 413 (payload too large) - split and retry
            if response.status_code == 413:
                logger.debug("Telemetry payload too large (HTTP 413), splitting", count=len(events))
                if len(events) == 1:
                    # Can't split further, drop the event
                    return
                mid = len(events) // 2
                await self._send_events_with_client(client, events[:mid])
                await self._send_events_with_client(client, events[mid:])
                return
            if response.status_code == 408 or response.status_code >= 500:
                logger.debug(
                    "Telemetry endpoint error, events moved to DLQ for retry",
                    status=response.status_code,
                )
                self._add_to_dlq(events)
        except Exception as e:
            logger.debug("Telemetry HTTP request failed (events moved to DLQ)", error=str(e))
            self._add_to_dlq(events)

    def _add_to_dlq(self, events: list[QueuedEvent]) -> None:
        """Add events to the dead letter queue for retry."""
        for queued in events:
            queued.retry_count += 1
            if queued.retry_count > self._max_retries:
                logger.debug(
                    "Dropping telemetry event after max retries",
                    event_name=queued.event._event_name, max_retries=self._max_retries,
                )
                continue
            self._dlq.append(queued)
