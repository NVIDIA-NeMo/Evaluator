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
"""Unit tests for the telemetry module."""

import os
import uuid
from unittest.mock import patch

import pytest


class TestTelemetryEnabled:
    """Tests for is_telemetry_enabled function."""

    def test_telemetry_enabled_default(self):
        """Test that telemetry is enabled by default."""
        from nemo_evaluator.telemetry import is_telemetry_enabled

        with patch.dict(os.environ, {}, clear=True):
            # Remove the env var if it exists
            os.environ.pop("NEMO_TELEMETRY_ENABLED", None)
            # Re-import to get fresh state
            import importlib
            import nemo_evaluator.telemetry as telemetry_module
            importlib.reload(telemetry_module)
            assert telemetry_module.is_telemetry_enabled() is True

    def test_telemetry_disabled_via_env(self):
        """Test that telemetry can be disabled via environment variable."""
        from nemo_evaluator.telemetry import is_telemetry_enabled

        with patch.dict(os.environ, {"NEMO_TELEMETRY_ENABLED": "false"}):
            assert is_telemetry_enabled() is False

        with patch.dict(os.environ, {"NEMO_TELEMETRY_ENABLED": "0"}):
            assert is_telemetry_enabled() is False

        with patch.dict(os.environ, {"NEMO_TELEMETRY_ENABLED": "no"}):
            assert is_telemetry_enabled() is False

    def test_telemetry_enabled_explicit(self):
        """Test that telemetry can be explicitly enabled."""
        from nemo_evaluator.telemetry import is_telemetry_enabled

        with patch.dict(os.environ, {"NEMO_TELEMETRY_ENABLED": "true"}):
            assert is_telemetry_enabled() is True

        with patch.dict(os.environ, {"NEMO_TELEMETRY_ENABLED": "1"}):
            assert is_telemetry_enabled() is True

        with patch.dict(os.environ, {"NEMO_TELEMETRY_ENABLED": "yes"}):
            assert is_telemetry_enabled() is True


class TestSessionId:
    """Tests for session ID generation and retrieval."""

    def test_session_id_generation(self):
        """Test that generated session IDs are valid UUIDs."""
        from nemo_evaluator.telemetry import generate_session_id

        session_id = generate_session_id()
        # Should be a valid UUID
        uuid.UUID(session_id)  # Raises ValueError if invalid

    def test_session_id_uniqueness(self):
        """Test that each call generates a unique session ID."""
        from nemo_evaluator.telemetry import generate_session_id

        ids = [generate_session_id() for _ in range(100)]
        assert len(set(ids)) == 100

    def test_get_session_id_from_env(self):
        """Test that session ID is read from environment variable."""
        from nemo_evaluator.telemetry import (
            SESSION_ID_ENV_VAR,
            get_session_id,
        )

        test_id = "test-session-123"
        with patch.dict(os.environ, {SESSION_ID_ENV_VAR: test_id}):
            assert get_session_id() == test_id

    def test_get_session_id_generates_new(self):
        """Test that a new session ID is generated when not in env."""
        from nemo_evaluator.telemetry import (
            SESSION_ID_ENV_VAR,
            get_session_id,
        )

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop(SESSION_ID_ENV_VAR, None)
            session_id = get_session_id()
            # Should be a valid UUID
            uuid.UUID(session_id)


class TestEventSerialization:
    """Tests for event serialization."""

    def test_evaluation_task_event_serialization(self):
        """Test that EvaluationTaskEvent serializes with correct camelCase aliases."""
        from nemo_evaluator.telemetry import EvaluationTaskEvent, TaskStatusEnum

        event = EvaluationTaskEvent(
            task="mmlu",
            eval_harness="lm-eval",
            model="llama-3.1-8b",
            execution_duration_seconds=123.45,
            task_status=TaskStatusEnum.SUCCESS,
        )

        serialized = event.model_dump(by_alias=True)
        assert "evalHarness" in serialized
        assert "executionDurationSeconds" in serialized
        assert "taskStatus" in serialized
        assert serialized["task"] == "mmlu"
        assert serialized["evalHarness"] == "lm-eval"
        assert serialized["model"] == "llama-3.1-8b"
        assert serialized["executionDurationSeconds"] == 123.45
        assert serialized["taskStatus"] == "success"

    def test_evaluation_task_event_started_without_duration(self):
        """Test that STARTED events can be created without execution_duration_seconds."""
        from nemo_evaluator.telemetry import EvaluationTaskEvent, TaskStatusEnum

        event = EvaluationTaskEvent(
            task="mmlu",
            eval_harness="unknown",
            model="llama-3.1-8b",
            task_status=TaskStatusEnum.STARTED,
        )

        serialized = event.model_dump(by_alias=True)
        assert serialized["taskStatus"] == "started"
        assert serialized["executionDurationSeconds"] is None

    def test_task_status_values(self):
        """Test that TaskStatusEnum has expected values."""
        from nemo_evaluator.telemetry import TaskStatusEnum

        assert TaskStatusEnum.STARTED.value == "started"
        assert TaskStatusEnum.SUCCESS.value == "success"
        assert TaskStatusEnum.FAILURE.value == "failure"


class TestTelemetryHandler:
    """Tests for TelemetryHandler."""

    def test_handler_initialization(self):
        """Test that TelemetryHandler initializes correctly."""
        from nemo_evaluator.telemetry import TelemetryHandler

        handler = TelemetryHandler(
            source_client_version="1.0.0",
            session_id="test-session",
        )
        assert handler._source_client_version == "1.0.0"
        assert handler._session_id == "test-session"
        assert handler._events == []
        assert handler._running is False

    def test_handler_enqueue_when_disabled(self):
        """Test that events are not enqueued when telemetry is disabled."""
        from nemo_evaluator.telemetry import (
            EvaluationTaskEvent,
            TaskStatusEnum,
            TelemetryHandler,
        )

        with patch.dict(os.environ, {"NEMO_TELEMETRY_ENABLED": "false"}):
            handler = TelemetryHandler()
            event = EvaluationTaskEvent(
                task="test",
                eval_harness="lm-eval",
                model="test-model",
                execution_duration_seconds=1.0,
                task_status=TaskStatusEnum.SUCCESS,
            )
            handler.enqueue(event)
            assert len(handler._events) == 0

    def test_handler_enqueue_when_enabled(self):
        """Test that events are enqueued when telemetry is enabled."""
        from nemo_evaluator.telemetry import (
            EvaluationTaskEvent,
            TaskStatusEnum,
            TelemetryHandler,
        )

        with patch.dict(os.environ, {"NEMO_TELEMETRY_ENABLED": "true"}):
            handler = TelemetryHandler()
            event = EvaluationTaskEvent(
                task="test",
                eval_harness="lm-eval",
                model="test-model",
                execution_duration_seconds=1.0,
                task_status=TaskStatusEnum.SUCCESS,
            )
            handler.enqueue(event)
            assert len(handler._events) == 1

    def test_handler_graceful_degradation(self):
        """Test that handler silently fails on invalid events."""
        from nemo_evaluator.telemetry import TelemetryHandler

        with patch.dict(os.environ, {"NEMO_TELEMETRY_ENABLED": "true"}):
            handler = TelemetryHandler()
            # Enqueue something that's not a TelemetryEvent
            handler.enqueue("not an event")  # type: ignore
            assert len(handler._events) == 0


class TestBuildPayload:
    """Tests for payload building."""

    def test_build_payload_structure(self):
        """Test that build_payload creates correct structure."""
        from datetime import datetime, timezone

        from nemo_evaluator.telemetry import (
            EvaluationTaskEvent,
            QueuedEvent,
            TaskStatusEnum,
            build_payload,
        )

        event = EvaluationTaskEvent(
            task="mmlu",
            eval_harness="lm-eval",
            model="test-model",
            execution_duration_seconds=60.5,
            task_status=TaskStatusEnum.SUCCESS,
        )
        queued = QueuedEvent(event=event, timestamp=datetime.now(timezone.utc))

        payload = build_payload(
            [queued],
            source_client_version="1.0.0",
            session_id="test-session",
        )

        assert payload["clientVer"] == "1.0.0"
        assert payload["sessionId"] == "test-session"
        assert len(payload["events"]) == 1
        assert payload["events"][0]["name"] == "EvaluationTaskEvent"
        assert payload["eventSchemaVer"] == "1.0"

    def test_build_payload_event_parameters(self):
        """Test that event parameters are correctly serialized in payload."""
        from datetime import datetime, timezone

        from nemo_evaluator.telemetry import (
            EvaluationTaskEvent,
            QueuedEvent,
            TaskStatusEnum,
            build_payload,
        )

        event = EvaluationTaskEvent(
            task="ifeval",
            eval_harness="helm",
            model="gpt-4",
            execution_duration_seconds=120.0,
            task_status=TaskStatusEnum.FAILURE,
        )
        queued = QueuedEvent(event=event, timestamp=datetime.now(timezone.utc))

        payload = build_payload(
            [queued],
            source_client_version="2.0.0",
            session_id="session-xyz",
        )

        event_params = payload["events"][0]["parameters"]
        assert event_params["task"] == "ifeval"
        assert event_params["evalHarness"] == "helm"
        assert event_params["model"] == "gpt-4"
        assert event_params["executionDurationSeconds"] == 120.0
        assert event_params["taskStatus"] == "failure"
