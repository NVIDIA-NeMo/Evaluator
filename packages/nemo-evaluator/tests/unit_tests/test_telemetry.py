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
from unittest.mock import MagicMock, patch

import pytest


class TestTelemetryLevel:
    """Tests for get_telemetry_level function."""

    def test_default_level_is_full(self):
        """Test that default telemetry level is DEFAULT (2) when nothing is set."""
        from nemo_evaluator.config import TelemetryLevel
        from nemo_evaluator.telemetry import get_telemetry_level

        mock_path = MagicMock()
        mock_path.exists.return_value = False
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("NEMO_EVALUATOR_TELEMETRY_LEVEL", None)
            with patch("nemo_evaluator.config.CONFIG_FILE", mock_path):
                assert get_telemetry_level() == TelemetryLevel.DEFAULT

    def test_level_0_disables_telemetry(self):
        """Test that level 0 disables telemetry."""
        from nemo_evaluator.config import TelemetryLevel
        from nemo_evaluator.telemetry import get_telemetry_level

        with patch.dict(os.environ, {"NEMO_EVALUATOR_TELEMETRY_LEVEL": "0"}):
            assert get_telemetry_level() == TelemetryLevel.OFF

    def test_level_1_is_minimal(self):
        """Test that level 1 enables minimal telemetry (no model_id)."""
        from nemo_evaluator.config import TelemetryLevel
        from nemo_evaluator.telemetry import get_telemetry_level

        with patch.dict(os.environ, {"NEMO_EVALUATOR_TELEMETRY_LEVEL": "1"}):
            assert get_telemetry_level() == TelemetryLevel.MINIMAL

    def test_level_2_is_full(self):
        """Test that level 2 enables full telemetry."""
        from nemo_evaluator.config import TelemetryLevel
        from nemo_evaluator.telemetry import get_telemetry_level

        with patch.dict(os.environ, {"NEMO_EVALUATOR_TELEMETRY_LEVEL": "2"}):
            assert get_telemetry_level() == TelemetryLevel.DEFAULT

    def test_invalid_value_falls_back_to_minimal(self):
        """Test that invalid values fall back to MINIMAL (1)."""
        from nemo_evaluator.config import TelemetryLevel
        from nemo_evaluator.telemetry import get_telemetry_level

        for invalid in ("true", "false", "yes", "no", "abc", "3", "-1"):
            with patch.dict(os.environ, {"NEMO_EVALUATOR_TELEMETRY_LEVEL": invalid}):
                assert get_telemetry_level() == TelemetryLevel.MINIMAL, (
                    f"Expected MINIMAL for invalid value '{invalid}'"
                )

    def test_config_file_fallback(self, tmp_path):
        """Test that config file is used when env var is not set."""
        from nemo_evaluator.config import (
            NemoEvaluatorConfig,
            TelemetryConfig,
            TelemetryLevel,
            save_config,
        )
        from nemo_evaluator.telemetry import get_telemetry_level

        config_file = tmp_path / "config.yaml"
        save_config(
            NemoEvaluatorConfig(telemetry=TelemetryConfig(level=TelemetryLevel.OFF)),
            path=config_file,
        )
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("NEMO_EVALUATOR_TELEMETRY_LEVEL", None)
            with patch("nemo_evaluator.config.CONFIG_FILE", config_file):
                assert get_telemetry_level() == TelemetryLevel.OFF

    def test_env_var_overrides_config_file(self, tmp_path):
        """Test that env var takes priority over config file."""
        from nemo_evaluator.config import (
            NemoEvaluatorConfig,
            TelemetryConfig,
            TelemetryLevel,
            save_config,
        )
        from nemo_evaluator.telemetry import get_telemetry_level

        config_file = tmp_path / "config.yaml"
        save_config(
            NemoEvaluatorConfig(telemetry=TelemetryConfig(level=TelemetryLevel.OFF)),
            path=config_file,
        )
        with patch.dict(os.environ, {"NEMO_EVALUATOR_TELEMETRY_LEVEL": "2"}):
            with patch("nemo_evaluator.config.CONFIG_FILE", config_file):
                assert get_telemetry_level() == TelemetryLevel.DEFAULT


class TestSessionId:
    """Tests for session ID generation and retrieval."""

    def test_session_id_generation(self):
        """Test that generated session IDs are valid UUIDs."""
        from nemo_evaluator.telemetry import _generate_session_id

        session_id = _generate_session_id()
        assert uuid.UUID(session_id)

    def test_session_id_uniqueness(self):
        """Test that each call generates a unique session ID."""
        from nemo_evaluator.telemetry import _generate_session_id

        ids = [_generate_session_id() for _ in range(100)]
        assert len(set(ids)) == 100

    def test_get_session_id_from_env(self):
        """Test that session ID is read from environment variable."""
        from nemo_evaluator.telemetry import (
            TELEMETRY_SESSION_ID_ENV_VAR,
            get_session_id,
        )

        test_id = "test-session-123"
        with patch.dict(os.environ, {TELEMETRY_SESSION_ID_ENV_VAR: test_id}):
            assert get_session_id() == test_id

    def test_get_session_id_generates_new(self):
        """Test that a new session ID is generated when not in env."""
        from nemo_evaluator.telemetry import (
            TELEMETRY_SESSION_ID_ENV_VAR,
            get_session_id,
        )

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop(TELEMETRY_SESSION_ID_ENV_VAR, None)
            session_id = get_session_id()
            assert uuid.UUID(session_id)


class TestEventSerialization:
    """Tests for event serialization."""

    def test_evaluation_task_event_serialization(self):
        """Test that EvaluationTaskEvent serializes with correct camelCase aliases."""
        from nemo_evaluator.telemetry import EvaluationTaskEvent, StatusEnum

        event = EvaluationTaskEvent(
            task="mmlu",
            framework_name="lm-eval",
            model="llama-3.1-8b",
            execution_duration_seconds=123.45,
            status=StatusEnum.SUCCESS,
        )

        serialized = event.model_dump(by_alias=True)
        assert serialized["task"] == "mmlu"
        assert serialized["frameworkName"] == "lm-eval"
        assert serialized["model"] == "llama-3.1-8b"
        assert serialized["executionDurationSeconds"] == 123.45
        assert serialized["status"] == "success"

    def test_evaluation_task_event_started_without_duration(self):
        """Test that STARTED events can be created without execution_duration_seconds."""
        from nemo_evaluator.telemetry import EvaluationTaskEvent, StatusEnum

        event = EvaluationTaskEvent(
            task="mmlu",
            framework_name="unknown",
            model="llama-3.1-8b",
            status=StatusEnum.STARTED,
        )

        serialized = event.model_dump(by_alias=True)
        assert serialized["status"] == "started"
        assert serialized["executionDurationSeconds"] == 0.0


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
        """Test that events are not enqueued when telemetry level is OFF."""
        from nemo_evaluator.config import TelemetryLevel
        from nemo_evaluator.telemetry import (
            EvaluationTaskEvent,
            StatusEnum,
            TelemetryHandler,
        )

        handler = TelemetryHandler(telemetry_level=TelemetryLevel.OFF)
        event = EvaluationTaskEvent(
            task="test",
            framework_name="lm-eval",
            model="test-model",
            execution_duration_seconds=1.0,
            status=StatusEnum.SUCCESS,
        )
        handler.enqueue(event)
        assert len(handler._events) == 0

    def test_handler_enqueue_when_enabled(self):
        """Test that events are enqueued when telemetry level is DEFAULT."""
        from nemo_evaluator.config import TelemetryLevel
        from nemo_evaluator.telemetry import (
            EvaluationTaskEvent,
            StatusEnum,
            TelemetryHandler,
        )

        handler = TelemetryHandler(telemetry_level=TelemetryLevel.DEFAULT)
        event = EvaluationTaskEvent(
            task="test",
            framework_name="lm-eval",
            model="test-model",
            execution_duration_seconds=1.0,
            status=StatusEnum.SUCCESS,
        )
        handler.enqueue(event)
        assert len(handler._events) == 1

    def test_handler_graceful_degradation(self):
        """Test that handler silently fails on invalid events."""
        from nemo_evaluator.config import TelemetryLevel
        from nemo_evaluator.telemetry import TelemetryHandler

        handler = TelemetryHandler(telemetry_level=TelemetryLevel.DEFAULT)
        handler.enqueue("not an event")  # type: ignore
        assert len(handler._events) == 0


class TestBuildPayload:
    """Tests for payload building."""

    def test_build_payload_empty_events_raises(self):
        """Test that build_payload raises ValueError when events is empty."""
        from nemo_evaluator.telemetry import build_payload

        with pytest.raises(ValueError, match="events must not be empty"):
            build_payload([], session_id="x", source_client_version="1.0")

    def test_build_payload_structure(self):
        """Test that build_payload creates correct structure."""
        from datetime import datetime, timezone

        from nemo_evaluator.telemetry import (
            EvaluationTaskEvent,
            QueuedEvent,
            StatusEnum,
            build_payload,
        )

        event = EvaluationTaskEvent(
            task="mmlu",
            framework_name="lm-eval",
            model="test-model",
            execution_duration_seconds=60.5,
            status=StatusEnum.SUCCESS,
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
            StatusEnum,
            build_payload,
        )

        event = EvaluationTaskEvent(
            task="ifeval",
            framework_name="helm",
            model="gpt-4",
            execution_duration_seconds=120.0,
            status=StatusEnum.FAILURE,
        )
        queued = QueuedEvent(event=event, timestamp=datetime.now(timezone.utc))

        payload = build_payload(
            [queued],
            source_client_version="2.0.0",
            session_id="session-xyz",
        )

        event_params = payload["events"][0]["parameters"]
        assert event_params["task"] == "ifeval"
        assert event_params["frameworkName"] == "helm"
        assert event_params["model"] == "gpt-4"
        assert event_params["executionDurationSeconds"] == 120.0
        assert event_params["status"] == "failure"
