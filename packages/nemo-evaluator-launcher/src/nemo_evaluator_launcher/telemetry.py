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
Telemetry handler for NeMo Evaluator Launcher.

This module re-exports shared telemetry infrastructure from nemo_evaluator.telemetry
and defines launcher-specific event types.

Environment variables:
- NEMO_TELEMETRY_ENABLED: Whether telemetry is enabled (default: true).
- NEMO_TELEMETRY_SESSION_ID: Session ID for correlating events across components.
- NEMO_TELEMETRY_ENDPOINT: The endpoint to send the telemetry events to.
"""

from __future__ import annotations

from enum import Enum
from typing import ClassVar, List

from pydantic import Field

# Re-export shared telemetry infrastructure from nemo_evaluator
from nemo_evaluator.telemetry import (
    SESSION_ID_ENV_VAR,
    QueuedEvent,
    TelemetryEvent,
    TelemetryHandler,
    build_payload,
    generate_session_id,
    get_session_id,
    is_telemetry_enabled,
    show_telemetry_notification,
)

__all__ = [
    # Re-exported from nemo_evaluator.telemetry
    "SESSION_ID_ENV_VAR",
    "QueuedEvent",
    "TelemetryEvent",
    "TelemetryHandler",
    "build_payload",
    "generate_session_id",
    "get_session_id",
    "is_telemetry_enabled",
    "show_telemetry_notification",
    # Launcher-specific
    "JobStatusEnum",
    "LauncherJobEvent",
]


class JobStatusEnum(str, Enum):
    """Status of a launcher job. Event is collected anonymously."""

    STARTED = "started"
    SUCCESS = "success"
    FAILURE = "failure"


class LauncherJobEvent(TelemetryEvent):
    """
    Telemetry event for launcher job execution.

    All fields are collected anonymously for understanding usage patterns.
    """

    _event_name: ClassVar[str] = "launcher_job"

    executor_type: str = Field(
        ...,
        alias="executorType",
        description="The executor type used (local/slurm/lepton). Event is collected anonymously.",
    )
    deployment_type: str = Field(
        ...,
        alias="deploymentType",
        description="The deployment type (none/vllm/sglang/nim). Event is collected anonymously.",
    )
    model: str = Field(
        ...,
        description="The model name being evaluated. Event is collected anonymously.",
    )
    tasks: List[str] = Field(
        default_factory=list,
        description="List of task names requested. Event is collected anonymously.",
    )
    exporters: List[str] = Field(
        default_factory=list,
        description="List of exporters used. Event is collected anonymously.",
    )
    status: JobStatusEnum = Field(
        ...,
        description="The status of the job (started/success/failure). Event is collected anonymously.",
    )

    model_config = {"populate_by_name": True}
