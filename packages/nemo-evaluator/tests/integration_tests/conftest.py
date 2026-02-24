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

"""Override parent conftest: enable telemetry with real HTTP sends to staging.

Send failures cause test failures by default. Set TELEMETRY_ALLOW_SEND_FAILURES=1
to suppress (e.g. when running without network access).
"""

import os
from unittest.mock import patch

import pytest

NEMO_TELEMETRY_STAGING_ENDPOINT = (
    "https://events.telemetry.data-uat.nvidia.com/v1.1/events/json"
)


class _SendTracker:
    """Wraps TelemetryHandler._send_events to detect failures."""

    def __init__(self):
        self.failures: list[str] = []

    def patch_handler_class(self):
        """Monkey-patch TelemetryHandler to track send failures."""
        from nemo_evaluator.telemetry import TelemetryHandler

        original_add_to_dlq = TelemetryHandler._add_to_dlq

        tracker = self

        def _tracking_add_to_dlq(handler_self, events):
            tracker.failures.append(f"{len(events)} event(s) moved to DLQ")
            return original_add_to_dlq(handler_self, events)

        return patch.object(TelemetryHandler, "_add_to_dlq", _tracking_add_to_dlq)


@pytest.fixture(autouse=True)
def _disable_telemetry(monkeypatch):
    """Override parent: enable telemetry, point to staging, allow real sends."""
    monkeypatch.setenv("NEMO_EVALUATOR_TELEMETRY_LEVEL", "2")
    monkeypatch.setenv(
        "NEMO_EVALUATOR_TELEMETRY_ENDPOINT", NEMO_TELEMETRY_STAGING_ENDPOINT
    )
    tracker = _SendTracker()
    with (
        patch(
            "nemo_evaluator.telemetry.NEMO_TELEMETRY_ENDPOINT",
            NEMO_TELEMETRY_STAGING_ENDPOINT,
        ),
        tracker.patch_handler_class(),
    ):
        yield

    allow_failures = os.environ.get("TELEMETRY_ALLOW_SEND_FAILURES", "0") == "1"
    if not allow_failures and tracker.failures:
        pytest.fail(
            f"Telemetry send failures: {tracker.failures}. "
            "Set TELEMETRY_ALLOW_SEND_FAILURES=1 to suppress."
        )
