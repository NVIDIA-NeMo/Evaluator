# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

from unittest.mock import AsyncMock, patch

import pytest

NEMO_TELEMETRY_STAGING_ENDPOINT = (
    "https://events.telemetry.data-uat.nvidia.com/v1.1/events/json"
)


@pytest.fixture(autouse=True)
def _disable_telemetry(monkeypatch):
    """Disable telemetry for all tests to avoid emitting events to prod.

    Three layers of protection:
    1. Set NEMO_EVALUATOR_TELEMETRY_LEVEL=0 so telemetry is disabled by default.
    2. Point the endpoint to staging so even if events leak they don't hit prod.
    3. Mock _send_events as a no-op so even tests that explicitly enable telemetry
       (e.g. to test enqueue behavior) cannot make real HTTP requests.
    """
    monkeypatch.setenv("NEMO_EVALUATOR_TELEMETRY_LEVEL", "0")
    monkeypatch.setenv(
        "NEMO_EVALUATOR_TELEMETRY_ENDPOINT", NEMO_TELEMETRY_STAGING_ENDPOINT
    )
    with (
        patch(
            "nemo_evaluator.telemetry.NEMO_TELEMETRY_ENDPOINT",
            NEMO_TELEMETRY_STAGING_ENDPOINT,
        ),
        patch(
            "nemo_evaluator.telemetry.TelemetryHandler._send_events",
            new_callable=AsyncMock,
        ),
    ):
        yield
