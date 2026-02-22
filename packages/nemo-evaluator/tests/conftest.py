# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest.mock import AsyncMock, patch

import pytest


@pytest.fixture(autouse=True)
def _disable_telemetry(monkeypatch):
    """Disable telemetry for all unit tests to avoid emitting events to prod.

    Two layers of protection:
    1. Set NEMO_EVALUATOR_TELEMETRY_ENABLED=false so telemetry is disabled by default.
    2. Mock _send_events as a no-op so even tests that explicitly enable telemetry
       (e.g. to test enqueue behavior) cannot make real HTTP requests.
    """
    monkeypatch.setenv("NEMO_EVALUATOR_TELEMETRY_ENABLED", "false")
    with patch(
        "nemo_evaluator.telemetry.TelemetryHandler._send_events",
        new_callable=AsyncMock,
    ):
        yield
