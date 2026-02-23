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
Integration tests that exercise run_eval() with real telemetry to the staging endpoint.

Only the executor is mocked — telemetry flows through the real pipeline:
run_eval() -> TelemetryHandler -> enqueue -> flush -> HTTP POST -> staging.

Run with:
    cd packages/nemo-evaluator-launcher
    uv run --group dev pytest tests/integration_tests/ -v --enable-network

After running, verify events appear in the staging telemetry dashboard.
"""

from unittest.mock import MagicMock, patch

import pytest
from omegaconf import OmegaConf

from nemo_evaluator_launcher.api.functional import run_eval


def _make_config(model_id="integration-test-model"):
    """Create a minimal OmegaConf config for run_eval()."""
    return OmegaConf.create(
        {
            "deployment": {"type": "none"},
            "execution": {"type": "local"},
            "target": {
                "api_endpoint": {
                    "model_id": model_id,
                    "url": "https://test.example.com/v1/chat/completions",
                }
            },
            "evaluation": {
                "tasks": [{"name": "integration-test-task"}],
            },
        }
    )


def test_run_eval_sends_telemetry_at_default_level():
    """Full run_eval() call at level 2 — events include model name."""
    mock_executor = MagicMock()
    mock_executor.execute_eval.return_value = "inv-123"

    with patch(
        "nemo_evaluator_launcher.api.functional.get_executor",
        return_value=mock_executor,
    ):
        run_eval(_make_config())


def test_run_eval_sends_telemetry_at_minimal_level(monkeypatch):
    """Full run_eval() call at level 1 — model should be 'redacted'."""
    monkeypatch.setenv("NEMO_EVALUATOR_TELEMETRY_LEVEL", "1")
    mock_executor = MagicMock()
    mock_executor.execute_eval.return_value = "inv-456"

    with patch(
        "nemo_evaluator_launcher.api.functional.get_executor",
        return_value=mock_executor,
    ):
        run_eval(_make_config())


def test_run_eval_no_telemetry_on_dry_run():
    """Dry run should not emit any telemetry events."""
    mock_executor = MagicMock()
    mock_executor.execute_eval.return_value = "inv-789"

    with patch(
        "nemo_evaluator_launcher.api.functional.get_executor",
        return_value=mock_executor,
    ):
        run_eval(_make_config(), dry_run=True)
