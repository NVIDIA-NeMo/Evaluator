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
Integration tests that exercise evaluate() with real telemetry to the staging endpoint.

Only the evaluation work is mocked — telemetry flows through the real pipeline:
evaluate() -> TelemetryHandler -> enqueue -> flush -> HTTP POST -> staging.

Run with:
    cd packages/nemo-evaluator
    uv run --group test pytest tests/integration_tests/ -v

After running, verify events appear in the staging telemetry dashboard.
"""

from unittest.mock import MagicMock, patch

import pytest

from nemo_evaluator.api.api_dataclasses import (
    ConfigParams,
    Evaluation,
    EvaluationConfig,
    EvaluationTarget,
)
from nemo_evaluator.core.evaluate import evaluate

pytestmark = pytest.mark.network


def _mock_eval(tmp_path):
    """Context managers that mock the expensive evaluation work."""
    return (
        patch(
            "nemo_evaluator.core.evaluate.validate_configuration",
            return_value=Evaluation(
                command="command",
                framework_name="lm-eval",
                pkg_name="pkg_name",
                config=EvaluationConfig(
                    output_dir=str(tmp_path),
                    params=ConfigParams(),
                ),
                target=EvaluationTarget(),
            ),
        ),
        patch(
            "nemo_evaluator.core.evaluate.monitor_memory_usage",
            return_value=(MagicMock(model_dump=lambda **k: {"test": "result"}), {}),
        ),
    )


def test_evaluate_sends_telemetry_at_default_level(tmp_path):
    """Full evaluate() call at level 2 — events include model name."""
    from nemo_evaluator.telemetry import TelemetryHandler

    enqueued_events = []
    original_enqueue = TelemetryHandler.enqueue

    def spy_enqueue(self, event):
        enqueued_events.append(event)
        return original_enqueue(self, event)

    mock_validate, mock_monitor = _mock_eval(tmp_path)
    with (
        mock_validate,
        mock_monitor,
        patch.object(TelemetryHandler, "enqueue", spy_enqueue),
    ):
        evaluate(
            eval_cfg=EvaluationConfig(
                output_dir=str(tmp_path),
                params=ConfigParams(),
                type="integration-test-task",
            ),
            target_cfg=EvaluationTarget(),
        )

    assert len(enqueued_events) == 2
    models = [e.model for e in enqueued_events]
    assert all(m != "redacted" for m in models)


def test_evaluate_sends_telemetry_at_minimal_level(tmp_path, monkeypatch):
    """Full evaluate() call at level 1 — model should be 'redacted'."""
    from nemo_evaluator.telemetry import TelemetryHandler

    monkeypatch.setenv("NEMO_EVALUATOR_TELEMETRY_LEVEL", "1")

    enqueued_events = []
    original_enqueue = TelemetryHandler.enqueue

    def spy_enqueue(self, event):
        enqueued_events.append(event)
        return original_enqueue(self, event)

    mock_validate, mock_monitor = _mock_eval(tmp_path)
    with (
        mock_validate,
        mock_monitor,
        patch.object(TelemetryHandler, "enqueue", spy_enqueue),
    ):
        evaluate(
            eval_cfg=EvaluationConfig(
                output_dir=str(tmp_path),
                params=ConfigParams(),
                type="integration-test-minimal",
            ),
            target_cfg=EvaluationTarget(),
        )

    assert len(enqueued_events) == 2
    models = [e.model for e in enqueued_events]
    assert all(m == "redacted" for m in models)
