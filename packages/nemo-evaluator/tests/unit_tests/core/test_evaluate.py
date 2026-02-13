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

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import yaml

from nemo_evaluator.api.api_dataclasses import (
    ConfigParams,
    Evaluation,
    EvaluationConfig,
    EvaluationTarget,
)
from nemo_evaluator.core.evaluate import evaluate


def test_evaluate_result_config_matches_run_config(tmp_path: Path):
    with (
        patch("nemo_evaluator.telemetry.is_telemetry_enabled", return_value=False),
        patch(
            "nemo_evaluator.core.evaluate.validate_configuration",
            return_value=Evaluation(
                command="command",
                framework_name="framework_name",
                pkg_name="pkg_name",
                config=EvaluationConfig(
                    output_dir=str(tmp_path),
                    params=ConfigParams(
                        temperature=1e-05,  # Framework default applied
                        top_p=1e-05,  # Framework default applied
                        parallelism=512,  # User value preserved
                        request_timeout=360,  # User value preserved
                    ),
                ),
                target=EvaluationTarget(),
            ),
        ),
        patch(
            "nemo_evaluator.core.evaluate.monitor_memory_usage",
            return_value=(MagicMock(model_dump=lambda **k: {"test": "result"}), {}),
        ),
    ):
        evaluate(
            eval_cfg=EvaluationConfig(
                output_dir=str(tmp_path),
                params=ConfigParams(
                    parallelism=512,  # User value (to be preserved)
                    request_timeout=360,  # User value (to be preserved)
                ),
            ),
            target_cfg=EvaluationTarget(),
        )
    with (tmp_path / "results.yml").open("r") as stream:
        results = yaml.safe_load(stream)
    assert results["config"]["params"] == {
        "temperature": 1e-05,
        "top_p": 1e-05,
        "parallelism": 512,
        "request_timeout": 360,
        "extra": {},
    }


def test_evaluate_telemetry_uses_framework_name(tmp_path: Path):
    """Test that telemetry completion event uses Evaluation.framework_name."""
    captured_events = []

    class CapturingHandler:
        """Minimal TelemetryHandler substitute that captures enqueued events."""

        def start(self):
            pass

        def stop(self):
            pass

        def enqueue(self, event):
            captured_events.append(event)

    with (
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
        patch("nemo_evaluator.telemetry.is_telemetry_enabled", return_value=True),
        patch("nemo_evaluator.telemetry.show_telemetry_notification"),
        patch("nemo_evaluator.telemetry.TelemetryHandler", return_value=CapturingHandler()),
        patch.dict(os.environ, {"NEMO_EVALUATOR_TELEMETRY_ENABLED": "true"}),
    ):
        evaluate(
            eval_cfg=EvaluationConfig(
                output_dir=str(tmp_path),
                params=ConfigParams(),
                type="test_task",
            ),
            target_cfg=EvaluationTarget(),
        )

    # Should have STARTED + SUCCESS events
    assert len(captured_events) == 2
    started_event = captured_events[0]
    assert started_event.framework_name == "lm-eval"
    assert started_event.status == "started"
    completion_event = captured_events[1]
    assert completion_event.framework_name == "lm-eval"
    assert completion_event.status == "success"
