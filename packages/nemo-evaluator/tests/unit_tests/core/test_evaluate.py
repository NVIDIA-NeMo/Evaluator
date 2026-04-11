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
import signal
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from nemo_evaluator.adapters.adapter_config import AdapterConfig
from nemo_evaluator.api.api_dataclasses import (
    ApiEndpoint,
    ConfigParams,
    Evaluation,
    EvaluationConfig,
    EvaluationTarget,
)
from nemo_evaluator.config import TelemetryLevel
from nemo_evaluator.core.evaluate import (
    INTERRUPTED_MARKER_FILENAME,
    SERVER_ERROR_MARKER_FILENAME,
    _run_evaluation,
    evaluate,
)


def test_evaluate_result_config_matches_run_config(tmp_path: Path):
    with (
        patch(
            "nemo_evaluator.telemetry.get_telemetry_level",
            return_value=TelemetryLevel.OFF,
        ),
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
        patch(
            "nemo_evaluator.telemetry.get_telemetry_level",
            return_value=TelemetryLevel.DEFAULT,
        ),
        patch("nemo_evaluator.telemetry.show_telemetry_notification"),
        patch(
            "nemo_evaluator.telemetry.TelemetryHandler", return_value=CapturingHandler()
        ),
        patch.dict(os.environ, {"NEMO_EVALUATOR_TELEMETRY_LEVEL": "2"}),
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
    assert started_event.task == "test_task"
    assert started_event.model == "unknown"
    assert started_event.status == "started"
    completion_event = captured_events[1]
    assert completion_event.framework_name == "lm-eval"
    assert completion_event.task == "test_task"
    assert completion_event.model == "unknown"
    assert completion_event.status == "success"


def _build_evaluation(tmp_path: Path, *, client_mode: bool = False):
    target_cfg = EvaluationTarget()
    if client_mode:
        target_cfg = EvaluationTarget(
            api_endpoint=ApiEndpoint(
                model_id="test-model",
                url="http://example.com",
                adapter_config=AdapterConfig(mode="client"),
            )
        )

    evaluation = Evaluation(
        command="command",
        framework_name="framework_name",
        pkg_name="pkg_name",
        config=EvaluationConfig(
            output_dir=str(tmp_path),
            params=ConfigParams(),
        ),
        target=target_cfg,
    )
    return evaluation, target_cfg


def _invoke_shutdown_handler(
    tmp_path: Path,
    signum: int,
    *,
    client_mode: bool = False,
    env: dict[str, str] | None = None,
    cleanup_side_effect: Exception | None = None,
    alive_children: list[MagicMock] | None = None,
):
    evaluation, target_cfg = _build_evaluation(tmp_path, client_mode=client_mode)
    handlers = {}
    child = MagicMock(pid=1234)
    parent = MagicMock()
    parent.children.return_value = [child]
    alive_children = alive_children or []

    def register_handler(sig, handler):
        handlers[sig] = handler

    def trigger_signal(*args, **kwargs):
        handlers[signum](signum, None)

    with (
        patch(
            "nemo_evaluator.core.evaluate.signal.signal",
            side_effect=register_handler,
        ),
        patch("nemo_evaluator.core.evaluate.psutil.Process", return_value=parent),
        patch(
            "nemo_evaluator.core.evaluate.psutil.wait_procs",
            return_value=([], alive_children),
        ) as wait_procs,
        patch(
            "nemo_evaluator.core.evaluate.monitor_memory_usage",
            side_effect=trigger_signal,
        ),
        patch.dict(os.environ, env or {}, clear=False),
    ):
        if client_mode:
            with patch(
                "nemo_evaluator.adapters.pipeline.AdapterPipeline"
            ) as pipeline_cls:
                if cleanup_side_effect is not None:
                    pipeline_cls.return_value.run_post_eval_hooks.side_effect = (
                        cleanup_side_effect
                    )
                with pytest.raises(SystemExit) as exc_info:
                    _run_evaluation(evaluation, target_cfg, None)
                return exc_info.value.code, child, wait_procs, pipeline_cls

        with pytest.raises(SystemExit) as exc_info:
            _run_evaluation(evaluation, target_cfg, None)
        return exc_info.value.code, child, wait_procs, None


def test_shutdown_handler_exits_zero_for_sigterm_after_graceful_cleanup(tmp_path: Path):
    exit_code, child, wait_procs, pipeline_cls = _invoke_shutdown_handler(
        tmp_path,
        signal.SIGTERM,
        client_mode=True,
    )

    assert exit_code == 0
    child.terminate.assert_called_once()
    child.send_signal.assert_not_called()
    wait_procs.assert_called_once_with([child], timeout=30.0)
    pipeline_cls.return_value.run_post_eval_hooks.assert_called_once_with(
        url="http://example.com"
    )
    marker_path = tmp_path / INTERRUPTED_MARKER_FILENAME
    assert marker_path.exists()
    marker = yaml.safe_load(marker_path.read_text())
    assert marker["signal"] == "SIGTERM"


def test_shutdown_handler_skips_interrupted_marker_on_server_error(tmp_path: Path):
    """When the server error marker exists, SIGTERM should not write the interrupted marker."""
    (tmp_path / SERVER_ERROR_MARKER_FILENAME).write_text("FatalErrorException")

    exit_code, child, wait_procs, pipeline_cls = _invoke_shutdown_handler(
        tmp_path,
        signal.SIGTERM,
        client_mode=True,
    )

    assert exit_code == 1
    child.terminate.assert_called_once()
    wait_procs.assert_called_once_with([child], timeout=30.0)
    pipeline_cls.return_value.run_post_eval_hooks.assert_called_once_with(
        url="http://example.com"
    )
    assert not (tmp_path / INTERRUPTED_MARKER_FILENAME).exists()
    assert (tmp_path / SERVER_ERROR_MARKER_FILENAME).exists()


def test_shutdown_handler_exits_130_for_sigint(tmp_path: Path):
    exit_code, child, wait_procs, pipeline_cls = _invoke_shutdown_handler(
        tmp_path,
        signal.SIGINT,
        client_mode=True,
    )

    assert exit_code == 130
    child.terminate.assert_not_called()
    child.send_signal.assert_called_once_with(signal.SIGINT)
    wait_procs.assert_called_once_with([child], timeout=1)
    pipeline_cls.assert_not_called()


def test_shutdown_handler_uses_env_override_timeout(tmp_path: Path):
    exit_code, child, wait_procs, _ = _invoke_shutdown_handler(
        tmp_path,
        signal.SIGTERM,
        env={"NEMO_EVALUATOR_SHUTDOWN_TIMEOUT_SECONDS": "45"},
    )

    assert exit_code == 0
    child.terminate.assert_called_once()
    wait_procs.assert_called_once_with([child], timeout=45.0)


def test_shutdown_handler_warns_and_falls_back_on_invalid_timeout(tmp_path: Path):
    with patch("nemo_evaluator.core.evaluate.logger.warning") as warning_logger:
        exit_code, child, wait_procs, _ = _invoke_shutdown_handler(
            tmp_path,
            signal.SIGTERM,
            env={"NEMO_EVALUATOR_SHUTDOWN_TIMEOUT_SECONDS": "invalid"},
        )

    assert exit_code == 0
    child.terminate.assert_called_once()
    wait_procs.assert_called_once_with([child], timeout=30.0)
    warning_logger.assert_called_once()
    assert "NEMO_EVALUATOR_SHUTDOWN_TIMEOUT_SECONDS" in warning_logger.call_args.args[0]


@pytest.mark.parametrize("raw_timeout", ["nan", "inf"])
def test_shutdown_handler_warns_and_falls_back_on_non_finite_timeout(
    tmp_path: Path, raw_timeout: str
):
    with patch("nemo_evaluator.core.evaluate.logger.warning") as warning_logger:
        exit_code, child, wait_procs, _ = _invoke_shutdown_handler(
            tmp_path,
            signal.SIGTERM,
            env={"NEMO_EVALUATOR_SHUTDOWN_TIMEOUT_SECONDS": raw_timeout},
        )

    assert exit_code == 0
    child.terminate.assert_called_once()
    wait_procs.assert_called_once_with([child], timeout=30.0)
    warning_logger.assert_called_once()
    assert "NEMO_EVALUATOR_SHUTDOWN_TIMEOUT_SECONDS" in warning_logger.call_args.args[0]


def test_shutdown_handler_force_kills_children_and_exits_one(tmp_path: Path):
    stuck_child = MagicMock(pid=5678)
    exit_code, child, wait_procs, _ = _invoke_shutdown_handler(
        tmp_path,
        signal.SIGTERM,
        alive_children=[stuck_child],
    )

    assert exit_code == 1
    child.terminate.assert_called_once()
    wait_procs.assert_called_once_with([child], timeout=30.0)
    stuck_child.kill.assert_called_once()


def test_shutdown_handler_exits_one_when_cleanup_fails(tmp_path: Path):
    exit_code, child, wait_procs, pipeline_cls = _invoke_shutdown_handler(
        tmp_path,
        signal.SIGTERM,
        client_mode=True,
        cleanup_side_effect=RuntimeError("cleanup failed"),
    )

    assert exit_code == 1
    child.terminate.assert_called_once()
    wait_procs.assert_called_once_with([child], timeout=30.0)
    pipeline_cls.return_value.run_post_eval_hooks.assert_called_once_with(
        url="http://example.com"
    )


def test_evaluate_clears_stale_markers(tmp_path: Path):
    marker_path = tmp_path / INTERRUPTED_MARKER_FILENAME
    marker_path.write_text("stale")
    server_error_path = tmp_path / SERVER_ERROR_MARKER_FILENAME
    server_error_path.write_text("stale")

    with (
        patch(
            "nemo_evaluator.telemetry.get_telemetry_level",
            return_value=TelemetryLevel.OFF,
        ),
        patch(
            "nemo_evaluator.core.evaluate.validate_configuration",
            return_value=Evaluation(
                command="command",
                framework_name="framework_name",
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
    ):
        evaluate(
            eval_cfg=EvaluationConfig(
                output_dir=str(tmp_path),
                params=ConfigParams(),
            ),
            target_cfg=EvaluationTarget(),
        )

    assert not marker_path.exists()
    assert not server_error_path.exists()
