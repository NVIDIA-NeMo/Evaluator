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
#
"""MLFlow exporter tests."""

from unittest.mock import patch

import pytest
from pydantic import ValidationError

from nemo_evaluator_launcher.common.execdb import ExecutionDB, JobData
from nemo_evaluator_launcher.exporters.mlflow import MLflowExporter
from nemo_evaluator_launcher.exporters.utils import DataForExport

_RUN_CONFIG = (
    "framework_name: test-harness\n"
    "config:\n"
    "  type: test_task\n"
    "target:\n"
    "  api_endpoint:\n"
    "    model_id: test-model\n"
)
_RESULTS = (
    "results:\n"
    "  tasks:\n"
    "    test_task:\n"
    "      metrics:\n"
    "        accuracy:\n"
    "          scores:\n"
    "            accuracy:\n"
    "              value: 0.9\n"
    "config: {}\n"
)


@pytest.fixture
def make_mlflow_job(tmp_path, mock_execdb):
    """Factory fixture: create a job with standard MLflow test artifacts."""

    def _make(inv: str, config=None) -> JobData:
        artifacts_dir = tmp_path / inv / f"{inv}.0" / "artifacts"
        artifacts_dir.mkdir(parents=True)
        (artifacts_dir / "run_config.yml").write_text(_RUN_CONFIG)
        (artifacts_dir / "results.yml").write_text(_RESULTS)
        jd = JobData(
            inv,
            f"{inv}.0",
            0.0,
            "local",
            {"output_dir": str(tmp_path / inv / f"{inv}.0")},
            config or {},
        )
        ExecutionDB().write_job(jd)
        return jd

    return _make


class TestMLflowExporter:
    def test_not_available(self, tmp_path):
        with patch("nemo_evaluator_launcher.exporters.mlflow.MLFLOW_AVAILABLE", False):
            data = DataForExport(
                artifacts_dir=tmp_path / "artifacts",
                logs_dir=None,
                config={},
                model_id="test-model",
                metrics={"acc": 0.9},
                harness="lm-eval",
                task="mmlu",
                container="test:latest",
                executor="local",
                invocation_id="test123",
                job_id="test123.0",
                timestamp=1234567890.0,
                job_data={},
            )
            successful, failed, skipped = MLflowExporter(
                {"tracking_uri": "http://mlflow"}
            ).export_jobs([data])
            assert successful == []
            assert failed == ["test123.0"]

    def test_export_job_ok(self, mlflow_fake, make_mlflow_job):
        _ML, _RunCtx = mlflow_fake
        jd = make_mlflow_job("test001")

        result = MLflowExporter({"tracking_uri": "http://mlflow"}).export([jd.job_id])
        assert result.successful_jobs == [jd.job_id]
        assert result.failed_jobs == []

    def test_missing_tracking_uri(self, monkeypatch, mock_execdb):
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        monkeypatch.setattr(
            "nemo_evaluator_launcher.exporters.mlflow.MLFLOW_AVAILABLE",
            True,
            raising=True,
        )

        with pytest.raises(
            ValidationError, match="MLflow requires 'tracking_uri' to be configured."
        ):
            MLflowExporter()

    def test_export_with_skip_existing(self, monkeypatch, mlflow_fake, make_mlflow_job):
        _ML, _RunCtx = mlflow_fake
        jd = make_mlflow_job("test002")

        exp = MLflowExporter({"tracking_uri": "http://ml", "skip_existing": True})
        monkeypatch.setattr(
            exp, "_get_existing_run_id", lambda *a, **k: "existing123", raising=False
        )

        result = exp.export([jd.job_id])
        assert result.skipped_jobs == [jd.job_id]

    def test_export_with_update_existing(
        self, monkeypatch, mlflow_fake, make_mlflow_job
    ):
        _ML, _RunCtx = mlflow_fake
        jd = make_mlflow_job("test003")

        exp = MLflowExporter({"tracking_uri": "http://ml"})
        monkeypatch.setattr(
            exp, "_get_existing_run_id", lambda *a, **k: "existing123", raising=False
        )

        result = exp.export([jd.job_id])
        assert result.successful_jobs == [jd.job_id]

    def test_log_config_params_flattens_config(
        self, monkeypatch, mlflow_fake, make_mlflow_job, tmp_path
    ):
        """Test that log_config_params=True flattens the config into MLflow params."""
        _ML, _RunCtx = mlflow_fake
        config = {
            "deployment": {"tensor_parallel_size": 8, "model": "test-model"},
            "evaluation": {
                "tasks": [
                    {"name": "task1", "config": {"param": "value1"}},
                    {"name": "task2"},
                ]
            },
        }
        jd = make_mlflow_job("test004", config=config)
        (tmp_path / "test004" / "test004.0" / "artifacts" / "metadata.yaml").write_text(
            "launcher_command: nel run test004\n"
        )

        logged_params = {}
        monkeypatch.setattr(
            "nemo_evaluator_launcher.exporters.mlflow.mlflow.log_params",
            lambda p: logged_params.update(p),
            raising=True,
        )

        result = MLflowExporter(
            {"tracking_uri": "http://mlflow", "log_config_params": True}
        ).export([jd.job_id])
        assert result.successful_jobs == [jd.job_id]
        assert "config.deployment.tensor_parallel_size" in logged_params
        assert logged_params["config.deployment.tensor_parallel_size"] == "8"
        assert "config.deployment.model" in logged_params
        assert logged_params["config.deployment.model"] == "test-model"
        assert logged_params.get("launcher_command") == "nel run test004"
        assert "results_dir" in logged_params

    def test_log_config_params_with_max_depth(
        self, monkeypatch, mlflow_fake, make_mlflow_job
    ):
        """Test that log_config_params_max_depth limits flattening depth."""
        _ML, _RunCtx = mlflow_fake
        jd = make_mlflow_job("test005", config={"a": {"b": {"c": {"d": "deep"}}}})

        logged_params = {}
        monkeypatch.setattr(
            "nemo_evaluator_launcher.exporters.mlflow.mlflow.log_params",
            lambda p: logged_params.update(p),
            raising=True,
        )

        result = MLflowExporter(
            {
                "tracking_uri": "http://mlflow",
                "log_config_params": True,
                "log_config_params_max_depth": 2,
            }
        ).export([jd.job_id])
        assert result.successful_jobs == [jd.job_id]
        # At depth 2, a.b should be stringified (not further flattened)
        assert "config.a.b" in logged_params
        # The value should be a string representation of the remaining dict
        assert "c" in logged_params["config.a.b"]
        # Deeper keys should not exist
        assert "config.a.b.c" not in logged_params
        assert "config.a.b.c.d" not in logged_params
