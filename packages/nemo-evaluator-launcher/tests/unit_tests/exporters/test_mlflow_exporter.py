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

from pathlib import Path
from unittest.mock import patch

from nemo_evaluator_launcher.common.execdb import ExecutionDB, JobData
from nemo_evaluator_launcher.exporters.mlflow import MLflowExporter


class TestMLflowExporter:
    def test_not_available(self):
        with patch("nemo_evaluator_launcher.exporters.mlflow.MLFLOW_AVAILABLE", False):
            exporter = MLflowExporter({"tracking_uri": "http://mlflow"})
            result = exporter.export(["test123.0"])
            assert not result.successful_jobs
            assert result.failed_jobs == ["test123.0"]

    def test_export_job_ok(self, monkeypatch, mlflow_fake, tmp_path: Path, mock_execdb):
        _ML, _RunCtx = mlflow_fake

        # Create job with artifacts
        inv = "test001"
        artifacts_dir = tmp_path / inv / f"{inv}.0" / "artifacts"
        artifacts_dir.mkdir(parents=True)

        # Create required files
        (artifacts_dir / "run_config.yml").write_text(
            "framework_name: test-harness\nconfig:\n  type: test_task\ntarget:\n  api_endpoint:\n    model_id: test-model\n"
        )
        (artifacts_dir / "results.yml").write_text(
            "results:\n  tasks:\n    test_task:\n      metrics:\n        accuracy:\n          scores:\n            accuracy:\n              value: 0.9\nconfig: {}\n"
        )

        jd = JobData(
            inv,
            f"{inv}.0",
            0.0,
            "local",
            {"output_dir": str(tmp_path / inv / f"{inv}.0")},
            {},
        )
        ExecutionDB().write_job(jd)

        exp = MLflowExporter({"tracking_uri": "http://mlflow"})

        result = exp.export([jd.job_id])
        assert result.successful_jobs == [jd.job_id]
        assert result.failed_jobs == []

    def test_missing_tracking_uri(self, monkeypatch, mock_execdb):
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        monkeypatch.setattr(
            "nemo_evaluator_launcher.exporters.mlflow.MLFLOW_AVAILABLE",
            True,
            raising=True,
        )

        exporter = MLflowExporter({})
        result = exporter.export(["test123.0"])
        assert not result.successful_jobs
        assert result.failed_jobs == ["test123.0"]

    def test_export_with_skip_existing(
        self, mock_execdb, monkeypatch, mlflow_fake, tmp_path: Path
    ):
        _ML, _RunCtx = mlflow_fake
        inv = "test002"

        # Create job with artifacts
        artifacts_dir = tmp_path / inv / f"{inv}.0" / "artifacts"
        artifacts_dir.mkdir(parents=True)

        (artifacts_dir / "run_config.yml").write_text(
            "framework_name: test-harness\nconfig:\n  type: test_task\ntarget:\n  api_endpoint:\n    model_id: test-model\n"
        )
        (artifacts_dir / "results.yml").write_text(
            "results:\n  tasks:\n    test_task:\n      metrics:\n        accuracy:\n          scores:\n            accuracy:\n              value: 0.9\nconfig: {}\n"
        )

        jd = JobData(
            inv,
            f"{inv}.0",
            0.0,
            "local",
            {"output_dir": str(tmp_path / inv / f"{inv}.0")},
            {},
        )
        ExecutionDB().write_job(jd)

        exp = MLflowExporter({"tracking_uri": "http://ml", "skip_existing": True})
        monkeypatch.setattr(
            exp,
            "_get_existing_run_id",
            lambda *a, **k: ("existing123"),
            raising=False,
        )

        result = exp.export([jd.job_id])
        assert result.skipped_jobs == [jd.job_id]

    def test_export_with_update_existing(
        self, mock_execdb, monkeypatch, mlflow_fake, tmp_path: Path
    ):
        _ML, _RunCtx = mlflow_fake
        inv = "test002"

        # Create job with artifacts
        artifacts_dir = tmp_path / inv / f"{inv}.0" / "artifacts"
        artifacts_dir.mkdir(parents=True)

        (artifacts_dir / "run_config.yml").write_text(
            "framework_name: test-harness\nconfig:\n  type: test_task\ntarget:\n  api_endpoint:\n    model_id: test-model\n"
        )
        (artifacts_dir / "results.yml").write_text(
            "results:\n  tasks:\n    test_task:\n      metrics:\n        accuracy:\n          scores:\n            accuracy:\n              value: 0.9\nconfig: {}\n"
        )

        jd = JobData(
            inv,
            f"{inv}.0",
            0.0,
            "local",
            {"output_dir": str(tmp_path / inv / f"{inv}.0")},
            {},
        )
        ExecutionDB().write_job(jd)

        exp = MLflowExporter({"tracking_uri": "http://ml"})
        monkeypatch.setattr(
            exp,
            "_get_existing_run_id",
            lambda *a, **k: ("existing123"),
            raising=False,
        )

        result = exp.export([jd.job_id])
        assert result.successful_jobs == [jd.job_id]

    def test_log_config_params_flattens_config(
        self, monkeypatch, mlflow_fake, tmp_path: Path, mock_execdb
    ):
        """Test that log_config_params=True flattens the config into MLflow params."""
        _ML, _RunCtx = mlflow_fake
        inv = "test004"

        # Job with nested config
        config = {
            "deployment": {"tensor_parallel_size": 8, "model": "test-model"},
            "evaluation": {
                "tasks": [
                    {"name": "task1", "config": {"param": "value1"}},
                    {"name": "task2"},
                ]
            },
        }

        # Create job with artifacts
        artifacts_dir = tmp_path / inv / f"{inv}.0" / "artifacts"
        artifacts_dir.mkdir(parents=True)

        (artifacts_dir / "run_config.yml").write_text(
            "framework_name: test-harness\nconfig:\n  type: test_task\ntarget:\n  api_endpoint:\n    model_id: test-model\n"
        )
        (artifacts_dir / "results.yml").write_text(
            "results:\n  tasks:\n    test_task:\n      metrics:\n        accuracy:\n          scores:\n            accuracy:\n              value: 0.9\nconfig: {}\n"
        )

        jd = JobData(
            inv,
            f"{inv}.0",
            0.0,
            "local",
            {"output_dir": str(tmp_path / inv / f"{inv}.0")},
            config,
        )
        ExecutionDB().write_job(jd)

        # Capture log_params calls
        logged_params = {}
        monkeypatch.setattr(
            "nemo_evaluator_launcher.exporters.mlflow.mlflow.log_params",
            lambda p: logged_params.update(p),
            raising=True,
        )

        exp = MLflowExporter(
            {
                "tracking_uri": "http://mlflow",
                "log_config_params": True,
            }
        )

        result = exp.export([jd.job_id])
        assert result.successful_jobs == [jd.job_id]

        # Verify flattened config params are logged
        assert "config.deployment.tensor_parallel_size" in logged_params
        assert logged_params["config.deployment.tensor_parallel_size"] == "8"
        assert "config.deployment.model" in logged_params
        assert logged_params["config.deployment.model"] == "test-model"

    def test_log_config_params_with_max_depth(
        self, monkeypatch, mlflow_fake, tmp_path: Path, mock_execdb
    ):
        """Test that log_config_params_max_depth limits flattening depth."""
        _ML, _RunCtx = mlflow_fake
        inv = "test005"

        config = {"a": {"b": {"c": {"d": "deep"}}}}

        # Create job with artifacts
        artifacts_dir = tmp_path / inv / f"{inv}.0" / "artifacts"
        artifacts_dir.mkdir(parents=True)

        (artifacts_dir / "run_config.yml").write_text(
            "framework_name: test-harness\nconfig:\n  type: test_task\ntarget:\n  api_endpoint:\n    model_id: test-model\n"
        )
        (artifacts_dir / "results.yml").write_text(
            "results:\n  tasks:\n    test_task:\n      metrics:\n        accuracy:\n          scores:\n            accuracy:\n              value: 0.9\nconfig: {}\n"
        )

        jd = JobData(
            inv,
            f"{inv}.0",
            0.0,
            "local",
            {"output_dir": str(tmp_path / inv / f"{inv}.0")},
            config,
        )
        ExecutionDB().write_job(jd)

        logged_params = {}
        monkeypatch.setattr(
            "nemo_evaluator_launcher.exporters.mlflow.mlflow.log_params",
            lambda p: logged_params.update(p),
            raising=True,
        )

        exp = MLflowExporter(
            {
                "tracking_uri": "http://mlflow",
                "log_config_params": True,
                "log_config_params_max_depth": 2,
            }
        )

        result = exp.export([jd.job_id])
        assert result.successful_jobs == [jd.job_id]

        # At depth 2, a.b should be stringified (not further flattened)
        assert "config.a.b" in logged_params
        # The value should be a string representation of the remaining dict
        assert "c" in logged_params["config.a.b"]
        # Deeper keys should not exist
        assert "config.a.b.c" not in logged_params
        assert "config.a.b.c.d" not in logged_params
