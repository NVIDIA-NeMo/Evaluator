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

import types
from pathlib import Path
from unittest.mock import Mock, patch

from nemo_evaluator_launcher.common.execdb import ExecutionDB, JobData
from nemo_evaluator_launcher.exporters.mlflow import MLflowExporter


class TestMLflowExporter:
    def test_not_available(self):
        with patch("nemo_evaluator_launcher.exporters.mlflow.MLFLOW_AVAILABLE", False):
            res = MLflowExporter().export_job(Mock())
            assert not res.success
            assert "not installed" in res.message

    def test_export_job_ok(self, monkeypatch, mlflow_fake, tmp_path: Path):
        _ML, _RunCtx = mlflow_fake
        jd = JobData("m1", "m1.0", 0.0, "local", {"output_dir": str(tmp_path)}, {})
        monkeypatch.setattr(
            "nemo_evaluator_launcher.exporters.mlflow.extract_accuracy_metrics",
            lambda *_: {"task_accuracy": 0.9},
            raising=True,
        )
        exp = MLflowExporter({"tracking_uri": "http://mlflow"})
        # Avoid real mlflow import/network inside _get_existing_run_info
        monkeypatch.setattr(
            exp, "_get_existing_run_info", lambda *a, **k: (False, None), raising=False
        )
        res = exp.export_job(jd)
        assert res.success
        assert res.metadata.get("run_id")

    def test_missing_tracking_uri(self, monkeypatch):
        jd = JobData("m2", "m2.0", 0.0, "local", {"output_dir": "/tmp"}, {})
        # Ensure no env fallback interferes with the test
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        monkeypatch.setattr(
            "nemo_evaluator_launcher.exporters.mlflow.MLFLOW_AVAILABLE",
            True,
            raising=True,
        )
        monkeypatch.setattr(
            "nemo_evaluator_launcher.exporters.mlflow.extract_accuracy_metrics",
            lambda *_: {"m": 1.0},
            raising=True,
        )
        res = MLflowExporter({}).export_job(jd)
        assert not res.success
        assert "tracking_uri" in res.message

    def test_export_invocation_connection_failure(
        self, mock_execdb, monkeypatch, tmp_path: Path
    ):
        # Create a job in the DB with metrics
        inv = "ab12cd34"
        ExecutionDB().write_job(
            JobData(inv, f"{inv}.0", 0.0, "local", {"output_dir": str(tmp_path)}, {})
        )

        # Mock metrics to return data (bypass early exit)
        monkeypatch.setattr(
            "nemo_evaluator_launcher.exporters.mlflow.extract_accuracy_metrics",
            lambda *_: {"task_accuracy": 0.9},
            raising=True,
        )
        monkeypatch.setattr(
            "nemo_evaluator_launcher.exporters.mlflow.MLFLOW_AVAILABLE",
            True,
            raising=True,
        )

        # Mock mlflow to fail on connection
        class _BrokenMLflow:
            @staticmethod
            def set_tracking_uri(*_):
                raise RuntimeError("MLflow connection failed")

        monkeypatch.setattr(
            "nemo_evaluator_launcher.exporters.mlflow.mlflow",
            _BrokenMLflow,
            raising=True,
        )

        res = MLflowExporter({"tracking_uri": "http://ml"}).export_invocation(inv)
        assert res["success"] is False
        assert "MLflow export failed" in res["error"]

    def test_export_invocation_success_with_webhook_tags_and_artifacts(
        self, mock_execdb, monkeypatch, mlflow_fake, tmp_path: Path
    ):
        _ML, _RunCtx = mlflow_fake
        inv = "deadbeef"
        db = ExecutionDB()
        for i in range(2):
            db.write_job(
                JobData(
                    inv, f"{inv}.{i}", 0.0, "local", {"output_dir": str(tmp_path)}, {}
                )
            )

        # Metrics for all jobs (ensures non-empty all_metrics; keys differ per job)
        monkeypatch.setattr(
            "nemo_evaluator_launcher.exporters.mlflow.extract_accuracy_metrics",
            lambda jd, getp, lm: {f"{jd.job_id}_accuracy": 0.9},
            raising=True,
        )

        # Capture mlflow calls
        calls = {
            "set_tags": None,
            "set_tag": [],
            "log_params": None,
            "log_metrics": None,
        }
        monkeypatch.setattr(
            "nemo_evaluator_launcher.exporters.mlflow.mlflow.set_tags",
            lambda t: calls.update(set_tags=dict(t)),
            raising=True,
        )
        monkeypatch.setattr(
            "nemo_evaluator_launcher.exporters.mlflow.mlflow.set_tag",
            lambda k, v: calls["set_tag"].append((k, v)),
            raising=True,
        )
        monkeypatch.setattr(
            "nemo_evaluator_launcher.exporters.mlflow.mlflow.log_params",
            lambda p: calls.update(log_params=dict(p)),
            raising=True,
        )
        monkeypatch.setattr(
            "nemo_evaluator_launcher.exporters.mlflow.mlflow.log_metrics",
            lambda m: calls.update(log_metrics=dict(m)),
            raising=True,
        )

        exp = MLflowExporter(
            {
                "tracking_uri": "http://ml",
                "experiment_name": "exp",
                "run_name": "explicit-run",
                "description": "desc",
                "triggered_by_webhook": True,
                "webhook_source": "gitlab",
                "source_artifact": "artifact.zip",
                "config_source": "snapshot",
                "extra_metadata": {"foo": "bar"},
                "tags": {"a": "b", "empty": None},
            }
        )
        monkeypatch.setattr(
            exp, "_get_existing_run_info", lambda *a, **k: (False, None), raising=False
        )

        # Each job contributes one artifact
        monkeypatch.setattr(
            MLflowExporter,
            "_log_artifacts",
            lambda self, jd, cfg, pre=None: ["results.yml"],
            raising=False,
        )

        res = exp.export_invocation(inv)
        assert res["success"] is True

        # Params include webhook + extras and correct jobs_count
        assert calls["log_params"]["invocation_id"] == inv
        assert calls["log_params"]["jobs_count"] == "2"
        assert calls["log_params"]["foo"] == "bar"
        assert calls["log_params"]["webhook_triggered"] == "true"

        # Tags merged and filtered (no None)
        assert calls["set_tags"]["invocation_id"] == inv
        assert "empty" not in calls["set_tags"]

        # Metrics merged from both jobs
        assert len(calls["log_metrics"]) == 2

        # Artifacts aggregated across jobs and run URL built
        assert res["metadata"]["artifacts_logged"] == 2
        assert res["metadata"]["tracking_uri"] == "http://ml"
        assert "run_url" in res["metadata"]

    def test_export_invocation_skip_existing(
        self, mock_execdb, monkeypatch, mlflow_fake, tmp_path: Path
    ):
        _ML, _RunCtx = mlflow_fake
        inv = "ivc0ffee"
        db = ExecutionDB()
        for i in range(2):
            db.write_job(
                JobData(
                    inv, f"{inv}.{i}", 0.0, "local", {"output_dir": str(tmp_path)}, {}
                )
            )

        # Ensure we pass the "no metrics" gate before skip_existing check
        monkeypatch.setattr(
            "nemo_evaluator_launcher.exporters.mlflow.extract_accuracy_metrics",
            lambda *_: {"dummy_accuracy": 0.1},
            raising=True,
        )

        exp = MLflowExporter({"tracking_uri": "http://ml", "skip_existing": True})
        monkeypatch.setattr(
            exp,
            "_get_existing_run_info",
            lambda *a, **k: (True, "existing123"),
            raising=False,
        )

        res = exp.export_invocation(inv)
        assert res["success"] is True
        assert res["metadata"]["run_id"] == "existing123"
        assert res["metadata"]["skipped"] is True
        assert set(res["jobs"].keys()) == {f"{inv}.0", f"{inv}.1"}

    def test_log_artifacts_disabled(self, monkeypatch, mlflow_fake, tmp_path: Path):
        _ML, _RunCtx = mlflow_fake
        jd = JobData("mX", "mX.0", 0.0, "local", {"output_dir": str(tmp_path)}, {})
        monkeypatch.setattr(
            "nemo_evaluator_launcher.exporters.mlflow.extract_accuracy_metrics",
            lambda *_: {"acc": 1.0},
            raising=True,
        )
        exp = MLflowExporter({"tracking_uri": "http://mlflow", "log_artifacts": False})
        monkeypatch.setattr(
            exp, "_get_existing_run_info", lambda *a, **k: (False, None), raising=False
        )
        res = exp.export_job(jd)
        assert res.success
        assert res.metadata["artifacts_logged"] == 0

    def test_log_artifacts_localexporter_failure(
        self, monkeypatch, mlflow_fake, tmp_path: Path
    ):
        _ML, _RunCtx = mlflow_fake
        jd = JobData("mY", "mY.0", 0.0, "local", {"output_dir": str(tmp_path)}, {})
        monkeypatch.setattr(
            "nemo_evaluator_launcher.exporters.mlflow.extract_accuracy_metrics",
            lambda *_: {"acc": 1.0},
            raising=True,
        )

        class _LE:
            def __init__(self, *_a, **_k): ...
            def export_job(self, *_a, **_k):
                return types.SimpleNamespace(
                    success=False, message="stage failed", dest=str(tmp_path / "noop")
                )

        monkeypatch.setattr(
            "nemo_evaluator_launcher.exporters.mlflow.LocalExporter", _LE, raising=True
        )
        exp = MLflowExporter({"tracking_uri": "http://mlflow"})
        monkeypatch.setattr(
            exp, "_get_existing_run_info", lambda *a, **k: (False, None), raising=False
        )
        res = exp.export_job(jd)
        assert res.success
        assert res.metadata["artifacts_logged"] == 0

    def test_log_artifacts_exception_returns_empty(
        self, monkeypatch, mlflow_fake, tmp_path: Path
    ):
        _ML, _RunCtx = mlflow_fake
        jd = JobData("mE", "mE.0", 0.0, "local", {"output_dir": str(tmp_path)}, {})
        monkeypatch.setattr(
            "nemo_evaluator_launcher.exporters.mlflow.extract_accuracy_metrics",
            lambda *_: {"acc": 1.0},
            raising=True,
        )

        class _LE:
            def __init__(self, *_a, **_k): ...
            def export_job(self, *_a, **_k):
                raise RuntimeError("boom")

        monkeypatch.setattr(
            "nemo_evaluator_launcher.exporters.mlflow.LocalExporter", _LE, raising=True
        )
        exp = MLflowExporter({"tracking_uri": "http://mlflow"})
        monkeypatch.setattr(
            exp, "_get_existing_run_info", lambda *a, **k: (False, None), raising=False
        )
        res = exp.export_job(jd)
        assert res.success
        assert res.metadata["artifacts_logged"] == 0

    def test_export_invocation_no_jobs_early_return(self, mlflow_fake):
        _ML, _RunCtx = mlflow_fake  # ensure MLFLOW_AVAILABLE = True
        res = MLflowExporter({"tracking_uri": "http://ml"}).export_invocation(
            "missing1234"
        )
        assert res["success"] is False
        assert "No jobs found" in res["error"]

    def test_log_artifacts_config_and_files(
        self, monkeypatch, mlflow_fake, tmp_path: Path
    ):
        _ML, _RunCtx = mlflow_fake
        # Prepare a staged artifacts dir returned by LocalExporter
        stage_dir = tmp_path / "stage"
        (stage_dir / "artifacts").mkdir(parents=True, exist_ok=True)
        (stage_dir / "artifacts" / "results.yml").write_text("ok", encoding="utf-8")

        class _LE:
            def __init__(self, *_a, **_k): ...
            def export_job(self, *_a, **_k):
                return types.SimpleNamespace(
                    success=True, message="ok", dest=str(stage_dir)
                )

        # Metrics and existence check
        monkeypatch.setattr(
            "nemo_evaluator_launcher.exporters.mlflow.extract_accuracy_metrics",
            lambda *_: {"acc": 1.0},
            raising=True,
        )
        # Ensure predictable artifact_path
        monkeypatch.setattr(
            "nemo_evaluator_launcher.exporters.mlflow.get_artifact_root",
            lambda *_: "taskX",
            raising=True,
        )
        # Use our LocalExporter stub
        monkeypatch.setattr(
            "nemo_evaluator_launcher.exporters.mlflow.LocalExporter", _LE, raising=True
        )
        # Only one available file
        monkeypatch.setattr(
            "nemo_evaluator_launcher.exporters.mlflow.get_available_artifacts",
            lambda *_: ["results.yml"],
            raising=True,
        )

        # Capture mlflow.log_artifact calls (config.yaml + results.yml)
        calls = {"config": 0, "files": []}

        def _log_artifact(path, artifact_path=None):
            p = Path(path)
            if p.name == "config.yaml":
                calls["config"] += 1
            else:
                calls["files"].append((p.name, artifact_path))

        monkeypatch.setattr(
            "nemo_evaluator_launcher.exporters.mlflow.mlflow.log_artifact",
            _log_artifact,
            raising=True,
        )

        jd = JobData("mA", "mA.0", 0.0, "local", {"output_dir": str(tmp_path)}, {})
        exp = MLflowExporter({"tracking_uri": "http://ml"})
        monkeypatch.setattr(
            exp, "_get_existing_run_info", lambda *a, **k: (False, None), raising=False
        )

        res = exp.export_job(jd)
        assert res.success
        assert res.metadata["artifacts_logged"] == 1
        assert calls["config"] == 1
        assert calls["files"] == [("results.yml", "taskX/artifacts")]
