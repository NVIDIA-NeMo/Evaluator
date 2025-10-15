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
"""Weight & Biases exporter tests."""

import sys
import types
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from nemo_evaluator_launcher.common.execdb import ExecutionDB, JobData
from nemo_evaluator_launcher.exporters.wandb import WandBExporter


class TestWandBExporter:
    def test_not_available(self):
        with patch("nemo_evaluator_launcher.exporters.wandb.WANDB_AVAILABLE", False):
            res = WandBExporter().export_job(Mock())
            assert not res.success
            assert "not installed" in res.message

    def test_wandb_export_no_metrics(self):
        job_data = JobData(
            invocation_id="test1234",
            job_id="test1234.0",
            timestamp=1234567890.0,
            executor="local",
            data={"output_dir": "/tmp/test"},
        )
        with patch("nemo_evaluator_launcher.exporters.wandb.WANDB_AVAILABLE", True):
            with patch(
                "nemo_evaluator_launcher.exporters.wandb.extract_accuracy_metrics",
                return_value={},
            ):
                exporter = WandBExporter({"entity": "test", "project": "test"})
                result = exporter.export_job(job_data)
                assert not result.success
                assert "No metrics found" in result.message

    @pytest.mark.parametrize("log_mode", ["per_task", "multi_task"])
    def test_export_job_ok(self, monkeypatch, wandb_fake, log_mode):
        _W, _Run = wandb_fake
        jd = JobData("i1", "i1.0", 0.0, "local", {"output_dir": "/tmp"}, {})
        monkeypatch.setattr(
            "nemo_evaluator_launcher.exporters.wandb.extract_accuracy_metrics",
            lambda *_: {"demo_accuracy": 0.9},
            raising=True,
        )
        if log_mode == "multi_task":
            # Mock the check_existing_run for multi_task mode
            exp = WandBExporter(
                {
                    "entity": "e",
                    "project": "p",
                    "log_mode": log_mode,
                    "log_artifacts": False,
                }
            )
            monkeypatch.setattr(
                exp, "_check_existing_run", lambda *a, **k: (False, None), raising=False
            )
        else:
            exp = WandBExporter(
                {
                    "entity": "e",
                    "project": "p",
                    "log_mode": log_mode,
                    "log_artifacts": False,
                }
            )

        res = exp.export_job(jd)
        assert res.success
        assert res.metadata.get("run_id")

    def test_per_task_ok(self, monkeypatch, wandb_fake):
        _W, _Run = wandb_fake
        jd = JobData("i1", "i1.0", 0.0, "local", {"output_dir": "/tmp"}, {})
        monkeypatch.setattr(
            "nemo_evaluator_launcher.exporters.wandb.extract_accuracy_metrics",
            lambda *_: {"demo_accuracy": 0.9},
            raising=True,
        )
        res = WandBExporter(
            {
                "entity": "e",
                "project": "p",
                "log_mode": "per_task",
                "log_artifacts": False,
            }
        ).export_job(jd)
        assert res.success
        assert res.metadata.get("run_id")

    def test_multi_task_resume(self, monkeypatch, wandb_fake):
        _W, _Run = wandb_fake
        jd = JobData("i2", "i2.1", 0.0, "local", {"output_dir": "/tmp"}, {})
        monkeypatch.setattr(
            "nemo_evaluator_launcher.exporters.wandb.extract_accuracy_metrics",
            lambda *_: {"x": 1.0},
            raising=True,
        )
        exp = WandBExporter(
            {
                "entity": "e",
                "project": "p",
                "log_mode": "multi_task",
                "log_artifacts": False,
            }
        )
        monkeypatch.setattr(
            exp, "_check_existing_run", lambda *a, **k: (True, "rid"), raising=False
        )
        res = exp.export_job(jd)
        assert res.success
        assert res.metadata.get("run_id")

    def test_config_update_exception_handling(self, monkeypatch, wandb_fake):
        _W, _Run = wandb_fake
        jd = JobData("i3", "i3.0", 0.0, "local", {"output_dir": "/tmp"}, {})

        # Make config.update raise
        class _RunWithBrokenConfig:
            def __init__(self, *a, **k):
                self.id = "run123"
                self.url = "http://wandb/run/123"
                self.summary = {}
                self.config = Mock()
                self.config.get.return_value = []
                self.config.update.side_effect = Exception("Config update failed")

            def define_metric(self, *a, **k): ...
            def log(self, *a, **k): ...
            def log_artifact(self, *a, **k): ...

        _W.init = staticmethod(lambda **kwargs: _RunWithBrokenConfig())

        monkeypatch.setattr(
            "nemo_evaluator_launcher.exporters.wandb.extract_accuracy_metrics",
            lambda *_: {"demo_accuracy": 0.9},
            raising=True,
        )

        exp = WandBExporter({"entity": "e", "project": "p", "log_mode": "multi_task"})
        res = exp.export_job(jd)
        assert res.success  # Should still succeed despite config update failure

    def test_export_job_exception_path(self, monkeypatch, wandb_fake):
        _W, _Run = wandb_fake
        jd = JobData("w1", "w1.0", 0.0, "local", {"output_dir": "/tmp"}, {})
        monkeypatch.setattr(
            "nemo_evaluator_launcher.exporters.wandb.extract_accuracy_metrics",
            lambda *_: {"acc": 1.0},
            raising=True,
        )
        # Force an exception inside export_job after metrics extraction
        monkeypatch.setattr(
            "nemo_evaluator_launcher.exporters.wandb.WandBExporter._create_wandb_run",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
            raising=True,
        )
        res = WandBExporter({"entity": "e", "project": "p"}).export_job(jd)
        assert not res.success
        assert "Failed" in res.message

    def test_export_invocation_no_jobs_early_return_wandb(self, wandb_fake):
        _W, _Run = wandb_fake
        res = WandBExporter({"entity": "e", "project": "p"}).export_invocation(
            "missing1234"
        )
        assert res["success"] is False
        assert "No jobs found" in res["error"]

    def test_export_invocation_exception_path_wandb(
        self, mock_execdb, monkeypatch, wandb_fake, tmp_path: Path
    ):
        _W, _Run = wandb_fake
        inv = "cafebabe"
        db = ExecutionDB()
        for i in range(2):
            db.write_job(
                JobData(
                    inv, f"{inv}.{i}", 0.0, "local", {"output_dir": str(tmp_path)}, {}
                )
            )
        # Ensure non-empty metrics so we reach _create_wandb_run
        monkeypatch.setattr(
            "nemo_evaluator_launcher.exporters.wandb.extract_accuracy_metrics",
            lambda *_: {"acc": 0.9},
            raising=True,
        )
        # Make the invocation path raise inside the try block
        monkeypatch.setattr(
            "nemo_evaluator_launcher.exporters.wandb.WandBExporter._create_wandb_run",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("invoke failed")),
            raising=True,
        )
        res = WandBExporter({"entity": "e", "project": "p"}).export_invocation(inv)
        assert res["success"] is False
        assert "W&B export failed" in res["error"]

    def test_log_artifacts_success(self, monkeypatch, wandb_fake, tmp_path: Path):
        _W, _Run = wandb_fake
        # Stage artifacts returned by LocalExporter
        stage_dir = tmp_path / "stage"
        (stage_dir / "artifacts").mkdir(parents=True, exist_ok=True)
        (stage_dir / "artifacts" / "results.yml").write_text("ok", encoding="utf-8")

        class _LE:
            def __init__(self, *_a, **_k): ...
            def export_job(self, *_a, **_k):
                return types.SimpleNamespace(
                    success=True, message="ok", dest=str(stage_dir)
                )

        calls = {"added": []}

        class _Artifact:
            def add_file(self, path, name=None):
                calls["added"].append((Path(path).name, name))

        monkeypatch.setattr(
            "nemo_evaluator_launcher.exporters.wandb.LocalExporter", _LE, raising=True
        )
        monkeypatch.setattr(
            "nemo_evaluator_launcher.exporters.wandb.get_available_artifacts",
            lambda *_: ["results.yml"],
            raising=True,
        )
        monkeypatch.setattr(
            "nemo_evaluator_launcher.exporters.wandb.get_task_name",
            lambda *_: "taskX",
            raising=True,
        )

        jd = JobData("wA", "wA.0", 0.0, "local", {"output_dir": str(tmp_path)}, {})
        exp = WandBExporter({})
        logged = exp._log_artifacts(jd, {"log_artifacts": True}, _Artifact())
        # wandb logs config.yaml fallback + results.yml when log_artifacts is True
        assert "config.yaml" in logged
        assert "results.yml" in logged
        assert len(calls["added"]) == 2  # config.yaml + results.yml

    def test_log_artifacts_exception_returns_empty(
        self, monkeypatch, wandb_fake, tmp_path: Path
    ):
        _W, _Run = wandb_fake

        class _LE:
            def __init__(self, *_a, **_k): ...
            def export_job(self, *_a, **_k):
                raise RuntimeError("boom")

        monkeypatch.setattr(
            "nemo_evaluator_launcher.exporters.wandb.LocalExporter", _LE, raising=True
        )

        jd = JobData("wB", "wB.0", 0.0, "local", {"output_dir": str(tmp_path)}, {})
        exp = WandBExporter({})
        logged = exp._log_artifacts(
            jd,
            {"log_artifacts": True},
            types.SimpleNamespace(add_file=lambda *a, **k: None),
        )
        assert logged == []

    def test_check_existing_run_missing_entity_project(self, monkeypatch, wandb_fake):
        _W, _Run = wandb_fake
        jd = JobData("wC", "wC.0", 0.0, "local", {"output_dir": "/tmp"}, {})
        exp = WandBExporter({})
        exists, run_id = exp._check_existing_run(
            "ident", jd, {"entity": None, "project": None}
        )
        assert exists is False and run_id is None

    def test_check_existing_run_webhook_id(self, monkeypatch, wandb_fake):
        _W, _Run = wandb_fake
        api = types.SimpleNamespace(
            run=lambda path: types.SimpleNamespace(id="abc123"), runs=lambda *_: []
        )
        fake_wandb = types.SimpleNamespace(Api=lambda: api)
        monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

        jd = JobData(
            "wD",
            "wD.0",
            0.0,
            "local",
            {
                "output_dir": "/tmp",
                "webhook_metadata": {"webhook_source": "wandb", "run_id": "abc123"},
            },
            {},
        )
        exp = WandBExporter({})
        exists, run_id = exp._check_existing_run(
            "ident", jd, {"entity": "e", "project": "p", "triggered_by_webhook": True}
        )
        assert exists is True and run_id == "abc123"

    def test_check_existing_run_name_match(self, monkeypatch, wandb_fake):
        _W, _Run = wandb_fake
        api = types.SimpleNamespace(
            run=lambda *_: types.SimpleNamespace(id="noop"),
            runs=lambda *_: [types.SimpleNamespace(display_name="my-name", id="rid1")],
        )
        fake_wandb = types.SimpleNamespace(Api=lambda: api)
        monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

        jd = JobData("wE", "wE.0", 0.0, "local", {"output_dir": "/tmp"}, {})
        exp = WandBExporter({})
        exists, run_id = exp._check_existing_run(
            "ident", jd, {"entity": "e", "project": "p", "name": "my-name"}
        )
        assert exists is True and run_id == "rid1"

    def test_check_existing_run_default_pattern_match(self, monkeypatch, wandb_fake):
        _W, _Run = wandb_fake
        api = types.SimpleNamespace(
            run=lambda *_: types.SimpleNamespace(id="noop"),
            runs=lambda *_: [
                types.SimpleNamespace(display_name="eval-ident", id="rid2")
            ],
        )
        fake_wandb = types.SimpleNamespace(Api=lambda: api)
        monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

        jd = JobData("wF", "wF.0", 0.0, "local", {"output_dir": "/tmp"}, {})
        exp = WandBExporter({})
        exists, run_id = exp._check_existing_run(
            "ident", jd, {"entity": "e", "project": "p"}
        )
        assert exists is True and run_id == "rid2"

    def test_check_existing_run_no_match(self, monkeypatch, wandb_fake):
        _W, _Run = wandb_fake
        api = types.SimpleNamespace(
            run=lambda *_: types.SimpleNamespace(id="noop"), runs=lambda *_: []
        )
        monkeypatch.setattr(
            "nemo_evaluator_launcher.exporters.wandb.wandb.Api",
            lambda: api,
            raising=True,
        )
        jd = JobData("wG", "wG.0", 0.0, "local", {"output_dir": "/tmp"}, {})
        exp = WandBExporter({})
        exists, run_id = exp._check_existing_run(
            "ident", jd, {"entity": "e", "project": "p"}
        )
        assert exists is False and run_id is None
