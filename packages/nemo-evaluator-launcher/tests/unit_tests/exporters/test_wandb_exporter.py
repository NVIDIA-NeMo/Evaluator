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

from nemo_evaluator_launcher.exporters.utils import DataForExport
from nemo_evaluator_launcher.exporters.wandb import WandBExporter


class TestWandBExporter:
    def test_not_available(self, tmp_path):
        with patch("nemo_evaluator_launcher.exporters.wandb.WANDB_AVAILABLE", False):
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
                invocation_id="test1234",
                job_id="test1234.0",
                timestamp=1234567890.0,
                job_data={},
            )
            successful, failed, skipped = WandBExporter().export_jobs([data])
            assert len(successful) == 0
            assert len(failed) == 1
            assert failed[0] == "test1234.0"

    def test_wandb_export_no_metrics(self, tmp_path):
        with patch("nemo_evaluator_launcher.exporters.wandb.WANDB_AVAILABLE", True):
            data = DataForExport(
                artifacts_dir=tmp_path / "artifacts",
                logs_dir=None,
                config={},
                model_id="test-model",
                metrics={},  # No metrics
                harness="lm-eval",
                task="mmlu",
                container="test:latest",
                executor="local",
                invocation_id="test1234",
                job_id="test1234.0",
                timestamp=1234567890.0,
                job_data={},
            )
            exporter = WandBExporter({"entity": "test", "project": "test"})
            successful, failed, skipped = exporter.export_jobs([data])
            assert len(successful) == 0
            assert len(skipped) == 1
            assert skipped[0] == "test1234.0"

    @pytest.mark.parametrize("log_mode", ["per_task", "multi_task"])
    def test_export_jobs_ok(self, monkeypatch, wandb_fake, log_mode, tmp_path):
        _W, _Run = wandb_fake
        data = DataForExport(
            artifacts_dir=tmp_path / "artifacts",
            logs_dir=None,
            config={},
            model_id="test-model",
            metrics={"demo_accuracy": 0.9},
            harness="lm-eval",
            task="mmlu",
            container="test:latest",
            executor="local",
            invocation_id="i1",
            job_id="i1.0",
            timestamp=0.0,
            job_data={},
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
                exp, "_check_existing_run", lambda *a, **k: None, raising=False
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

        successful, failed, skipped = exp.export_jobs([data])
        assert len(successful) == 1
        assert successful[0] == "i1.0"
        assert len(failed) == 0

    def test_per_task_ok(self, monkeypatch, wandb_fake, tmp_path):
        _W, _Run = wandb_fake
        data = DataForExport(
            artifacts_dir=tmp_path / "artifacts",
            logs_dir=None,
            config={},
            model_id="test-model",
            metrics={"demo_accuracy": 0.9},
            harness="lm-eval",
            task="mmlu",
            container="test:latest",
            executor="local",
            invocation_id="i1",
            job_id="i1.0",
            timestamp=0.0,
            job_data={},
        )
        successful, failed, skipped = WandBExporter(
            {
                "entity": "e",
                "project": "p",
                "log_mode": "per_task",
                "log_artifacts": False,
            }
        ).export_jobs([data])
        assert len(successful) == 1
        assert successful[0] == "i1.0"

    def test_multi_task_resume(self, monkeypatch, wandb_fake, tmp_path):
        _W, _Run = wandb_fake
        data = DataForExport(
            artifacts_dir=tmp_path / "artifacts",
            logs_dir=None,
            config={},
            model_id="test-model",
            metrics={"x": 1.0},
            harness="lm-eval",
            task="mmlu",
            container="test:latest",
            executor="local",
            invocation_id="i2",
            job_id="i2.1",
            timestamp=0.0,
            job_data={},
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
            exp, "_check_existing_run", lambda *a, **k: "rid", raising=False
        )
        successful, failed, skipped = exp.export_jobs([data])
        assert len(successful) == 1
        assert successful[0] == "i2.1"

    def test_config_update_exception_handling(self, monkeypatch, wandb_fake, tmp_path):
        _W, _Run = wandb_fake
        data = DataForExport(
            artifacts_dir=tmp_path / "artifacts",
            logs_dir=None,
            config={},
            model_id="test-model",
            metrics={"demo_accuracy": 0.9},
            harness="lm-eval",
            task="mmlu",
            container="test:latest",
            executor="local",
            invocation_id="i3",
            job_id="i3.0",
            timestamp=0.0,
            job_data={},
        )

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
            def finish(self): ...

        _W.init = staticmethod(lambda **kwargs: _RunWithBrokenConfig())

        exp = WandBExporter({"entity": "e", "project": "p", "log_mode": "multi_task"})
        successful, failed, skipped = exp.export_jobs([data])
        assert (
            len(successful) == 1
        )  # Should still succeed despite config update failure

    def test_export_jobs_exception_path(self, monkeypatch, wandb_fake, tmp_path):
        _W, _Run = wandb_fake
        data = DataForExport(
            artifacts_dir=tmp_path / "artifacts",
            logs_dir=None,
            config={},
            model_id="test-model",
            metrics={"acc": 1.0},
            harness="lm-eval",
            task="mmlu",
            container="test:latest",
            executor="local",
            invocation_id="w1",
            job_id="w1.0",
            timestamp=0.0,
            job_data={},
        )
        # Force an exception inside export_jobs after metrics extraction
        monkeypatch.setattr(
            "nemo_evaluator_launcher.exporters.wandb.WandBExporter._create_wandb_run",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
            raising=True,
        )
        successful, failed, skipped = WandBExporter(
            {"entity": "e", "project": "p"}
        ).export_jobs([data])
        assert len(failed) == 1
        assert failed[0] == "w1.0"

    def test_export_jobs_multi_task_exception_path(
        self, monkeypatch, wandb_fake, tmp_path: Path
    ):
        _W, _Run = wandb_fake
        # Create multiple data objects for the same invocation
        data1 = DataForExport(
            artifacts_dir=tmp_path / "artifacts1",
            logs_dir=None,
            config={},
            model_id="test-model",
            metrics={"acc": 0.9},
            harness="lm-eval",
            task="mmlu",
            container="test:latest",
            executor="local",
            invocation_id="cafebabe",
            job_id="cafebabe.0",
            timestamp=0.0,
            job_data={},
        )
        data2 = DataForExport(
            artifacts_dir=tmp_path / "artifacts2",
            logs_dir=None,
            config={},
            model_id="test-model",
            metrics={"f1": 0.8},
            harness="lm-eval",
            task="gsm8k",
            container="test:latest",
            executor="local",
            invocation_id="cafebabe",
            job_id="cafebabe.1",
            timestamp=0.0,
            job_data={},
        )
        # Make the invocation path raise inside the try block
        monkeypatch.setattr(
            "nemo_evaluator_launcher.exporters.wandb.WandBExporter._create_wandb_run",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("invoke failed")),
            raising=True,
        )
        successful, failed, skipped = WandBExporter(
            {"entity": "e", "project": "p", "log_mode": "multi_task"}
        ).export_jobs([data1, data2])
        assert len(failed) == 2
        assert "cafebabe.0" in failed
        assert "cafebabe.1" in failed

    def test_log_artifacts_success(self, monkeypatch, wandb_fake, tmp_path: Path):
        _W, _Run = wandb_fake
        # Create artifacts directory with files
        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        (artifacts_dir / "results.yml").write_text("ok", encoding="utf-8")

        calls = {"added": []}

        class _Artifact:
            def add_file(self, path, name=None):
                calls["added"].append((Path(path).name, name))

        data = DataForExport(
            artifacts_dir=artifacts_dir,
            logs_dir=None,
            config={"test": "config"},
            model_id="test-model",
            metrics={"acc": 0.9},
            harness="lm-eval",
            task="taskX",
            container="test:latest",
            executor="local",
            invocation_id="wA",
            job_id="wA.0",
            timestamp=0.0,
            job_data={},
        )

        exp = WandBExporter({"log_artifacts": True})
        logged = exp._log_artifacts(data, _Artifact())
        # wandb logs config.yaml fallback + results.yml when log_artifacts is True
        assert "config.yml" in logged
        assert "results.yml" in logged
        assert len(calls["added"]) == 2  # config.yaml + results.yml

    def test_log_artifacts_exception_returns_empty(
        self, monkeypatch, wandb_fake, tmp_path: Path
    ):
        _W, _Run = wandb_fake

        # Create data with non-existent artifacts directory to trigger exception
        data = DataForExport(
            artifacts_dir=tmp_path / "nonexistent",  # This directory doesn't exist
            logs_dir=None,
            config={},
            model_id="test-model",
            metrics={"acc": 0.9},
            harness="lm-eval",
            task="taskX",
            container="test:latest",
            executor="local",
            invocation_id="wB",
            job_id="wB.0",
            timestamp=0.0,
            job_data={},
        )

        exp = WandBExporter({"log_artifacts": True})
        logged = exp._log_artifacts(
            data,
            types.SimpleNamespace(add_file=lambda *a, **k: None),
        )
        assert logged == []

    def test_check_existing_run_missing_entity_project(
        self, monkeypatch, wandb_fake, tmp_path
    ):
        _W, _Run = wandb_fake
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
            invocation_id="wC",
            job_id="wC.0",
            timestamp=0.0,
            job_data={},
        )
        exp = WandBExporter({"entity": None, "project": None})
        run_id = exp._check_existing_run("ident", data)
        assert run_id is None

    def test_check_existing_run_webhook_id(self, monkeypatch, wandb_fake, tmp_path):
        _W, _Run = wandb_fake
        api = types.SimpleNamespace(
            run=lambda path: types.SimpleNamespace(id="abc123"), runs=lambda *_: []
        )
        fake_wandb = types.SimpleNamespace(Api=lambda: api)
        monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

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
            invocation_id="wD",
            job_id="wD.0",
            timestamp=0.0,
            job_data={
                "webhook_metadata": {"webhook_source": "wandb", "run_id": "abc123"}
            },
        )
        exp = WandBExporter(
            {"entity": "e", "project": "p", "triggered_by_webhook": True}
        )
        run_id = exp._check_existing_run("ident", data)
        assert run_id == "abc123"

    def test_check_existing_run_name_match(self, monkeypatch, wandb_fake, tmp_path):
        _W, _Run = wandb_fake
        api = types.SimpleNamespace(
            run=lambda *_: types.SimpleNamespace(id="noop"),
            runs=lambda *_: [types.SimpleNamespace(display_name="my-name", id="rid1")],
        )
        fake_wandb = types.SimpleNamespace(Api=lambda: api)
        monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

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
            invocation_id="wE",
            job_id="wE.0",
            timestamp=0.0,
            job_data={},
        )
        exp = WandBExporter({"entity": "e", "project": "p", "name": "my-name"})
        run_id = exp._check_existing_run("ident", data)
        assert run_id == "rid1"

    def test_check_existing_run_default_pattern_match(
        self, monkeypatch, wandb_fake, tmp_path
    ):
        _W, _Run = wandb_fake
        api = types.SimpleNamespace(
            run=lambda *_: types.SimpleNamespace(id="noop"),
            runs=lambda *_: [
                types.SimpleNamespace(display_name="eval-ident", id="rid2")
            ],
        )
        fake_wandb = types.SimpleNamespace(Api=lambda: api)
        monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

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
            invocation_id="wF",
            job_id="wF.0",
            timestamp=0.0,
            job_data={},
        )
        exp = WandBExporter({"entity": "e", "project": "p"})
        run_id = exp._check_existing_run("ident", data)
        assert run_id == "rid2"

    def test_check_existing_run_no_match(self, monkeypatch, wandb_fake, tmp_path):
        _W, _Run = wandb_fake
        api = types.SimpleNamespace(
            run=lambda *_: types.SimpleNamespace(id="noop"), runs=lambda *_: []
        )
        monkeypatch.setattr(
            "nemo_evaluator_launcher.exporters.wandb.wandb.Api",
            lambda: api,
            raising=True,
        )
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
            invocation_id="wG",
            job_id="wG.0",
            timestamp=0.0,
            job_data={},
        )
        exp = WandBExporter({"entity": "e", "project": "p"})
        run_id = exp._check_existing_run("ident", data)
        assert run_id is None
