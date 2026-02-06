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
"""Google Sheets exporter tests."""

import types
from pathlib import Path

import pytest

from nemo_evaluator_launcher.exporters.gsheets import GSheetsExporter
from nemo_evaluator_launcher.exporters.utils import DataForExport


class TestGSheetsExporter:
    @pytest.mark.parametrize("executor", ["local", "slurm"])
    def test_not_available(self, monkeypatch, tmp_path: Path, executor):
        monkeypatch.setattr(
            "nemo_evaluator_launcher.exporters.gsheets.GSPREAD_AVAILABLE",
            False,
            raising=True,
        )
        data = DataForExport(
            artifacts_dir=tmp_path / "artifacts",
            logs_dir=None,
            metrics={"task_acc": 0.9},
            harness="lm-eval",
            task="task",
            invocation_id="i1",
            job_id="i1.0",
            executor=executor,
            model_id="test-model",
            container="local",
            timestamp=0.0,
            config={},
            job_data={},
        )
        successful, failed, skipped = GSheetsExporter({}).export_jobs([data])
        assert len(successful) == 0
        assert len(failed) == 1
        assert failed[0] == "i1.0"

    def test_export_job_ok(self, monkeypatch, tmp_path: Path):
        rows = []
        headers = []

        class _WS:
            def get_all_values(self):
                return []

            def update(self, _range, values):
                nonlocal headers
                headers = values[0]

            def format(self, *_a, **_k): ...
            def append_row(self, row):
                rows.append(row)

        class _SH:
            url = "http://fake-sheet"
            sheet1 = _WS()

            def share(self, *a, **k): ...

        class _Client:
            def open(self, *_):
                raise SpreadsheetNotFound()

            def create(self, *_):
                return _SH()

        SpreadsheetNotFound = type("SpreadsheetNotFound", (Exception,), {})
        fake_gspread = types.SimpleNamespace(
            service_account=lambda filename=None: _Client(),
            SpreadsheetNotFound=SpreadsheetNotFound,
        )

        # Enable gspread availability
        monkeypatch.setattr(
            "nemo_evaluator_launcher.exporters.gsheets.GSPREAD_AVAILABLE",
            True,
            raising=True,
        )

        # Add gspread attribute to the module
        import nemo_evaluator_launcher.exporters.gsheets as gs_mod

        monkeypatch.setattr(gs_mod, "gspread", fake_gspread, raising=False)

        data = DataForExport(
            artifacts_dir=tmp_path / "artifacts",
            logs_dir=None,
            metrics={"task_accuracy": 0.9},
            harness="lm-eval",
            task="task",
            invocation_id="inv1",
            job_id="inv1.0",
            executor="local",
            model_id="test-model",
            container="local",
            timestamp=0.0,
            config={},
            job_data={},
        )
        successful, failed, skipped = GSheetsExporter({}).export_jobs([data])

        assert len(successful) == 1
        assert successful[0] == "inv1.0"
        assert len(failed) == 0
        assert "accuracy" in headers
        assert len(rows) == 1

    def test_export_multiple_jobs_ok(self, monkeypatch, tmp_path: Path):
        rows = []
        headers = ["Model Name", "Task Name", "Invocation ID", "Job ID", "Executor"]

        class _WS:
            def get_all_values(self):
                return [headers]

            def update(self, _range, values):
                nonlocal headers
                headers = values[0]

            def format(self, *_a, **_k): ...
            def append_row(self, row):
                rows.append(row)

        class _SH:
            url = "http://fake-sheet"
            sheet1 = _WS()

            def share(self, *a, **k): ...

        class _Client:
            def open(self, *_):
                return _SH()

            def create(self, *_):
                return _SH()

        SpreadsheetNotFound = type("SpreadsheetNotFound", (Exception,), {})
        fake_gspread = types.SimpleNamespace(
            service_account=lambda filename=None: _Client(),
            SpreadsheetNotFound=SpreadsheetNotFound,
        )

        # Enable gspread availability
        monkeypatch.setattr(
            "nemo_evaluator_launcher.exporters.gsheets.GSPREAD_AVAILABLE",
            True,
            raising=True,
        )

        # Add gspread attribute to the module
        import nemo_evaluator_launcher.exporters.gsheets as gs_mod

        monkeypatch.setattr(gs_mod, "gspread", fake_gspread, raising=False)

        inv = "zz11yy22"
        data_list = [
            DataForExport(
                artifacts_dir=tmp_path / "artifacts0",
                logs_dir=None,
                metrics={"t0_accuracy": 0.8},
                harness="lm-eval",
                task="t0",
                invocation_id=inv,
                job_id=f"{inv}.0",
                executor="local",
                model_id="test-model",
                container="local",
                timestamp=0.0,
                config={},
                job_data={},
            ),
            DataForExport(
                artifacts_dir=tmp_path / "artifacts1",
                logs_dir=None,
                metrics={"t1_accuracy": 0.9},
                harness="lm-eval",
                task="t1",
                invocation_id=inv,
                job_id=f"{inv}.1",
                executor="local",
                model_id="test-model",
                container="local",
                timestamp=0.0,
                config={},
                job_data={},
            ),
        ]

        successful, failed, skipped = GSheetsExporter({}).export_jobs(data_list)
        assert len(successful) == 2
        assert len(failed) == 0
        assert "accuracy" in headers
        assert len(rows) == 2

    def test_export_job_no_metrics(self, monkeypatch, tmp_path: Path):
        # Enable gspread availability
        monkeypatch.setattr(
            "nemo_evaluator_launcher.exporters.gsheets.GSPREAD_AVAILABLE",
            True,
            raising=True,
        )

        data = DataForExport(
            artifacts_dir=tmp_path / "artifacts",
            logs_dir=None,
            metrics={},  # No metrics
            harness="lm-eval",
            task="task",
            invocation_id="i5",
            job_id="i5.0",
            executor="local",
            model_id="test-model",
            container="local",
            timestamp=0.0,
            config={},
            job_data={},
        )

        successful, failed, skipped = GSheetsExporter({}).export_jobs([data])
        assert len(successful) == 0
        assert len(failed) == 0
        assert len(skipped) == 1
        assert skipped[0] == "i5.0"

    def test_export_open_existing_and_headers_update(self, monkeypatch, tmp_path: Path):
        rows = []
        headers = [
            "Model Name",
            "Task Name",
            "Invocation ID",
            "Job ID",
            "Executor",
        ]  # no accuracy yet

        class _WS:
            def get_all_values(self):
                return [headers]

            def update(self, _range, values):
                nonlocal headers
                headers = values[0]

            def format(self, *_a, **_k): ...
            def append_row(self, row):
                rows.append(row)

        class _SH:
            url = "http://fake-sheet"
            sheet1 = _WS()

            def share(self, *a, **k): ...

        class _Client:
            def open(self, *_):
                return _SH()

            def create(self, *_):
                return _SH()

        fake_gspread = types.SimpleNamespace(
            service_account=lambda filename=None: _Client(),
            SpreadsheetNotFound=type("SpreadsheetNotFound", (Exception,), {}),
        )

        # Enable gspread availability
        monkeypatch.setattr(
            "nemo_evaluator_launcher.exporters.gsheets.GSPREAD_AVAILABLE",
            True,
            raising=True,
        )

        # Add gspread attribute to the module
        import nemo_evaluator_launcher.exporters.gsheets as gs_mod

        monkeypatch.setattr(gs_mod, "gspread", fake_gspread, raising=False)

        inv = "aa11bb22"
        data = DataForExport(
            artifacts_dir=tmp_path / "artifacts",
            logs_dir=None,
            metrics={"t0_accuracy": 1.0},
            harness="lm-eval",
            task="t0",
            invocation_id=inv,
            job_id=f"{inv}.0",
            executor="local",
            model_id="test-model",
            container="local",
            timestamp=0.0,
            config={},
            job_data={},
        )

        successful, failed, skipped = GSheetsExporter({}).export_jobs([data])
        assert len(successful) == 1
        assert "accuracy" in headers
        assert len(rows) == 1

    def test_export_exception_path(self, monkeypatch, tmp_path: Path):
        # Enable gspread
        monkeypatch.setattr(
            "nemo_evaluator_launcher.exporters.gsheets.GSPREAD_AVAILABLE",
            True,
            raising=True,
        )
        import nemo_evaluator_launcher.exporters.gsheets as gs_mod

        # Create fake gspread that raises during service_account()
        fake_gspread = types.SimpleNamespace(
            service_account=lambda **kwargs: (_ for _ in ()).throw(
                RuntimeError("GSheets connection failed")
            ),
            SpreadsheetNotFound=type("SpreadsheetNotFound", (Exception,), {}),
        )
        monkeypatch.setattr(gs_mod, "gspread", fake_gspread, raising=False)

        data = DataForExport(
            artifacts_dir=tmp_path / "artifacts",
            logs_dir=None,
            metrics={"task_acc": 0.9},
            harness="lm-eval",
            task="task",
            invocation_id="gx",
            job_id="gx.0",
            executor="local",
            model_id="test-model",
            container="local",
            timestamp=0.0,
            config={},
            job_data={},
        )

        successful, failed, skipped = GSheetsExporter({}).export_jobs([data])
        assert len(successful) == 0
        assert len(failed) == 1
        assert failed[0] == "gx.0"

    def test_export_service_account_file_and_create_sheet(
        self, monkeypatch, tmp_path: Path
    ):
        # Enable gspread
        monkeypatch.setattr(
            "nemo_evaluator_launcher.exporters.gsheets.GSPREAD_AVAILABLE",
            True,
            raising=True,
        )
        import nemo_evaluator_launcher.exporters.gsheets as gs_mod

        calls = {"service_account": [], "create": 0, "share": 0}

        class _WS:
            def get_all_values(self):
                return []

            def update(self, *_): ...
            def format(self, *_): ...
            def append_row(self, *_): ...

        class _SH:
            url = "http://fake-sheet"
            sheet1 = _WS()

            def share(self, *a, **k):
                calls["share"] += 1

        class _Client:
            def open(self, name):
                raise SpreadsheetNotFound()

            def create(self, name):
                calls["create"] += 1
                return _SH()

        SpreadsheetNotFound = type("SpreadsheetNotFound", (Exception,), {})
        fake_gspread = types.SimpleNamespace(
            service_account=lambda filename=None: (
                calls["service_account"].append(filename),
                _Client(),
            )[1],
            SpreadsheetNotFound=SpreadsheetNotFound,
        )
        monkeypatch.setattr(gs_mod, "gspread", fake_gspread, raising=False)
        monkeypatch.setattr(
            "nemo_evaluator_launcher.exporters.gsheets.GSheetsExporter._get_or_update_headers",
            lambda *_: ["Model Name", "Task Name", "accuracy"],
            raising=True,
        )
        monkeypatch.setattr(
            "nemo_evaluator_launcher.exporters.gsheets.GSheetsExporter._prepare_row_data",
            lambda *_: ["model", "task", "0.95"],
            raising=True,
        )

        data = DataForExport(
            artifacts_dir=tmp_path / "artifacts",
            logs_dir=None,
            metrics={"task_accuracy": 0.95},
            harness="lm-eval",
            task="task",
            invocation_id="gy",
            job_id="gy.0",
            executor="local",
            model_id="test-model",
            container="local",
            timestamp=0.0,
            config={},
            job_data={},
        )

        exp = GSheetsExporter({"service_account_file": "/path/to/creds.json"})
        successful, failed, skipped = exp.export_jobs([data])

        assert len(successful) == 1
        assert calls["service_account"] == [
            "/path/to/creds.json"
        ]  # service_account_file branch
        assert calls["create"] == 1  # SpreadsheetNotFound -> create

    def test_export_spreadsheet_not_found_creates_new(
        self, monkeypatch, tmp_path: Path
    ):
        # Enable gspread
        monkeypatch.setattr(
            "nemo_evaluator_launcher.exporters.gsheets.GSPREAD_AVAILABLE",
            True,
            raising=True,
        )
        import nemo_evaluator_launcher.exporters.gsheets as gs_mod

        calls = {"open": 0, "create": 0, "share": 0}

        class _WS:
            def get_all_values(self):
                return []

            def update(self, *_): ...
            def format(self, *_): ...
            def append_row(self, *_): ...

        class _SH:
            url = "http://new-sheet"
            sheet1 = _WS()

            def share(self, email, perm_type=None, role=None):
                calls["share"] += 1

        class _Client:
            def open(self, name):
                calls["open"] += 1
                raise SpreadsheetNotFound("Sheet not found")

            def create(self, name):
                calls["create"] += 1
                return _SH()

        SpreadsheetNotFound = type("SpreadsheetNotFound", (Exception,), {})
        fake_gspread = types.SimpleNamespace(
            service_account=lambda filename=None: _Client(),
            SpreadsheetNotFound=SpreadsheetNotFound,
        )
        monkeypatch.setattr(gs_mod, "gspread", fake_gspread, raising=False)

        # Mock header/row methods to complete the flow
        monkeypatch.setattr(
            "nemo_evaluator_launcher.exporters.gsheets.GSheetsExporter._get_or_update_headers",
            lambda *_: ["Model Name", "Task Name", "accuracy"],
            raising=True,
        )
        monkeypatch.setattr(
            "nemo_evaluator_launcher.exporters.gsheets.GSheetsExporter._prepare_row_data",
            lambda *_: ["model", "task", "0.95"],
            raising=True,
        )

        data = DataForExport(
            artifacts_dir=tmp_path / "artifacts",
            logs_dir=None,
            metrics={"task_accuracy": 0.95},
            harness="lm-eval",
            task="task",
            invocation_id="create1",
            job_id="create1.0",
            executor="local",
            model_id="test-model",
            container="local",
            timestamp=0.0,
            config={},
            job_data={},
        )

        successful, failed, skipped = GSheetsExporter({}).export_jobs([data])

        assert len(successful) == 1
        assert calls["open"] == 1  # Tried to open first
        assert calls["create"] == 1  # SpreadsheetNotFound -> create

    def test_export_mixed_success_and_failures(self, monkeypatch, tmp_path: Path):
        # Enable gspread
        monkeypatch.setattr(
            "nemo_evaluator_launcher.exporters.gsheets.GSPREAD_AVAILABLE",
            True,
            raising=True,
        )
        import nemo_evaluator_launcher.exporters.gsheets as gs_mod

        rows = []

        class _WS:
            def get_all_values(self):
                return []

            def update(self, *_): ...
            def format(self, *_): ...
            def append_row(self, row):
                if "fail" in str(row):
                    raise RuntimeError("Simulated append failure")
                rows.append(row)

        class _SH:
            url = "http://sheet1"
            sheet1 = _WS()

        class _Client:
            def open(self, *_):
                return _SH()

        fake_gspread = types.SimpleNamespace(
            service_account=lambda filename=None: _Client(),
            SpreadsheetNotFound=type("SpreadsheetNotFound", (Exception,), {}),
        )
        monkeypatch.setattr(gs_mod, "gspread", fake_gspread, raising=False)

        # Mock prepare_row_data to return different data
        call_count = {"calls": 0}

        def _mock_prepare(self, data, metrics, headers):
            call_count["calls"] += 1
            if data.job_id.endswith(".1"):
                return ["fail"]  # This will trigger append failure
            return ["success"]

        monkeypatch.setattr(
            "nemo_evaluator_launcher.exporters.gsheets.GSheetsExporter._prepare_row_data",
            _mock_prepare,
            raising=False,
        )

        data_list = [
            DataForExport(
                artifacts_dir=tmp_path / "artifacts0",
                logs_dir=None,
                metrics={"task_acc": 0.9},
                harness="lm-eval",
                task="task",
                invocation_id="inv1",
                job_id="inv1.0",
                executor="local",
                model_id="test-model",
                container="local",
                timestamp=0.0,
                config={},
                job_data={},
            ),
            DataForExport(
                artifacts_dir=tmp_path / "artifacts1",
                logs_dir=None,
                metrics={"task_acc": 0.8},
                harness="lm-eval",
                task="task",
                invocation_id="inv1",
                job_id="inv1.1",  # This will fail
                executor="local",
                model_id="test-model",
                container="local",
                timestamp=0.0,
                config={},
                job_data={},
            ),
            DataForExport(
                artifacts_dir=tmp_path / "artifacts2",
                logs_dir=None,
                metrics={},  # No metrics - will be skipped
                harness="lm-eval",
                task="task",
                invocation_id="inv1",
                job_id="inv1.2",
                executor="local",
                model_id="test-model",
                container="local",
                timestamp=0.0,
                config={},
                job_data={},
            ),
        ]

        successful, failed, skipped = GSheetsExporter({}).export_jobs(data_list)

        assert len(successful) == 1
        assert successful[0] == "inv1.0"
        assert len(failed) == 1
        assert failed[0] == "inv1.1"
        assert len(skipped) == 1
        assert skipped[0] == "inv1.2"
        assert len(rows) == 1  # Only one successful append

    def test_export_with_metrics_skipping(self, monkeypatch, tmp_path: Path):
        # Enable gspread
        monkeypatch.setattr(
            "nemo_evaluator_launcher.exporters.gsheets.GSPREAD_AVAILABLE",
            True,
            raising=True,
        )
        import nemo_evaluator_launcher.exporters.gsheets as gs_mod

        class _WS:
            def get_all_values(self):
                return []

            def update(self, *_): ...
            def format(self, *_): ...
            def append_row(self, *_): ...

        class _SH:
            url = "http://fake"
            sheet1 = _WS()

        monkeypatch.setattr(
            gs_mod,
            "gspread",
            types.SimpleNamespace(
                service_account=lambda **k: types.SimpleNamespace(
                    open=lambda name: _SH()
                ),
                SpreadsheetNotFound=type("SpreadsheetNotFound", (Exception,), {}),
            ),
            raising=False,
        )

        monkeypatch.setattr(
            "nemo_evaluator_launcher.exporters.gsheets.GSheetsExporter._get_or_update_headers",
            lambda *_: ["Model Name", "accuracy"],
            raising=True,
        )
        monkeypatch.setattr(
            "nemo_evaluator_launcher.exporters.gsheets.GSheetsExporter._prepare_row_data",
            lambda *_: ["model", "0.9"],
            raising=True,
        )

        inv = "metric_fail"
        data_list = [
            DataForExport(
                artifacts_dir=tmp_path / "artifacts0",
                logs_dir=None,
                metrics={"task_accuracy": 0.9},  # Has metrics
                harness="lm-eval",
                task="task",
                invocation_id=inv,
                job_id=f"{inv}.0",
                executor="local",
                model_id="test-model",
                container="local",
                timestamp=0.0,
                config={},
                job_data={},
            ),
            DataForExport(
                artifacts_dir=tmp_path / "artifacts1",
                logs_dir=None,
                metrics={},  # No metrics - will be skipped
                harness="lm-eval",
                task="task",
                invocation_id=inv,
                job_id=f"{inv}.1",
                executor="local",
                model_id="test-model",
                container="local",
                timestamp=0.0,
                config={},
                job_data={},
            ),
        ]

        successful, failed, skipped = GSheetsExporter({}).export_jobs(data_list)

        assert len(successful) == 1  # Only job with metrics
        assert successful[0] == f"{inv}.0"
        assert len(failed) == 0
        assert len(skipped) == 1  # Job without metrics
        assert skipped[0] == f"{inv}.1"
