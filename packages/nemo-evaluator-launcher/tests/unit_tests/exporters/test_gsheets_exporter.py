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

from nemo_evaluator_launcher.common.execdb import ExecutionDB, JobData
from nemo_evaluator_launcher.exporters.gsheets import GSheetsExporter


class TestGSheetsExporter:
    @pytest.mark.parametrize("executor", ["local", "slurm"])
    def test_not_available(self, monkeypatch, executor):
        monkeypatch.setattr(
            "nemo_evaluator_launcher.exporters.gsheets.GSPREAD_AVAILABLE",
            False,
            raising=True,
        )
        jd = JobData("i1", "i1.0", 0.0, executor, {"output_dir": "/tmp"})
        res = GSheetsExporter().export_job(jd)
        assert res.success is False
        assert "not installed" in res.message

    def test_export_invocation_no_jobs(self, monkeypatch):
        monkeypatch.setattr(
            "nemo_evaluator_launcher.exporters.gsheets.GSPREAD_AVAILABLE",
            True,
            raising=True,
        )
        res = GSheetsExporter({}).export_invocation("deadbeef")
        assert res["success"] is False
        assert "No jobs found" in res["error"]

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

        # Add gspread attribute to the module (since it wasn't imported if GSPREAD_AVAILABLE was False)
        import nemo_evaluator_launcher.exporters.gsheets as gs_mod

        monkeypatch.setattr(gs_mod, "gspread", fake_gspread, raising=False)

        monkeypatch.setattr(
            "nemo_evaluator_launcher.exporters.gsheets.GSheetsExporter._get_artifacts_locally",
            lambda *a, **k: (tmp_path, None),
            raising=True,
        )
        monkeypatch.setattr(
            "nemo_evaluator_launcher.exporters.gsheets.extract_accuracy_metrics",
            lambda *_: {"task_accuracy": 0.9},
            raising=True,
        )

        jd = JobData(
            "inv1",
            "inv1.0",
            0.0,
            "local",
            {"output_dir": str(tmp_path)},
            {"evaluation": {"tasks": [{"name": "task"}]}},
        )
        res = GSheetsExporter({}).export_job(jd)

        assert res.success
        assert res.metadata["metrics_logged"] == 1
        assert "accuracy" in headers
        assert len(rows) == 1

    def test_export_invocation_ok(self, monkeypatch, tmp_path: Path):
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

        # Add gspread attribute to the module (since it wasn't imported if GSPREAD_AVAILABLE was False)
        import nemo_evaluator_launcher.exporters.gsheets as gs_mod

        monkeypatch.setattr(gs_mod, "gspread", fake_gspread, raising=False)

        monkeypatch.setattr(
            "nemo_evaluator_launcher.exporters.gsheets.GSheetsExporter._get_artifacts_locally",
            lambda *a, **k: (tmp_path, None),
            raising=True,
        )

        def _metrics(jd, *_):
            return (
                {"task_accuracy": 0.8}
                if jd.job_id.endswith(".0")
                else {"task_accuracy": 0.9}
            )

        monkeypatch.setattr(
            "nemo_evaluator_launcher.exporters.gsheets.extract_accuracy_metrics",
            _metrics,
            raising=True,
        )

        db = ExecutionDB()
        inv = "zz11yy22"
        db.write_job(
            JobData(
                inv,
                f"{inv}.0",
                0.0,
                "local",
                {"output_dir": str(tmp_path)},
                {"evaluation": {"tasks": [{"name": "t0"}]}},
            )
        )
        db.write_job(
            JobData(
                inv,
                f"{inv}.1",
                0.0,
                "local",
                {"output_dir": str(tmp_path)},
                {"evaluation": {"tasks": [{"name": "t1"}]}},
            )
        )

        res = GSheetsExporter({}).export_invocation(inv)
        assert res["success"]
        assert res["metadata"]["rows_added"] == 2
        assert "accuracy" in headers
        assert len(rows) == 2

    def test_get_artifacts_locally_localexporter_failure(
        self, monkeypatch, tmp_path: Path
    ):
        exporter = GSheetsExporter({})
        jd = JobData("i1", "i1.0", 0.0, "local", {"output_dir": str(tmp_path)})

        # Force mkdtemp to return a predictable path and ensure cleanup is called
        temp_dir = tmp_path / "gs_temp"
        monkeypatch.setattr(
            "tempfile.mkdtemp", lambda prefix="": str(temp_dir), raising=True
        )

        calls = {"rm": []}
        monkeypatch.setattr(
            "shutil.rmtree",
            lambda p: calls["rm"].append(str(p)),
            raising=True,
        )

        class _LE:
            def __init__(self, *a, **k): ...
            def export_job(self, *a, **k):
                return types.SimpleNamespace(
                    success=False, message="boom", dest=str(tmp_path / "x")
                )

        monkeypatch.setattr(
            "nemo_evaluator_launcher.exporters.gsheets.LocalExporter", _LE, raising=True
        )

        artifacts_dir, tmpd = exporter._get_artifacts_locally(jd)
        assert artifacts_dir is None and tmpd is None
        assert str(temp_dir) in calls["rm"]

    def test_get_artifacts_locally_missing_artifacts_dir(
        self, monkeypatch, tmp_path: Path
    ):
        exporter = GSheetsExporter({})
        jd = JobData("i2", "i2.0", 0.0, "local", {"output_dir": str(tmp_path)})

        temp_dir = tmp_path / "gs_temp2"
        monkeypatch.setattr(
            "tempfile.mkdtemp", lambda prefix="": str(temp_dir), raising=True
        )

        calls = {"rm": []}
        monkeypatch.setattr(
            "shutil.rmtree", lambda p: calls["rm"].append(str(p)), raising=True
        )

        class _LE:
            def __init__(self, *a, **k): ...
            def export_job(self, *a, **k):
                # success True but missing artifacts dir
                d = tmp_path / "le_dest"
                d.mkdir(parents=True, exist_ok=True)
                return types.SimpleNamespace(success=True, message="ok", dest=str(d))

        monkeypatch.setattr(
            "nemo_evaluator_launcher.exporters.gsheets.LocalExporter", _LE, raising=True
        )

        artifacts_dir, tmpd = exporter._get_artifacts_locally(jd)
        assert artifacts_dir is None and tmpd is None
        assert str(temp_dir) in calls["rm"]

    def test_get_artifacts_locally_exception_cleanup(self, monkeypatch, tmp_path: Path):
        exporter = GSheetsExporter({})
        jd = JobData("i3", "i3.0", 0.0, "local", {"output_dir": str(tmp_path)})

        temp_dir = tmp_path / "gs_temp3"
        monkeypatch.setattr(
            "tempfile.mkdtemp", lambda prefix="": str(temp_dir), raising=True
        )

        calls = {"rm": []}
        monkeypatch.setattr(
            "shutil.rmtree", lambda p: calls["rm"].append(str(p)), raising=True
        )

        class _LE:
            def __init__(self, *a, **k): ...
            def export_job(self, *a, **k):
                raise RuntimeError("explode")

        monkeypatch.setattr(
            "nemo_evaluator_launcher.exporters.gsheets.LocalExporter", _LE, raising=True
        )

        artifacts_dir, tmpd = exporter._get_artifacts_locally(jd)
        assert artifacts_dir is None and tmpd is None
        assert str(temp_dir) in calls["rm"]

    def test_export_job_get_artifacts_failed(self, monkeypatch, tmp_path: Path):
        jd = JobData("i4", "i4.0", 0.0, "local", {"output_dir": str(tmp_path)})
        monkeypatch.setattr(
            "nemo_evaluator_launcher.exporters.gsheets.GSPREAD_AVAILABLE",
            True,
            raising=True,
        )
        monkeypatch.setattr(
            "nemo_evaluator_launcher.exporters.gsheets.GSheetsExporter._get_artifacts_locally",
            lambda *a, **k: (None, None),
            raising=True,
        )
        res = GSheetsExporter({}).export_job(jd)
        assert res.success is False
        assert "Failed to get artifacts locally" in res.message

    def test_export_job_no_metrics(self, monkeypatch, tmp_path: Path):
        jd = JobData("i5", "i5.0", 0.0, "local", {"output_dir": str(tmp_path)})

        # minimal gspread stub to avoid network
        class _WS:
            def get_all_values(self):
                return []

            def update(self, *_): ...
            def format(self, *_): ...
            def append_row(self, *_): ...

        class _SH:
            url = "http://fake"
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

        # Add gspread attribute to the module (since it wasn't imported if GSPREAD_AVAILABLE was False)
        import nemo_evaluator_launcher.exporters.gsheets as gs_mod

        monkeypatch.setattr(gs_mod, "gspread", fake_gspread, raising=False)

        monkeypatch.setattr(
            "nemo_evaluator_launcher.exporters.gsheets.GSheetsExporter._get_artifacts_locally",
            lambda *a, **k: (tmp_path, None),
            raising=True,
        )
        monkeypatch.setattr(
            "nemo_evaluator_launcher.exporters.gsheets.extract_accuracy_metrics",
            lambda *_: {},
            raising=True,
        )

        res = GSheetsExporter({}).export_job(jd)
        assert res.success is False
        assert "No accuracy metrics found" in res.message

    def test_export_invocation_open_existing_and_headers_update(
        self, monkeypatch, tmp_path: Path
    ):
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

        # Add gspread attribute to the module (since it wasn't imported if GSPREAD_AVAILABLE was False)
        import nemo_evaluator_launcher.exporters.gsheets as gs_mod

        monkeypatch.setattr(gs_mod, "gspread", fake_gspread, raising=False)

        monkeypatch.setattr(
            "nemo_evaluator_launcher.exporters.gsheets.GSheetsExporter._get_artifacts_locally",
            lambda *a, **k: (tmp_path, None),
            raising=True,
        )
        monkeypatch.setattr(
            "nemo_evaluator_launcher.exporters.gsheets.extract_accuracy_metrics",
            lambda *_: {"task_accuracy": 1.0},
            raising=True,
        )

        inv = "aa11bb22"
        db = ExecutionDB()
        db.write_job(
            JobData(
                inv,
                f"{inv}.0",
                0.0,
                "local",
                {"output_dir": str(tmp_path)},
                {"evaluation": {"tasks": [{"name": "t0"}]}},
            )
        )

        res = GSheetsExporter({}).export_invocation(inv)
        assert res["success"] is True
        assert "accuracy" in headers
        assert len(rows) == 1

    def test_export_multiple_invocations_aggregates(self, monkeypatch):
        # Enable gspread availability so we don't get early "not installed" return
        monkeypatch.setattr(
            "nemo_evaluator_launcher.exporters.gsheets.GSPREAD_AVAILABLE",
            True,
            raising=True,
        )

        exp = GSheetsExporter({})

        # stub out export_invocation
        def _exp(inv):
            return {
                "success": True,
                "metadata": {"rows_added": 2, "spreadsheet_url": "http://fake"},
            }

        monkeypatch.setattr(
            exp, "export_invocation", lambda inv: _exp(inv), raising=False
        )
        res = exp.export_multiple_invocations(["i1", "i2"])
        assert res["success"] is True
        assert res["metadata"]["total_invocations"] == 2
        assert res["metadata"]["total_rows_added"] == 4

    def test_export_invocation_exception_path(self, monkeypatch, tmp_path: Path):
        # Enable gspread
        monkeypatch.setattr(
            "nemo_evaluator_launcher.exporters.gsheets.GSPREAD_AVAILABLE",
            True,
            raising=True,
        )
        import nemo_evaluator_launcher.exporters.gsheets as gs_mod

        # Create fake gspread that raises an exception during service_account()
        fake_gspread = types.SimpleNamespace(
            service_account=lambda **kwargs: (_ for _ in ()).throw(
                RuntimeError("Connection failed")
            ),
            SpreadsheetNotFound=type("SpreadsheetNotFound", (Exception,), {}),
        )
        monkeypatch.setattr(gs_mod, "gspread", fake_gspread, raising=False)

        # Create a job so we get past the "no jobs" check
        inv = "exception123"
        db = ExecutionDB()
        db.write_job(
            JobData(inv, f"{inv}.0", 0.0, "local", {"output_dir": str(tmp_path)}, {})
        )

        res = GSheetsExporter({}).export_invocation(inv)
        assert res["success"] is False
        assert "Sheets export failed" in res["error"]

    def test_export_job_exception_path(self, monkeypatch, tmp_path: Path):
        # Enable gspread
        monkeypatch.setattr(
            "nemo_evaluator_launcher.exporters.gsheets.GSPREAD_AVAILABLE",
            True,
            raising=True,
        )
        import nemo_evaluator_launcher.exporters.gsheets as gs_mod

        # Mock _get_artifacts_locally to succeed so we reach the gspread code
        monkeypatch.setattr(
            "nemo_evaluator_launcher.exporters.gsheets.GSheetsExporter._get_artifacts_locally",
            lambda self, jd: (tmp_path / "artifacts", str(tmp_path / "temp")),
            raising=True,
        )

        # Create fake gspread that raises during service_account()
        fake_gspread = types.SimpleNamespace(
            service_account=lambda **kwargs: (_ for _ in ()).throw(
                RuntimeError("GSheets connection failed")
            ),
            SpreadsheetNotFound=type("SpreadsheetNotFound", (Exception,), {}),
        )
        monkeypatch.setattr(gs_mod, "gspread", fake_gspread, raising=False)

        jd = JobData("gx", "gx.0", 0.0, "local", {"output_dir": str(tmp_path)}, {})
        res = GSheetsExporter({}).export_job(jd)
        assert res.success is False
        assert "Failed" in res.message

    def test_export_job_service_account_file_and_create_sheet(
        self, monkeypatch, tmp_path: Path
    ):
        # Enable gspread
        monkeypatch.setattr(
            "nemo_evaluator_launcher.exporters.gsheets.GSPREAD_AVAILABLE",
            True,
            raising=True,
        )
        import nemo_evaluator_launcher.exporters.gsheets as gs_mod

        # Mock successful artifacts - create the temp directory that finally block expects
        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir()
        temp_dir = tmp_path / "temp"
        temp_dir.mkdir()  # Create the temp dir so finally block can clean it up

        monkeypatch.setattr(
            "nemo_evaluator_launcher.exporters.gsheets.GSheetsExporter._get_artifacts_locally",
            lambda self, jd: (artifacts_dir, str(temp_dir)),
            raising=True,
        )
        monkeypatch.setattr(
            "nemo_evaluator_launcher.exporters.gsheets.extract_accuracy_metrics",
            lambda *_: {"task_accuracy": 0.95},
            raising=True,
        )

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

        jd = JobData("gy", "gy.0", 0.0, "local", {"output_dir": str(tmp_path)}, {})
        exp = GSheetsExporter({"service_account_file": "/path/to/creds.json"})
        res = exp.export_job(jd)

        assert res.success
        assert calls["service_account"] == [
            "/path/to/creds.json"
        ]  # service_account_file branch
        assert calls["create"] == 1  # SpreadsheetNotFound -> create
        assert calls["share"] == 1  # share after create

    def test_export_job_spreadsheet_not_found_creates_new(
        self, monkeypatch, tmp_path: Path
    ):
        # Enable gspread
        monkeypatch.setattr(
            "nemo_evaluator_launcher.exporters.gsheets.GSPREAD_AVAILABLE",
            True,
            raising=True,
        )
        import nemo_evaluator_launcher.exporters.gsheets as gs_mod

        # Mock successful artifacts
        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir()
        temp_dir = tmp_path / "temp"
        temp_dir.mkdir()

        monkeypatch.setattr(
            "nemo_evaluator_launcher.exporters.gsheets.GSheetsExporter._get_artifacts_locally",
            lambda self, jd: (artifacts_dir, str(temp_dir)),
            raising=True,
        )
        monkeypatch.setattr(
            "nemo_evaluator_launcher.exporters.gsheets.extract_accuracy_metrics",
            lambda *_: {"task_accuracy": 0.95},
            raising=True,
        )

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

        jd = JobData(
            "create1", "create1.0", 0.0, "local", {"output_dir": str(tmp_path)}, {}
        )
        res = GSheetsExporter({}).export_job(jd)

        assert res.success
        assert calls["open"] == 1  # Tried to open first
        assert calls["create"] == 1  # SpreadsheetNotFound -> create
        assert calls["share"] == 1  # share after create

    def test_export_multiple_invocations_detailed_logic(self, monkeypatch):
        # Enable gspread
        monkeypatch.setattr(
            "nemo_evaluator_launcher.exporters.gsheets.GSPREAD_AVAILABLE",
            True,
            raising=True,
        )

        exp = GSheetsExporter({"spreadsheet_name": "Custom Sheet"})

        # Mock export_invocation to return different results per invocation
        call_count = {"calls": 0}

        def _mock_export(invocation_id):  # Fix parameter name
            call_count["calls"] += 1
            if invocation_id == "success1":  # Use invocation_id not inv_id
                return {
                    "success": True,
                    "metadata": {"rows_added": 3, "spreadsheet_url": "http://sheet1"},
                }
            elif invocation_id == "success2":
                return {
                    "success": True,
                    "metadata": {"rows_added": 2},  # No spreadsheet_url (already set)
                }
            else:  # inv_id == "failed"
                return {
                    "success": False,
                    "error": "Export failed",
                }

        monkeypatch.setattr(exp, "export_invocation", _mock_export, raising=False)

        res = exp.export_multiple_invocations(["success1", "success2", "failed"])

        assert res["success"] is True
        assert res["metadata"]["total_invocations"] == 3
        assert res["metadata"]["total_rows_added"] == 5  # 3 + 2 + 0
        assert (
            res["metadata"]["spreadsheet_url"] == "http://sheet1"
        )  # From first success
        assert res["metadata"]["spreadsheet_name"] == "Custom Sheet"
        assert call_count["calls"] == 3

        # Check individual results
        assert res["invocations"]["success1"]["success"] is True
        assert res["invocations"]["success2"]["success"] is True
        assert res["invocations"]["failed"]["success"] is False

    def test_export_invocation_metric_extraction_exception(
        self, monkeypatch, tmp_path: Path
    ):
        # Enable gspread
        monkeypatch.setattr(
            "nemo_evaluator_launcher.exporters.gsheets.GSPREAD_AVAILABLE",
            True,
            raising=True,
        )
        import nemo_evaluator_launcher.exporters.gsheets as gs_mod

        # Create minimal fake gspread (won't be used since exception happens before)
        fake_gspread = types.SimpleNamespace(
            service_account=lambda filename=None: types.SimpleNamespace(),
            SpreadsheetNotFound=type("SpreadsheetNotFound", (Exception,), {}),
        )
        monkeypatch.setattr(gs_mod, "gspread", fake_gspread, raising=False)

        # Create jobs in DB
        inv = "metric_fail"
        db = ExecutionDB()
        for i in range(2):
            db.write_job(
                JobData(
                    inv, f"{inv}.{i}", 0.0, "local", {"output_dir": str(tmp_path)}, {}
                )
            )

        # Mock _get_artifacts_locally to succeed for first job, then make extract_accuracy_metrics raise
        call_count = {"calls": 0}

        def _get_artifacts(self, jd):  # Add self parameter
            call_count["calls"] += 1
            temp_dir = tmp_path / f"temp_{call_count['calls']}"
            temp_dir.mkdir(exist_ok=True)
            return tmp_path / "artifacts", str(temp_dir)

        monkeypatch.setattr(
            "nemo_evaluator_launcher.exporters.gsheets.GSheetsExporter._get_artifacts_locally",
            _get_artifacts,
            raising=True,
        )

        # Make extract_accuracy_metrics raise an exception
        def _extract_metrics(jd, get_paths_fn):
            if jd.job_id.endswith(".0"):
                # First job succeeds
                return {"task_accuracy": 0.9}
            else:
                # Second job raises exception during metric extraction
                raise RuntimeError("Metric extraction failed for job")

        monkeypatch.setattr(
            "nemo_evaluator_launcher.exporters.gsheets.extract_accuracy_metrics",
            _extract_metrics,
            raising=True,
        )

        # Mock the rest to avoid hitting real gspread
        monkeypatch.setattr(
            "nemo_evaluator_launcher.exporters.gsheets.GSheetsExporter._get_or_update_headers",
            lambda *_: ["Model Name", "accuracy"],
            raising=True,
        )

        class _WS:
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

        res = GSheetsExporter({}).export_invocation(inv)

        assert res["success"] is True  # Overall success even with one job failing
        assert res["metadata"]["rows_added"] == 1  # Only successful job added

        # Check individual job results
        assert res["jobs"][f"{inv}.0"]["success"] is True
        assert res["jobs"][f"{inv}.1"]["success"] is False
        assert "Metric extraction failed" in res["jobs"][f"{inv}.1"]["message"]
