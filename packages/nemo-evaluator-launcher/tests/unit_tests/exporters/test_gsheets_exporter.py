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


@pytest.fixture
def fake_worksheet():
    class _WS:
        def __init__(self):
            self.headers = []
            self.rows = []

        def get_all_values(self):
            return [self.headers] + self.rows

        def update(self, values):
            self.headers[:] = values[0]
            if len(values) > 1:
                self.rows[:] = values[1:]

        def format(self, *_a, **_k): ...

    return _WS()


@pytest.fixture
def fake_spreadsheet(fake_worksheet):
    """Reusable fixture: fake gspread client that always creates a new sheet."""

    class _SH:
        def __init__(self):
            self.title = "Fake Sheet"
            self.id = "fake-sheet-id"
            self.url = "http://fake-sheet"
            self.sheet1 = fake_worksheet

        def share(self, *a, **k): ...

    return _SH()


@pytest.fixture
def fake_account_new_sheet(fake_spreadsheet):
    from gspread import SpreadsheetNotFound

    class _Client:
        def __init__(self):
            self.calls = {"open": 0, "create": 0}
            self._spreadsheet = fake_spreadsheet

        def open(self, *_):
            self.calls["open"] += 1
            raise SpreadsheetNotFound()

        def create(self, *_):
            self.calls["create"] += 1
            return self._spreadsheet

    return _Client()


@pytest.fixture
def fake_account_existing_sheet(fake_spreadsheet):
    class _Client:
        def __init__(self):
            self.calls = {"open": 0, "create": 0}
            self._spreadsheet = fake_spreadsheet

        def open(self, *_):
            self.calls["open"] += 1
            return self._spreadsheet

        def create(self, *_):
            self.calls["create"] += 1
            return self._spreadsheet

    return _Client()


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
        assert len(skipped) == 0
        assert len(failed) == 1
        assert failed[0] == "i1.0"

    def test_export_job_ok(self, monkeypatch, tmp_path: Path, fake_account_new_sheet):
        monkeypatch.setattr(
            "nemo_evaluator_launcher.exporters.gsheets.GSPREAD_AVAILABLE",
            True,
            raising=True,
        )

        monkeypatch.setattr(
            "gspread.service_account",
            lambda *a, **k: fake_account_new_sheet,
            raising=False,
        )

        # patch so that Path(service_account_file).exists() is True
        monkeypatch.setattr(Path, "exists", lambda self: True)

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

        assert successful == ["inv1.0"]
        assert len(failed) == 0
        assert len(skipped) == 0
        assert "accuracy" in fake_account_new_sheet._spreadsheet.sheet1.headers
        assert len(fake_account_new_sheet._spreadsheet.sheet1.rows) == 1

    def test_export_multiple_jobs_ok(
        self, monkeypatch, tmp_path: Path, fake_account_new_sheet
    ):
        monkeypatch.setattr(
            "nemo_evaluator_launcher.exporters.gsheets.GSPREAD_AVAILABLE",
            True,
            raising=True,
        )

        monkeypatch.setattr(
            "gspread.service_account",
            lambda *a, **k: fake_account_new_sheet,
            raising=False,
        )

        # patch so that Path(service_account_file).exists() is True
        monkeypatch.setattr(Path, "exists", lambda self: True)

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
        headers = fake_account_new_sheet._spreadsheet.sheet1.headers
        assert "accuracy" in headers
        assert "t1_accuracy" not in headers
        assert "t0_accuracy" not in headers
        assert len(fake_account_new_sheet._spreadsheet.sheet1.rows) == 2

    def test_export_job_no_metrics(
        self, monkeypatch, tmp_path: Path, fake_account_new_sheet
    ):
        # Enable gspread availability
        monkeypatch.setattr(
            "nemo_evaluator_launcher.exporters.gsheets.GSPREAD_AVAILABLE",
            True,
            raising=True,
        )

        monkeypatch.setattr(
            "gspread.service_account",
            lambda *a, **k: fake_account_new_sheet,
            raising=False,
        )

        # patch so that Path(service_account_file).exists() is True
        monkeypatch.setattr(Path, "exists", lambda self: True)

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

    def test_export_open_existing_and_headers_update(
        self, monkeypatch, tmp_path: Path, fake_account_existing_sheet
    ):
        headers = [
            "Model Name",
            "Task Name",
            "Invocation ID",
            "Job ID",
            "Executor",
            "some_metric",
        ]  # no accuracy yet
        rows = [["model", "task", "inv1", "inv1.0", "local", "1.0"]]

        fake_account_existing_sheet._spreadsheet.sheet1.headers = headers
        fake_account_existing_sheet._spreadsheet.sheet1.rows = rows

        # Enable gspread availability
        monkeypatch.setattr(
            "nemo_evaluator_launcher.exporters.gsheets.GSPREAD_AVAILABLE",
            True,
            raising=True,
        )

        monkeypatch.setattr(
            "gspread.service_account",
            lambda *a, **k: fake_account_existing_sheet,
            raising=False,
        )

        # patch so that Path(service_account_file).exists() is True
        monkeypatch.setattr(Path, "exists", lambda self: True)

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
        assert successful == [f"{inv}.0"]
        assert failed == []
        assert skipped == []
        assert "accuracy" in fake_account_existing_sheet._spreadsheet.sheet1.headers
        assert "some_metric" in fake_account_existing_sheet._spreadsheet.sheet1.headers

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
            auth=types.SimpleNamespace(
                DEFAULT_SERVICE_ACCOUNT_FILENAME="creds.json",
                DEFAULT_CREDENTIALS_FILENAME="creds.json",
            ),
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
        self,
        monkeypatch,
        tmp_path: Path,
        fake_account_new_sheet,
    ):
        # Enable gspread
        monkeypatch.setattr(
            "nemo_evaluator_launcher.exporters.gsheets.GSPREAD_AVAILABLE",
            True,
            raising=True,
        )
        # So that "if service_account_file and Path(service_account_file).exists()" is True
        monkeypatch.setattr(Path, "exists", lambda self: True)

        service_account_calls = []

        def _track_service_account(*args, **kwargs):
            service_account_calls.append({"args": args, "kwargs": kwargs})
            return fake_account_new_sheet

        monkeypatch.setattr(
            "nemo_evaluator_launcher.exporters.gsheets.gspread.service_account",
            _track_service_account,
            raising=False,
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
        assert len(service_account_calls) == 1
        assert (
            service_account_calls[0]["kwargs"].get("filename") == "/path/to/creds.json"
        )
        assert len(fake_account_new_sheet._spreadsheet.sheet1.rows) == 1
        assert "accuracy" in fake_account_new_sheet._spreadsheet.sheet1.headers

    def test_export_spreadsheet_not_found_creates_new(
        self,
        monkeypatch,
        tmp_path: Path,
        fake_account_new_sheet,
    ):
        # Enable gspread
        monkeypatch.setattr(
            "nemo_evaluator_launcher.exporters.gsheets.GSPREAD_AVAILABLE",
            True,
            raising=True,
        )

        monkeypatch.setattr(
            "gspread.service_account",
            lambda *a, **k: fake_account_new_sheet,
            raising=False,
        )
        # patch so that Path(service_account_file).exists() is True
        monkeypatch.setattr(Path, "exists", lambda self: True)

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
        assert fake_account_new_sheet.calls["open"] == 1  # Tried to open first
        assert (
            fake_account_new_sheet.calls["create"] == 1
        )  # SpreadsheetNotFound -> create

    def test_export_mixed_success_and_skipped(
        self, monkeypatch, tmp_path: Path, fake_account_new_sheet
    ):
        # Enable gspread
        monkeypatch.setattr(
            "nemo_evaluator_launcher.exporters.gsheets.GSPREAD_AVAILABLE",
            True,
            raising=True,
        )
        monkeypatch.setattr(
            "gspread.service_account",
            lambda *a, **k: fake_account_new_sheet,
            raising=False,
        )

        # patch so that Path(service_account_file).exists() is True
        monkeypatch.setattr(Path, "exists", lambda self: True)

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
                metrics={},  # No metrics - will be skipped
                harness="lm-eval",
                task="task",
                invocation_id="inv1",
                job_id="inv1.1",
                executor="local",
                model_id="test-model",
                container="local",
                timestamp=0.0,
                config={},
                job_data={},
            ),
        ]

        successful, failed, skipped = GSheetsExporter({}).export_jobs(data_list)

        assert successful == ["inv1.0"]
        assert failed == []
        assert skipped == ["inv1.1"]
        assert (
            len(fake_account_new_sheet._spreadsheet.sheet1.rows) == 1
        )  # Only one successful append
