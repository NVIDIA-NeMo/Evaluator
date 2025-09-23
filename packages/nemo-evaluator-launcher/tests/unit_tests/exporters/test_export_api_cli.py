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
"""Test export API and CLI functionality."""

import time

from nemo_evaluator_launcher.api.functional import export_results
from nemo_evaluator_launcher.cli.export import ExportCmd
from nemo_evaluator_launcher.common.execdb import ExecutionDB, JobData


class _DummyExporter:
    """Minimal exporter to validate API behavior."""

    def __init__(self, config=None):
        self.config = config or {}
        self.calls = []

    # API shim: export_invocation
    def export_invocation(self, invocation_id: str):
        self.calls.append(("inv", invocation_id, self.config))
        # Return shape must match BaseExporter.export_invocation
        return {
            "success": True,
            "invocation_id": invocation_id,
            "jobs": {
                f"{invocation_id}.0": {"success": True, "message": "ok"},
            },
        }

    # API shim: export_job
    def export_job(self, job_data: JobData):
        self.calls.append(("job", job_data.job_id, self.config))
        from nemo_evaluator_launcher.exporters.base import ExportResult

        return ExportResult(
            success=True,
            dest="dummy",
            message="ok",
            metadata={"jid": job_data.job_id},
        )

    # Optional consolidated path contract for local
    def export_multiple_invocations(self, inv_ids):
        self.calls.append(("multi", tuple(inv_ids), self.config))
        return {"success": True, "invocations": {i: {"success": True} for i in inv_ids}}


def _register_dummy(monkeypatch):
    # Force factory to return dummy exporter
    monkeypatch.setattr(
        "nemo_evaluator_launcher.exporters.get_exporter",
        lambda name: (lambda cfg=None: _DummyExporter(cfg)),
        raising=True,
    )


def _execdb_add_job(inv: str, idx: int) -> JobData:
    jd = JobData(
        invocation_id=inv,
        job_id=f"{inv}.{idx}",
        timestamp=time.time(),
        executor="local",
        data={"output_dir": "/tmp/unused"},
        config={"evaluation": {"tasks": [{"name": "simple_evals.mmlu"}]}},
    )
    ExecutionDB().write_job(jd)
    return jd


def test_single_invocation_calls_exporter(mock_execdb, monkeypatch):
    _register_dummy(monkeypatch)
    inv = "a1b2c3d4"
    _execdb_add_job(inv, 0)

    res = export_results(inv, dest="dummy", config={"k": 1})

    assert res["success"] is True
    # metadata for jobs must include metadata key (API guarantees)
    assert res["jobs"][f"{inv}.0"]["metadata"] == {}
    # sanity on structure
    assert res["invocation_id"] == inv


def test_single_job_calls_exporter(mock_execdb, monkeypatch):
    _register_dummy(monkeypatch)
    inv = "d4c3b2a1"
    jd = _execdb_add_job(inv, 0)

    res = export_results(jd.job_id, dest="dummy", config={"x": 2})

    assert res["success"] is True
    assert res["invocation_id"] == inv
    # job result normalized under jobs dict
    jr = res["jobs"][jd.job_id]
    assert jr["success"] is True
    assert jr["metadata"] == {"jid": jd.job_id}


def test_pipeline_id_resolution(mock_execdb, monkeypatch):
    _register_dummy(monkeypatch)
    inv = "11223344"
    jd = _execdb_add_job(inv, 0)
    # attach pipeline_id into job data
    db = ExecutionDB()
    db_job = db.get_job(jd.job_id)
    db_job.data["pipeline_id"] = 3579171
    db.write_job(db_job)

    res = export_results("3579171", dest="dummy", config={})
    assert res["success"] is True
    assert res["invocation_id"] == inv
    assert list(res["jobs"].keys()) == [jd.job_id]


def test_mixed_ids_no_consolidation(mock_execdb, monkeypatch):
    # Mixed IDs but dest != local → do not use consolidated path
    flags = {"inv": 0, "job": 0}

    class _TraceExporter(_DummyExporter):
        def export_invocation(self, invocation_id):
            flags["inv"] += 1
            return super().export_invocation(invocation_id)

        def export_job(self, job_data):
            flags["job"] += 1
            return super().export_job(job_data)

    monkeypatch.setattr(
        "nemo_evaluator_launcher.exporters.get_exporter",
        lambda name: (lambda cfg=None: _TraceExporter(cfg)),
        raising=True,
    )

    inv = "55667788"
    _execdb_add_job(inv, 0)
    _execdb_add_job(inv, 1)

    res = export_results([inv, f"{inv}.1"], dest="dummy", config={})
    assert res["success"] is True
    assert "invocations" in res and inv in res["invocations"]
    assert res["invocations"][inv]["success"] is True
    # and ensure at least one job was exported
    assert len(res["invocations"][inv].get("jobs", {})) >= 1


def test_cli_arg_mapping_and_format_note(monkeypatch, capsys):
    # Verify CLI maps args to config and prints note when format is ignored
    from nemo_evaluator_launcher.cli.export import ExportCmd

    called = {}

    def _fake_export(ids, dest, config):
        called["ids"] = ids
        called["dest"] = dest
        called["config"] = config
        return {"success": True, "jobs": {ids[0]: {"success": True}}}

    monkeypatch.setattr(
        "nemo_evaluator_launcher.api.functional.export_results",
        _fake_export,
        raising=True,
    )

    cmd = ExportCmd(
        invocation_ids=["deadbeef"],
        dest="wandb",
        output_dir="/tmp/results",
        output_filename="x.json",
        format="json",
        copy_logs=True,
        log_metrics=["score"],
        only_required=False,
    )
    cmd.execute()
    out = capsys.readouterr().out
    assert "will be ignored" in out  # note about format for non-local
    assert called["dest"] == "wandb"
    assert called["config"]["output_dir"] == "/tmp/results"
    assert called["config"]["output_filename"] == "x.json"
    assert called["config"]["copy_logs"] is True
    assert called["config"]["only_required"] is False
    assert called["config"]["format"] == "json"  # carried but unused by dest
    assert called["config"]["log_metrics"] == ["score"]


def test_handles_exporter_exception(monkeypatch):
    class _Boom:
        def __init__(self, *_args, **_kw): ...
        def export_invocation(self, *_a, **_k):
            raise RuntimeError("boom")

    monkeypatch.setattr(
        "nemo_evaluator_launcher.exporters.get_exporter",
        lambda name: (lambda cfg=None: _Boom()),
        raising=True,
    )
    res = export_results("abcdef12", dest="dummy", config={})
    assert res["success"] is False
    assert "error" in res

    def test_cli_export_single_with_summary_and_failed_job(self, monkeypatch, capsys):
        # Mock export_results to return mixed success/failure with summary
        def _mock_export(ids, dest, config):
            return {
                "success": True,
                "jobs": {
                    "job1.0": {
                        "success": True,
                        "message": "Success",
                        "metadata": {
                            "run_url": "http://wandb/run/123",
                            "summary_path": "/path/to/summary.json",
                        },
                    },
                    "job1.1": {
                        "success": False,
                        "message": "Failed to extract metrics",
                    },
                },
            }

        monkeypatch.setattr(
            "nemo_evaluator_launcher.api.functional.export_results", _mock_export
        )

        cmd = ExportCmd(invocation_ids=["job1"], dest="local")
        cmd.execute()

        captured = capsys.readouterr()
        assert "Export completed for job1" in captured.out
        assert "URL: http://wandb/run/123" in captured.out
        assert "Summary: /path/to/summary.json" in captured.out
        assert "job1.1 failed: Failed to extract metrics" in captured.out

    def test_cli_export_multiple_with_summary_and_mixed_results(
        self, monkeypatch, capsys
    ):
        # Mock export_results for multiple invocations
        def _mock_export(ids, dest, config):
            return {
                "success": True,
                "metadata": {
                    "successful_invocations": 2,
                    "total_invocations": 3,
                    "summary_path": "/path/to/multi_summary.csv",
                },
                "invocations": {
                    "inv1": {"success": True, "jobs": {"inv1.0": {}, "inv1.1": {}}},
                    "inv2": {"success": True, "jobs": {"inv2.0": {}}},
                    "inv3": {"success": False, "error": "No metrics found"},
                },
            }

        monkeypatch.setattr(
            "nemo_evaluator_launcher.api.functional.export_results", _mock_export
        )

        cmd = ExportCmd(invocation_ids=["inv1", "inv2", "inv3"], dest="wandb")
        cmd.execute()

        captured = capsys.readouterr()
        assert "Export completed: 2/3 successful" in captured.out
        assert "Summary: /path/to/multi_summary.csv" in captured.out
        assert "inv1: 2 jobs" in captured.out
        assert "inv2: 1 jobs" in captured.out
        assert "inv3: failed, No metrics found" in captured.out
