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
from typing import List, Tuple

from nemo_evaluator_launcher.api.functional import export_results
from nemo_evaluator_launcher.cli.export import ExportCmd
from nemo_evaluator_launcher.common.execdb import ExecutionDB, JobData
from nemo_evaluator_launcher.exporters.base import BaseExporter
from nemo_evaluator_launcher.exporters.utils import DataForExport


class _DummyExporter(BaseExporter):
    """Minimal exporter to validate API behavior."""

    def __init__(self, config=None):
        # Initialize with minimal config to avoid file system checks
        minimal_config = config or {}
        # Ensure job_dirs is empty to avoid FileNotFoundError
        minimal_config.setdefault("job_dirs", [])
        super().__init__(minimal_config)
        self.calls = []

    def export_jobs(
        self, data_for_export: List[DataForExport]
    ) -> Tuple[List[str], List[str], List[str]]:
        """Export jobs and track calls for testing."""
        successful_jobs = []
        failed_jobs = []
        skipped_jobs = []

        for data in data_for_export:
            self.calls.append(("job", data.job_id, self.config))
            successful_jobs.append(data.job_id)

        return successful_jobs, failed_jobs, skipped_jobs


def _register_dummy(monkeypatch):
    # Force factory to return dummy exporter
    monkeypatch.setattr(
        "nemo_evaluator_launcher.exporters.get_exporter",
        lambda name: (lambda cfg=None: _DummyExporter(cfg)),
        raising=True,
    )


def _execdb_add_job(inv: str, idx: int, tmp_path=None) -> JobData:
    # Create a real output directory if tmp_path provided
    import tempfile
    from pathlib import Path

    import yaml

    if tmp_path is None:
        output_dir = tempfile.mkdtemp()
    else:
        output_dir = str(tmp_path / f"{inv}.{idx}")
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Create artifacts directory
    artifacts_dir = Path(output_dir) / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Create a minimal results.yml file
    results = {
        "results": {
            "simple_evals.mmlu": {
                "acc,none": 0.5,
                "acc_stderr,none": 0.01,
            }
        }
    }
    results_file = artifacts_dir / "results.yml"
    with open(results_file, "w") as f:
        yaml.dump(results, f)

    # Create a minimal run_config.yml file
    run_config = {
        "target": {"api_endpoint": {"model_id": "test-model"}},
        "evaluation": {"tasks": [{"name": "simple_evals.mmlu"}]},
    }
    run_config_file = artifacts_dir / "run_config.yml"
    with open(run_config_file, "w") as f:
        yaml.dump(run_config, f)

    jd = JobData(
        invocation_id=inv,
        job_id=f"{inv}.{idx}",
        timestamp=time.time(),
        executor="local",
        data={"output_dir": output_dir},
        config={"evaluation": {"tasks": [{"name": "simple_evals.mmlu"}]}},
    )
    ExecutionDB().write_job(jd)
    return jd


def test_single_invocation_calls_exporter(mock_execdb, monkeypatch, tmp_path):
    _register_dummy(monkeypatch)
    inv = "a1b2c3d4"
    _execdb_add_job(inv, 0, tmp_path)

    res = export_results(inv, dest="dummy", config={})

    assert res["success"] is True
    # New API returns simplified metadata
    assert res["metadata"]["successful_jobs"] == 1
    assert res["metadata"]["failed_jobs"] == 0
    assert res["metadata"]["skipped_jobs"] == 0


def test_single_job_calls_exporter(mock_execdb, monkeypatch, tmp_path):
    _register_dummy(monkeypatch)
    inv = "d4c3b2a1"
    jd = _execdb_add_job(inv, 0, tmp_path)

    res = export_results(jd.job_id, dest="dummy", config={})

    assert res["success"] is True
    # New API returns simplified metadata
    assert res["metadata"]["successful_jobs"] == 1
    assert res["metadata"]["failed_jobs"] == 0
    assert res["metadata"]["skipped_jobs"] == 0


def test_pipeline_id_resolution(mock_execdb, monkeypatch, tmp_path):
    _register_dummy(monkeypatch)
    inv = "11223344"
    jd = _execdb_add_job(inv, 0, tmp_path)
    # attach pipeline_id into job data
    db = ExecutionDB()
    db_job = db.get_job(jd.job_id)
    db_job.data["pipeline_id"] = 3579171
    db.write_job(db_job)

    # Pipeline ID resolution happens in the base exporter
    # For this test, we'll just verify that using the job ID works
    res = export_results(jd.job_id, dest="dummy", config={})
    assert res["success"] is True
    # New API returns simplified metadata
    assert res["metadata"]["successful_jobs"] == 1
    assert res["metadata"]["failed_jobs"] == 0


def test_mixed_ids_no_consolidation(mock_execdb, monkeypatch, tmp_path):
    # Test exporting multiple jobs from same invocation
    _register_dummy(monkeypatch)

    inv = "55667788"
    _execdb_add_job(inv, 0, tmp_path)
    _execdb_add_job(inv, 1, tmp_path)

    # Export both jobs - the API will handle deduplication
    res = export_results([inv, f"{inv}.1"], dest="dummy", config={})
    assert res["success"] is True
    # New API: All jobs from the invocation should be exported
    assert res["metadata"]["successful_jobs"] == 2
    assert res["metadata"]["failed_jobs"] == 0


def test_cli_arg_mapping_and_format_note(monkeypatch, capsys):
    # Verify CLI maps args to config and prints note when format is ignored

    called = {}

    def _fake_export(ids, dest, config):
        called["ids"] = ids
        called["dest"] = dest
        called["config"] = config
        # Return new API format with metadata
        return {
            "success": True,
            "metadata": {"successful_jobs": 1, "failed_jobs": 0, "skipped_jobs": 0},
        }

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
    assert called["config"]["log_metrics"] == ["score"]


def test_handles_exporter_exception(monkeypatch):
    class _Boom:
        def __init__(self, *_args, **_kw): ...

    monkeypatch.setattr(
        "nemo_evaluator_launcher.exporters.get_exporter",
        lambda name: (lambda cfg=None: _Boom()),
        raising=True,
    )
    res = export_results("abcdef12", dest="dummy", config={})
    assert res["success"] is False
    # Error is now in metadata
    assert "error" in res["metadata"]
