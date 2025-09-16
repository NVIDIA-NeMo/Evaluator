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
"""Tests for local functionality."""

import csv
import json
from pathlib import Path

from nemo_evaluator_launcher.api.functional import export_results
from nemo_evaluator_launcher.common.execdb import ExecutionDB, JobData
from nemo_evaluator_launcher.exporters.local import LocalExporter


def _make_job(inv_id: str, idx: int, task_name: str) -> JobData:
    # Build a config where the idx-th task has the given name
    tasks = [{"name": None} for _ in range(idx + 1)]
    tasks[idx] = {"name": task_name}
    return JobData(
        invocation_id=inv_id,
        job_id=f"{inv_id}.{idx}",
        timestamp=0.0,
        executor="local",
        data={},  # output_dir is set by prepare_local_job fixture
        config={"evaluation": {"tasks": tasks}},
    )


class TestConsolidatedLocalExports:
    def test_export_results_consolidated_mixed_inputs_json(
        self, tmp_path: Path, mock_execdb, make_job_fs, prepare_local_job
    ):
        # Arrange jobs across 3 invocations: inv1(2 jobs), inv2(1 job), inv3(1 job for standalone)
        inv1, inv2, inv3 = "aaa11111", "bbb22222", "ccc33333"
        j11 = _make_job(inv1, 0, "simple_evals.mmlu")
        j12 = _make_job(inv1, 1, "simple_evals.humaneval")
        j21 = _make_job(inv2, 0, "simple_evals.mmlu")
        j31 = _make_job(inv3, 0, "simple_evals.hle")

        # Materialize per-job artifacts under <tmp>/<inv>/<job>/artifacts
        # and point job.data['output_dir'] to <tmp>/<inv> (as LocalExporter expects)
        for jd in (j11, j12, j21, j31):
            _, job_dir = prepare_local_job(jd, with_required=True, with_optional=True)
            # Ensure structure is <tmp>/<inv>/<job>/...
            assert job_dir.parent.name == jd.invocation_id

        # Register jobs in the ExecutionDB
        db = ExecutionDB()
        for jd in (j11, j12, j21, j31):
            db.write_job(jd)

        # Act: mixed inputs (two invocations + one explicit job id) with consolidated JSON
        res = export_results(
            [inv1, inv2, j31.job_id],
            dest="local",
            config={"format": "json", "output_dir": str(tmp_path)},
        )

        # Assert: success and summary file exists
        assert res["success"] is True
        summary_path = Path(res["metadata"]["summary_path"])
        assert summary_path.exists()
        data = json.loads(summary_path.read_text(encoding="utf-8"))

        # Collect all job_ids present in the summary payload
        seen_job_ids = set()
        for bench in (data.get("benchmarks") or {}).values():
            for model_entries in (bench.get("models") or {}).values():
                for entry in model_entries:
                    jid = entry.get("job_id")
                    if jid:
                        seen_job_ids.add(jid)

        # All jobs from inv1, inv2, and the standalone job must be present
        expected = {j11.job_id, j12.job_id, j21.job_id, j31.job_id}
        assert expected.issubset(seen_job_ids)

    def test_export_results_consolidated_mixed_inputs_csv(
        self, tmp_path: Path, mock_execdb, make_job_fs, prepare_local_job
    ):
        # Arrange: reuse a smaller set to validate CSV path
        inv1, inv2 = "ddd44444", "eee55555"
        j11 = _make_job(inv1, 0, "simple_evals.mmlu")
        j12 = _make_job(inv1, 1, "simple_evals.humaneval")
        j21 = _make_job(inv2, 0, "simple_evals.hle")

        for jd in (j11, j12, j21):
            _, job_dir = prepare_local_job(jd, with_required=True, with_optional=True)
            assert job_dir.parent.name == jd.invocation_id

        db = ExecutionDB()
        for jd in (j11, j12, j21):
            db.write_job(jd)

        res = export_results(
            [inv1, j21.job_id],
            dest="local",
            config={"format": "csv", "output_dir": str(tmp_path)},
        )
        assert res["success"] is True
        summary_path = Path(res["metadata"]["summary_path"])
        assert summary_path.exists()

        # Parse CSV and assert all job_ids are present
        rows = summary_path.read_text(encoding="utf-8").strip().splitlines()
        reader = csv.reader(rows)
        headers = next(reader)
        idx_job = headers.index("Job ID")
        csv_job_ids = {r[idx_job] for r in reader}

        expected = {j11.job_id, j12.job_id, j21.job_id}
        assert expected.issubset(csv_job_ids)


def test_copy_all_tree(tmp_path: Path):
    artifacts_dir = tmp_path / "in" / "artifacts"
    logs_dir = tmp_path / "in" / "logs"
    (artifacts_dir / "a").mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "a" / "a.txt").write_text("x")
    (artifacts_dir / "extra.json").write_text("{}")
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "log.txt").write_text("l")

    exporter = LocalExporter({"only_required": False, "copy_logs": True})
    out_dir = tmp_path / "out"
    files = exporter._copy_local_artifacts(
        {
            "artifacts_dir": artifacts_dir,
            "logs_dir": logs_dir,
            "storage_type": "local_filesystem",
        },
        out_dir,
        exporter.config,
    )

    assert (out_dir / "artifacts" / "a" / "a.txt").exists()
    assert (out_dir / "artifacts" / "extra.json").exists()
    assert (out_dir / "logs" / "log.txt").exists()
    assert str(out_dir / "artifacts" / "extra.json") in files
