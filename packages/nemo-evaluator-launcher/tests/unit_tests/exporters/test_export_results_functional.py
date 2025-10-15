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
"""Tests for export_results functional API."""

import json
from pathlib import Path

import pytest

from nemo_evaluator_launcher.api.functional import export_results
from nemo_evaluator_launcher.common.execdb import ExecutionDB, JobData


def _make_job(
    inv_id: str, idx: int, task_name: str, *, executor: str = "local"
) -> JobData:
    tasks = [{"name": None} for _ in range(idx + 1)]
    tasks[idx] = {"name": task_name}
    return JobData(
        invocation_id=inv_id,
        job_id=f"{inv_id}.{idx}",
        timestamp=0.0,
        executor=executor,
        data={},
        config={"evaluation": {"tasks": tasks}},
    )


def _register_jobs(*jobs: JobData):
    db = ExecutionDB()
    for jd in jobs:
        db.write_job(jd)


class TestExportResultsFunctional:
    def test_single_invocation_json_and_csv(
        self, tmp_path: Path, mock_execdb, prepare_local_job
    ):
        inv = "aa11bb22"
        j1 = _make_job(inv, 0, "simple_evals.mmlu")
        j2 = _make_job(inv, 1, "simple_evals.humaneval")
        for jd in (j1, j2):
            _, job_dir = prepare_local_job(jd, with_required=True, with_optional=True)
            jd.data["output_dir"] = str(job_dir)
            assert job_dir.parent.name == inv
        _register_jobs(j1, j2)

        export_root = Path(tmp_path) / "exports"
        rj = export_results(
            inv, dest="local", config={"format": "json", "output_dir": str(export_root)}
        )
        assert rj["success"] is True
        assert j1.job_id in rj["jobs"] and j2.job_id in rj["jobs"]

        rc = export_results(
            inv, dest="local", config={"format": "csv", "output_dir": str(export_root)}
        )
        assert rc["success"] is True
        csv_path = export_root / inv / "processed_results.csv"
        assert csv_path.exists()

    def test_single_job_invocation_like_response(
        self, tmp_path: Path, mock_execdb, prepare_local_job
    ):
        inv = "cc33dd44"
        j = _make_job(inv, 0, "simple_evals.mmlu")
        _, job_dir = prepare_local_job(j, with_required=True, with_optional=True)
        j.data["output_dir"] = str(job_dir)
        assert job_dir.parent.name == inv
        _register_jobs(j)

        export_root = Path(tmp_path) / "exports"
        r = export_results(
            j.job_id,
            dest="local",
            config={"format": "json", "output_dir": str(export_root)},
        )
        assert r["success"] is True
        assert r["invocation_id"] == inv
        assert list(r["jobs"].keys()) == [j.job_id]

    def test_multiple_invocations_consolidated_json_and_csv(
        self, tmp_path: Path, mock_execdb, prepare_local_job
    ):
        inv1, inv2 = "ee55ff66", "gg77hh88"
        j11 = _make_job(inv1, 0, "simple_evals.mmlu")
        j12 = _make_job(inv1, 1, "simple_evals.humaneval")
        j21 = _make_job(inv2, 0, "simple_evals.hle")
        for jd in (j11, j12, j21):
            _, job_dir = prepare_local_job(jd, with_required=True, with_optional=True)
            jd.data["output_dir"] = str(job_dir)
            assert job_dir.parent.name in (inv1, inv2)
        _register_jobs(j11, j12, j21)

        export_root = Path(tmp_path) / "exports"
        rj = export_results(
            [inv1, inv2],
            dest="local",
            config={"format": "json", "output_dir": str(export_root)},
        )
        assert rj["success"] is True
        summary_path = Path(rj["metadata"]["summary_path"])
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        seen = set()
        for bench in (data.get("benchmarks") or {}).values():
            for model_entries in (bench.get("models") or {}).values():
                for entry in model_entries:
                    jid = entry.get("job_id")
                    if jid:
                        seen.add(jid)
        assert {j11.job_id, j12.job_id, j21.job_id}.issubset(seen)

        rc = export_results(
            [inv1, inv2],
            dest="local",
            config={"format": "csv", "output_dir": str(export_root)},
        )
        assert rc["success"] is True
        csv_path = Path(rc["metadata"]["summary_path"])
        assert csv_path.exists()

    def test_multiple_jobs_non_consolidated_partial_invocations(
        self, tmp_path: Path, mock_execdb, prepare_local_job
    ):
        invA, invB = "ii99jj00", "kk11ll22"
        a0 = _make_job(invA, 0, "simple_evals.mmlu")
        b0 = _make_job(invB, 0, "simple_evals.hle")
        for jd in (a0, b0):
            _, job_dir = prepare_local_job(jd, with_required=True, with_optional=True)
            jd.data["output_dir"] = str(job_dir)
            assert job_dir.parent.name in (invA, invB)
        _register_jobs(a0, b0)

        export_root = Path(tmp_path) / "exports"
        r = export_results(
            [a0.job_id, b0.job_id],
            dest="local",
            config={"output_dir": str(export_root)},
        )
        assert r["success"] is True
        assert invA in r["invocations"] and invB in r["invocations"]
        assert r["invocations"][invA]["partial"] is True
        assert r["invocations"][invB]["partial"] is True

    def test_mixed_invocation_plus_job_triggers_consolidated_json(
        self, tmp_path: Path, mock_execdb, prepare_local_job
    ):
        inv1, inv2 = "mm33nn44", "oo55pp66"
        j11 = _make_job(inv1, 0, "simple_evals.mmlu")
        j12 = _make_job(inv1, 1, "simple_evals.humaneval")
        j21 = _make_job(inv2, 0, "simple_evals.hle")
        for jd in (j11, j12, j21):
            _, job_dir = prepare_local_job(jd, with_required=True, with_optional=True)
            jd.data["output_dir"] = str(job_dir)
            assert job_dir.parent.name in (inv1, inv2)
        _register_jobs(j11, j12, j21)

        export_root = Path(tmp_path) / "exports"
        r = export_results(
            [inv1, j21.job_id],
            dest="local",
            config={"format": "json", "output_dir": str(export_root)},
        )
        assert r["success"] is True
        summary_path = Path(r["metadata"]["summary_path"])
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        seen = set()
        for bench in (data.get("benchmarks") or {}).values():
            for model_entries in (bench.get("models") or {}).values():
                for entry in model_entries:
                    jid = entry.get("job_id")
                    if jid:
                        seen.add(jid)
        assert {j11.job_id, j12.job_id, j21.job_id}.issubset(seen)

    @pytest.mark.skip(reason="gitlab_ci_local not supported")
    def test_single_pipeline_id_gitlab_ci_local(
        self, tmp_path: Path, mock_execdb, monkeypatch
    ):
        pipeline_id = 123456
        inv = "qq77rr88"
        j = _make_job(inv, 0, "simple_evals.mmlu", executor="gitlab")
        j.data["pipeline_id"] = pipeline_id
        _register_jobs(j)

        # Isolate working dir for gitlab_ci_local path
        monkeypatch.chdir(tmp_path)
        (Path("artifacts") / str(pipeline_id)).mkdir(parents=True, exist_ok=True)
        (Path("artifacts") / str(pipeline_id) / "results.yml").write_text(
            "results: {tasks: {demo: {metrics: {accuracy: {value: 0.9}}}}}",
            encoding="utf-8",
        )
        (Path("artifacts") / str(pipeline_id) / "eval_factory_metrics.json").write_text(
            json.dumps({"total_time": 1.23}), encoding="utf-8"
        )
        monkeypatch.setenv("CI", "true")

        r = export_results(
            str(pipeline_id),
            dest="local",
            config={"output_dir": str(tmp_path), "format": "json"},
        )
        assert r["success"] is True
