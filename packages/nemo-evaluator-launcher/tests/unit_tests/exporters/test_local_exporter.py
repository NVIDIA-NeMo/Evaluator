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

from pathlib import Path
from typing import List, Tuple
from unittest.mock import patch

from nemo_evaluator_launcher.common.execdb import ExecutionDB, JobData
from nemo_evaluator_launcher.exporters.local import LocalExporter


def test_export_with_format_json(tmp_path: Path, mock_execdb, prepare_local_job):
    """Test export with JSON format."""
    inv = "test0001"
    j1 = JobData(
        invocation_id=inv,
        job_id=f"{inv}.0",
        timestamp=0.0,
        executor="local",
        data={},
        config={"evaluation": {"tasks": [{"name": "simple_evals.mmlu"}]}},
    )

    prepare_local_job(j1, with_required=True, with_optional=True)
    ExecutionDB().write_job(j1)

    output_dir = tmp_path / "output"
    exporter = LocalExporter({"format": "json", "output_dir": str(output_dir)})

    with (
        patch(
            "nemo_evaluator_launcher.exporters.base.extract_accuracy_metrics",
            return_value={"accuracy": 0.9},
        ),
        patch(
            "nemo_evaluator_launcher.exporters.base.load_benchmark_info",
            return_value=("test-harness", "simple_evals.mmlu"),
        ),
        patch(
            "nemo_evaluator_launcher.exporters.base.get_model_id",
            return_value="test-model",
        ),
    ):
        result = exporter.export([j1.job_id])

    assert result.successful_jobs == [j1.job_id]
    assert result.failed_jobs == []

    # Check that output directory was created with artifacts
    job_export_dir = output_dir / inv / j1.job_id
    assert job_export_dir.exists()


def test_export_with_format_csv(tmp_path: Path, mock_execdb, prepare_local_job):
    """Test export with CSV format."""
    inv = "test0002"
    j1 = JobData(
        invocation_id=inv,
        job_id=f"{inv}.0",
        timestamp=0.0,
        executor="local",
        data={},
        config={"evaluation": {"tasks": [{"name": "simple_evals.mmlu"}]}},
    )

    prepare_local_job(j1, with_required=True, with_optional=True)
    ExecutionDB().write_job(j1)

    output_dir = tmp_path / "output"
    exporter = LocalExporter({"format": "csv", "output_dir": str(output_dir)})

    with (
        patch(
            "nemo_evaluator_launcher.exporters.base.extract_accuracy_metrics",
            return_value={"accuracy": 0.9},
        ),
        patch(
            "nemo_evaluator_launcher.exporters.base.load_benchmark_info",
            return_value=("test-harness", "simple_evals.mmlu"),
        ),
        patch(
            "nemo_evaluator_launcher.exporters.base.get_model_id",
            return_value="test-model",
        ),
    ):
        result = exporter.export([j1.job_id])

    assert result.successful_jobs == [j1.job_id]
    assert result.failed_jobs == []


class TestLocalExporterManualScenarios:
    """Test cases based on manual testing scenarios."""

    def test_successful_export_multiple_jobs(
        self, tmp_path: Path, mock_execdb, prepare_local_job
    ):
        """Test case 1: Successful export of 2 jobs with CSV format.

        Simulates: uv run nel export d96db8.1 38015bbf34550e78 --output_dir ./test-export-local --format csv
        Expected: {'successful_jobs': 2, 'failed_jobs': 0, 'skipped_jobs': 0}
        """
        # Create two jobs from different invocations
        inv1, inv2 = "d96db8f1e74c630d", "38015bbf34550e78"
        j1 = JobData(
            invocation_id=inv1,
            job_id=f"{inv1}.1",
            timestamp=1234567890.0,
            executor="local",
            data={},
            config={"evaluation": {"tasks": [None, {"name": "simple_evals.mmlu"}]}},
        )
        j2 = JobData(
            invocation_id=inv2,
            job_id=f"{inv2}.0",
            timestamp=1234567891.0,
            executor="local",
            data={},
            config={"evaluation": {"tasks": [{"name": "simple_evals.humaneval"}]}},
        )

        # Create artifacts for both jobs
        for jd in [j1, j2]:
            _, job_dir = prepare_local_job(jd, with_required=True, with_optional=True)
            artifacts_dir = job_dir / "artifacts"
            # Add required files
            (artifacts_dir / "run_config.yml").write_text(
                "framework_name: test-harness\nconfig:\n  type: test_task\ntarget:\n  api_endpoint:\n    model_id: test-model\n"
            )
            (artifacts_dir / "results.yml").write_text(
                "results:\n  tasks:\n    test_task:\n      metrics:\n        accuracy:\n          scores:\n            accuracy:\n              value: 0.9\nconfig: {}\n"
            )
            ExecutionDB().write_job(jd)

        output_dir = tmp_path / "test-export-local"
        exporter = LocalExporter({"format": "csv", "output_dir": str(output_dir)})

        result = exporter.export([j1.job_id, j2.job_id])

        assert len(result.successful_jobs) == 2
        assert j1.job_id in result.successful_jobs
        assert j2.job_id in result.successful_jobs
        assert result.failed_jobs == []
        assert result.skipped_jobs == []

    def test_partial_failure_missing_artifacts(
        self, tmp_path: Path, mock_execdb, prepare_local_job
    ):
        """Test case 2: Export with one successful and one failed job.

        Simulates: uv run nel export d96db8 --output_dir ./test-export-local --format json
        Expected: {'successful_jobs': 1, 'failed_jobs': 1}
        Error: No such file or directory: '.../artifacts/results.yml'
        """
        inv1 = "d96db8f1e74c630d"

        # Job 1: Successful
        j1 = JobData(
            invocation_id=inv1,
            job_id=f"{inv1}.0",
            timestamp=1234567890.0,
            executor="local",
            data={},
            config={"evaluation": {"tasks": [{"name": "simple_evals.mmlu"}]}},
        )
        _, job_dir = prepare_local_job(j1, with_required=True, with_optional=True)
        artifacts_dir = job_dir / "artifacts"
        (artifacts_dir / "run_config.yml").write_text(
            "framework_name: test-harness\nconfig:\n  type: test_task\ntarget:\n  api_endpoint:\n    model_id: test-model\n"
        )
        (artifacts_dir / "results.yml").write_text(
            "results:\n  tasks:\n    test_task:\n      metrics:\n        accuracy:\n          scores:\n            accuracy:\n              value: 0.9\nconfig: {}\n"
        )
        ExecutionDB().write_job(j1)

        # Job 2: Failed - missing artifacts/results.yml
        j2 = JobData(
            invocation_id=inv1,
            job_id=f"{inv1}.1",
            timestamp=1234567891.0,
            executor="local",
            data={},
            config={
                "evaluation": {"tasks": [None, {"name": "simple_evals.humaneval"}]}
            },
        )
        _, job_dir2 = prepare_local_job(j2, with_required=False, with_optional=False)
        # Create artifacts dir but don't add results.yml
        artifacts_dir2 = job_dir2 / "artifacts"
        artifacts_dir2.mkdir(parents=True, exist_ok=True)
        ExecutionDB().write_job(j2)

        output_dir = tmp_path / "test-export-local"
        output_dir.mkdir(parents=True)

        # Export both jobs - j1 succeeds, j2 fails
        exporter = LocalExporter({"format": "json", "output_dir": str(output_dir)})
        result = exporter.export([j1.job_id, j2.job_id])

        assert len(result.successful_jobs) == 1
        assert j1.job_id in result.successful_jobs
        assert len(result.failed_jobs) == 1
        assert j2.job_id in result.failed_jobs

    def test_non_existing_job_id(self, tmp_path: Path, mock_execdb, prepare_local_job):
        """Test case 3: Export with non-existing job ID.

        Simulates: uv run nel export d96db8.1 38015bbf34550e78 12345 --output_dir ./test-export-local --format csv
        Expected: {'successful_jobs': 0, 'failed_jobs': 1, 'skipped_jobs': 2}
        Error: Invocation 12345 not found in ExecutionDB nor job directories
        """
        inv1, inv2 = "d96db8f1e74c630d", "38015bbf34550e78"

        # Create two valid jobs
        j1 = JobData(
            invocation_id=inv1,
            job_id=f"{inv1}.1",
            timestamp=1234567890.0,
            executor="local",
            data={},
            config={"evaluation": {"tasks": [None, {"name": "simple_evals.mmlu"}]}},
        )
        j2 = JobData(
            invocation_id=inv2,
            job_id=f"{inv2}.0",
            timestamp=1234567891.0,
            executor="local",
            data={},
            config={"evaluation": {"tasks": [{"name": "simple_evals.humaneval"}]}},
        )

        for jd in [j1, j2]:
            _, job_dir = prepare_local_job(jd, with_required=True, with_optional=True)
            artifacts_dir = job_dir / "artifacts"
            (artifacts_dir / "run_config.yml").write_text(
                "framework_name: test-harness\nconfig:\n  type: test_task\ntarget:\n  api_endpoint:\n    model_id: test-model\n"
            )
            (artifacts_dir / "results.yml").write_text(
                "results:\n  tasks:\n    test_task:\n      metrics:\n        accuracy:\n          scores:\n            accuracy:\n              value: 0.9\nconfig: {}\n"
            )
            ExecutionDB().write_job(jd)

        output_dir = tmp_path / "test-export-local"
        output_dir.mkdir(parents=True)

        # Pre-export to make them skipped
        exporter_pre = LocalExporter({"format": "csv", "output_dir": str(output_dir)})
        exporter_pre.export([j1.job_id, j2.job_id])

        # Try to export with a non-existing ID
        exporter = LocalExporter({"format": "csv", "output_dir": str(output_dir)})
        result = exporter.export([j1.job_id, j2.job_id, "12345"])

        assert len(result.successful_jobs) == 0
        assert len(result.failed_jobs) == 1
        assert "12345" in result.failed_jobs
        assert len(result.skipped_jobs) == 2

    def test_remote_job_export_with_copy(
        self, tmp_path: Path, mock_execdb, monkeypatch
    ):
        """Test case 4: Export remote job that requires copying artifacts.

        Simulates: uv run nel export a64bf890f4d84c12 --output_dir ./test-export-local-from-slurm --format csv
        Expected: {'successful_jobs': 4, 'failed_jobs': 1, 'skipped_jobs': 0}
        Error: No such file or directory: '/tmp/.../artifacts/results.yml' (after copy)
        """
        inv = "a64bf890f4d84c12"

        # Create 5 remote jobs (4 successful, 1 will fail after copy)
        jobs = []
        for i in range(5):
            jd = JobData(
                invocation_id=inv,
                job_id=f"{inv}.{i}",
                timestamp=1234567890.0 + i,
                executor="slurm",
                data={
                    "remote_rundir_path": f"/remote/path/job{i}",
                    "hostname": "slurm-server.example.com",
                    "username": "testuser",
                },
                config={
                    "evaluation": {
                        "tasks": [{"name": f"task{i}"} for _ in range(i + 1)]
                    }
                },
            )
            jobs.append(jd)
            ExecutionDB().write_job(jd)

        def mock_copy_artifacts(
            jobs_data: List[JobData],
            export_dir: Path,
            copy_local: bool = False,
            only_required: bool = True,
            copy_logs: bool = False,
            copy_artifacts: bool = True,
        ) -> Tuple[List[JobData], List[str]]:
            # Simulate successful copy for jobs 0, 1, 3, 4
            # Job 2 will fail (no artifacts copied)
            prepared_jobs_data = []
            failed_job_ids = []

            for job_data in jobs_data:
                job_num = int(job_data.job_id.split(".")[-1])

                if job_num == 2:
                    # Job 2 fails - no artifacts copied
                    failed_job_ids.append(job_data.job_id)
                    continue

                # Create artifacts for successful jobs
                job_local_dir = export_dir / job_data.job_id
                artifacts_dir = job_local_dir / "artifacts"
                artifacts_dir.mkdir(parents=True, exist_ok=True)

                (artifacts_dir / "run_config.yml").write_text(
                    "framework_name: test-harness\nconfig:\n  type: test_task\ntarget:\n  api_endpoint:\n    model_id: test-model\n"
                )
                (artifacts_dir / "results.yml").write_text(
                    "results:\n  tasks:\n    test_task:\n      metrics:\n        accuracy:\n          scores:\n            accuracy:\n              value: 0.9\nconfig: {}\n"
                )
                (artifacts_dir / "eval_factory_metrics.json").write_text("{}")

                # Update job data with output_dir
                job_data.data["output_dir"] = str(job_local_dir)
                prepared_jobs_data.append(job_data)

            return prepared_jobs_data, failed_job_ids

        monkeypatch.setattr(
            "nemo_evaluator_launcher.exporters.base.copy_artifacts",
            mock_copy_artifacts,
        )

        output_dir = tmp_path / "test-export-local-from-slurm"
        exporter = LocalExporter({"format": "csv", "output_dir": str(output_dir)})

        result = exporter.export([jd.job_id for jd in jobs])

        assert len(result.successful_jobs) == 4
        assert len(result.failed_jobs) == 1
        assert f"{inv}.2" in result.failed_jobs
        assert result.skipped_jobs == []

    def test_json_format_updates_existing_file(
        self, tmp_path: Path, mock_execdb, prepare_local_job
    ):
        """Test that JSON format properly updates existing results file."""
        inv = "test0003"
        j1 = JobData(
            invocation_id=inv,
            job_id=f"{inv}.0",
            timestamp=1234567890.0,
            executor="local",
            data={},
            config={"evaluation": {"tasks": [{"name": "simple_evals.mmlu"}]}},
        )

        _, job_dir = prepare_local_job(j1, with_required=True, with_optional=True)
        artifacts_dir = job_dir / "artifacts"
        (artifacts_dir / "run_config.yml").write_text(
            "framework_name: test-harness\nconfig:\n  type: test_task\ntarget:\n  api_endpoint:\n    model_id: test-model\n"
        )
        (artifacts_dir / "results.yml").write_text(
            "results:\n  tasks:\n    test_task:\n      metrics:\n        accuracy:\n          scores:\n            accuracy:\n              value: 0.9\nconfig: {}\n"
        )
        ExecutionDB().write_job(j1)

        output_dir = tmp_path / "output"
        exporter = LocalExporter({"format": "json", "output_dir": str(output_dir)})

        # First export
        result1 = exporter.export([j1.job_id])
        assert result1.successful_jobs == [j1.job_id]

        # Second export of same job (should be skipped)
        result2 = exporter.export([j1.job_id])
        assert result2.skipped_jobs == [j1.job_id]
        assert result2.successful_jobs == []
