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
"""Tests for BaseExporter core behavior."""

from pathlib import Path
from typing import List, Tuple
from unittest.mock import patch

import pytest

from nemo_evaluator_launcher.common.execdb import JobData
from nemo_evaluator_launcher.exporters.base import BaseExporter
from nemo_evaluator_launcher.exporters.utils import DataForExport, ExportResult


class ConcreteExporter(BaseExporter):
    """Concrete implementation of BaseExporter for testing."""

    def export_jobs(
        self, data_for_export: List[DataForExport]
    ) -> Tuple[List[str], List[str], List[str]]:
        """Concrete implementation for testing."""
        successful = [data.job_id for data in data_for_export]
        return successful, [], []


class TestBaseExporter:
    def test_base_exporter_init_with_config(self, tmp_path):
        """Test BaseExporter initialization with config."""
        job_dir = tmp_path / "jobs"
        job_dir.mkdir()
        config = {
            "format": "json",
            "output_dir": "/tmp/test",
            "job_dirs": [str(job_dir)],
            "copy_logs": True,
            "copy_artifacts": False,
            "only_required": False,
        }
        exporter = ConcreteExporter(config)
        assert exporter.config == config
        assert exporter.db is not None
        assert exporter.job_dirs == [job_dir]
        assert exporter.copy_logs is True
        assert exporter.copy_artifacts is False
        assert exporter.only_required is False

    def test_base_exporter_init_without_config(self):
        """Test BaseExporter initialization without config."""
        exporter = ConcreteExporter()
        assert exporter.config == {}
        assert exporter.db is not None
        assert exporter.job_dirs == []
        assert exporter.copy_logs is False
        assert exporter.copy_artifacts is True
        assert exporter.only_required is True

    def test_base_exporter_init_with_none_config(self):
        """Test BaseExporter initialization with None config."""
        exporter = ConcreteExporter(None)
        assert exporter.config == {}
        assert exporter.db is not None

    def test_base_exporter_init_with_invalid_job_dir(self):
        """Test BaseExporter initialization with non-existent job directory."""
        config = {"job_dirs": ["/nonexistent/path"]}
        with pytest.raises(FileNotFoundError, match="Job directory .* not found"):
            ConcreteExporter(config)

    def test_abstract_methods_enforcement(self):
        """Test that abstract methods must be implemented."""
        with pytest.raises(TypeError):
            # This should fail because BaseExporter is abstract
            BaseExporter()  # type: ignore[abstract]

    def test_export_with_jobs_from_db(self, tmp_path):
        """Test export method with jobs retrieved from database."""
        # Create mock artifacts directory
        job_output_dir = tmp_path / "job_output"
        artifacts_dir = job_output_dir / "artifacts"
        artifacts_dir.mkdir(parents=True)

        # Create required files
        (artifacts_dir / "metadata.yaml").write_text("model_id: test-model\n")
        (artifacts_dir / "results.yml").write_text("accuracy: 0.95\n")
        (artifacts_dir / "tasks.json").write_text('{"task": "test_task"}')

        exporter = ConcreteExporter()

        # Mock the database to return jobs
        mock_job_data = JobData(
            invocation_id="test123",
            job_id="test123.0",
            timestamp=123.0,
            executor="local",
            data={"output_dir": str(job_output_dir)},
            config={"evaluation": {"tasks": ["test_task"]}},
        )

        with (
            patch.object(exporter.db, "get_job", return_value=mock_job_data),
            patch.object(exporter.db, "get_jobs", return_value={}),
            patch(
                "nemo_evaluator_launcher.exporters.base.extract_accuracy_metrics",
                return_value={"accuracy": 0.95},
            ),
            patch(
                "nemo_evaluator_launcher.exporters.base.load_benchmark_info",
                return_value=("harness", "test_task"),
            ),
            patch(
                "nemo_evaluator_launcher.exporters.base.get_model_id",
                return_value="test-model",
            ),
        ):
            result = exporter.export(["test123.0"])

            assert isinstance(result, ExportResult)
            assert result.successful_jobs == ["test123.0"]
            assert result.failed_jobs == []
            assert result.skipped_jobs == []

    def test_export_invocation_with_multiple_jobs(self, tmp_path):
        """Test export with multiple jobs in an invocation."""
        exporter = ConcreteExporter()

        # Create mock job data
        jobs = {
            "test123.0": JobData(
                invocation_id="test123",
                job_id="test123.0",
                timestamp=123.0,
                executor="local",
                data={"output_dir": str(tmp_path / "job0")},
                config={},
            ),
            "test123.1": JobData(
                invocation_id="test123",
                job_id="test123.1",
                timestamp=124.0,
                executor="local",
                data={"output_dir": str(tmp_path / "job1")},
                config={},
            ),
        }

        # Create artifacts directories
        for job_data in jobs.values():
            artifacts_dir = Path(job_data.data["output_dir"]) / "artifacts"
            artifacts_dir.mkdir(parents=True)
            (artifacts_dir / "metadata.yaml").write_text("model_id: test-model\n")

        with (
            patch.object(exporter.db, "get_jobs", return_value=jobs),
            patch.object(exporter.db, "get_job", return_value=None),
            patch(
                "nemo_evaluator_launcher.exporters.base.extract_accuracy_metrics",
                return_value={"accuracy": 0.95},
            ),
            patch(
                "nemo_evaluator_launcher.exporters.base.load_benchmark_info",
                return_value=("harness", "test_task"),
            ),
            patch(
                "nemo_evaluator_launcher.exporters.base.get_model_id",
                return_value="test-model",
            ),
        ):
            result = exporter.export(["test123"])

            assert len(result.successful_jobs) == 2
            assert "test123.0" in result.successful_jobs
            assert "test123.1" in result.successful_jobs

    def test_export_with_missing_artifacts(self, tmp_path):
        """Test export with missing artifacts directory."""
        job_output_dir = tmp_path / "job_output"
        job_output_dir.mkdir()
        # Don't create artifacts directory

        exporter = ConcreteExporter()

        mock_job_data = JobData(
            invocation_id="test123",
            job_id="test123.0",
            timestamp=123.0,
            executor="local",
            data={"output_dir": str(job_output_dir)},
            config={},
        )

        with patch.object(exporter.db, "get_job", return_value=mock_job_data):
            result = exporter.export(["test123.0"])

            assert result.failed_jobs == ["test123.0"]
            assert result.successful_jobs == []

    def test_prepare_data_for_export(self, tmp_path):
        """Test prepare_data_for_export method."""
        job_output_dir = tmp_path / "job_output"
        artifacts_dir = job_output_dir / "artifacts"
        artifacts_dir.mkdir(parents=True)

        # Create required files
        (artifacts_dir / "metadata.yaml").write_text("model_id: test-model\n")
        (artifacts_dir / "results.yml").write_text("accuracy: 0.95\n")
        (artifacts_dir / "tasks.json").write_text('{"task": "test_task"}')

        exporter = ConcreteExporter()

        job_data = JobData(
            invocation_id="test123",
            job_id="test123.0",
            timestamp=123.0,
            executor="local",
            data={"output_dir": str(job_output_dir), "eval_image": "test-container"},
            config={"model": "test-model"},
        )

        with (
            patch(
                "nemo_evaluator_launcher.exporters.base.extract_accuracy_metrics",
                return_value={"accuracy": 0.95},
            ),
            patch(
                "nemo_evaluator_launcher.exporters.base.load_benchmark_info",
                return_value=("harness", "test_task"),
            ),
            patch(
                "nemo_evaluator_launcher.exporters.base.get_model_id",
                return_value="test-model",
            ),
        ):
            data = exporter.prepare_data_for_export(job_data)

            assert data is not None
            assert isinstance(data, DataForExport)
            assert data.artifacts_dir == artifacts_dir
            assert data.logs_dir is None  # copy_logs is False by default
            assert data.model_id == "test-model"
            assert data.metrics == {"accuracy": 0.95}
            assert data.harness == "harness"
            assert data.task == "test_task"
            assert data.container == "test-container"
            assert data.executor == "local"
            assert data.invocation_id == "test123"
            assert data.job_id == "test123.0"
            assert data.timestamp == 123.0

    def test_prepare_data_for_export_with_logs(self, tmp_path):
        """Test prepare_data_for_export with copy_logs enabled."""
        job_output_dir = tmp_path / "job_output"
        artifacts_dir = job_output_dir / "artifacts"
        logs_dir = job_output_dir / "logs"
        artifacts_dir.mkdir(parents=True)
        logs_dir.mkdir(parents=True)

        (artifacts_dir / "metadata.yaml").write_text("model_id: test-model\n")

        exporter = ConcreteExporter({"copy_logs": True})

        job_data = JobData(
            invocation_id="test123",
            job_id="test123.0",
            timestamp=123.0,
            executor="local",
            data={"output_dir": str(job_output_dir)},
            config={},
        )

        with (
            patch(
                "nemo_evaluator_launcher.exporters.base.extract_accuracy_metrics",
                return_value={},
            ),
            patch(
                "nemo_evaluator_launcher.exporters.base.load_benchmark_info",
                return_value=(None, "test_task"),
            ),
            patch(
                "nemo_evaluator_launcher.exporters.base.get_model_id",
                return_value="test-model",
            ),
        ):
            data = exporter.prepare_data_for_export(job_data)

            assert data is not None
            assert data.logs_dir == logs_dir

    def test_prepare_data_for_export_missing_output_dir(self):
        """Test prepare_data_for_export with missing output_dir in job data."""
        exporter = ConcreteExporter()

        job_data = JobData(
            invocation_id="test123",
            job_id="test123.0",
            timestamp=123.0,
            executor="local",
            data={},  # No output_dir
            config={},
        )

        data = exporter.prepare_data_for_export(job_data)
        assert data is None

    def test_get_jobs_data_from_db(self, tmp_path):
        """Test _get_jobs_data retrieves jobs from database."""
        exporter = ConcreteExporter()

        mock_job_data = JobData(
            invocation_id="test123",
            job_id="test123.0",
            timestamp=123.0,
            executor="local",
            data={"output_dir": str(tmp_path)},
            config={},
        )

        with (
            patch.object(exporter.db, "get_job", return_value=mock_job_data),
            patch.object(exporter.db, "get_jobs", return_value={}),
        ):
            jobs_data, failed_jobs = exporter._get_jobs_data(["test123.0"])

            assert jobs_data == [mock_job_data]
            assert failed_jobs == []

    def test_get_jobs_data_from_invocation(self, tmp_path):
        """Test _get_jobs_data retrieves all jobs from an invocation."""
        exporter = ConcreteExporter()

        jobs = {
            "test123.0": JobData(
                invocation_id="test123",
                job_id="test123.0",
                timestamp=123.0,
                executor="local",
                data={"output_dir": str(tmp_path)},
                config={},
            ),
            "test123.1": JobData(
                invocation_id="test123",
                job_id="test123.1",
                timestamp=124.0,
                executor="local",
                data={"output_dir": str(tmp_path)},
                config={},
            ),
        }

        with (
            patch.object(exporter.db, "get_jobs", return_value=jobs),
            patch.object(exporter.db, "get_job", return_value=None),
        ):
            jobs_data, failed_jobs = exporter._get_jobs_data(["test123"])

            assert len(jobs_data) == 2
            assert jobs_data[0].job_id == "test123.0"
            assert jobs_data[1].job_id == "test123.1"
            assert failed_jobs == []

    def test_get_jobs_data_missing_job(self):
        """Test _get_jobs_data with missing job ID."""
        exporter = ConcreteExporter()

        with (
            patch.object(exporter.db, "get_job", return_value=None),
            patch.object(exporter.db, "get_jobs", return_value={}),
        ):
            jobs_data, failed_jobs = exporter._get_jobs_data(["missing.0"])

            assert len(jobs_data) == 0
            assert "missing.0" in failed_jobs

    def test_get_jobs_in_dir(self, tmp_path):
        """Test _get_jobs_in_dir extracts job data from directory."""
        # Create invocation directory structure
        invocation_dir = tmp_path / "20251219_112732-test123abc"
        invocation_dir.mkdir()

        # Create task subdirectories
        task1_dir = invocation_dir / "task1"
        task1_artifacts = task1_dir / "artifacts"
        task1_artifacts.mkdir(parents=True)

        task2_dir = invocation_dir / "task2"
        task2_artifacts = task2_dir / "artifacts"
        task2_artifacts.mkdir(parents=True)

        # Create metadata files
        config = {
            "evaluation": {"tasks": ["task1", "task2"]},
            "execution": {"type": "local"},
        }
        (task1_artifacts / "metadata.yaml").write_text(
            "config:\n  evaluation:\n    tasks:\n      - task1\n      - task2\n  execution:\n    type: local\n"
        )
        (task2_artifacts / "metadata.yaml").write_text(
            "config:\n  evaluation:\n    tasks:\n      - task1\n      - task2\n  execution:\n    type: local\n"
        )

        exporter = ConcreteExporter()

        with patch(
            "nemo_evaluator_launcher.exporters.base.load_config_from_metadata",
            return_value=config,
        ):
            jobs_data = exporter._get_jobs_in_dir(invocation_dir)

            assert len(jobs_data) == 2
            assert "test123abc.0" in jobs_data  # task1
            assert "test123abc.1" in jobs_data  # task2
            assert jobs_data["test123abc.0"].invocation_id == "test123abc"
            assert jobs_data["test123abc.1"].invocation_id == "test123abc"

    def test_copy_remote_artifacts(self, tmp_path):
        """Test _copy_remote_artifacts for remote jobs."""
        export_dir = tmp_path / "export"
        export_dir.mkdir()

        exporter = ConcreteExporter()

        remote_job_data = JobData(
            invocation_id="test123",
            job_id="test123.0",
            timestamp=123.0,
            executor="slurm",
            data={
                "remote_rundir_path": "/remote/path",
                "hostname": "server.com",
                "username": "user",
            },
            config={},
        )

        jobs_data = [remote_job_data]

        with (
            patch(
                "nemo_evaluator_launcher.exporters.base.ssh_setup_masters",
                return_value={("user", "server.com"): "control_path"},
            ),
            patch(
                "nemo_evaluator_launcher.exporters.base.ssh_download_artifacts",
                return_value=["file1.txt", "file2.txt"],
            ),
            patch("nemo_evaluator_launcher.exporters.base.ssh_cleanup_masters"),
        ):
            prepared_jobs, failed_jobs = exporter._copy_remote_artifacts(
                jobs_data, export_dir
            )

            assert len(prepared_jobs) == 1
            assert prepared_jobs[0].job_id == "test123.0"
            assert prepared_jobs[0].data["output_dir"] == str(export_dir / "test123.0")
            assert failed_jobs == []

    def test_copy_remote_artifacts_no_files_copied(self, tmp_path):
        """Test _copy_remote_artifacts when no artifacts are copied."""
        export_dir = tmp_path / "export"
        export_dir.mkdir()

        exporter = ConcreteExporter()

        remote_job_data = JobData(
            invocation_id="test123",
            job_id="test123.0",
            timestamp=123.0,
            executor="slurm",
            data={
                "remote_rundir_path": "/remote/path",
                "hostname": "server.com",
                "username": "user",
            },
            config={},
        )

        jobs_data = [remote_job_data]

        with (
            patch(
                "nemo_evaluator_launcher.exporters.base.ssh_setup_masters",
                return_value={("user", "server.com"): "control_path"},
            ),
            patch(
                "nemo_evaluator_launcher.exporters.base.ssh_download_artifacts",
                return_value=[],  # No files copied
            ),
            patch("nemo_evaluator_launcher.exporters.base.ssh_cleanup_masters"),
        ):
            prepared_jobs, failed_jobs = exporter._copy_remote_artifacts(
                jobs_data, export_dir
            )

            assert len(prepared_jobs) == 0
            assert "test123.0" in failed_jobs

    def test_copy_remote_artifacts_local_job(self, tmp_path):
        """Test _copy_remote_artifacts with local job (no copying needed)."""
        export_dir = tmp_path / "export"
        export_dir.mkdir()

        exporter = ConcreteExporter()

        local_job_data = JobData(
            invocation_id="test123",
            job_id="test123.0",
            timestamp=123.0,
            executor="local",
            data={"output_dir": str(tmp_path / "local_output")},
            config={},
        )

        jobs_data = [local_job_data]

        prepared_jobs, failed_jobs = exporter._copy_remote_artifacts(
            jobs_data, export_dir
        )

        # Local job should be returned as-is
        assert len(prepared_jobs) == 1
        assert prepared_jobs[0].job_id == "test123.0"
        assert failed_jobs == []


class TestExportResult:
    """Test ExportResult dataclass functionality."""

    def test_export_result_creation(self):
        """Test ExportResult creation with all fields."""
        result = ExportResult(
            successful_jobs=["job1", "job2"],
            failed_jobs=["job3"],
            skipped_jobs=["job4"],
            metadata={"key": "value"},
        )
        assert result.successful_jobs == ["job1", "job2"]
        assert result.failed_jobs == ["job3"]
        assert result.skipped_jobs == ["job4"]
        assert result.metadata == {"key": "value"}

    def test_export_result_default_metadata(self):
        """Test ExportResult with default metadata."""
        result = ExportResult(
            successful_jobs=["job1"],
            failed_jobs=[],
            skipped_jobs=[],
        )
        assert result.metadata == {}


class TestDataForExport:
    """Test DataForExport dataclass functionality."""

    def test_data_for_export_creation(self, tmp_path):
        """Test DataForExport creation with all fields."""
        artifacts_dir = tmp_path / "artifacts"
        logs_dir = tmp_path / "logs"

        data = DataForExport(
            artifacts_dir=artifacts_dir,
            logs_dir=logs_dir,
            config={"key": "value"},
            model_id="test-model",
            metrics={"accuracy": 0.95},
            harness="test_harness",
            task="test_task",
            container="test-container",
            executor="local",
            invocation_id="inv123",
            job_id="inv123.0",
            timestamp=123.0,
            job_data={"extra": "data"},
        )

        assert data.artifacts_dir == artifacts_dir
        assert data.logs_dir == logs_dir
        assert data.config == {"key": "value"}
        assert data.model_id == "test-model"
        assert data.metrics == {"accuracy": 0.95}
        assert data.harness == "test_harness"
        assert data.task == "test_task"
        assert data.container == "test-container"
        assert data.executor == "local"
        assert data.invocation_id == "inv123"
        assert data.job_id == "inv123.0"
        assert data.timestamp == 123.0
        assert data.job_data == {"extra": "data"}

    def test_data_for_export_optional_fields(self, tmp_path):
        """Test DataForExport with optional fields set to None."""
        artifacts_dir = tmp_path / "artifacts"

        data = DataForExport(
            artifacts_dir=artifacts_dir,
            logs_dir=None,
            config={},
            model_id="test-model",
            metrics={},
            harness=None,
            task="test_task",
            container="test-container",
            executor="local",
            invocation_id="inv123",
            job_id="inv123.0",
            timestamp=123.0,
        )

        assert data.logs_dir is None
        assert data.harness is None
        assert data.job_data is None
