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
from unittest.mock import patch

import pytest

from nemo_evaluator_launcher.common.execdb import JobData
from nemo_evaluator_launcher.exporters.base import BaseExporter, ExportResult


class ConcreteExporter(BaseExporter):
    """Concrete implementation of BaseExporter for testing."""

    def export_job(self, job_data: JobData) -> ExportResult:
        """Concrete implementation for testing."""
        return ExportResult(
            success=True,
            dest="test_dest",
            message="Test export successful",
            metadata={"test": "data"},
        )

    def supports_executor(self, executor_type: str) -> bool:
        """Concrete implementation for testing."""
        return executor_type in ["local", "slurm", "gitlab", "lepton"]


class TestBaseExporter:
    def test_base_exporter_init_with_config(self):
        """Test BaseExporter initialization with config."""
        config = {"format": "json", "output_dir": "/tmp/test"}
        exporter = ConcreteExporter(config)
        assert exporter.config == config
        assert exporter.db is not None

    def test_base_exporter_init_without_config(self):
        """Test BaseExporter initialization without config."""
        exporter = ConcreteExporter()
        assert exporter.config == {}
        assert exporter.db is not None

    def test_base_exporter_init_with_none_config(self):
        """Test BaseExporter initialization with None config."""
        exporter = ConcreteExporter(None)
        assert exporter.config == {}
        assert exporter.db is not None

    def test_export_invocation_with_jobs(self):
        """Test export_invocation with actual jobs."""
        exporter = ConcreteExporter()

        # Mock the database to return jobs
        mock_job_data = JobData(
            invocation_id="test123",
            job_id="test123.0",
            timestamp=123.0,
            executor="local",
            data={"output_dir": "/tmp/test"},
        )

        with patch.object(
            exporter.db, "get_jobs", return_value={"test123.0": mock_job_data}
        ):
            result = exporter.export_invocation("test123")

            assert result["success"] is True
            assert result["invocation_id"] == "test123"
            assert "jobs" in result
            assert "test123.0" in result["jobs"]

            job_result = result["jobs"]["test123.0"]
            assert job_result["success"] is True
            assert job_result["dest"] == "test_dest"
            assert job_result["message"] == "Test export successful"
            assert job_result["metadata"]["test"] == "data"

    def test_get_job_paths_local(self):
        """Test get_job_paths for local executor."""
        exporter = ConcreteExporter()
        job_data = JobData(
            invocation_id="test123",
            job_id="test123.0",
            timestamp=123.0,
            executor="local",
            data={"output_dir": "/tmp/test"},
        )
        paths = exporter.get_job_paths(job_data)
        expected = {
            "artifacts_dir": Path("/tmp/test/artifacts"),
            "logs_dir": Path("/tmp/test/logs"),
            "storage_type": "local_filesystem",
        }
        assert paths == expected

    def test_get_job_paths_remote_ssh(self):
        """Test get_job_paths for slurm executor."""
        exporter = ConcreteExporter()
        job_data = JobData(
            invocation_id="test123",
            job_id="test123.0",
            timestamp=123.0,
            executor="slurm",
            data={
                "remote_rundir_path": "/remote/path",
                "hostname": "server.com",
                "username": "user",
            },
        )
        paths = exporter.get_job_paths(job_data)
        expected = {
            "remote_path": "/remote/path",
            "hostname": "server.com",
            "username": "user",
            "storage_type": "remote_ssh",
        }
        assert paths == expected

    def test_get_job_paths_gitlab_ci_local(self, monkeypatch):
        """Test get_job_paths for gitlab executor with CI environment."""
        exporter = ConcreteExporter()
        job_data = JobData(
            invocation_id="test123",
            job_id="test123.0",
            timestamp=123.0,
            executor="gitlab",
            data={"pipeline_id": "12345"},
        )
        monkeypatch.setenv("CI", "true")
        paths = exporter.get_job_paths(job_data)
        expected = {
            "artifacts_dir": Path("artifacts/12345"),
            "storage_type": "gitlab_ci_local",
        }
        assert paths == expected

    def test_get_job_paths_gitlab_remote(self, monkeypatch):
        """Test get_job_paths for gitlab executor without CI environment."""
        exporter = ConcreteExporter()
        job_data = JobData(
            invocation_id="test123",
            job_id="test123.0",
            timestamp=123.0,
            executor="gitlab",
            data={"pipeline_id": "12345", "project_id": "67890"},
        )
        # Ensure CI is not set
        monkeypatch.delenv("CI", raising=False)
        paths = exporter.get_job_paths(job_data)
        expected = {
            "pipeline_id": "12345",
            "project_id": "67890",
            "storage_type": "gitlab_remote",
        }
        assert paths == expected

    def test_get_job_paths_gitlab_remote_default_project_id(self, monkeypatch):
        """Test get_job_paths for gitlab executor with default project_id."""
        exporter = ConcreteExporter()
        job_data = JobData(
            invocation_id="test123",
            job_id="test123.0",
            timestamp=123.0,
            executor="gitlab",
            data={"pipeline_id": "12345"},  # No project_id
        )
        # Ensure CI is not set
        monkeypatch.delenv("CI", raising=False)
        paths = exporter.get_job_paths(job_data)
        expected = {
            "pipeline_id": "12345",
            "project_id": 155749,  # Default value
            "storage_type": "gitlab_remote",
        }
        assert paths == expected

    def test_get_job_paths_lepton(self):
        """Test get_job_paths for lepton executor."""
        exporter = ConcreteExporter()
        job_data = JobData(
            invocation_id="test123",
            job_id="test123.0",
            timestamp=123.0,
            executor="lepton",
            data={"output_dir": "/tmp/lepton"},
        )
        paths = exporter.get_job_paths(job_data)
        expected = {
            "artifacts_dir": Path("/tmp/lepton/artifacts"),
            "logs_dir": Path("/tmp/lepton/logs"),
            "storage_type": "local_filesystem",
        }
        assert paths == expected

    def test_get_job_paths_unknown_executor(self):
        """Test get_job_paths for unknown executor."""
        exporter = ConcreteExporter()
        job_data = JobData(
            invocation_id="test123",
            job_id="test123.0",
            timestamp=123.0,
            executor="unknown",
            data={},
        )
        with pytest.raises(ValueError, match="Unknown executor: unknown"):
            exporter.get_job_paths(job_data)

    def test_export_invocation_no_jobs(self):
        """Test export_invocation when no jobs are found."""
        exporter = ConcreteExporter()
        with patch.object(exporter.db, "get_jobs", return_value={}):
            result = exporter.export_invocation("nonexistent")
            assert result["success"] is False
            assert "No jobs found" in result["error"]

    def test_concrete_exporter_export_job(self):
        """Test the concrete export_job implementation."""
        exporter = ConcreteExporter()
        job_data = JobData(
            invocation_id="test123",
            job_id="test123.0",
            timestamp=123.0,
            executor="local",
            data={"output_dir": "/tmp/test"},
        )
        result = exporter.export_job(job_data)
        assert isinstance(result, ExportResult)
        assert result.success is True
        assert result.dest == "test_dest"
        assert result.message == "Test export successful"
        assert result.metadata["test"] == "data"

    def test_concrete_exporter_supports_executor(self):
        """Test the concrete supports_executor implementation."""
        exporter = ConcreteExporter()
        assert exporter.supports_executor("local") is True
        assert exporter.supports_executor("slurm") is True
        assert exporter.supports_executor("gitlab") is True
        assert exporter.supports_executor("lepton") is True
        assert exporter.supports_executor("unknown") is False

    def test_export_result_dataclass(self):
        """Test ExportResult dataclass functionality."""
        result = ExportResult(
            success=True,
            dest="test_destination",
            message="Test message",
        )
        assert result.success is True
        assert result.dest == "test_destination"
        assert result.message == "Test message"
        assert result.metadata == {}  # Default empty dict

        # Test with metadata
        result_with_metadata = ExportResult(
            success=False,
            dest="failed_dest",
            message="Failed",
            metadata={"error": "test_error"},
        )
        assert result_with_metadata.metadata["error"] == "test_error"

    def test_abstract_methods_enforcement(self):
        """Test that abstract methods must be implemented."""
        with pytest.raises(TypeError):
            # This should fail because BaseExporter is abstract
            BaseExporter()  # type: ignore[abstract]


class TestBaseExporterEdgeCases:
    """Additional edge case tests for BaseExporter."""

    def test_export_invocation_with_multiple_jobs(self):
        """Test export_invocation with multiple jobs."""
        exporter = ConcreteExporter()

        jobs = {
            "test123.0": JobData(
                invocation_id="test123",
                job_id="test123.0",
                timestamp=123.0,
                executor="local",
                data={"output_dir": "/tmp/test0"},
            ),
            "test123.1": JobData(
                invocation_id="test123",
                job_id="test123.1",
                timestamp=124.0,
                executor="local",
                data={"output_dir": "/tmp/test1"},
            ),
        }

        with patch.object(exporter.db, "get_jobs", return_value=jobs):
            result = exporter.export_invocation("test123")

            assert result["success"] is True
            assert result["invocation_id"] == "test123"
            assert len(result["jobs"]) == 2
            assert "test123.0" in result["jobs"]
            assert "test123.1" in result["jobs"]

    def test_get_job_paths_with_missing_pipeline_id(self, monkeypatch):
        """Test get_job_paths for gitlab executor with missing pipeline_id."""
        exporter = ConcreteExporter()
        job_data = JobData(
            invocation_id="test123",
            job_id="test123.0",
            timestamp=123.0,
            executor="gitlab",
            data={},  # No pipeline_id
        )
        # Ensure CI is not set to test the remote path
        monkeypatch.delenv("CI", raising=False)
        paths = exporter.get_job_paths(job_data)
        expected = {
            "pipeline_id": None,
            "project_id": 155749,  # Default value
            "storage_type": "gitlab_remote",
        }
        assert paths == expected
