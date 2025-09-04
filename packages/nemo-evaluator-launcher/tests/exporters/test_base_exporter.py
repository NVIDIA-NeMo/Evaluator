"""Tests for BaseExporter core behavior."""

from pathlib import Path
from unittest.mock import Mock

from nemo_evaluator_launcher.common.execdb import JobData
from nemo_evaluator_launcher.exporters.base import BaseExporter


class TestBaseExporter:
    def test_get_job_paths_local(self):
        mock_exporter = Mock(spec=BaseExporter)
        mock_exporter.get_job_paths = BaseExporter.get_job_paths
        job_data = JobData(
            invocation_id="test123",
            job_id="test123.0",
            timestamp=123.0,
            executor="local",
            data={"output_dir": "/tmp/test"},
        )
        paths = mock_exporter.get_job_paths(mock_exporter, job_data)
        expected = {
            "artifacts_dir": Path("/tmp/test/artifacts"),
            "logs_dir": Path("/tmp/test/logs"),
            "storage_type": "local_filesystem",
        }
        assert paths == expected

    def test_get_job_paths_remote_ssh(self):
        mock_exporter = Mock(spec=BaseExporter)
        mock_exporter.get_job_paths = BaseExporter.get_job_paths
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
        paths = mock_exporter.get_job_paths(mock_exporter, job_data)
        expected = {
            "remote_path": "/remote/path",
            "hostname": "server.com",
            "username": "user",
            "storage_type": "remote_ssh",
        }
        assert paths == expected

    def test_get_job_paths_gitlab_ci_local(self, monkeypatch):
        mock_exporter = Mock(spec=BaseExporter)
        mock_exporter.get_job_paths = BaseExporter.get_job_paths
        job_data = JobData(
            invocation_id="test123",
            job_id="test123.0",
            timestamp=123.0,
            executor="gitlab",
            data={"pipeline_id": "12345"},
        )
        monkeypatch.setenv("CI", "true")
        paths = mock_exporter.get_job_paths(mock_exporter, job_data)
        expected = {
            "artifacts_dir": Path("artifacts/12345"),
            "storage_type": "gitlab_ci_local",
        }
        assert paths == expected

    def test_export_invocation_no_jobs(self):
        mock_exporter = Mock(spec=BaseExporter)
        mock_exporter.export_invocation = BaseExporter.export_invocation
        mock_exporter.db = Mock()
        mock_exporter.db.get_jobs.return_value = {}
        result = mock_exporter.export_invocation(mock_exporter, "nonexistent")
        assert result["success"] is False
        assert "No jobs found" in result["error"]
