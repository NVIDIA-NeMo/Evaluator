"""Tests for SSH helpers (setup/cleanup and artifact download)."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from nemo_evaluator_launcher.common.execdb import JobData
from nemo_evaluator_launcher.exporters import utils as U


class TestExportPipelines:
    def test_ssh_setup_and_cleanup_masters(self):
        jobs = {
            "a.0": JobData(
                invocation_id="a",
                job_id="a.0",
                timestamp=0.0,
                executor="slurm",
                data={
                    "paths": {
                        "storage_type": "remote_ssh",
                        "username": "user",
                        "hostname": "host",
                    }
                },
            ),
            "a.1": JobData(
                invocation_id="a",
                job_id="a.1",
                timestamp=0.0,
                executor="slurm",
                data={
                    "paths": {
                        "storage_type": "remote_ssh",
                        "username": "user",
                        "hostname": "host",
                    }
                },
            ),
            "b.0": JobData(
                invocation_id="b",
                job_id="b.0",
                timestamp=0.0,
                executor="local",
                data={"paths": {"storage_type": "local_filesystem"}},
            ),
        }
        with patch(
            "subprocess.run", return_value=SimpleNamespace(returncode=0)
        ) as mock_run:
            cp = U.ssh_setup_masters(jobs)
            # Just check that the path ends with the expected socket name
            assert len(cp) == 1
            assert ("user", "host") in cp
            assert cp[("user", "host")].endswith("user_host.sock")
            U.ssh_cleanup_masters(cp)
            assert mock_run.call_count >= 2

    def test_ssh_download_artifacts_only_required_with_logs(self, tmp_path: Path):
        paths = {"username": "user", "hostname": "host", "remote_path": "/remote"}
        # Pre-make local log file so it's picked up after mocked 'scp -r'
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        (logs_dir / "foo.log").write_text("x")

        with patch("subprocess.run", return_value=SimpleNamespace(returncode=0)):
            out = U.ssh_download_artifacts(
                paths,
                tmp_path,
                config={"copy_logs": True, "only_required": True},
                control_paths=None,
            )

        # Should request all relevant artifacts (required + optional)
        expected_artifacts = {
            str(tmp_path / "artifacts" / name) for name in U.get_relevant_artifacts()
        }
        assert set(out).issuperset(expected_artifacts)

    def test_ssh_download_artifacts_available_only(self, tmp_path: Path):
        # When only_required=False, use locally available artifact names for selection
        local_artifacts = tmp_path / "local_artifacts"
        local_artifacts.mkdir()
        (local_artifacts / "results.yml").write_text("x")

        paths = {
            "username": "user",
            "hostname": "host",
            "remote_path": "/remote",
            "artifacts_dir": local_artifacts,
        }

        with patch("subprocess.run", return_value=SimpleNamespace(returncode=0)):
            out = U.ssh_download_artifacts(
                paths, tmp_path, config={"only_required": False}, control_paths=None
            )

        assert str(tmp_path / "artifacts" / "results.yml") in out
