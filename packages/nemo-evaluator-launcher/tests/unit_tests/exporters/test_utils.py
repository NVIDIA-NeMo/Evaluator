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
"""Tests for exporters utilities: artifacts, metrics, SSH, GitLab."""

import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

import nemo_evaluator_launcher.api.functional as F
from nemo_evaluator_launcher.api.functional import (
    export_results,
    get_status,
    kill_job_or_invocation,
)
from nemo_evaluator_launcher.common.execdb import ExecutionDB, JobData
from nemo_evaluator_launcher.exporters import utils as U
from nemo_evaluator_launcher.exporters.utils import (
    OPTIONAL_ARTIFACTS,
    REQUIRED_ARTIFACTS,
    MetricConflictError,
    _safe_update_metrics,
    extract_accuracy_metrics,
    get_available_artifacts,
    get_benchmark_info,
    get_container_from_mapping,
    get_model_name,
    get_pipeline_id,
    get_relevant_artifacts,
    validate_artifacts,
)


class TestArtifactUtils:
    def test_get_relevant_artifacts(self):
        all_artifacts = get_relevant_artifacts()
        expected = REQUIRED_ARTIFACTS + OPTIONAL_ARTIFACTS
        assert all_artifacts == expected
        assert "results.yml" in all_artifacts
        assert "eval_factory_metrics.json" in all_artifacts
        assert "omni-info.json" in all_artifacts

    def test_validate_artifacts_missing_dir(self):
        result = validate_artifacts(Path("/nonexistent"))
        assert result["can_export"] is False
        assert result["missing_required"] == REQUIRED_ARTIFACTS
        assert result["missing_optional"] == OPTIONAL_ARTIFACTS
        assert "not found" in result["message"]

    def test_validate_artifacts_all_present(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifacts_dir = Path(tmpdir)
            for artifact in get_relevant_artifacts():
                (artifacts_dir / artifact).touch()
            result = validate_artifacts(artifacts_dir)
            assert result["can_export"] is True
            assert result["missing_required"] == []
            assert result["missing_optional"] == []
            assert "All artifacts available" in result["message"]

    def test_get_available_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifacts_dir = Path(tmpdir)
            (artifacts_dir / "results.yml").touch()
            (artifacts_dir / "omni-info.json").touch()
            available = get_available_artifacts(artifacts_dir)
            assert "results.yml" in available
            assert "omni-info.json" in available
            assert "eval_factory_metrics.json" not in available


class TestMetricsExtraction:
    def test_merge_and_filter(self, tmp_path: Path):
        artifacts = tmp_path / "artifacts"
        artifacts.mkdir(parents=True)
        (artifacts / "results.yml").write_text(
            "results: {tasks: {demo: {metrics: {metric: {scores: {accuracy: {value: 0.9}, f1: {value: 0.5}}}}}}}",
            encoding="utf-8",
        )
        (artifacts / "foo.json").write_text('{"score": 0.75}', encoding="utf-8")

        jd = JobData(
            "abcd1234",
            "abcd1234.0",
            0.0,
            "local",
            {},
            {"evaluation": {"tasks": [{"name": "lm-eval.mmlu"}]}},
        )

        def get_paths(_):
            return {"artifacts_dir": artifacts, "storage_type": "local_filesystem"}

        all_metrics = extract_accuracy_metrics(jd, get_paths)
        filtered = extract_accuracy_metrics(jd, get_paths, log_metrics=["acc"])
        assert all_metrics.get("demo_metric_accuracy") == 0.9
        assert all_metrics.get("foo_score") == 0.75
        assert "demo_metric_f1" in all_metrics
        assert set(filtered.keys()) == {"demo_metric_accuracy"}

    def test_metric_conflict_raises(self):
        target = {"k": 1.0}
        with pytest.raises(MetricConflictError):
            _safe_update_metrics(target, {"k": 2.0}, context=" test")

    def test_nested_scores_numeric_and_broken(self, tmp_path: Path):
        # results.yml with nested scores, numeric metric, and a broken metric
        (tmp_path / "artifacts").mkdir(parents=True)
        (tmp_path / "artifacts" / "results.yml").write_text(
            """
results:
  tasks:
    demo:
      metrics:
        accuracy:
          scores:
            macro: { value: 0.81 }
            micro: { value: 0.86 }
            broken: { value: "not-a-number" }
            """.strip(),
            encoding="utf-8",
        )
        jd = JobData(
            "i1",
            "i1.0",
            0.0,
            "local",
            {},
            {"evaluation": {"tasks": [{"name": "demo"}]}},
        )

        def get_paths(_):
            return {
                "artifacts_dir": tmp_path / "artifacts",
                "storage_type": "local_filesystem",
            }

        metrics = extract_accuracy_metrics(jd, get_paths)
        assert metrics["demo_accuracy_macro"] == 0.81
        assert metrics["demo_accuracy_micro"] == 0.86
        # 'broken' is ignored due to ValueError in float cast

    def test_nested_groups(self, tmp_path: Path):
        # results.yml with nested scores, numeric metric, and a broken metric
        (tmp_path / "artifacts").mkdir(parents=True)
        (tmp_path / "artifacts" / "results.yml").write_text(
            """
results:
  groups:
    demo:
      groups:
        subgroup_one:
          metrics:
            accuracy:
              scores:
                macro: { value: 0.4 }
        subgroup_two:
          metrics:
            accuracy:
              scores:
                macro: { value: 0.8 }
      metrics:
        accuracy:
          scores:
            macro: { value: 0.6 }
            """.strip(),
            encoding="utf-8",
        )
        jd = JobData(
            "i1",
            "i1.0",
            0.0,
            "local",
            {},
            {"evaluation": {"tasks": [{"name": "demo"}]}},
        )

        def get_paths(_):
            return {
                "artifacts_dir": tmp_path / "artifacts",
                "storage_type": "local_filesystem",
            }

        metrics = extract_accuracy_metrics(jd, get_paths)

        assert metrics["demo_accuracy_macro"] == 0.6
        assert metrics["demo_subgroup_one_accuracy_macro"] == 0.4
        assert metrics["demo_subgroup_two_accuracy_macro"] == 0.8

    def test_remote_storage_and_get_paths_error(self, tmp_path: Path):
        jd = JobData(
            "i2", "i2.0", 0.0, "local", {}, {"evaluation": {"tasks": [{"name": "x"}]}}
        )

        # remote_ssh => _get_artifacts_dir returns None => extract returns {}
        def paths_remote(_):
            return {"storage_type": "remote_ssh"}

        assert extract_accuracy_metrics(jd, paths_remote) == {}

        # get_paths raises => extract returns {}
        def paths_raises(_):
            raise RuntimeError("boom")

        assert extract_accuracy_metrics(jd, paths_raises) == {}


class TestMappingHelpers:
    def test_mapping_lookups(self, monkeypatch):
        monkeypatch.setattr(
            "nemo_evaluator_launcher.exporters.utils.load_tasks_mapping",
            lambda: {
                ("lm-eval", "mmlu"): {"harness": "lm-eval", "container": "cont:tag"}
            },
            raising=True,
        )

        jd = JobData(
            "abcd1234",
            "abcd1234.0",
            0.0,
            "local",
            {"model_id": "foo/bar"},
            {"evaluation": {"tasks": [{"name": "lm-eval.mmlu"}]}},
        )

        bench = get_benchmark_info(jd)
        container = get_container_from_mapping(jd)
        model = get_model_name(jd, {})

        assert bench["harness"] == "lm-eval"
        assert bench["benchmark"] == "mmlu"
        assert container == "cont:tag"
        assert model in ("foo/bar", f"unknown_model_{jd.job_id}")

    def test_pipeline_and_model_helpers(self):
        jd = JobData(
            "xx", "xx", 0.0, "gitlab", {"pipeline_id": 123, "model_name": "x"}, None
        )
        assert get_pipeline_id(jd) == 123
        assert get_model_name(jd) == "x"


class TestSSHHelpers:
    def test_setup_and_cleanup_masters(self):
        jobs = {
            "a.0": JobData(
                "a",
                "a.0",
                0.0,
                "slurm",
                {
                    "paths": {
                        "storage_type": "remote_ssh",
                        "username": "user",
                        "hostname": "host",
                    }
                },
            ),
            "a.1": JobData(
                "a",
                "a.1",
                0.0,
                "slurm",
                {
                    "paths": {
                        "storage_type": "remote_ssh",
                        "username": "user",
                        "hostname": "host",
                    }
                },
            ),
            "b.0": JobData(
                "b",
                "b.0",
                0.0,
                "local",
                {"paths": {"storage_type": "local_filesystem"}},
            ),
        }
        with patch(
            "subprocess.run", return_value=SimpleNamespace(returncode=0)
        ) as mock_run:
            cp = U.ssh_setup_masters(jobs)
            assert len(cp) == 1
            assert ("user", "host") in cp
            assert cp[("user", "host")].endswith("user_host.sock")
            U.ssh_cleanup_masters(cp)
            assert mock_run.call_count >= 2

    def test_download_artifacts_only_required_with_logs(self, tmp_path: Path):
        paths = {"username": "user", "hostname": "host", "remote_path": "/remote"}
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

        expected_artifacts = {
            str(tmp_path / "artifacts" / name) for name in U.get_relevant_artifacts()
        }
        assert set(out).issuperset(expected_artifacts)

    def test_download_artifacts_available_only(self, tmp_path: Path):
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

    def test_download_with_control_paths(self, tmp_path: Path, monkeypatch):
        paths = {"username": "u", "hostname": "h", "remote_path": "/remote"}
        control_paths = {("u", "h"): str(tmp_path / "u_h.sock")}
        calls = []

        def fake_run(cmd, capture_output=True, check=False):
            calls.append(cmd)
            return SimpleNamespace(returncode=0)

        monkeypatch.setattr("subprocess.run", fake_run, raising=True)

        U.ssh_download_artifacts(
            paths, tmp_path, config={"only_required": True}, control_paths=control_paths
        )

        # Assert ControlPath option was used in scp commands
        assert any(
            "-o" in c and any(str(control_paths[("u", "h")]) in part for part in c)
            for c in calls
        )


class TestArtifactsDirHelper:
    def test_get_artifacts_dir_variants(self, tmp_path: Path):
        # local_filesystem
        assert (
            U._get_artifacts_dir(
                {"storage_type": "local_filesystem", "artifacts_dir": tmp_path}
            )
            == tmp_path
        )
        # gitlab_ci_local
        assert (
            U._get_artifacts_dir(
                {"storage_type": "gitlab_ci_local", "artifacts_dir": tmp_path}
            )
            == tmp_path
        )
        # remote_ssh
        assert U._get_artifacts_dir({"storage_type": "remote_ssh"}) is None
        # unsupported
        assert U._get_artifacts_dir({"storage_type": "unsupported"}) is None


class TestConfigMissingValidation:
    def test_validate_missing_root_raises(self, monkeypatch):
        cfg = OmegaConf.create({"a": 1})
        monkeypatch.setattr(
            "nemo_evaluator_launcher.api.functional.OmegaConf.is_missing",
            lambda c, k: k == "a",
            raising=True,
        )
        with pytest.raises(ValueError, match="MISSING value at path: a"):
            F._validate_no_missing_values(cfg)

    def test_validate_missing_nested_raises(self, monkeypatch):
        cfg = OmegaConf.create({"x": {"y": 1}})
        monkeypatch.setattr(
            "nemo_evaluator_launcher.api.functional.OmegaConf.is_missing",
            lambda c, k: k == "y",
            raising=True,
        )
        with pytest.raises(ValueError, match="MISSING value at path: x.y"):
            F._validate_no_missing_values(cfg)


class TestGetStatusErrorBranches:
    def test_invocation_executor_valueerror(self, mock_execdb, monkeypatch):
        db = ExecutionDB()
        inv = "deadbeef"
        db.write_job(JobData(inv, f"{inv}.0", 0.0, "bogus", {"k": "v"}))

        monkeypatch.setattr(
            F,
            "get_executor",
            lambda *_: (_ for _ in ()).throw(ValueError("unknown exec")),
        )
        out = get_status([inv])
        assert len(out) >= 1
        assert out[0]["invocation"] == inv
        assert out[0]["status"] == "error"
        assert "unknown exec" in out[0]["data"]["error"]

    def test_invocation_executor_get_status_exception(self, mock_execdb, monkeypatch):
        db = ExecutionDB()
        inv = "feedfa11"
        db.write_job(JobData(inv, f"{inv}.0", 0.0, "local", {"k": "v"}))

        class DummyExec:
            @staticmethod
            def get_status(_):
                raise RuntimeError("boom")

        monkeypatch.setattr(F, "get_executor", lambda *_: DummyExec)
        out = get_status([inv])
        assert out[0]["invocation"] == inv
        assert out[0]["status"] == "error"
        assert "boom" in out[0]["data"]["error"]

    def test_job_executor_get_status_exception(self, mock_execdb, monkeypatch):
        db = ExecutionDB()
        inv = "c0ffee00"
        job_id = f"{inv}.0"
        db.write_job(JobData(inv, job_id, 0.0, "local", {"k": "v"}))

        class DummyExec:
            @staticmethod
            def get_status(_):
                raise RuntimeError("kaboom")

        monkeypatch.setattr(F, "get_executor", lambda *_: DummyExec)
        out = get_status([job_id])
        assert out[0]["invocation"] == inv
        assert out[0]["job_id"] == job_id
        assert out[0]["status"] == "error"
        assert "kaboom" in out[0]["data"]["error"]

    def test_job_executor_returns_empty_list_unknown(self, mock_execdb, monkeypatch):
        db = ExecutionDB()
        inv = "a1b2c3d4"
        job_id = f"{inv}.1"
        db.write_job(JobData(inv, job_id, 0.0, "local", {"k": "v"}))

        class DummyExec:
            @staticmethod
            def get_status(_):
                return []

        monkeypatch.setattr(F, "get_executor", lambda *_: DummyExec)
        out = get_status([job_id])
        assert out[0]["status"] == "unknown"


class TestKillJobOrInvocation:
    def test_kill_single_job_success(self, mock_execdb, monkeypatch):
        db = ExecutionDB()
        inv = "deafbead"
        job_id = f"{inv}.0"
        db.write_job(JobData(inv, job_id, 0.0, "local", {}))

        class Exec:
            @staticmethod
            def kill_job(_):
                return None

        monkeypatch.setattr(F, "get_executor", lambda *_: Exec)
        out = kill_job_or_invocation(job_id)
        assert out[0]["status"] == "killed"
        assert out[0]["data"]["result"] == "Successfully killed job"

    def test_kill_single_job_no_kill_support(self, mock_execdb, monkeypatch):
        db = ExecutionDB()
        inv = "beadfeed"
        job_id = f"{inv}.0"
        db.write_job(JobData(inv, job_id, 0.0, "local", {}))

        class Exec:
            pass

        monkeypatch.setattr(F, "get_executor", lambda *_: Exec)
        out = kill_job_or_invocation(job_id)
        assert out[0]["status"] == "error"
        assert "does not support" in out[0]["data"]["error"]

    def test_kill_single_job_expected_error(self, mock_execdb, monkeypatch):
        db = ExecutionDB()
        inv = "fadedcab"
        job_id = f"{inv}.0"
        db.write_job(JobData(inv, job_id, 0.0, "local", {}))

        class Exec:
            @staticmethod
            def kill_job(_):
                raise ValueError("expected")

        monkeypatch.setattr(F, "get_executor", lambda *_: Exec)
        out = kill_job_or_invocation(job_id)
        assert out[0]["status"] == "error"
        assert out[0]["data"]["error"] == "expected"

    def test_kill_single_job_unexpected_error(self, mock_execdb, monkeypatch):
        db = ExecutionDB()
        inv = "cabfabed"
        job_id = f"{inv}.0"
        db.write_job(JobData(inv, job_id, 0.0, "local", {}))

        class Exec:
            @staticmethod
            def kill_job(_):
                raise RuntimeError("boom!")

        def get_exec(*_):
            # wrap to raise non-(ValueError, RuntimeError) inside kill_single_job
            class Wrapper:
                @staticmethod
                def kill_job(_):
                    raise Exception("unexpected")

            return Wrapper

        monkeypatch.setattr(F, "get_executor", get_exec)
        out = kill_job_or_invocation(job_id)
        assert out[0]["status"] == "error"
        assert "Unexpected error" in out[0]["data"]["error"]

    def test_kill_job_not_found(self):
        out = kill_job_or_invocation("unknown.0")
        assert out[0]["status"] == "not_found"

    def test_kill_invocation_not_found(self):
        out = kill_job_or_invocation("1234ffff")
        assert out[0]["status"] == "not_found"

    def test_kill_invocation_multiple_jobs(self, mock_execdb, monkeypatch):
        db = ExecutionDB()
        inv = "aa11bb22"
        db.write_job(JobData(inv, f"{inv}.0", 0.0, "local", {}))
        db.write_job(JobData(inv, f"{inv}.1", 0.0, "local", {}))

        class Exec:
            @staticmethod
            def kill_job(_):
                return None

        monkeypatch.setattr(F, "get_executor", lambda *_: Exec)
        out = kill_job_or_invocation(inv)
        assert len(out) == 2
        assert all(r["status"] == "killed" for r in out)


class TestExportResultsInvocationPath:
    def test_multi_ids_invocation_path_injects_metadata(self, monkeypatch):
        class FakeExporter:
            def export_invocation(self, inv_id):
                return {
                    "success": True,
                    "invocation_id": inv_id,
                    "jobs": {f"{inv_id}.0": {"success": True}},
                }

        monkeypatch.setattr(F, "create_exporter", lambda *_: FakeExporter())
        res = export_results(["inv1"], dest="dummy", config={})
        # Using multiple-IDs path requires >1; ensure it goes through that branch
        res = export_results(["inv1", "inv2"], dest="dummy", config={})
        assert res["success"] is True
        assert "invocations" in res
        for inv_id, payload in res["invocations"].items():
            assert payload["success"] is True
            # metadata injected for each job
            for job in payload["jobs"].values():
                assert "metadata" in job
