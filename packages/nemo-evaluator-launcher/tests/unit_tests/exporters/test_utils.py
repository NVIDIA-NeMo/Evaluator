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
    get_status,
    kill_job_or_invocation,
)
from nemo_evaluator_launcher.common.execdb import ExecutionDB, JobData
from nemo_evaluator_launcher.exporters import utils as U
from nemo_evaluator_launcher.exporters.utils import (
    EXCLUDED_PATTERNS,
    OPTIONAL_ARTIFACTS,
    REQUIRED_ARTIFACTS,
    MetricConflictError,
    _safe_update_metrics,
    copy_artifacts,
    extract_accuracy_metrics,
    flatten_config,
    get_available_artifacts,
    get_benchmark_info,
    get_copytree_ignore,
    get_model_name,
    get_relevant_artifacts,
    should_exclude_artifact,
    validate_artifacts,
)


class TestArtifactExclusions:
    """Tests for artifact exclusion patterns (should_exclude_artifact, get_copytree_ignore)."""

    def test_excluded_patterns_defined(self):
        """Verify EXCLUDED_PATTERNS contains expected patterns."""
        assert "*cache*" in EXCLUDED_PATTERNS
        assert "*.db" in EXCLUDED_PATTERNS
        assert "*.lock" in EXCLUDED_PATTERNS
        assert "synthetic" in EXCLUDED_PATTERNS

    def test_should_exclude_cache_patterns(self):
        """Test that *cache* pattern matches various cache directories."""
        # Should match
        assert should_exclude_artifact("cache") is True
        assert should_exclude_artifact("Cache") is True  # case insensitive
        assert should_exclude_artifact("response_stats_cache") is True
        assert should_exclude_artifact("lm_cache_rank0") is True
        # Should not match
        assert should_exclude_artifact("results.yml") is False
        assert should_exclude_artifact("my_data") is False

    def test_should_exclude_db_pattern(self):
        """Test that *.db pattern matches database files/dirs."""
        # Should match
        assert should_exclude_artifact("cache.db") is True
        assert should_exclude_artifact("lm_cache_rank0.db") is True
        assert should_exclude_artifact("DATA.DB") is True  # case insensitive
        # Should not match
        assert should_exclude_artifact("database") is False
        assert should_exclude_artifact("db_config.yml") is False

    def test_should_exclude_lock_pattern(self):
        """Test that *.lock pattern matches lock files."""
        # Should match
        assert should_exclude_artifact("tb.lock") is True
        assert should_exclude_artifact("process.lock") is True
        assert should_exclude_artifact("FILE.LOCK") is True  # case insensitive
        # Should not match
        assert should_exclude_artifact("lockfile") is False
        assert should_exclude_artifact("lock_manager.py") is False

    def test_should_exclude_synthetic_pattern(self):
        """Test that synthetic pattern matches synthetic dirs exactly."""
        # Should match (rsync pattern is normalized to exact match)
        assert should_exclude_artifact("synthetic") is True
        assert should_exclude_artifact("Synthetic") is True  # case insensitive
        # Should not match
        assert should_exclude_artifact("synthetic_data") is False
        assert should_exclude_artifact("my_synthetic") is False

    def test_should_exclude_debug_json(self):
        """Test that debug.json files are excluded (LiteLLM traces)."""
        # Should match exact filename
        assert should_exclude_artifact("debug.json") is True
        assert should_exclude_artifact("DEBUG.JSON") is True  # case insensitive
        # Should not match similar names
        assert should_exclude_artifact("debug.json.bak") is False
        assert should_exclude_artifact("my_debug.json") is False
        assert should_exclude_artifact("debug.jsonl") is False

    def test_get_copytree_ignore_filters_correctly(self):
        """Test that get_copytree_ignore returns proper ignore function."""
        ignore_func = get_copytree_ignore()
        contents = [
            "results.yml",
            "cache",
            "response_stats_cache",
            "data.db",
            "tb.lock",
            "synthetic",
            "task_output",
            "report.json",
        ]
        excluded = ignore_func("/some/dir", contents)
        # Should exclude
        assert "cache" in excluded
        assert "response_stats_cache" in excluded
        assert "data.db" in excluded
        assert "tb.lock" in excluded
        assert "synthetic" in excluded
        # Should not exclude
        assert "results.yml" not in excluded
        assert "task_output" not in excluded
        assert "report.json" not in excluded


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

    def test_copy_remote_artifacts(self, tmp_path):
        """Test copy_artifacts for remote jobs."""
        export_dir = tmp_path / "export"
        export_dir.mkdir()

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
                "nemo_evaluator_launcher.exporters.utils.ssh_setup_masters",
                return_value={("user", "server.com"): "control_path"},
            ),
            patch(
                "nemo_evaluator_launcher.exporters.utils.ssh_download_artifacts",
                return_value=["artifacts/results.yml"],  # No files copied
            ),
            patch("nemo_evaluator_launcher.exporters.utils.ssh_cleanup_masters"),
        ):
            prepared_jobs, failed_jobs = copy_artifacts(jobs_data, export_dir)

            assert len(prepared_jobs) == 1
            assert prepared_jobs[0].job_id == "test123.0"
            assert prepared_jobs[0].data["output_dir"] == str(export_dir / "test123.0")
            assert failed_jobs == []

    def test_copy_remote_artifacts_no_files_copied(self, tmp_path):
        """Test _copy_remote_artifacts when no artifacts are copied."""
        export_dir = tmp_path / "export"
        export_dir.mkdir()

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
                "nemo_evaluator_launcher.exporters.utils.ssh_setup_masters",
                return_value={("user", "server.com"): "control_path"},
            ),
            patch(
                "nemo_evaluator_launcher.exporters.utils.ssh_download_artifacts",
                return_value=[],  # No files copied
            ),
            patch("nemo_evaluator_launcher.exporters.utils.ssh_cleanup_masters"),
        ):
            prepared_jobs, failed_jobs = copy_artifacts(jobs_data, export_dir)

            assert len(prepared_jobs) == 0
            assert "test123.0" in failed_jobs

    def test_copy_remote_artifacts_local_job(self, tmp_path):
        """Test _copy_remote_artifacts with local job (no copying needed)."""
        export_dir = tmp_path / "export"
        export_dir.mkdir()

        local_job_data = JobData(
            invocation_id="test123",
            job_id="test123.0",
            timestamp=123.0,
            executor="local",
            data={"output_dir": str(tmp_path / "local_output")},
            config={},
        )

        jobs_data = [local_job_data]

        prepared_jobs, failed_jobs = copy_artifacts(jobs_data, export_dir)

        # Local job should be returned as-is
        assert len(prepared_jobs) == 1
        assert prepared_jobs[0].job_id == "test123.0"
        assert failed_jobs == []


class TestMetricsExtraction:
    def test_merge_and_filter(self, tmp_path: Path):
        artifacts = tmp_path / "artifacts"
        artifacts.mkdir(parents=True)
        (artifacts / "results.yml").write_text(
            "results: {tasks: {demo: {metrics: {metric: {scores: {accuracy: {value: 0.9}, f1: {value: 0.5}}}}}}}",
            encoding="utf-8",
        )

        all_metrics = extract_accuracy_metrics(artifacts)
        filtered = extract_accuracy_metrics(artifacts, log_metrics=["acc"])
        assert all_metrics.get("demo_metric_accuracy") == 0.9
        assert "demo_metric_f1" in all_metrics
        assert set(filtered.keys()) == {"demo_metric_accuracy"}

    def test_metric_conflict_raises(self):
        target = {"k": 1.0}
        with pytest.raises(MetricConflictError):
            _safe_update_metrics(target, {"k": 2.0}, context=" test")

    def test_nested_scores_numeric_and_broken(self, tmp_path: Path):
        # results.yml with nested scores, numeric metric, and a broken metric
        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir(parents=True)
        (artifacts_dir / "results.yml").write_text(
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

        metrics = extract_accuracy_metrics(artifacts_dir)
        assert metrics["demo_accuracy_macro"] == 0.81
        assert metrics["demo_accuracy_micro"] == 0.86
        # 'broken' is ignored due to ValueError in float cast

    def test_nested_groups(self, tmp_path: Path):
        # results.yml with nested groups
        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir(parents=True)
        (artifacts_dir / "results.yml").write_text(
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

        metrics = extract_accuracy_metrics(artifacts_dir)

        assert metrics["demo_accuracy_macro"] == 0.6
        assert metrics["demo_subgroup_one_accuracy_macro"] == 0.4
        assert metrics["demo_subgroup_two_accuracy_macro"] == 0.8

    def test_extract_with_missing_artifacts_dir(self, tmp_path: Path):
        artifacts_dir = tmp_path / "nonexistent"

        with pytest.raises(RuntimeError, match="Artifacts directory .* not found"):
            extract_accuracy_metrics(artifacts_dir)


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
        model = get_model_name(jd, {})

        assert bench["harness"] == "lm-eval"
        assert bench["benchmark"] == "mmlu"
        assert model in ("foo/bar", f"unknown_model_{jd.job_id}")

    def test_model_name_helper(self):
        jd = JobData("xx", "xx", 0.0, "local", {"model_name": "x"}, None)
        assert get_model_name(jd) == "x"


class TestSSHHelpers:
    def test_setup_and_cleanup_masters(self):
        remotes = [("user", "host"), ("user", "host"), ("other", "otherhost")]

        with patch(
            "subprocess.run", return_value=SimpleNamespace(returncode=0)
        ) as mock_run:
            cp = U.ssh_setup_masters(remotes)
            assert len(cp) == 2  # Two unique remote combinations
            assert ("user", "host") in cp
            assert ("other", "otherhost") in cp
            assert cp[("user", "host")].endswith("user_host.sock")
            U.ssh_cleanup_masters(cp)
            assert mock_run.call_count >= 2

    def test_download_artifacts_only_required_with_logs(self, tmp_path: Path):
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        (logs_dir / "foo.log").write_text("x")

        with patch("subprocess.run", return_value=SimpleNamespace(returncode=0)):
            out = U.ssh_download_artifacts(
                username="user",
                hostname="host",
                remote_path="/remote",
                export_dir=tmp_path,
                copy_logs=True,
                only_required=True,
                control_paths=None,
            )

        expected_artifacts = {
            str(tmp_path / "artifacts" / name) for name in U.get_relevant_artifacts()
        }
        assert set(out).issuperset(expected_artifacts)

    def test_download_artifacts_only_required_false_uses_tar_with_exclusions(
        self, tmp_path: Path
    ):
        """Test only_required=False uses tar+ssh with --exclude to filter artifacts."""
        ssh_commands = []

        # Mock Popen for SSH tar streaming
        class FakePopen:
            def __init__(self, cmd, stdout=None, stderr=None):
                ssh_commands.append(cmd)
                self.returncode = 0
                self.stdout = None

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def wait(self):
                pass

        # Mock subprocess.run for tar extraction - simulate creating files
        def fake_run(cmd, stdin=None, capture_output=True):
            if "tar" in cmd and "-xzf" in cmd:
                # Simulate tar extraction by creating files
                art_dir = tmp_path / "artifacts"
                art_dir.mkdir(parents=True, exist_ok=True)
                (art_dir / "results.yml").write_text("x")
                (art_dir / "extra.json").write_text("{}")
                (art_dir / "subdir").mkdir(exist_ok=True)
                (art_dir / "subdir" / "nested.txt").write_text("nested")
            return SimpleNamespace(returncode=0)

        with patch("subprocess.Popen", FakePopen):
            with patch("subprocess.run", side_effect=fake_run):
                out = U.ssh_download_artifacts(
                    username="user",
                    hostname="host",
                    remote_path="/remote",
                    export_dir=tmp_path,
                    only_required=False,
                    control_paths=None,
                )

        # Verify SSH command was called with tar and exclusion patterns
        assert len(ssh_commands) > 0
        ssh_cmd = ssh_commands[0]
        # The remote tar command should be in the SSH args
        remote_cmd = ssh_cmd[-1]  # Last arg is the remote command
        assert "tar -czf -" in remote_cmd
        # Verify exclusion patterns are included (rsync patterns converted to tar format)
        assert "--exclude=*cache*" in remote_cmd
        assert "--exclude=*.db" in remote_cmd
        assert "--exclude=*.lock" in remote_cmd
        assert "--exclude=synthetic" in remote_cmd
        assert "--exclude=debug.json" in remote_cmd
        # Verify all files including nested are listed
        assert str(tmp_path / "artifacts" / "results.yml") in out
        assert str(tmp_path / "artifacts" / "extra.json") in out
        assert str(tmp_path / "artifacts" / "subdir" / "nested.txt") in out

    def test_download_with_control_paths(self, tmp_path: Path, monkeypatch):
        control_paths = {("u", "h"): str(tmp_path / "u_h.sock")}
        calls = []

        def fake_run(cmd, capture_output=True, check=False):
            calls.append(cmd)
            return SimpleNamespace(returncode=0)

        monkeypatch.setattr("subprocess.run", fake_run, raising=True)

        U.ssh_download_artifacts(
            username="u",
            hostname="h",
            remote_path="/remote",
            export_dir=tmp_path,
            only_required=True,
            control_paths=control_paths,
        )

        # Assert ControlPath option was used in scp commands
        assert any(
            "-o" in c and any(str(control_paths[("u", "h")]) in part for part in c)
            for c in calls
        )


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


class TestFlattenConfig:
    def test_simple_dict(self):
        config = {"a": 1, "b": "hello"}
        result = flatten_config(config)
        assert result == {"a": "1", "b": "hello"}

    def test_nested_dict(self):
        config = {"a": {"b": {"c": 42}}}
        result = flatten_config(config)
        assert result == {"a.b.c": "42"}

    def test_with_parent_key(self):
        config = {"x": 1}
        result = flatten_config(config, parent_key="config")
        assert result == {"config.x": "1"}

    def test_list_with_scalars(self):
        config = {"items": ["a", "b", "c"]}
        result = flatten_config(config)
        assert result == {"items.0": "a", "items.1": "b", "items.2": "c"}

    def test_list_with_dicts(self):
        config = {"tasks": [{"name": "foo"}, {"name": "bar"}]}
        result = flatten_config(config)
        assert result == {"tasks.0.name": "foo", "tasks.1.name": "bar"}

    def test_nested_list_of_dicts(self):
        config = {
            "evaluation": {
                "tasks": [
                    {"name": "task1", "config": {"param": "value1"}},
                    {"name": "task2", "config": {"param": "value2"}},
                ]
            }
        }
        result = flatten_config(config, parent_key="config")
        assert result["config.evaluation.tasks.0.name"] == "task1"
        assert result["config.evaluation.tasks.0.config.param"] == "value1"
        assert result["config.evaluation.tasks.1.name"] == "task2"
        assert result["config.evaluation.tasks.1.config.param"] == "value2"

    def test_null_values(self):
        config = {"a": None, "b": {"c": None}}
        result = flatten_config(config)
        assert result == {"a": "null", "b.c": "null"}

    def test_max_depth_limit(self):
        config = {"a": {"b": {"c": {"d": "deep"}}}}
        result = flatten_config(config, max_depth=2)
        # At depth 2, the inner dict should be stringified
        assert "a.b" in result
        assert "{'c': {'d': 'deep'}}" in result["a.b"]

    def test_empty_dict(self):
        result = flatten_config({})
        assert result == {}

    def test_empty_list(self):
        config = {"items": []}
        result = flatten_config(config)
        assert result == {}

    def test_mixed_types(self):
        config = {
            "string": "hello",
            "number": 42,
            "float": 3.14,
            "bool": True,
            "none": None,
        }
        result = flatten_config(config)
        assert result["string"] == "hello"
        assert result["number"] == "42"
        assert result["float"] == "3.14"
        assert result["bool"] == "True"
        assert result["none"] == "null"

    def test_custom_separator(self):
        config = {"a": {"b": 1}}
        result = flatten_config(config, sep="/")
        assert result == {"a/b": "1"}
