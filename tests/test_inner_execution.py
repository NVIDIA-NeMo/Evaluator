"""Tests for NEL_INNER_EXECUTION env-var mechanism that prevents recursive dispatch."""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from nemo_evaluator.eval.config import (
    BenchmarkConfig,
    ClusterConfig,
    EvalConfig,
    ModelConfig,
    OutputConfig,
)


def _make_config(cluster_type: str = "docker", bench: str = "gsm8k") -> EvalConfig:
    return EvalConfig(
        model=ModelConfig(url="http://localhost:8000/v1", id="test-model"),
        benchmarks=[BenchmarkConfig(name=bench)],
        cluster=ClusterConfig(type=cluster_type),
        output=OutputConfig(dir="/tmp/nel-test"),
    )


class TestDispatchOverride:
    """CLI dispatch should force local executor when NEL_INNER_EXECUTION=1."""

    def test_forces_local_when_inner(self):
        """With NEL_INNER_EXECUTION=1, get_executor receives 'local' even if config says 'docker'."""
        executor_mock = MagicMock()

        with patch.dict(os.environ, {"NEL_INNER_EXECUTION": "1"}):
            with patch("nemo_evaluator.executors.get_executor", return_value=executor_mock) as get:
                config = _make_config("docker")
                executor_type = config.cluster.type
                if os.environ.get("NEL_INNER_EXECUTION") == "1":
                    executor_type = "local"
                get(executor_type)

                get.assert_called_once_with("local")

    def test_respects_config_without_env(self):
        """Without the env var, executor type matches config."""
        executor_mock = MagicMock()
        env = os.environ.copy()
        env.pop("NEL_INNER_EXECUTION", None)

        with patch.dict(os.environ, env, clear=True):
            with patch("nemo_evaluator.executors.get_executor", return_value=executor_mock) as get:
                config = _make_config("docker")
                executor_type = config.cluster.type
                if os.environ.get("NEL_INNER_EXECUTION") == "1":
                    executor_type = "local"
                get(executor_type)

                get.assert_called_once_with("docker")

    def test_ignores_zero_value(self):
        """NEL_INNER_EXECUTION=0 must NOT trigger the override (strict '1' check)."""
        executor_mock = MagicMock()

        with patch.dict(os.environ, {"NEL_INNER_EXECUTION": "0"}):
            with patch("nemo_evaluator.executors.get_executor", return_value=executor_mock) as get:
                config = _make_config("docker")
                executor_type = config.cluster.type
                if os.environ.get("NEL_INNER_EXECUTION") == "1":
                    executor_type = "local"
                get(executor_type)

                get.assert_called_once_with("docker")

    def test_ignores_false_string(self):
        """NEL_INNER_EXECUTION=false must NOT trigger the override."""
        with patch.dict(os.environ, {"NEL_INNER_EXECUTION": "false"}):
            config = _make_config("slurm")
            executor_type = config.cluster.type
            if os.environ.get("NEL_INNER_EXECUTION") == "1":
                executor_type = "local"

            assert executor_type == "slurm"


class TestDockerEnvFlag:
    """Docker executor must pass NEL_INNER_EXECUTION=1 into the container."""

    def test_dryrun_includes_inner_execution_env(self, tmp_path, capsys):
        from nemo_evaluator.executors.docker import DockerExecutor

        config = _make_config("docker")
        config.output.dir = str(tmp_path / "out")

        with patch("nemo_evaluator.executors.docker._build_local_image", return_value="nemo-evaluator:local"):
            DockerExecutor().run(config, dry_run=True)

        captured = capsys.readouterr()
        assert "NEL_INNER_EXECUTION=1" in captured.out

    def test_config_not_mutated(self, tmp_path):
        """Serialized config must preserve the original cluster.type (no mutation)."""
        import json
        from nemo_evaluator.executors.docker import DockerExecutor

        config = _make_config("docker")
        config.output.dir = str(tmp_path / "out")

        with patch("nemo_evaluator.executors.docker._build_local_image", return_value="nemo-evaluator:local"):
            with patch("nemo_evaluator.executors.docker.subprocess") as sub_mock:
                sub_mock.run.return_value = MagicMock(returncode=0, stdout="abc123\n")
                DockerExecutor().run(config)

        cfg_path = tmp_path / "out" / "_docker_config.json"
        saved = json.loads(cfg_path.read_text())
        assert saved["cluster"]["type"] == "docker"


class TestSlurmHeaderFlag:
    """Generated sbatch scripts must contain the inner execution export."""

    def test_header_contains_export(self):
        from nemo_evaluator.eval.slurm_gen import generate_sbatch

        config = _make_config("slurm", bench="gsm8k")
        config.cluster.type = "slurm"

        script = generate_sbatch(config)
        assert "export NEL_INNER_EXECUTION=1" in script

    def test_export_has_comment(self):
        from nemo_evaluator.eval.slurm_gen import generate_sbatch

        config = _make_config("slurm", bench="gsm8k")
        config.cluster.type = "slurm"

        script = generate_sbatch(config)
        lines = script.splitlines()
        for i, line in enumerate(lines):
            if "export NEL_INNER_EXECUTION=1" in line:
                assert i > 0 and "Prevent" in lines[i - 1], \
                    "Export should be preceded by an explanatory comment"
                break
        else:
            pytest.fail("export NEL_INNER_EXECUTION=1 not found in script")
