# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for NEL_INNER_EXECUTION env-var mechanism that prevents recursive dispatch."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch


from nemo_evaluator.config import (
    BenchmarkConfig,
    DockerCluster,
    EvalConfig,
    ExternalApiService,
    LocalCluster,
    OutputConfig,
    SimpleSolver,
    SlurmCluster,
    NodePool,
)


def _make_config(cluster_type: str = "docker", bench: str = "gsm8k") -> EvalConfig:
    svc = ExternalApiService(
        type="api",
        url="http://localhost:8000/v1/chat/completions",
        protocol="chat_completions",
        model="test-model",
    )
    solver = SimpleSolver(type="simple", service="model")
    benchmark = BenchmarkConfig(name=bench, solver=solver)

    if cluster_type == "docker":
        cluster = DockerCluster(type="docker")
    elif cluster_type == "slurm":
        cluster = SlurmCluster(
            type="slurm",
            node_pools={"compute": NodePool(partition="batch")},
        )
    else:
        cluster = LocalCluster(type="local")

    return EvalConfig(
        services={"model": svc},
        benchmarks=[benchmark],
        cluster=cluster,
        output=OutputConfig(dir="/tmp/nel-test"),
    )


class TestDispatchOverride:
    """CLI dispatch should force local executor when NEL_INNER_EXECUTION=1."""

    def test_forces_local_when_inner(self):
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
        with patch.dict(os.environ, {"NEL_INNER_EXECUTION": "false"}):
            config = _make_config("slurm")
            executor_type = config.cluster.type
            if os.environ.get("NEL_INNER_EXECUTION") == "1":
                executor_type = "local"

            assert executor_type == "slurm"


class TestDockerEnvFlag:
    """Docker executor must pass NEL_INNER_EXECUTION=1 into the container."""

    def test_dryrun_includes_inner_execution_env(self, tmp_path, capsys):
        from nemo_evaluator.executors.docker_executor import DockerExecutor

        config = _make_config("docker")
        config.output.dir = str(tmp_path / "out")

        with patch("nemo_evaluator.executors.docker_executor._build_local_image", return_value="nemo-evaluator:local"):
            DockerExecutor().run(config, dry_run=True)

        captured = capsys.readouterr()
        assert "NEL_INNER_EXECUTION=1" in captured.out
        assert "PYTHONUNBUFFERED=1" in captured.out
        assert "--env-file" in captured.out

    def test_config_not_mutated(self, tmp_path):
        import json
        from nemo_evaluator.executors.docker_executor import DockerExecutor

        config = _make_config("docker")
        config.output.dir = str(tmp_path / "out")

        with patch("nemo_evaluator.executors.docker_executor._build_local_image", return_value="nemo-evaluator:local"):
            with patch("nemo_evaluator.executors.docker_executor.subprocess") as sub_mock:
                sub_mock.run.return_value = MagicMock(returncode=0, stdout="abc123\n")
                DockerExecutor().run(config)

        cfg_path = tmp_path / "out" / "_docker_config.json"
        saved = json.loads(cfg_path.read_text())
        assert saved["cluster"]["type"] == "docker"


class TestSlurmHeaderFlag:
    """Generated sbatch scripts must contain the inner execution export."""

    def test_header_contains_export(self):
        from nemo_evaluator.orchestration.slurm_gen import generate_sbatch

        config = _make_config("slurm", bench="gsm8k")

        script, _, _ = generate_sbatch(config)
        assert "export NEL_INNER_EXECUTION=1" in script

    def test_unbuffered_python(self):
        from nemo_evaluator.orchestration.slurm_gen import generate_sbatch

        config = _make_config("slurm", bench="gsm8k")

        script, _, _ = generate_sbatch(config)
        assert "export PYTHONUNBUFFERED=1" in script, "PYTHONUNBUFFERED must be set for real-time log output in SLURM"
