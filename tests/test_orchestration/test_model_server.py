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
"""Tests for nemo_evaluator.orchestration.model_server — all subprocess mocked."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from nemo_evaluator.orchestration.model_server import (
    APIDeployment,
    DeployConfig,
    DockerModelDeployment,
    ProcessModelDeployment,
    get_deployment,
)


class TestDeployConfig:
    def test_defaults(self):
        c = DeployConfig()
        assert c.type == "api"
        assert c.gpus == 1
        assert c.port == 8000
        assert c.health_path == "/health"
        assert c.startup_timeout == 600.0

    def test_custom(self):
        c = DeployConfig(type="vllm", model="llama-3", gpus=4, port=9000)
        assert c.type == "vllm"
        assert c.model == "llama-3"


class TestAPIDeployment:
    def test_start_returns_url(self):
        dep = APIDeployment("http://api.example.com/v1")
        assert dep.start() == "http://api.example.com/v1"

    def test_health_wait_noop(self):
        dep = APIDeployment("http://x")
        dep.health_wait()

    def test_stop_noop(self):
        dep = APIDeployment("http://x")
        dep.stop()


class TestDockerModelDeployment:
    def test_build_docker_cmd(self):
        c = DeployConfig(type="docker", image="my-image:latest", gpus=2, port=8080)
        dep = DockerModelDeployment(c)
        cmd = dep._build_docker_cmd()
        assert "docker" in cmd
        assert "run" in cmd
        assert "my-image:latest" in cmd

    def test_build_docker_cmd_with_env(self):
        c = DeployConfig(
            type="docker",
            image="img",
            extra_env={"HF_TOKEN": "abc"},
            cluster_env={"CUDA_VISIBLE_DEVICES": "0,1"},
        )
        dep = DockerModelDeployment(c)
        cmd = dep._build_docker_cmd()
        env_flags = [cmd[i + 1] for i, v in enumerate(cmd) if v == "-e"]
        assert any("HF_TOKEN=abc" in f for f in env_flags)

    @patch("nemo_evaluator.orchestration.model_server.subprocess.run")
    def test_start_failure_raises(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stderr="no image")
        c = DeployConfig(type="docker", image="bad")
        dep = DockerModelDeployment(c)
        with pytest.raises(RuntimeError, match="Failed to start"):
            dep.start()


class TestProcessModelDeployment:
    def test_build_cmd_vllm(self):
        c = DeployConfig(type="vllm", model="meta-llama/Llama-3", port=8001)
        dep = ProcessModelDeployment(c)
        cmd = dep._build_cmd()
        assert "vllm.entrypoints.openai.api_server" in " ".join(cmd)
        assert "--model" in cmd
        assert "meta-llama/Llama-3" in cmd

    def test_build_cmd_sglang(self):
        c = DeployConfig(type="sglang", model="m", port=8002)
        dep = ProcessModelDeployment(c)
        cmd = dep._build_cmd()
        assert "sglang.launch_server" in " ".join(cmd)

    def test_build_cmd_unknown_raises(self):
        c = DeployConfig(type="unknown", model="m")
        dep = ProcessModelDeployment(c)
        with pytest.raises(ValueError, match="does not support"):
            dep._build_cmd()


class TestGetDeployment:
    def test_api(self):
        c = DeployConfig(type="api", extra_env={"base_url": "http://x"})
        dep = get_deployment(c)
        assert isinstance(dep, APIDeployment)

    def test_docker(self):
        c = DeployConfig(type="docker", image="img")
        dep = get_deployment(c)
        assert isinstance(dep, DockerModelDeployment)

    def test_vllm_single_node(self):
        c = DeployConfig(type="vllm", model="m", nodes=1)
        dep = get_deployment(c)
        assert isinstance(dep, ProcessModelDeployment)

    def test_unknown_raises(self):
        c = DeployConfig(type="nonexistent")
        with pytest.raises(ValueError, match="Unknown deployment type"):
            get_deployment(c)

    def test_sglang(self):
        c = DeployConfig(type="sglang", model="m")
        dep = get_deployment(c)
        assert isinstance(dep, ProcessModelDeployment)
