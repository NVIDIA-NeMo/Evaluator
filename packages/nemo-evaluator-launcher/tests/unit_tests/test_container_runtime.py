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
"""Tests for ContainerRuntime -- all tests mock shutil.which so they run in CI
without docker or podman installed."""

from unittest.mock import patch

import pytest

from nemo_evaluator_launcher.common.container_runtime import ContainerRuntime


class TestContainerRuntimeDetection:
    @pytest.mark.parametrize("runtime", ["docker", "podman"], ids=["docker", "podman"])
    def test_env_var_selects_runtime(self, monkeypatch, runtime):
        monkeypatch.setenv("CONTAINER_RUNTIME", runtime)
        with patch("shutil.which", return_value=f"/usr/bin/{runtime}"):
            rt = ContainerRuntime()
        assert rt.runtime == runtime

    def test_env_var_not_in_path_raises(self, monkeypatch):
        monkeypatch.setenv("CONTAINER_RUNTIME", "docker")
        with patch("shutil.which", return_value=None):
            with pytest.raises(RuntimeError, match="not found in PATH"):
                ContainerRuntime()

    def test_auto_detect_prefers_docker(self, monkeypatch):
        monkeypatch.delenv("CONTAINER_RUNTIME", raising=False)

        def which_side_effect(name):
            return "/usr/bin/docker" if name == "docker" else None

        with patch("shutil.which", side_effect=which_side_effect):
            rt = ContainerRuntime()
        assert rt.runtime == "docker"

    def test_auto_detect_falls_back_to_podman(self, monkeypatch):
        monkeypatch.delenv("CONTAINER_RUNTIME", raising=False)

        def which_side_effect(name):
            return "/usr/bin/podman" if name == "podman" else None

        with patch("shutil.which", side_effect=which_side_effect):
            rt = ContainerRuntime()
        assert rt.runtime == "podman"

    def test_auto_detect_neither_raises(self, monkeypatch):
        monkeypatch.delenv("CONTAINER_RUNTIME", raising=False)
        with patch("shutil.which", return_value=None):
            with pytest.raises(RuntimeError, match="No container runtime found"):
                ContainerRuntime()


class TestContainerRuntimeHelpers:
    def _make(self, runtime_name: str) -> ContainerRuntime:
        rt = ContainerRuntime.__new__(ContainerRuntime)
        rt.runtime = runtime_name
        return rt

    def test_get_config_path_docker_custom_dir(self, monkeypatch):
        monkeypatch.setenv("DOCKER_CONFIG", "/custom/docker")
        rt = self._make("docker")
        assert rt.get_config_path() == "/custom/docker/config.json"

    def test_get_config_path_podman_xdg(self, monkeypatch):
        monkeypatch.setenv("XDG_RUNTIME_DIR", "/run/user/1000")
        rt = self._make("podman")
        assert rt.get_config_path() == "/run/user/1000/containers/auth.json"

    def test_get_stop_command(self):
        rt = self._make("docker")
        assert rt.get_stop_command("mycontainer") == "docker stop mycontainer"
