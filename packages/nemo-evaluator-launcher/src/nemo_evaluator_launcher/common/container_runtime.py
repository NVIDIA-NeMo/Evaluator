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
"""Container runtime abstraction for Docker and Podman."""

import os
import shutil
from typing import Optional


class ContainerRuntime:
    """Abstraction for container runtime (Docker or Podman)."""

    def __init__(self):
        self.runtime = self._detect_runtime()

    def _detect_runtime(self) -> str:
        """Detect available container runtime."""
        # Check environment variable first
        runtime = os.getenv("CONTAINER_RUNTIME")
        if runtime in ("docker", "podman"):
            if shutil.which(runtime):
                return runtime
            else:
                raise RuntimeError(
                    f"CONTAINER_RUNTIME={runtime} specified but not found in PATH"
                )

        # Auto-detect: prefer Docker for backward compatibility
        if shutil.which("docker"):
            return "docker"
        elif shutil.which("podman"):
            return "podman"
        else:
            raise RuntimeError(
                "No container runtime found. Please install Docker or Podman, "
                "or set CONTAINER_RUNTIME environment variable."
            )

    @property
    def command(self) -> str:
        """Get the container runtime command name."""
        return self.runtime

    def get_config_path(self) -> Optional[str]:
        """Get path to container runtime config file."""
        if self.runtime == "docker":
            docker_config_dir = os.getenv("DOCKER_CONFIG")
            if docker_config_dir:
                return os.path.join(docker_config_dir, "config.json")
            return os.path.expanduser("~/.docker/config.json")
        elif self.runtime == "podman":
            # Podman uses different config location
            xdg_runtime_dir = os.getenv("XDG_RUNTIME_DIR")
            if xdg_runtime_dir:
                return os.path.join(xdg_runtime_dir, "containers/auth.json")
            return os.path.expanduser("~/.config/containers/auth.json")
        return None

    def get_stop_command(self, container_name: str) -> str:
        """Get command to stop a container."""
        return f"{self.runtime} stop {container_name}"

    def get_kill_process_pattern(self, container_name: str) -> str:
        """Get pattern to kill container processes."""
        return f"{self.runtime} run.*{container_name}"
