# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
Context manager for spawning the generic NeMo Evaluator resource server
(``ne_resource_server``) as a subprocess.

Usage::

    with ResourceServerProcess(
        eval_type="AIME_2025",
        framework_config=framework_config,
    ) as rsp:
        print(rsp.url)       # http://localhost:12345
        print(rsp.data_dir)  # /tmp/nemo_evaluator_data
"""

from __future__ import annotations

import json
import shutil
import socket
import subprocess
import time
from typing import Any, Optional

from nemo_evaluator.logging import get_logger

logger = get_logger(__name__)

DEFAULT_HOST = "localhost"
DEFAULT_DATA_DIR = "/tmp/nemo_evaluator_data"


def _find_free_port(host: str = DEFAULT_HOST) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


def _wait_for_server(
    host: str, port: int, max_wait: float = 120, interval: float = 0.5
) -> bool:
    deadline = time.time() + max_wait
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            time.sleep(interval)
    return False


class ResourceServerProcess:
    """Spawn ``ne_resource_server`` / ``nemo_evaluator_resource_server`` as a subprocess.

    The process is started in ``__enter__`` and terminated in ``__exit__``.

    Configuration is derived from the harness's ``framework.yml``.
    """

    def __init__(
        self,
        eval_type: str,
        framework_config: dict[str, Any],
        host: str = DEFAULT_HOST,
        port: Optional[int] = None,
        data_dir: str = DEFAULT_DATA_DIR,
    ) -> None:
        self.eval_type = eval_type
        self.host = host
        self.port = port or _find_free_port(host)
        self.data_dir = data_dir

        # Derive harness_package from framework.yml
        fw = framework_config.get("framework", {})
        self.harness_package = fw.get("pkg_name")
        if not self.harness_package:
            raise ValueError("framework_config['framework']['pkg_name'] is required")

        # Build judge config from defaults (if present)
        defaults = framework_config.get("defaults", {})
        judge_defaults = (
            defaults.get("config", {})
            .get("params", {})
            .get("extra", {})
            .get("judge", {})
        )
        self.harness_kwargs: dict[str, Any] = {}
        if judge_defaults and judge_defaults.get("url"):
            self.harness_kwargs["judge_config"] = {
                "backend": judge_defaults.get("backend", "openai"),
                "model": judge_defaults.get("model_id"),
                "url": judge_defaults.get("url"),
            }

        self._process: Optional[subprocess.Popen] = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def __enter__(self) -> ResourceServerProcess:
        exe = shutil.which("ne_resource_server") or shutil.which(
            "nemo_evaluator_resource_server"
        )
        if exe is None:
            raise RuntimeError(
                "Cannot find ne_resource_server or nemo_evaluator_resource_server on PATH. "
                "Is nemo-gym installed?"
            )

        cmd = [
            exe,
            "--harness_package",
            self.harness_package,
            "--eval_type",
            self.eval_type,
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--data_dir",
            self.data_dir,
        ]
        if self.harness_kwargs:
            cmd += ["--harness_kwargs_json", json.dumps(self.harness_kwargs)]

        logger.info("Spawning resource server: %s", " ".join(cmd))
        self._process = subprocess.Popen(cmd)

        if _wait_for_server(self.host, self.port):
            logger.info("Resource server ready at %s", self.url)
            return self

        self._process.terminate()
        self._process.wait(timeout=5)
        raise RuntimeError(f"Resource server failed to start at {self.url}")

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning(
                    "Force-killing resource server (pid=%s)", self._process.pid
                )
                self._process.kill()
                self._process.wait()
        return False
