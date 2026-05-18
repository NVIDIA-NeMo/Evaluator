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
"""ToolSandbox benchmark — Apple's stateful multi-turn tool-use evaluation.

Registers ``toolsandbox`` as a built-in benchmark.  Bypasses the standard
seed/solve/verify loop via ``run_batch()``, which spawns a pre-built Docker
image containing ToolSandbox and parses the resulting ``result_summary.json``.

Config usage::

    benchmarks:
      - name: toolsandbox
        params:
          image: toolsandbox-nel:latest           # pre-built Docker image
          user_model: meta/llama-3.1-70b-instruct # user-simulator model
          parallel: 4                             # concurrent scenarios
          test_mode: false                        # true = small subset only
          scenarios: []                           # [] = all scenarios
        solver:
          type: simple
          service: my_model
        timeout: 7200.0

Build the image once before running::

    docker build -f docker/Dockerfile.toolsandbox -t toolsandbox-nel:latest .

The agent under evaluation is taken from the ``solver.service`` entry.
The user simulator calls the same NVIDIA API base URL with ``user_model``.
Both require ``NVIDIA_API_KEY`` in the environment.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from nemo_evaluator.environments.base import EvalEnvironment, SeedResult, VerifyResult
from nemo_evaluator.environments.registry import register

if TYPE_CHECKING:
    from nemo_evaluator.sandbox.base import Sandbox

logger = logging.getLogger(__name__)

_DEFAULT_IMAGE = "toolsandbox-nel:latest"
_DEFAULT_USER_MODEL = "meta/llama-3.1-70b-instruct"
_DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
_CONTAINER_OUTPUT = "/output"

# ToolSandbox uses --agent Gorilla / --user GPT_4_o_2024_05_13 as the CLI
# selectors; the entrypoint script patches those factory entries to point at
# NVIDIANIMAgent / NVIDIANIMUser backed by NVIDIA_BASE_URL.
_CLI_AGENT = "Gorilla"
_CLI_USER = "GPT_4_o_2024_05_13"


def _to_openai_base_url(url: str) -> str:
    """Normalize a NEL service URL to the OpenAI SDK base_url format.

    NEL service URLs include the full path (e.g. /v1/chat/completions).
    ToolSandbox expects just the base (e.g. https://host/v1).
    """
    for suffix in ("/chat/completions", "/completions", "/responses"):
        if url.endswith(suffix):
            return url[: -len(suffix)]
    return url.rstrip("/")


@register("toolsandbox")
class ToolSandboxEnvironment(EvalEnvironment):
    """Runs ToolSandbox in a Docker container and parses aggregate metrics.

    The entire scenario suite executes as a single batch inside the container.
    ``seed()`` and ``verify()`` are not used.
    """

    def __init__(
        self,
        image: str = _DEFAULT_IMAGE,
        user_model: str = _DEFAULT_USER_MODEL,
        scenarios: list[str] | None = None,
        parallel: int = 4,
        timeout: float = 7200.0,
        test_mode: bool = False,
    ) -> None:
        super().__init__()
        self._image = image
        self._user_model = user_model
        self._scenarios: list[str] = scenarios or []
        self._parallel = parallel
        self._timeout = timeout
        self._test_mode = test_mode

    # ------------------------------------------------------------------
    # EvalEnvironment interface
    # ------------------------------------------------------------------

    async def dataset_size(self) -> int:
        return 0

    async def seed(self, idx: int) -> SeedResult:
        raise NotImplementedError("ToolSandboxEnvironment uses run_batch()")

    async def verify(
        self,
        response: str,
        expected: str,
        sandbox: "Sandbox | None" = None,
        **metadata: Any,
    ) -> VerifyResult:
        raise NotImplementedError("ToolSandboxEnvironment uses run_batch()")

    # ------------------------------------------------------------------
    # Batch execution
    # ------------------------------------------------------------------

    async def run_batch(self, solver: Any = None, config: dict[str, Any] | None = None) -> dict[str, Any]:
        config = config or {}
        model_url = config.get("base_url", "") or _DEFAULT_BASE_URL
        model_id = config.get("model", "")
        api_key = config.get("api_key") or os.environ.get("NVIDIA_API_KEY", "")

        base_url = _to_openai_base_url(model_url)

        with tempfile.TemporaryDirectory(prefix="nel_toolsandbox_") as tmpdir:
            output_dir = Path(tmpdir) / "output"
            output_dir.mkdir()

            cmd = self._build_docker_cmd(output_dir, base_url, model_id, api_key)
            logger.info("Launching ToolSandbox: %s", " ".join(cmd[:10]) + " ...")

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=self._timeout)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                logger.error("ToolSandbox timed out after %.0fs", self._timeout)
                stderr = b"timeout"

            rc = proc.returncode or 0
            if rc != 0:
                logger.error(
                    "ToolSandbox container exited %d:\n%s",
                    rc,
                    (stderr or b"").decode(errors="replace")[:2000],
                )

            return self._parse_results(output_dir, rc, model_id)

    # ------------------------------------------------------------------
    # Docker command builder
    # ------------------------------------------------------------------

    def _build_docker_cmd(
        self,
        output_dir: Path,
        base_url: str,
        model_id: str,
        api_key: str,
    ) -> list[str]:
        cmd = [
            "docker", "run", "--rm",
            "-v", f"{output_dir}:{_CONTAINER_OUTPUT}",
            "-e", f"NVIDIA_BASE_URL={base_url}",
            "-e", f"NVIDIA_AGENT_MODEL={model_id}",
            "-e", f"NVIDIA_USER_MODEL={self._user_model}",
        ]
        if api_key:
            cmd.extend(["-e", f"NVIDIA_API_KEY={api_key}"])

        cmd.append(self._image)

        cmd.extend(["--agent", _CLI_AGENT, "--user", _CLI_USER])
        cmd.extend(["--output_dir", _CONTAINER_OUTPUT])
        cmd.extend(["--parallel", str(self._parallel)])

        if self._test_mode:
            cmd.append("--test_mode")
        elif self._scenarios:
            cmd.extend(["--scenarios"] + list(self._scenarios))

        return cmd

    # ------------------------------------------------------------------
    # Results parsing
    # ------------------------------------------------------------------

    def _parse_results(self, output_dir: Path, exit_code: int, model_id: str) -> dict[str, Any]:
        summary = self._load_result_summary(output_dir)
        scores = self._extract_scores(summary)

        return {
            "benchmark": {
                "name": self.name,
                "samples": summary.get("num_scenarios", len(summary.get("per_category", {}))),
                "scores": scores,
            },
            "config": {
                "benchmark": self.name,
                "image": self._image,
                "model": model_id,
                "user_model": self._user_model,
                "framework": "toolsandbox",
                "scenarios": self._scenarios or "all",
                "test_mode": self._test_mode,
            },
            "_container_exit_code": exit_code,
        }

    def _load_result_summary(self, output_dir: Path) -> dict[str, Any]:
        for candidate in sorted(output_dir.rglob("result_summary.json")):
            try:
                return json.loads(candidate.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Could not parse %s: %s", candidate, exc)
        logger.warning("No result_summary.json found in %s", output_dir)
        return {}

    @staticmethod
    def _extract_scores(summary: dict[str, Any]) -> dict[str, Any]:
        scores: dict[str, Any] = {}

        for metric in ("similarity", "turn_count"):
            if metric in summary:
                scores[metric] = {"value": round(float(summary[metric]), 4)}

        per_category = summary.get("per_category") or {}
        for cat_name, cat_data in per_category.items():
            if isinstance(cat_data, dict) and "similarity" in cat_data:
                scores[f"per_category/{cat_name}/similarity"] = {
                    "value": round(float(cat_data["similarity"]), 4)
                }

        return scores
