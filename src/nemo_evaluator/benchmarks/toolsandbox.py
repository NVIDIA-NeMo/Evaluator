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
seed/solve/verify loop via ``run_batch()``, which runs ToolSandbox in one of
three modes and parses the resulting ``result_summary.json``.

Runner modes
------------
docker (default)
    Spawns a pre-built Docker image.  Requires Docker on the eval host.

    Build the image once::

        docker build -f docker/Dockerfile.toolsandbox -t toolsandbox-nel:latest .

apptainer
    Same image as ``docker`` mode but executed via ``apptainer run``.
    Use on SLURM clusters where Docker is unavailable.  ``image`` should be
    a path to a ``.sif`` or ``.sqsh`` file on the shared filesystem.

subprocess
    Runs the ToolSandbox entrypoint directly as a Python subprocess — no
    container needed.  Use when the eval container already has ToolSandbox
    pre-installed (e.g. ``Dockerfile.toolsandbox-combined``).  Set
    ``python_exe`` to the venv Python that has ToolSandbox, and
    ``entrypoint`` to the patch script path.

Config usage::

    benchmarks:
      - name: toolsandbox
        params:
          # --- runner selection ---
          runner: docker                           # docker | apptainer | subprocess
          image: toolsandbox-nel:latest            # docker image name / sif path
          # --- subprocess-mode overrides ---
          python_exe: /opt/toolsandbox-venv/bin/python
          entrypoint: /opt/toolsandbox_entrypoint.py
          # --- benchmark settings ---
          user_model: meta/llama-3.1-70b-instruct  # user-simulator model
          parallel: 4                              # concurrent scenarios
          test_mode: false                         # true = small subset only
          scenarios: []                            # [] = all scenarios
        solver:
          type: simple
          service: my_model
        timeout: 7200.0

Both agent and user simulator call the NVIDIA Inference API — no OpenAI
key required.  ``NVIDIA_API_KEY`` must be set in the environment.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from nemo_evaluator.environments.base import EvalEnvironment, SeedResult, VerifyResult
from nemo_evaluator.environments.registry import register

if TYPE_CHECKING:
    from nemo_evaluator.sandbox.base import Sandbox

logger = logging.getLogger(__name__)

_DEFAULT_IMAGE = "toolsandbox-nel:latest"
_DEFAULT_USER_MODEL = "meta/llama-3.1-70b-instruct"
_DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
_DEFAULT_ENTRYPOINT = "/opt/toolsandbox_entrypoint.py"
_CONTAINER_OUTPUT = "/output"

_CLI_AGENT = "Gorilla"
_CLI_USER = "GPT_4_o_2024_05_13"


def _to_openai_base_url(url: str) -> str:
    """Strip /chat/completions, /completions, /responses path suffix from NEL service URLs."""
    for suffix in ("/chat/completions", "/completions", "/responses"):
        if url.endswith(suffix):
            return url[: -len(suffix)]
    return url.rstrip("/")


@register("toolsandbox")
class ToolSandboxEnvironment(EvalEnvironment):
    """Runs ToolSandbox and parses aggregate metrics.

    The entire scenario suite executes as a single batch.
    ``seed()`` and ``verify()`` are not used.
    """

    def __init__(
        self,
        runner: Literal["docker", "apptainer", "subprocess"] = "docker",
        image: str = _DEFAULT_IMAGE,
        python_exe: str | None = None,
        entrypoint: str = _DEFAULT_ENTRYPOINT,
        user_model: str = _DEFAULT_USER_MODEL,
        scenarios: list[str] | None = None,
        parallel: int = 4,
        timeout: float = 7200.0,
        test_mode: bool = False,
    ) -> None:
        super().__init__()
        self._runner = runner
        self._image = image
        self._python_exe = python_exe or sys.executable
        self._entrypoint = entrypoint
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

            cmd, env = self._build_cmd(output_dir, base_url, model_id, api_key)
            logger.info("Launching ToolSandbox (%s): %s", self._runner, " ".join(cmd[:10]) + " ...")

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
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
                    "ToolSandbox exited %d:\n%s",
                    rc,
                    (stderr or b"").decode(errors="replace")[:2000],
                )

            return self._parse_results(output_dir, rc, model_id)

    # ------------------------------------------------------------------
    # Command builders
    # ------------------------------------------------------------------

    def _build_cmd(
        self,
        output_dir: Path,
        base_url: str,
        model_id: str,
        api_key: str,
    ) -> tuple[list[str], dict[str, str] | None]:
        """Return (cmd, env) for the selected runner."""
        if self._runner == "subprocess":
            return self._build_subprocess_cmd(output_dir, base_url, model_id, api_key)
        if self._runner == "apptainer":
            return self._build_apptainer_cmd(output_dir, base_url, model_id, api_key), None
        return self._build_docker_cmd(output_dir, base_url, model_id, api_key), None

    def _toolsandbox_cli_args(self, output_dir_str: str) -> list[str]:
        args = ["--agent", _CLI_AGENT, "--user", _CLI_USER]
        args.extend(["--output_dir", output_dir_str])
        args.extend(["--parallel", str(self._parallel)])
        if self._test_mode:
            args.append("--test_mode")
        elif self._scenarios:
            args.extend(["--scenarios"] + list(self._scenarios))
        return args

    def _container_env_flags(self, base_url: str, model_id: str, api_key: str) -> list[str]:
        flags = [
            "-e", f"NVIDIA_BASE_URL={base_url}",
            "-e", f"NVIDIA_AGENT_MODEL={model_id}",
            "-e", f"NVIDIA_USER_MODEL={self._user_model}",
        ]
        if api_key:
            flags.extend(["-e", f"NVIDIA_API_KEY={api_key}"])
        return flags

    def _build_docker_cmd(self, output_dir: Path, base_url: str, model_id: str, api_key: str) -> list[str]:
        cmd = ["docker", "run", "--rm", "-v", f"{output_dir}:{_CONTAINER_OUTPUT}"]
        cmd.extend(self._container_env_flags(base_url, model_id, api_key))
        cmd.append(self._image)
        cmd.extend(self._toolsandbox_cli_args(_CONTAINER_OUTPUT))
        return cmd

    def _build_apptainer_cmd(self, output_dir: Path, base_url: str, model_id: str, api_key: str) -> list[str]:
        env_flags: list[str] = []
        for flag in self._container_env_flags(base_url, model_id, api_key):
            if flag == "-e":
                continue
            env_flags.extend(["--env", flag])

        cmd = [
            "apptainer", "run", "--bind", f"{output_dir}:{_CONTAINER_OUTPUT}",
            *env_flags,
            self._image,
        ]
        cmd.extend(self._toolsandbox_cli_args(_CONTAINER_OUTPUT))
        return cmd

    def _build_subprocess_cmd(
        self, output_dir: Path, base_url: str, model_id: str, api_key: str
    ) -> tuple[list[str], dict[str, str]]:
        # Env is passed via environment variables to the subprocess
        env = {
            **os.environ,
            "NVIDIA_BASE_URL": base_url,
            "NVIDIA_AGENT_MODEL": model_id,
            "NVIDIA_USER_MODEL": self._user_model,
        }
        if api_key:
            env["NVIDIA_API_KEY"] = api_key

        cmd = [self._python_exe, self._entrypoint]
        cmd.extend(self._toolsandbox_cli_args(str(output_dir)))
        return cmd, env

    # ------------------------------------------------------------------
    # Results parsing
    # ------------------------------------------------------------------

    def _parse_results(self, output_dir: Path, exit_code: int, model_id: str) -> dict[str, Any]:
        summary = self._load_result_summary(output_dir)
        scores = self._extract_scores(summary)

        return {
            "benchmark": {
                "name": self.name,
                "samples": len(summary.get("per_scenario_results", [])),
                "scores": scores,
            },
            "config": {
                "benchmark": self.name,
                "runner": self._runner,
                "image": self._image if self._runner != "subprocess" else None,
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
        """Extract scores from ToolSandbox result_summary.json.

        Real format (confirmed from smoke test):
          category_aggregated_results:
            ALL_CATEGORIES: {similarity: float, turn_count: float}
            STATE_DEPENDENCY: {similarity: float, turn_count: float}
            ...
        """
        scores: dict[str, Any] = {}

        cat_results: dict[str, Any] = summary.get("category_aggregated_results") or {}

        # Overall score comes from the ALL_CATEGORIES aggregate
        all_cat = cat_results.get("ALL_CATEGORIES") or {}
        if "similarity" in all_cat:
            scores["similarity"] = {"value": round(float(all_cat["similarity"]), 4)}
        if "turn_count" in all_cat:
            scores["turn_count"] = {"value": round(float(all_cat["turn_count"]), 2)}

        # Per-category breakdown (skip ALL_CATEGORIES to avoid duplication)
        for cat_name, cat_data in cat_results.items():
            if cat_name == "ALL_CATEGORIES":
                continue
            if isinstance(cat_data, dict) and "similarity" in cat_data:
                scores[f"per_category/{cat_name}/similarity"] = {
                    "value": round(float(cat_data["similarity"]), 4)
                }

        return scores
