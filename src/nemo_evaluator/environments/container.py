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
"""ContainerEnvironment: run legacy eval-factory containers as opaque environments.

Wraps any legacy evaluation container image, runs it via ``docker run``,
and injects a **legacy-format** ``run_config.yaml`` so the container's
``nemo-evaluator run_eval`` entrypoint can consume it directly.

The generated run-config follows the canonical Evaluator schema::

    config:
      type: <task>
      output_dir: /results
      params: { ... }          # optional pass-through
    target:
      api_endpoint:
        url: <model_url>
        model_id: <model_id>
        api_key_name: NEMO_API_KEY
        type: chat              # or completions

Usage in NEL config:
    benchmarks:
      - name: "container://registry/image:tag#eval_type"
        solver:
          type: container
          service: nemotron
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nemo_evaluator.sandbox.base import Sandbox

import yaml

from nemo_evaluator.environments.base import EvalEnvironment, SeedResult, VerifyResult

logger = logging.getLogger(__name__)

_CONTAINER_RESULTS_DIR = "/results"
_CONTAINER_CONFIG_PATH = "/config/run_config.yaml"
_API_KEY_ENV = "NEMO_API_KEY"

_NEL_PROTOCOL_TO_LEGACY_TYPE = {
    "chat_completions": "chat",
    "completions": "completions",
    "responses": "chat",
}


class ContainerEnvironment(EvalEnvironment):
    """Runs a legacy eval-factory container and parses its results.

    The container is launched with a mounted ``run_config.yaml`` in the
    canonical legacy Evaluator format.  Results are read from
    ``/results/results.yml`` (legacy) or ``/results/results.json``.
    """

    def __init__(
        self,
        image: str,
        task: str = "",
        model_url: str | None = None,
        model_id: str | None = None,
        api_key: str | None = None,
        endpoint_type: str = "chat",
        legacy_params: dict[str, Any] | None = None,
        extra_env: dict[str, str] | None = None,
        extra_mounts: list[str] | None = None,
        pre_cmd: str | None = None,
        timeout: float = 3600.0,
    ) -> None:
        super().__init__()
        self.name = f"container/{task or image.split('/')[-1].split(':')[0]}"
        self._image = image
        self._task = task
        self._model_url = model_url
        self._model_id = model_id
        self._api_key = api_key
        self._endpoint_type = endpoint_type
        self._legacy_params = legacy_params or {}
        self._extra_env = extra_env or {}
        self._extra_mounts = extra_mounts or []
        self._pre_cmd = pre_cmd
        self._timeout = timeout

    async def dataset_size(self) -> int:
        return 0

    async def seed(self, idx: int) -> SeedResult:
        raise NotImplementedError("ContainerEnvironment uses run_batch()")

    async def verify(
        self, response: str, expected: str, sandbox: Sandbox | None = None, **metadata: Any
    ) -> VerifyResult:
        raise NotImplementedError("ContainerEnvironment uses run_batch()")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def run_batch(self, solver: Any = None, config: dict[str, Any] | None = None) -> dict[str, Any]:
        """Run the container and parse results."""
        config = config or {}
        model_url = self._model_url or config.get("base_url", "")
        model_id = self._model_id or config.get("model", "")
        api_key = self._api_key or config.get("api_key", "")
        endpoint_type = config.get("endpoint_type", self._endpoint_type)
        extra_params = {**self._legacy_params, **config.get("params", {})}

        with tempfile.TemporaryDirectory(prefix="nel_container_") as tmpdir:
            results_dir = Path(tmpdir) / "results"
            results_dir.mkdir()

            run_config = self._build_legacy_run_config(
                model_url,
                model_id,
                endpoint_type,
                extra_params,
            )
            config_path = Path(tmpdir) / "run_config.yaml"
            config_path.write_text(yaml.dump(run_config, sort_keys=False), encoding="utf-8")

            cmd = self._build_docker_cmd(results_dir, config_path)

            env_vars: dict[str, str] = {}
            if api_key:
                env_vars[_API_KEY_ENV] = api_key
            env_vars.update(self._extra_env)

            for k, v in env_vars.items():
                cmd.extend(["-e", f"{k}={v}"])

            cmd.append(self._image)

            if self._pre_cmd:
                cmd.extend(["bash", "-c", self._pre_cmd])
            else:
                cmd.extend(
                    [
                        "run_eval",
                        "--run_config",
                        _CONTAINER_CONFIG_PATH,
                        "--output_dir",
                        _CONTAINER_RESULTS_DIR,
                    ]
                )

            logger.info("Running container: %s", " ".join(cmd[:12]) + " ...")

            import asyncio as _aio

            proc = await _aio.create_subprocess_exec(
                *cmd,
                stdout=_aio.subprocess.PIPE,
                stderr=_aio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await _aio.wait_for(proc.communicate(), timeout=self._timeout)
            except _aio.TimeoutError:
                proc.kill()
                await proc.wait()
                stderr = b"timeout"

            if proc.returncode != 0:
                logger.error("Container failed (exit %d): %s", proc.returncode, (stderr or b"").decode()[:2000])

            return self._parse_results(results_dir, proc.returncode or 0)

    # ------------------------------------------------------------------
    # Legacy run_config generation
    # ------------------------------------------------------------------

    def _build_legacy_run_config(
        self,
        model_url: str,
        model_id: str,
        endpoint_type: str,
        extra_params: dict[str, Any],
    ) -> dict[str, Any]:
        """Build a legacy Evaluator ``run_config.yaml``.

        The generated YAML is consumed by the container's
        ``nemo-evaluator run_eval --run_config <path>`` entrypoint, which
        merges it with the baked-in ``framework.yml`` defaults.
        """
        run_config: dict[str, Any] = {
            "config": {
                "type": self._task,
                "output_dir": _CONTAINER_RESULTS_DIR,
            },
            "target": {
                "api_endpoint": {
                    "url": model_url,
                    "model_id": model_id,
                    "api_key_name": _API_KEY_ENV,
                    "type": endpoint_type,
                },
            },
        }
        if extra_params:
            run_config["config"]["params"] = extra_params
        return run_config

    # ------------------------------------------------------------------
    # Docker plumbing
    # ------------------------------------------------------------------

    def _build_docker_cmd(self, results_dir: Path, config_path: Path) -> list[str]:
        cmd = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{results_dir}:{_CONTAINER_RESULTS_DIR}",
            "-v",
            f"{config_path}:{_CONTAINER_CONFIG_PATH}:ro",
        ]
        for mount in self._extra_mounts:
            cmd.extend(["-v", mount])
        return cmd

    # ------------------------------------------------------------------
    # Results parsing (handles both legacy and simple formats)
    # ------------------------------------------------------------------

    def _parse_results(self, results_dir: Path, exit_code: int) -> dict[str, Any]:
        """Parse container output into NEL bundle format.

        Supports two result layouts:

        1. **Legacy Evaluator** — ``results.yml`` with nested
           ``results.tasks.<task>.metrics.<group>.scores.<metric>.value``
        2. **Simple** — flat ``metrics:`` or ``scores:`` dict at the top
           level of ``results.yml`` / ``results.json``.
        """
        results_file = results_dir / "results.yml"
        results_json = results_dir / "results.json"

        raw: dict[str, Any] = {}
        if results_file.exists():
            raw = yaml.safe_load(results_file.read_text()) or {}
        elif results_json.exists():
            raw = json.loads(results_json.read_text())

        scores = self._extract_scores(raw)

        return {
            "benchmark": {
                "name": self.name,
                "samples": raw.get("samples", raw.get("n_samples", 0)),
                "scores": scores,
            },
            "config": {
                "benchmark": self.name,
                "image": self._image,
                "task": self._task,
                "framework": "container",
            },
            "_container_exit_code": exit_code,
        }

    @staticmethod
    def _extract_scores(raw: dict[str, Any]) -> dict[str, Any]:
        """Extract scores from either legacy or simple format."""

        # Legacy Evaluator format:
        #   results:
        #     tasks:
        #       <task>:
        #         metrics:
        #           <group>:
        #             scores:
        #               <metric>:
        #                 value: 0.85
        #                 stats: { ... }
        results_block = raw.get("results")
        if isinstance(results_block, dict):
            tasks = results_block.get("tasks") or {}
            scores: dict[str, Any] = {}
            for task_name, task_data in tasks.items():
                if not isinstance(task_data, dict):
                    continue
                for group_name, group_data in (task_data.get("metrics") or {}).items():
                    if not isinstance(group_data, dict):
                        continue
                    for metric_name, metric_data in (group_data.get("scores") or {}).items():
                        if not isinstance(metric_data, dict) or "value" not in metric_data:
                            continue
                        key = f"{task_name}/{group_name}/{metric_name}"
                        scores[key] = {
                            "value": round(float(metric_data["value"]), 4),
                        }
                        if "stats" in metric_data:
                            scores[key]["stats"] = metric_data["stats"]
            if scores:
                return scores

            # Legacy format may also have groups
            groups = results_block.get("groups") or {}
            for group_name, group_data in groups.items():
                if not isinstance(group_data, dict):
                    continue
                for mg_name, mg_data in (group_data.get("metrics") or {}).items():
                    if not isinstance(mg_data, dict):
                        continue
                    for metric_name, metric_data in (mg_data.get("scores") or {}).items():
                        if not isinstance(metric_data, dict) or "value" not in metric_data:
                            continue
                        key = f"{group_name}/{mg_name}/{metric_name}"
                        scores[key] = {
                            "value": round(float(metric_data["value"]), 4),
                        }
                        if "stats" in metric_data:
                            scores[key]["stats"] = metric_data["stats"]
            if scores:
                return scores

        # Simple flat format: metrics: {acc: 0.85} or scores: {acc: {value: 0.85}}
        source = raw.get("metrics") or raw.get("scores") or {}
        scores = {}
        for key, value in source.items():
            if isinstance(value, (int, float)):
                scores[key] = {"value": round(float(value), 4)}
            elif isinstance(value, dict) and "value" in value:
                scores[key] = value
        return scores
