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
"""GymSolver: delegate agent execution to a running nemo-gym server."""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, Any

import aiohttp

from nemo_evaluator.environments.base import SeedResult
from nemo_evaluator.environments.gym_protocol import (
    extract_assistant_text,
    wrap_text_as_responses_create_params,
)
from nemo_evaluator.observability.types import ModelResponse

from .base import SolveResult
from .trajectory_util import build_atif_trajectory

if TYPE_CHECKING:
    from nemo_evaluator.sandbox.base import Sandbox

logger = logging.getLogger(__name__)


class GymSolver:
    """Delegates agent execution to a nemo-gym server.

    When ``trust_reward=True``, the gym's reward is used directly (skips nel verify).
    When ``False`` (default), only the agent response is extracted for nel's own scorer.
    """

    def __init__(
        self,
        gym_url: str,
        *,
        gym_agent: str | None = None,
        trust_reward: bool = False,
        model_id: str = "",
        model_url: str = "",
        api_key: str | None = None,
        timeout: float = 3600.0,
    ) -> None:
        self._gym_url = gym_url.rstrip("/")
        self._gym_agent = (gym_agent or "").lower()
        self._trust_reward = trust_reward
        self._model_id = model_id
        self._model_url = model_url
        self._api_key = api_key
        self._timeout = timeout

    def _build_request(self, task: SeedResult) -> dict[str, Any]:
        meta = task.metadata or {}

        if self._gym_agent == "miniswe":
            return self._build_miniswe_request(task, meta)
        return self._build_swebench_request(task, meta)

    def _build_swebench_request(self, task: SeedResult, meta: dict) -> dict[str, Any]:
        rcp = wrap_text_as_responses_create_params(task.prompt, model=self._model_id or "evaluator")
        rcp["metadata"] = {
            "problem_statement": meta.get("problem_statement", task.prompt),
            "instance_id": meta.get("instance_id", ""),
            "base_commit": meta.get("base_commit", ""),
            "dataset_name": meta.get("dataset_name", ""),
            "split": meta.get("split", "test"),
            "instance_dict": meta.get("instance_dict", json.dumps(meta)),
        }
        if self._api_key:
            rcp["metadata"]["api_key"] = self._api_key
        if self._model_url:
            rcp["metadata"]["model_url"] = self._model_url
        return {"responses_create_params": rcp}

    def _build_miniswe_request(self, task: SeedResult, meta: dict) -> dict[str, Any]:
        rcp: dict[str, Any] = {
            "temperature": 0.0,
            "top_p": 1.0,
            "input": [],
        }
        if self._model_id:
            rcp["model"] = self._model_id
        return {
            "responses_create_params": rcp,
            "instance_id": meta.get("instance_id", ""),
            "subset": meta.get("subset", "gym"),
            "split": meta.get("split", "test"),
        }

    async def solve(
        self,
        task: SeedResult,
        sandbox: Sandbox | None = None,
    ) -> SolveResult:
        body = self._build_request(task)

        t0 = time.monotonic()
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self._timeout),
        ) as client:
            url = f"{self._gym_url}/run"
            logger.info(
                "GymSolver: POST %s (agent=%s, trust_reward=%s)", url, self._gym_agent or "auto", self._trust_reward
            )
            async with client.post(url, json=body) as resp:
                resp.raise_for_status()
                result = await resp.json()
        latency_ms = (time.monotonic() - t0) * 1000

        response_text = self._extract_response(result)
        reward = float(result.get("reward", 0.0)) if self._trust_reward else None

        scoring_details: dict[str, Any] = {}
        if self._trust_reward:
            scoring_details = {
                k: v for k, v in result.items() if k not in ("responses_create_params", "response", "reward")
            }

        trajectory = self._extract_trajectory(result)

        model_resp = ModelResponse(
            content=response_text,
            model=self._model_id or "gym-agent",
            total_tokens=0,
            completion_tokens=0,
            latency_ms=round(latency_ms, 2),
        )

        logger.info(
            "GymSolver: %.1fs, %d chars response%s",
            latency_ms / 1000,
            len(response_text),
            f", reward={reward}" if reward is not None else "",
        )

        return SolveResult(
            response=response_text,
            model_response=model_resp,
            trajectory=trajectory,
            reward=reward,
            scoring_details=scoring_details,
        )

    def _extract_response(self, result: dict[str, Any]) -> str:
        gym_resp = result.get("response")
        if gym_resp is not None:
            return extract_assistant_text(gym_resp)
        rcp = result.get("responses_create_params", {})
        output = rcp.get("output") or rcp.get("input")
        if output:
            return extract_assistant_text({"output": output} if isinstance(output, list) else output)
        return ""

    def _extract_trajectory(self, result: dict[str, Any]) -> list[dict[str, Any]]:
        gym_resp = result.get("response", {})
        raw_items: list[dict[str, Any]] = []
        if isinstance(gym_resp, dict):
            output = gym_resp.get("output", [])
            if isinstance(output, list):
                raw_items = [item for item in output if isinstance(item, dict)]
        if not raw_items:
            return []
        steps = [
            {
                "source": "agent",
                "message": item.get("content", str(item)),
                "extra": {k: v for k, v in item.items() if k != "content"},
            }
            for item in raw_items
        ]
        return build_atif_trajectory(
            steps,
            agent_name=self._gym_agent or "gym-agent",
            model_name=self._model_id or None,
        )

    async def close(self) -> None:
        pass
