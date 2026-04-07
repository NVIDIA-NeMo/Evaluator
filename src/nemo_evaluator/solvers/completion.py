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
"""CompletionSolver: uses the /completions endpoint instead of /chat/completions."""

from __future__ import annotations

from typing import Any

from nemo_evaluator.environments.base import SeedResult
from nemo_evaluator.observability.types import ModelResponse

from .base import SolveResult
from .trajectory_util import build_single_turn_atif


class CompletionSolver:
    """Uses the /completions endpoint instead of /chat/completions."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        seed: int | None = None,
        stop: list[str] | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
    ) -> None:
        from nemo_evaluator.engine.model_client import ModelClient

        self._model_client = ModelClient(
            base_url=base_url.rstrip("/"),
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            seed=seed,
            stop=stop,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
        )
        self._url = f"{base_url.rstrip('/')}/completions"
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._top_p = top_p
        self._seed = seed
        self._stop = stop
        self._frequency_penalty = frequency_penalty
        self._presence_penalty = presence_penalty

    async def solve(self, task: SeedResult) -> SolveResult:
        import time

        payload: dict[str, Any] = {
            "model": self._model,
            "prompt": task.prompt,
        }
        if self._temperature is not None:
            payload["temperature"] = self._temperature
        if self._max_tokens is not None:
            payload["max_tokens"] = self._max_tokens
        if self._top_p is not None:
            payload["top_p"] = self._top_p
        if self._seed is not None:
            payload["seed"] = self._seed
        if self._stop is not None:
            payload["stop"] = self._stop
        if self._frequency_penalty is not None:
            payload["frequency_penalty"] = self._frequency_penalty
        if self._presence_penalty is not None:
            payload["presence_penalty"] = self._presence_penalty
        t0 = time.monotonic()
        data = await self._model_client._post_with_retry(self._url, payload)
        latency = (time.monotonic() - t0) * 1000

        text = data["choices"][0].get("text", "")
        usage = data.get("usage", {})
        pt = usage.get("prompt_tokens", 0)
        ct = usage.get("completion_tokens", 0)
        model_resp = ModelResponse(
            content=text,
            model=data.get("model", self._model),
            finish_reason=data["choices"][0].get("finish_reason", ""),
            prompt_tokens=pt,
            completion_tokens=ct,
            total_tokens=usage.get("total_tokens", 0),
            latency_ms=round(latency, 2),
            raw_response=data,
        )
        trajectory = build_single_turn_atif(
            task.prompt,
            text,
            model_name=self._model,
            prompt_tokens=pt,
            completion_tokens=ct,
        )
        return SolveResult(response=text, model_response=model_resp, trajectory=trajectory)

    async def close(self) -> None:
        await self._model_client.close()
