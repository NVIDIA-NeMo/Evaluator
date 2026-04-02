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
"""VLMSolver: vision-language model solver."""

from __future__ import annotations

from typing import Any

from nemo_evaluator.environments.base import SeedResult

from .base import SolveResult
from .trajectory_util import build_single_turn_atif


class VLMSolver:
    """Vision-language model solver. Uses vlm_chat() when images are present,
    falls back to regular chat() for text-only tasks."""

    def __init__(self, client: Any, system_prompt: str | None = None, image_detail: str = "auto") -> None:
        self._client = client
        self._system = system_prompt
        self._detail = image_detail

    async def solve(self, task: SeedResult) -> SolveResult:
        effective_system = self._system or task.system
        if task.images:
            resp = await self._client.vlm_chat(
                prompt=task.prompt,
                images=task.images,
                system=effective_system,
                detail=self._detail,
            )
        elif task.messages:
            resp = await self._client.chat(messages=task.messages)
        else:
            resp = await self._client.chat(task.prompt, system=effective_system)
        trajectory = build_single_turn_atif(
            task.prompt,
            resp.content,
            system=effective_system,
            model_name=getattr(resp, "model", None),
            prompt_tokens=getattr(resp, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(resp, "completion_tokens", 0) or 0,
        )
        return SolveResult(response=resp.content, model_response=resp, trajectory=trajectory)

    async def close(self) -> None:
        await self._client.close()
