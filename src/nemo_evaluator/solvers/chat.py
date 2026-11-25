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
"""ChatSolver: single model call. Default for standard benchmarks."""

from __future__ import annotations

from typing import Any

from nemo_evaluator.environments.base import SeedResult

from .base import SolveResult
from .trajectory_util import build_single_turn_atif


def with_system_prompt(messages: list[dict[str, Any]], system: str | None) -> list[dict[str, Any]]:
    """Apply a solver-level system prompt to a seeded conversation.

    Mirrors ``system_prompt or task.system`` on the prompt path: the solver's
    prompt replaces a leading system turn, or is prepended if there is none.
    """
    if not system:
        return messages
    rest = messages[1:] if messages and messages[0].get("role") == "system" else messages
    return [{"role": "system", "content": system}, *rest]


class ChatSolver:
    """Single model call. Default for standard benchmarks."""

    def __init__(self, client: Any, system_prompt: str | None = None) -> None:
        self._client = client
        self._system = system_prompt

    async def solve(self, task: SeedResult) -> SolveResult:
        effective_system = self._system or task.system
        if task.messages:
            resp = await self._client.chat(messages=with_system_prompt(task.messages, self._system))
        else:
            resp = await self._client.chat(task.prompt, system=effective_system)
        trajectory = build_single_turn_atif(
            task.prompt,
            resp.content,
            system=effective_system,
            model_name=getattr(resp, "model", None),
            prompt_tokens=getattr(resp, "prompt_tokens", None),
            completion_tokens=getattr(resp, "completion_tokens", None),
        )
        return SolveResult(response=resp.content, model_response=resp, trajectory=trajectory)

    async def close(self) -> None:
        await self._client.close()
