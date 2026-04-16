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
"""Shared ATIF trajectory helpers for all solvers.

Produces trajectories conforming to Harbor ATIF-v1.6
(Agent Trajectory Interchange Format, RFC 0001).
"""

from __future__ import annotations

import uuid
from typing import Any


def build_atif_trajectory(
    steps: list[dict[str, Any]],
    *,
    agent_name: str = "nemo-evaluator",
    agent_version: str = "0.1.0",
    model_name: str | None = None,
    session_id: str | None = None,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    status: str | None = None,
    extra: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build a single-element list containing one ATIF-v1.6 trajectory document.

    Returns ``[atif_doc]`` — the list wrapper that ``SolveResult.trajectory``
    expects.  Each step is auto-numbered (``step_id`` 1, 2, 3, ...) and must
    have at minimum ``source`` and ``message``.
    """
    numbered_steps = []
    for i, step in enumerate(steps, start=1):
        s = dict(step)
        s.setdefault("step_id", i)
        numbered_steps.append(s)

    agent_obj: dict[str, Any] = {
        "name": agent_name,
        "version": agent_version,
    }
    if model_name:
        agent_obj["model_name"] = model_name

    doc: dict[str, Any] = {
        "schema_version": "ATIF-v1.6",
        "session_id": session_id or uuid.uuid4().hex[:16],
        "agent": agent_obj,
        "steps": numbered_steps,
    }

    if prompt_tokens or completion_tokens:
        doc["final_metrics"] = {
            "total_prompt_tokens": prompt_tokens,
            "total_completion_tokens": completion_tokens,
        }

    if status:
        doc.setdefault("extra", {})["status"] = status

    if extra:
        doc.setdefault("extra", {}).update(extra)

    return [doc]


def build_single_turn_atif(
    prompt: str,
    response: str,
    *,
    system: str | None = None,
    model_name: str | None = None,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> list[dict[str, Any]]:
    """Build an ATIF trajectory for a single-turn model call (chat/completion/VLM).

    Produces 2-3 steps: optional system, user prompt, and agent response.
    The agent response ``message`` is the model output (may be empty string
    for agent benchmarks where the result is file modifications).
    """
    steps: list[dict[str, Any]] = []
    if system:
        steps.append({"source": "system", "message": system})
    steps.append({"source": "user", "message": prompt})
    steps.append(
        {
            "source": "agent",
            "message": response,
            "model_name": model_name,
        }
    )
    return build_atif_trajectory(
        steps,
        model_name=model_name,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
