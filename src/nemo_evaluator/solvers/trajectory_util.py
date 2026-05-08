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
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    status: str | None = None,
    extra: dict[str, Any] | None = None,
    user_prompt: str | None = None,
    system_prompt: str | None = None,
) -> list[dict[str, Any]]:
    """Build a single-element list containing one ATIF-v1.6 trajectory document.

    Steps are auto-numbered (``step_id`` 1, 2, 3, ...). When ``user_prompt``
    is supplied and ``steps`` does not already begin with a user/system
    step, an initiating user step (optionally preceded by ``system_prompt``)
    is prepended. Idempotent when the caller already seeds a user turn.
    """
    normalized = [dict(s) for s in steps]
    if user_prompt:
        leading_source = normalized[0].get("source") if normalized else None
        if leading_source not in ("user", "system"):
            prelude: list[dict[str, Any]] = []
            if system_prompt:
                prelude.append({"source": "system", "message": system_prompt})
            prelude.append({"source": "user", "message": user_prompt})
            normalized = prelude + normalized

    numbered_steps = []
    for i, step in enumerate(normalized, start=1):
        s = dict(step)
        s["step_id"] = i
        numbered_steps.append(s)

    if prompt_tokens is None and completion_tokens is None:
        prompt_tokens = 0
        completion_tokens = 0
        for s in numbered_steps:
            m = s.get("metrics") or {}
            prompt_tokens += m.get("prompt_tokens", 0)
            completion_tokens += m.get("completion_tokens", 0)
    else:
        prompt_tokens = prompt_tokens or 0
        completion_tokens = completion_tokens or 0

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

    doc["final_metrics"] = {
        "total_prompt_tokens": prompt_tokens,
        "total_completion_tokens": completion_tokens,
        "total_steps": len(numbered_steps),
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
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    system: str | None = None,
    model_name: str | None = None,
) -> list[dict[str, Any]]:
    """Build an ATIF trajectory for a single-turn model call."""
    steps: list[dict[str, Any]] = []
    if system:
        steps.append({"source": "system", "message": system})
    steps.append({"source": "user", "message": prompt})
    agent_step: dict[str, Any] = {
        "source": "agent",
        "message": response,
        "model_name": model_name,
    }
    if prompt_tokens is not None and completion_tokens is not None:
        agent_step["metrics"] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }
    steps.append(agent_step)
    return build_atif_trajectory(
        steps,
        model_name=model_name,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
