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
"""Gym protocol: response parsing and NeMoGymResponse envelope construction.

Shared by GymEnvironment (native protocol mode) and the evaluator server
(gym-compat mode).
"""

from __future__ import annotations

from typing import Any


def extract_assistant_text(response: Any) -> str:
    """Extract plain text from various response formats.

    Handles: plain strings, NeMoGymResponse (output[].content[].text),
    and OpenAI ChatCompletion (choices[].message.content).
    """
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        texts: list[str] = []
        for item in response.get("output", []):
            if item.get("type") == "message" and item.get("role") == "assistant":
                content = item.get("content", [])
                if isinstance(content, list):
                    texts.extend(c["text"] for c in content if isinstance(c, dict) and "text" in c)
                elif isinstance(content, str):
                    texts.append(content)
        if texts:
            return "\n".join(texts).strip()
        for ch in response.get("choices", []):
            msg = ch.get("message", {}).get("content")
            if msg:
                return msg
    return str(response)


def wrap_text_as_gym_response(text: str) -> dict[str, Any]:
    """Wrap plain text into a NeMoGymResponse-compatible dict.

    Native Gym resource servers validate ``body.response`` against
    ``NeMoGymResponse`` (an OpenAI ``Response`` subclass).  The
    ``output_text`` property concatenates text from output message items.
    Also set as a top-level key for servers that read it directly.
    """
    return {
        "id": "eval-synthetic",
        "object": "response",
        "created_at": 0,
        "status": "completed",
        "model": "evaluator",
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "output": [
            {
                "id": "msg-eval-synthetic",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": text,
                        "annotations": [],
                    }
                ],
            }
        ],
        "output_text": text,
    }


def wrap_text_as_responses_create_params(
    prompt: str,
    model: str = "evaluator",
) -> dict[str, Any]:
    """Build a minimal NeMoGymResponseCreateParamsNonStreaming-compatible dict."""
    return {
        "input": [{"role": "user", "content": prompt}],
        "model": model,
    }


def extract_prompt_from_rcp(rcp: dict[str, Any]) -> str:
    """Extract the user prompt from a responses_create_params dict."""
    inp = rcp.get("input", "")
    if isinstance(inp, str):
        return inp
    if isinstance(inp, list):
        for msg in reversed(inp):
            if isinstance(msg, dict) and msg.get("role") == "user":
                content = msg.get("content", "")
                return content if isinstance(content, str) else str(content)
    return ""


def _as_text(content: Any) -> str:
    """Coerce a Responses/chat ``content`` value to a plain string."""
    return content if isinstance(content, str) else str(content) if content is not None else ""


def messages_from_rcp(rcp: dict[str, Any]) -> list[dict[str, Any]]:
    """Build a chat-completions messages list from responses_create_params.input.

    Tool-call structure is preserved so tool-use trajectories survive:

    * role-based messages keep any ``tool_calls`` / ``tool_call_id`` / ``name``
      fields (an assistant tool-call turn or a ``role: "tool"`` result stays
      intact instead of collapsing to an empty ``content``);
    * Responses-API typed items are translated to the equivalent messages —
      ``function_call`` -> an assistant message with ``tool_calls``,
      ``function_call_output`` -> a ``role: "tool"`` message whose ``content``
      carries the tool output (which otherwise rode in ``output``, not
      ``content``, and was silently dropped).
    """
    inp = rcp.get("input", [])
    if not isinstance(inp, list):
        return []
    messages: list[dict[str, Any]] = []
    for m in inp:
        if not isinstance(m, dict):
            continue
        item_type = m.get("type")
        if item_type == "function_call":
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": m.get("call_id", ""),
                            "type": "function",
                            "function": {"name": m.get("name", ""), "arguments": _as_text(m.get("arguments", ""))},
                        }
                    ],
                }
            )
            continue
        if item_type == "function_call_output":
            messages.append({"role": "tool", "tool_call_id": m.get("call_id", ""), "content": _as_text(m.get("output"))})
            continue
        msg: dict[str, Any] = {"role": m.get("role", "user"), "content": _as_text(m.get("content", ""))}
        for key in ("tool_calls", "tool_call_id", "name"):
            if m.get(key) is not None:
                msg[key] = m[key]
        messages.append(msg)
    return messages
