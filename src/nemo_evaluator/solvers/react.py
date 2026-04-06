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
"""ReActSolver: NEL-driven agent loop with pluggable tool backends.

NEL drives the model-call → parse-tool-calls → dispatch cycle directly,
giving full observability over every turn.  Works with Gym Resource Server
tools (``HttpToolBackend``), sandbox tools (``SandboxToolBackend``), or
both (``CompositeToolBackend``).
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from nemo_evaluator.environments.base import SeedResult
from nemo_evaluator.errors import GracefulError
from nemo_evaluator.observability.types import ModelResponse
from nemo_evaluator.engine.model_client import ModelClient, ToolCallInfo, ToolCallingResponse
from nemo_evaluator.solvers.base import SolveResult
from nemo_evaluator.solvers.tool_backend import (
    CompositeToolBackend,
    HttpToolBackend,
    SandboxToolBackend,
    ToolBackend,
    ToolInfraError,
    ToolResult,
)
from nemo_evaluator.solvers.trajectory_util import build_atif_trajectory

if TYPE_CHECKING:
    from nemo_evaluator.sandbox.base import Sandbox

logger = logging.getLogger(__name__)


class ReActSolver:
    """NEL-native ReAct loop.

    Parameters
    ----------
    client:
        ``ModelClient`` configured for the target LLM.
    http_backend:
        Optional ``HttpToolBackend`` for Gym Resource Server tools.
    use_sandbox_tools:
        Whether to create a ``SandboxToolBackend`` per-problem.
    max_turns:
        Maximum model-call rounds before stopping.
    system_prompt:
        Optional system message prepended to every conversation.
    generation:
        Optional per-solver generation overrides (temperature, max_tokens, etc.).
    tool_timeout:
        Per-tool-call timeout in seconds.
    max_output_chars:
        Truncation limit for tool output fed back to the model.
    response_mode:
        ``"last_message"`` — return last model text (default).
        ``"sandbox_artifact"`` — return ``""``, rely on sandbox ``capture_cmd``.
    """

    def __init__(
        self,
        client: ModelClient,
        http_backend: HttpToolBackend | None = None,
        use_sandbox_tools: bool = False,
        max_turns: int = 50,
        system_prompt: str | None = None,
        generation: Any = None,
        tool_timeout: float = 180.0,
        max_output_chars: int = 16384,
        response_mode: str = "last_message",
    ) -> None:
        self._client = client
        self._http_backend = http_backend
        self._use_sandbox_tools = use_sandbox_tools
        self._max_turns = max_turns
        self._system_prompt = system_prompt
        self._generation = generation
        self._tool_timeout = tool_timeout
        self._max_output_chars = max_output_chars
        self._response_mode = response_mode

    async def solve(
        self,
        task: SeedResult,
        sandbox: Sandbox | None = None,
    ) -> SolveResult:
        t0 = time.monotonic()

        # 1. Build per-problem backend
        backend = self._build_backend(sandbox)

        try:
            atif_steps: list[dict[str, Any]] = []
            total_prompt_tokens = 0
            total_completion_tokens = 0

            # 2. Discover tools (prefer seed metadata, fall back to backend)
            tools = task.metadata.get("tools") if task.metadata else None
            if not tools:
                tools = await backend.list_tools()
            if not tools:
                raise ToolInfraError("No tools available from any backend")

            # 3. Build initial messages
            messages = self._build_initial_messages(task)
            last_response: ModelResponse | None = None
            last_tcr: ToolCallingResponse | None = None

            gen_overrides = self._gen_overrides()

            for turn in range(self._max_turns):
                # 4a. Call model with tools — exceptions propagate to eval loop
                tcr = await self._client.chat_with_tools(messages, tools, **gen_overrides)

                last_response = tcr.model_response
                last_tcr = tcr
                total_prompt_tokens += tcr.model_response.prompt_tokens
                total_completion_tokens += tcr.model_response.completion_tokens

                # 4b. Record ATIF agent step
                atif_steps.append(
                    {
                        "source": "agent",
                        "message": tcr.content or "",
                        "model_name": tcr.model_response.model,
                        "tool_calls": [
                            {
                                "function_name": tc.name,
                                "arguments": tc.arguments,
                                "tool_call_id": tc.id,
                            }
                            for tc in tcr.tool_calls
                        ],
                        "metrics": {
                            "prompt_tokens": tcr.model_response.prompt_tokens,
                            "completion_tokens": tcr.model_response.completion_tokens,
                            "latency_ms": tcr.model_response.latency_ms,
                        },
                    }
                )

                # 4c. No tool calls → model is done
                if not tcr.tool_calls:
                    break

                # 4d. Append assistant message with tool_calls to conversation
                messages.append(self._assistant_msg(tcr))

                # 4e. Dispatch each tool call sequentially (sandbox state is ordered)
                for tc in tcr.tool_calls:
                    result = await self._dispatch_tool(backend, tc)
                    truncated = self._truncate(result.content)

                    # Record ATIF observation
                    atif_steps.append(
                        {
                            "source": "system",
                            "message": truncated,
                            "observation": {
                                "results": [
                                    {
                                        "source_call_id": tc.id,
                                        "content": truncated,
                                        "is_error": result.is_error,
                                    }
                                ],
                            },
                        }
                    )

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": truncated,
                        }
                    )

                # 4f. Context management
                messages = self._manage_context(messages)

            # 5. Extract response
            if last_tcr and last_tcr.tool_calls and turn >= self._max_turns - 1:
                raise GracefulError(f"max_turns_exhausted ({self._max_turns} turns)")

            if self._response_mode == "sandbox_artifact":
                final_text = ""
            else:
                final_text = self._extract_last_text(messages, last_tcr)

            # 6. Build trajectory
            trajectory = build_atif_trajectory(
                atif_steps,
                agent_name="nemo-evaluator-react",
                model_name=last_response.model if last_response else None,
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
            )

            latency_ms = (time.monotonic() - t0) * 1000
            if last_response:
                last_response = ModelResponse(
                    content=final_text,
                    model=last_response.model,
                    finish_reason=last_response.finish_reason,
                    prompt_tokens=total_prompt_tokens,
                    completion_tokens=total_completion_tokens,
                    total_tokens=total_prompt_tokens + total_completion_tokens,
                    latency_ms=round(latency_ms, 2),
                    raw_response={},
                )

            return SolveResult(
                response=final_text,
                model_response=last_response,
                trajectory=trajectory,
            )

        except GracefulError as exc:
            logger.warning("ReActSolver: graceful failure: %s", exc)
            latency_ms = (time.monotonic() - t0) * 1000
            return SolveResult(
                response="",
                model_response=ModelResponse(
                    content="",
                    model=self._client.model,
                    total_tokens=total_prompt_tokens + total_completion_tokens,
                    latency_ms=round(latency_ms, 2),
                ),
                trajectory=build_atif_trajectory(
                    atif_steps + [{"source": "system", "message": str(exc)}],
                    agent_name="nemo-evaluator-react",
                    status="error",
                ),
                error=str(exc),
            )
        finally:
            await backend.close()

    # ── Helpers ────────────────────────────────────────────────────────

    def _build_backend(self, sandbox: Sandbox | None) -> ToolBackend:
        backends: list[ToolBackend] = []
        if self._http_backend:
            backends.append(self._http_backend)
        if sandbox and self._use_sandbox_tools:
            backends.append(SandboxToolBackend(sandbox, timeout=self._tool_timeout))
        if not backends:
            raise ToolInfraError("No tool backends configured")
        if len(backends) == 1:
            return backends[0]
        return CompositeToolBackend(backends)

    def _build_initial_messages(self, task: SeedResult) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        system = self._system_prompt or task.system
        if system:
            messages.append({"role": "system", "content": system})
        if task.messages:
            messages.extend(task.messages)
        else:
            messages.append({"role": "user", "content": task.prompt})
        return messages

    def _gen_overrides(self) -> dict[str, Any]:
        overrides: dict[str, Any] = {}
        if self._generation is not None:
            gen = self._generation
            if getattr(gen, "temperature", None) is not None:
                overrides["temperature"] = gen.temperature
            if getattr(gen, "max_tokens", None) is not None:
                overrides["max_tokens"] = gen.max_tokens
            if getattr(gen, "top_p", None) is not None:
                overrides["top_p"] = gen.top_p
            if getattr(gen, "seed", None) is not None:
                overrides["seed"] = gen.seed
            if getattr(gen, "stop", None) is not None:
                overrides["stop"] = gen.stop
            if getattr(gen, "frequency_penalty", None) is not None:
                overrides["frequency_penalty"] = gen.frequency_penalty
            if getattr(gen, "presence_penalty", None) is not None:
                overrides["presence_penalty"] = gen.presence_penalty
        return overrides

    @staticmethod
    def _assistant_msg(tcr: ToolCallingResponse) -> dict[str, Any]:
        msg: dict[str, Any] = {"role": "assistant"}
        if tcr.content:
            msg["content"] = tcr.content
        if tcr.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": __import__("json").dumps(tc.arguments),
                    },
                }
                for tc in tcr.tool_calls
            ]
        return msg

    async def _dispatch_tool(self, backend: ToolBackend, tc: ToolCallInfo) -> ToolResult:
        try:
            return await backend.call_tool(tc.name, tc.arguments)
        except ToolInfraError:
            raise
        except Exception as exc:
            logger.warning("Tool %s raised %s: %s", tc.name, type(exc).__name__, exc)
            return ToolResult(content=str(exc), is_error=True)

    def _truncate(self, text: str) -> str:
        if len(text) <= self._max_output_chars:
            return text
        half = self._max_output_chars // 2
        return text[:half] + f"\n\n... [truncated {len(text) - self._max_output_chars} chars] ...\n\n" + text[-half:]

    def _manage_context(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Drop middle turns to keep context size manageable.

        Heuristic: estimate ~4 chars/token.  If estimated tokens exceed
        a conservative threshold, drop the oldest middle messages while
        keeping system + first user + last K messages.
        """
        total_chars = sum(len(m.get("content", "")) for m in messages)
        estimated_tokens = total_chars // 4

        threshold = 100_000
        if estimated_tokens <= threshold:
            return messages

        keep_start = 2  # system + first user
        keep_end = 10  # recent messages
        if len(messages) <= keep_start + keep_end:
            return messages

        head = messages[:keep_start]
        tail = messages[-keep_end:]
        dropped = len(messages) - keep_start - keep_end
        head.append(
            {
                "role": "system",
                "content": f"[{dropped} earlier messages omitted to manage context length]",
            }
        )
        return head + tail

    @staticmethod
    def _extract_last_text(
        messages: list[dict[str, Any]],
        last_tcr: ToolCallingResponse | None,
    ) -> str:
        if last_tcr and last_tcr.content:
            return last_tcr.content
        for msg in reversed(messages):
            if msg.get("role") == "assistant" and msg.get("content"):
                return msg["content"]
        return ""

    async def close(self) -> None:
        if self._http_backend:
            await self._http_backend.close()
