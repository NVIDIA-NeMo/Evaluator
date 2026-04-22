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
"""Tests for the NEL-driven ReActSolver and tool backends."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nemo_evaluator.environments.base import SeedResult
from nemo_evaluator.observability.types import ModelResponse
from nemo_evaluator.engine.model_client import ToolCallInfo, ToolCallingResponse
from nemo_evaluator.solvers.base import SolveResult
from nemo_evaluator.solvers.tool_backend import (
    CompositeToolBackend,
    SandboxToolBackend,
    ToolInfraError,
    ToolResult,
    _openapi_to_tools,
)


# ── Fixtures ──────────────────────────────────────────────────────────


def _make_seed(prompt: str = "Fix the bug.", expected: str = "") -> SeedResult:
    return SeedResult(prompt=prompt, expected_answer=expected)


def _make_tool_calling_response(
    content: str = "",
    tool_calls: list[ToolCallInfo] | None = None,
    finish_reason: str = "stop",
) -> ToolCallingResponse:
    return ToolCallingResponse(
        content=content,
        tool_calls=tool_calls or [],
        finish_reason=finish_reason,
        model_response=ModelResponse(
            content=content,
            model="test-model",
            finish_reason=finish_reason,
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            latency_ms=42.0,
        ),
    )


def _make_tool_call(
    name: str = "bash",
    arguments: dict[str, Any] | None = None,
    tc_id: str = "tc_1",
) -> ToolCallInfo:
    return ToolCallInfo(id=tc_id, name=name, arguments=arguments or {"command": "ls"})


# ── ToolResult ────────────────────────────────────────────────────────


class TestToolResult:
    def test_default_not_error(self):
        r = ToolResult(content="ok")
        assert not r.is_error
        assert r.content == "ok"
        assert r.metadata == {}

    def test_error_result(self):
        r = ToolResult(content="boom", is_error=True, metadata={"exit_code": 1})
        assert r.is_error
        assert r.metadata["exit_code"] == 1


# ── OpenAPI to tools conversion ───────────────────────────────────────


class TestOpenApiToTools:
    def test_converts_post_endpoints(self):
        spec = {
            "paths": {
                "/web_search": {
                    "post": {
                        "summary": "Search the web",
                        "requestBody": {
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "query": {"type": "string"},
                                        },
                                        "required": ["query"],
                                    },
                                },
                            },
                        },
                    },
                },
                "/health": {"get": {"summary": "Health check"}},
            },
        }
        tools = _openapi_to_tools(spec)
        assert len(tools) == 1
        assert tools[0]["function"]["name"] == "web_search"
        assert tools[0]["function"]["parameters"]["properties"]["query"]["type"] == "string"

    def test_skips_internal_endpoints(self):
        spec = {
            "paths": {
                "/seed_session": {"post": {"summary": "Seed"}},
                "/verify": {"post": {"summary": "Verify"}},
                "/run": {"post": {"summary": "Run"}},
                "/real_tool": {"post": {"summary": "A tool"}},
            },
        }
        tools = _openapi_to_tools(spec)
        assert len(tools) == 1
        assert tools[0]["function"]["name"] == "real_tool"

    def test_resolves_refs(self):
        spec = {
            "paths": {
                "/search": {
                    "post": {
                        "summary": "Search",
                        "requestBody": {
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/SearchInput"},
                                },
                            },
                        },
                    },
                },
            },
            "components": {
                "schemas": {
                    "SearchInput": {
                        "type": "object",
                        "properties": {"q": {"type": "string"}},
                        "required": ["q"],
                    },
                },
            },
        }
        tools = _openapi_to_tools(spec)
        assert tools[0]["function"]["parameters"]["properties"]["q"]["type"] == "string"


# ── SandboxToolBackend ────────────────────────────────────────────────


class TestSandboxToolBackend:
    def _make_sandbox(self, exec_result=None):
        sandbox = AsyncMock()
        spec = MagicMock()
        spec.workdir = "/workspace"
        sandbox.spec = spec

        if exec_result is None:
            exec_result = MagicMock()
            exec_result.stdout = "hello\n__NEL_CWD__=/workspace"
            exec_result.stderr = ""
            exec_result.return_code = 0
        sandbox.exec = AsyncMock(return_value=exec_result)
        sandbox.upload = AsyncMock()
        sandbox.download = AsyncMock()
        return sandbox

    @pytest.mark.asyncio
    async def test_list_tools_returns_canonical_set(self):
        sb = self._make_sandbox()
        backend = SandboxToolBackend(sb)
        tools = await backend.list_tools()
        names = {t["function"]["name"] for t in tools}
        assert names == {"bash", "file_read", "file_write", "str_replace_editor"}
        await backend.close()

    @pytest.mark.asyncio
    async def test_bash_returns_stdout(self):
        sb = self._make_sandbox()
        backend = SandboxToolBackend(sb)
        result = await backend.call_tool("bash", {"command": "echo hello"})
        assert not result.is_error
        assert "hello" in result.content
        await backend.close()

    @pytest.mark.asyncio
    async def test_bash_error_sets_is_error(self):
        exec_result = MagicMock()
        exec_result.stdout = "not found\n__NEL_CWD__=/workspace"
        exec_result.stderr = "error"
        exec_result.return_code = 1
        sb = self._make_sandbox(exec_result)

        backend = SandboxToolBackend(sb)
        result = await backend.call_tool("bash", {"command": "bad_cmd"})
        assert result.is_error
        assert result.metadata["exit_code"] == 1
        assert "exit code 1" in result.content
        await backend.close()

    @pytest.mark.asyncio
    async def test_bash_updates_cwd(self):
        exec_result = MagicMock()
        exec_result.stdout = "ok\n__NEL_CWD__=/home/user"
        exec_result.stderr = ""
        exec_result.return_code = 0
        sb = self._make_sandbox(exec_result)

        backend = SandboxToolBackend(sb)
        await backend.call_tool("bash", {"command": "cd /home/user"})
        assert backend._cwd == "/home/user"
        await backend.close()

    @pytest.mark.asyncio
    async def test_file_write_uses_upload(self):
        sb = self._make_sandbox()
        backend = SandboxToolBackend(sb)
        result = await backend.call_tool(
            "file_write",
            {
                "path": "/workspace/test.py",
                "content": "print('hi')",
            },
        )
        assert not result.is_error
        assert "Written" in result.content
        sb.upload.assert_awaited_once()
        await backend.close()

    @pytest.mark.asyncio
    async def test_file_read_uses_download(self):
        sb = self._make_sandbox()

        async def _download(remote, local):
            local.write_text("file content")

        sb.download = AsyncMock(side_effect=_download)
        backend = SandboxToolBackend(sb)
        result = await backend.call_tool("file_read", {"path": "/workspace/test.py"})
        assert not result.is_error
        assert result.content == "file content"
        await backend.close()

    @pytest.mark.asyncio
    async def test_str_replace_editor_view(self):
        sb = self._make_sandbox()

        async def _download(remote, local):
            local.write_text("line1\nline2\nline3\n")

        sb.download = AsyncMock(side_effect=_download)
        backend = SandboxToolBackend(sb)
        result = await backend.call_tool(
            "str_replace_editor",
            {
                "command": "view",
                "path": "/workspace/test.py",
            },
        )
        assert not result.is_error
        assert "line2" in result.content
        await backend.close()

    @pytest.mark.asyncio
    async def test_str_replace_editor_replace(self):
        sb = self._make_sandbox()

        async def _download(remote, local):
            local.write_text("old_value = 1")

        sb.download = AsyncMock(side_effect=_download)
        backend = SandboxToolBackend(sb)
        result = await backend.call_tool(
            "str_replace_editor",
            {
                "command": "str_replace",
                "path": "/workspace/test.py",
                "old_str": "old_value",
                "new_str": "new_value",
            },
        )
        assert not result.is_error
        await backend.close()

    @pytest.mark.asyncio
    async def test_unknown_tool_raises(self):
        sb = self._make_sandbox()
        backend = SandboxToolBackend(sb)
        with pytest.raises(ToolInfraError, match="Unknown sandbox tool"):
            await backend.call_tool("nonexistent", {})
        await backend.close()

    @pytest.mark.asyncio
    async def test_sandbox_exec_failure_raises_infra_error(self):
        sb = self._make_sandbox()
        sb.exec = AsyncMock(side_effect=RuntimeError("container crashed"))
        backend = SandboxToolBackend(sb)
        with pytest.raises(ToolInfraError, match="Sandbox exec failed"):
            await backend.call_tool("bash", {"command": "ls"})
        await backend.close()


# ── CompositeToolBackend ──────────────────────────────────────────────


class TestCompositeToolBackend:
    @pytest.mark.asyncio
    async def test_merges_tools_and_routes(self):
        b1 = AsyncMock()
        b1.list_tools = AsyncMock(
            return_value=[
                {"type": "function", "function": {"name": "web_search"}},
            ]
        )
        b1.call_tool = AsyncMock(return_value=ToolResult(content="result1"))

        b2 = AsyncMock()
        b2.list_tools = AsyncMock(
            return_value=[
                {"type": "function", "function": {"name": "bash"}},
            ]
        )
        b2.call_tool = AsyncMock(return_value=ToolResult(content="result2"))

        comp = CompositeToolBackend([b1, b2])
        tools = await comp.list_tools()
        assert len(tools) == 2

        r1 = await comp.call_tool("web_search", {"q": "test"})
        assert r1.content == "result1"
        b1.call_tool.assert_awaited_once()

        r2 = await comp.call_tool("bash", {"command": "ls"})
        assert r2.content == "result2"
        b2.call_tool.assert_awaited_once()

        await comp.close()

    @pytest.mark.asyncio
    async def test_unknown_tool_raises(self):
        b1 = AsyncMock()
        b1.list_tools = AsyncMock(
            return_value=[
                {"type": "function", "function": {"name": "tool1"}},
            ]
        )
        comp = CompositeToolBackend([b1])
        await comp.list_tools()
        with pytest.raises(ToolInfraError, match="not found"):
            await comp.call_tool("nonexistent", {})


# ── ReActSolver ───────────────────────────────────────────────────────


class TestReActSolver:
    def _make_solver(self, **kwargs):
        from nemo_evaluator.solvers.react import ReActSolver

        client = AsyncMock()
        client.model = "test-model"
        defaults = {
            "client": client,
            "http_backend": None,
            "use_sandbox_tools": False,
            "max_turns": 10,
        }
        defaults.update(kwargs)
        return ReActSolver(**defaults), defaults["client"]

    def _mock_backend(self, tools=None, call_result=None):
        backend = AsyncMock()
        backend.list_tools = AsyncMock(
            return_value=tools
            or [
                {"type": "function", "function": {"name": "bash"}},
            ]
        )
        backend.call_tool = AsyncMock(return_value=call_result or ToolResult(content="ok"))
        backend.close = AsyncMock()
        return backend

    @pytest.mark.asyncio
    async def test_single_turn_no_tools(self):
        solver, client = self._make_solver()
        client.chat_with_tools = AsyncMock(return_value=_make_tool_calling_response(content="The answer is 42."))
        backend = self._mock_backend()
        with patch.object(solver, "_build_backend", return_value=backend):
            result = await solver.solve(_make_seed("What is 6*7?"))

        assert isinstance(result, SolveResult)
        assert result.response == "The answer is 42."
        assert result.error is None
        assert result.trajectory

    @pytest.mark.asyncio
    async def test_multi_turn_tool_use(self):
        solver, client = self._make_solver()

        turn1 = _make_tool_calling_response(
            content="Let me check...",
            tool_calls=[_make_tool_call("bash", {"command": "ls"}, "tc_1")],
            finish_reason="tool_calls",
        )
        turn2 = _make_tool_calling_response(
            content="I found the file. The bug is fixed.",
            finish_reason="stop",
        )
        client.chat_with_tools = AsyncMock(side_effect=[turn1, turn2])

        backend = self._mock_backend(call_result=ToolResult(content="README.md\nsetup.py"))
        with patch.object(solver, "_build_backend", return_value=backend):
            result = await solver.solve(_make_seed())

        assert result.response == "I found the file. The bug is fixed."
        assert result.error is None
        assert client.chat_with_tools.await_count == 2
        backend.call_tool.assert_awaited_once_with("bash", {"command": "ls"})

    @pytest.mark.asyncio
    async def test_max_turns_exhausted(self):
        solver, client = self._make_solver(max_turns=2)

        always_tool = _make_tool_calling_response(
            content="",
            tool_calls=[_make_tool_call("bash", {"command": "ls"})],
            finish_reason="tool_calls",
        )
        client.chat_with_tools = AsyncMock(return_value=always_tool)

        backend = self._mock_backend(call_result=ToolResult(content="files"))
        with patch.object(solver, "_build_backend", return_value=backend):
            result = await solver.solve(_make_seed())

        assert result.error.startswith("max_turns_exhausted")

    @pytest.mark.asyncio
    async def test_tool_error_fed_back_to_model(self):
        solver, client = self._make_solver()

        turn1 = _make_tool_calling_response(
            content="",
            tool_calls=[_make_tool_call("bash", {"command": "bad"})],
            finish_reason="tool_calls",
        )
        turn2 = _make_tool_calling_response(
            content="Oops, that failed. Let me try another approach.",
            finish_reason="stop",
        )
        client.chat_with_tools = AsyncMock(side_effect=[turn1, turn2])

        backend = self._mock_backend(
            call_result=ToolResult(content="command not found", is_error=True),
        )
        with patch.object(solver, "_build_backend", return_value=backend):
            result = await solver.solve(_make_seed())

        assert result.response == "Oops, that failed. Let me try another approach."
        assert result.error is None

    @pytest.mark.asyncio
    async def test_infra_error_aborts_loop(self):
        solver, client = self._make_solver()

        turn1 = _make_tool_calling_response(
            content="",
            tool_calls=[_make_tool_call("bash", {"command": "ls"})],
            finish_reason="tool_calls",
        )
        client.chat_with_tools = AsyncMock(return_value=turn1)

        backend = self._mock_backend()
        backend.call_tool = AsyncMock(side_effect=ToolInfraError("container died"))
        with patch.object(solver, "_build_backend", return_value=backend):
            result = await solver.solve(_make_seed())

        assert result.error is not None
        assert "container died" in result.error

    @pytest.mark.asyncio
    async def test_sandbox_artifact_response_mode(self):
        solver, client = self._make_solver(response_mode="sandbox_artifact")

        turn1 = _make_tool_calling_response(
            content="",
            tool_calls=[_make_tool_call("bash", {"command": "git diff"})],
            finish_reason="tool_calls",
        )
        turn2 = _make_tool_calling_response(
            content="Done. The fix has been applied.",
            finish_reason="stop",
        )
        client.chat_with_tools = AsyncMock(side_effect=[turn1, turn2])

        backend = self._mock_backend(call_result=ToolResult(content="patch output"))
        with patch.object(solver, "_build_backend", return_value=backend):
            result = await solver.solve(_make_seed())

        assert result.response == ""
        assert result.error is None

    @pytest.mark.asyncio
    async def test_system_prompt_included(self):
        solver, client = self._make_solver(system_prompt="You are a coding agent.")

        client.chat_with_tools = AsyncMock(
            return_value=_make_tool_calling_response(content="ok"),
        )
        backend = self._mock_backend()
        with patch.object(solver, "_build_backend", return_value=backend):
            await solver.solve(_make_seed())

        call_args = client.chat_with_tools.call_args
        messages = call_args[0][0]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a coding agent."

    @pytest.mark.asyncio
    async def test_seed_metadata_tools_used(self):
        solver, client = self._make_solver()
        client.chat_with_tools = AsyncMock(
            return_value=_make_tool_calling_response(content="done"),
        )

        seed_tools = [{"type": "function", "function": {"name": "custom_tool"}}]
        seed = _make_seed()
        seed.metadata = {"tools": seed_tools}

        backend = self._mock_backend()
        with patch.object(solver, "_build_backend", return_value=backend):
            await solver.solve(seed)

        call_args = client.chat_with_tools.call_args
        tools_passed = call_args[0][1]
        assert tools_passed == seed_tools
        backend.list_tools.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_atif_trajectory_structure(self):
        solver, client = self._make_solver()

        turn1 = _make_tool_calling_response(
            content="Let me look...",
            tool_calls=[_make_tool_call("bash", {"command": "ls"}, "tc_1")],
            finish_reason="tool_calls",
        )
        turn2 = _make_tool_calling_response(
            content="Found it.",
            finish_reason="stop",
        )
        client.chat_with_tools = AsyncMock(side_effect=[turn1, turn2])

        backend = self._mock_backend(call_result=ToolResult(content="file.py"))
        with patch.object(solver, "_build_backend", return_value=backend):
            result = await solver.solve(_make_seed())

        assert len(result.trajectory) == 1
        doc = result.trajectory[0]
        assert doc["schema_version"] == "ATIF-v1.6"
        assert doc["agent"]["name"] == "nemo-evaluator-react"

        steps = doc["steps"]
        # user + agent(tool_call) + system(observation) + agent(final) = 4
        assert len(steps) == 4
        assert steps[0]["source"] == "user"
        assert steps[0]["message"] == "Fix the bug."
        assert steps[1]["source"] == "agent"
        assert steps[1]["tool_calls"][0]["function_name"] == "bash"
        assert steps[2]["source"] == "system"
        assert "file.py" in steps[2]["message"]
        assert steps[3]["source"] == "agent"
        assert steps[3]["message"] == "Found it."

        # Per-step metrics on agent steps (ATIF spec requirement)
        for step in steps:
            if step["source"] == "agent":
                assert "metrics" in step, f"agent step missing metrics: {step}"
                assert "prompt_tokens" in step["metrics"]
                assert "completion_tokens" in step["metrics"]

        # final_metrics must always be present and include total_steps
        fm = doc["final_metrics"]
        assert fm["total_steps"] == 4
        assert fm["total_prompt_tokens"] == 200  # 2 turns x 100 prompt tokens
        assert fm["total_completion_tokens"] == 100  # 2 turns x 50 completion tokens

    @pytest.mark.asyncio
    async def test_output_truncation(self):
        solver, client = self._make_solver(max_output_chars=100)

        turn1 = _make_tool_calling_response(
            content="",
            tool_calls=[_make_tool_call("bash", {"command": "cat big_file"})],
            finish_reason="tool_calls",
        )
        turn2 = _make_tool_calling_response(content="done", finish_reason="stop")
        client.chat_with_tools = AsyncMock(side_effect=[turn1, turn2])

        big_output = "x" * 1000
        backend = self._mock_backend(call_result=ToolResult(content=big_output))
        with patch.object(solver, "_build_backend", return_value=backend):
            result = await solver.solve(_make_seed())

        doc = result.trajectory[0]
        # steps: user(0), agent(1), system/observation(2), agent(3)
        observation_step = doc["steps"][2]
        assert "truncated" in observation_step["message"]
        assert len(observation_step["message"]) < len(big_output)


# ── Config validation tests ───────────────────────────────────────────


class TestToolCallingSolverConfig:
    def test_requires_at_least_one_tool_source(self):
        from nemo_evaluator.config import ToolCallingSolverConfig

        with pytest.raises(Exception, match="at least one tool source"):
            ToolCallingSolverConfig(
                type="tool_calling",
                service="model",
                resource_service=None,
                sandbox_tools=False,
            )

    def test_resource_service_only(self):
        from nemo_evaluator.config import ToolCallingSolverConfig

        cfg = ToolCallingSolverConfig(
            type="tool_calling",
            service="model",
            resource_service="tools",
        )
        assert cfg.resource_service == "tools"
        assert not cfg.sandbox_tools

    def test_sandbox_tools_only(self):
        from nemo_evaluator.config import ToolCallingSolverConfig

        cfg = ToolCallingSolverConfig(
            type="tool_calling",
            service="model",
            sandbox_tools=True,
        )
        assert cfg.sandbox_tools
        assert cfg.resource_service is None

    def test_both_tools_ok(self):
        from nemo_evaluator.config import ToolCallingSolverConfig

        cfg = ToolCallingSolverConfig(
            type="tool_calling",
            service="model",
            resource_service="tools",
            sandbox_tools=True,
        )
        assert cfg.resource_service == "tools"
        assert cfg.sandbox_tools

    def test_sandbox_tools_requires_sandbox(self):
        from nemo_evaluator.config import (
            BenchmarkConfig,
            EvalConfig,
            ExternalApiService,
            ToolCallingSolverConfig,
        )

        with pytest.raises(Exception, match="requires a sandbox"):
            EvalConfig(
                services={
                    "model": ExternalApiService(
                        type="api",
                        url="http://x/v1/chat/completions",
                        protocol="chat_completions",
                    ),
                },
                benchmarks=[
                    BenchmarkConfig(
                        name="swebench",
                        solver=ToolCallingSolverConfig(
                            type="tool_calling",
                            service="model",
                            sandbox_tools=True,
                        ),
                    ),
                ],
            )

    def test_resource_service_validated(self):
        from nemo_evaluator.config import (
            BenchmarkConfig,
            EvalConfig,
            ExternalApiService,
            ToolCallingSolverConfig,
        )

        with pytest.raises(Exception, match="not in services"):
            EvalConfig(
                services={
                    "model": ExternalApiService(
                        type="api",
                        url="http://x/v1/chat/completions",
                        protocol="chat_completions",
                    ),
                },
                benchmarks=[
                    BenchmarkConfig(
                        name="test",
                        solver=ToolCallingSolverConfig(
                            type="tool_calling",
                            service="model",
                            resource_service="nonexistent",
                        ),
                    ),
                ],
            )

    def test_defaults(self):
        from nemo_evaluator.config import ToolCallingSolverConfig

        cfg = ToolCallingSolverConfig(
            type="tool_calling",
            service="model",
            sandbox_tools=True,
        )
        assert cfg.max_turns == 50
        assert cfg.tool_timeout == 180.0
        assert cfg.max_output_chars == 16384
        assert cfg.response_mode == "last_message"


# ── ModelClient.chat_with_tools tests ─────────────────────────────────


class TestChatWithTools:
    @pytest.mark.asyncio
    async def test_parses_tool_calls(self):
        from nemo_evaluator.engine.model_client import ModelClient

        api_response = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "bash",
                                    "arguments": '{"command": "ls"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "model": "test-model",
            "usage": {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70},
        }

        client = ModelClient(base_url="http://fake:8000/v1", model="test-model")
        with patch.object(client, "_post_with_retry", new_callable=AsyncMock, return_value=api_response):
            result = await client.chat_with_tools(
                messages=[{"role": "user", "content": "list files"}],
                tools=[],
            )

        assert len(result.tool_calls) == 1
        tc = result.tool_calls[0]
        assert tc.id == "call_1"
        assert tc.name == "bash"
        assert tc.arguments == {"command": "ls"}
        assert result.finish_reason == "tool_calls"
        assert result.model_response.prompt_tokens == 50

    @pytest.mark.asyncio
    async def test_no_tool_calls(self):
        from nemo_evaluator.engine.model_client import ModelClient

        api_response = {
            "choices": [
                {
                    "message": {"content": "The answer is 42."},
                    "finish_reason": "stop",
                }
            ],
            "model": "test-model",
            "usage": {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60},
        }

        client = ModelClient(base_url="http://fake:8000/v1", model="test-model")
        with patch.object(client, "_post_with_retry", new_callable=AsyncMock, return_value=api_response):
            result = await client.chat_with_tools(
                messages=[{"role": "user", "content": "What is 6*7?"}],
                tools=[],
            )

        assert result.content == "The answer is 42."
        assert result.tool_calls == []
        assert result.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_malformed_arguments_handled(self):
        from nemo_evaluator.engine.model_client import ModelClient

        api_response = {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {
                                    "name": "bash",
                                    "arguments": "not valid json {{{",
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "model": "test-model",
            "usage": {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60},
        }

        client = ModelClient(base_url="http://fake:8000/v1", model="test-model")
        with patch.object(client, "_post_with_retry", new_callable=AsyncMock, return_value=api_response):
            result = await client.chat_with_tools(
                messages=[{"role": "user", "content": "test"}],
                tools=[],
            )

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].arguments == {"raw": "not valid json {{{"}
