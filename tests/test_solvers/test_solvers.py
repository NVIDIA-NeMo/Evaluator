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
"""Tests for solver implementations with mocked dependencies."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from nemo_evaluator.environments.base import SeedResult
from nemo_evaluator.observability.types import ModelResponse
from nemo_evaluator.solvers.base import SolveResult


def _make_seed(prompt: str = "What is 2+2?", expected: str = "4") -> SeedResult:
    return SeedResult(
        prompt=prompt,
        expected_answer=expected,
        messages=[{"role": "user", "content": prompt}],
    )


def _make_model_response(content: str = "The answer is 4.") -> ModelResponse:
    return ModelResponse(
        content=content,
        model="test-model",
        total_tokens=50,
        prompt_tokens=20,
        completion_tokens=30,
    )


# ---------------------------------------------------------------------------
# ChatSolver
# ---------------------------------------------------------------------------


class TestChatSolver:
    @pytest.mark.asyncio
    async def test_basic_solve(self):
        from nemo_evaluator.solvers.chat import ChatSolver

        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(return_value=_make_model_response("Answer: B"))

        solver = ChatSolver(client=mock_client)
        result = await solver.solve(_make_seed())

        assert isinstance(result, SolveResult)
        assert result.response == "Answer: B"
        mock_client.chat.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_system_prompt_passed(self):
        from nemo_evaluator.solvers.chat import ChatSolver

        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(return_value=_make_model_response("42"))

        seed = _make_seed()
        seed.system = "You are a math tutor."

        solver = ChatSolver(client=mock_client, system_prompt="Override system")
        await solver.solve(seed)

        assert mock_client.chat.call_args is not None

    @pytest.mark.asyncio
    async def test_empty_response_handled(self):
        from nemo_evaluator.solvers.chat import ChatSolver

        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(return_value=_make_model_response(""))

        solver = ChatSolver(client=mock_client)
        result = await solver.solve(_make_seed())
        assert result.response == ""


# ---------------------------------------------------------------------------
# NatSolver
# ---------------------------------------------------------------------------


class TestNatSolver:
    @pytest.mark.asyncio
    async def test_solve_calls_nat_endpoint(self):
        from contextlib import asynccontextmanager

        from nemo_evaluator.solvers.nat import NatSolver

        solver = NatSolver(nat_url="http://fake-nat:8000")

        class _FakeResp:
            def raise_for_status(self):
                pass

            async def text(self):
                return 'data: {"output": "Fixed the bug"}\n\n'

        class _FakeSession:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                pass

            @asynccontextmanager
            async def post(self, url, **kw):
                yield _FakeResp()

            @asynccontextmanager
            async def get(self, url, **kw):
                class _Health:
                    def raise_for_status(self):
                        pass

                yield _Health()

        with patch("aiohttp.ClientSession", return_value=_FakeSession()):
            result = await solver.solve(_make_seed("Fix the bug"))
            assert isinstance(result, SolveResult)


# ---------------------------------------------------------------------------
# OpenClawSolver
# ---------------------------------------------------------------------------


class TestOpenClawSolver:
    def test_init_with_skip_preflight(self):
        """OpenClawSolver can be created with skip_preflight=True."""
        from nemo_evaluator.solvers.openclaw import OpenClawSolver

        solver = OpenClawSolver(
            model_url="http://localhost:8000/v1",
            model_id="test-model",
            skip_preflight=True,
        )
        assert solver is not None

    @pytest.mark.asyncio
    async def test_solve_signature_accepts_sandbox(self):
        import inspect

        from nemo_evaluator.solvers.openclaw import OpenClawSolver

        sig = inspect.signature(OpenClawSolver.solve)
        assert "sandbox" in sig.parameters
