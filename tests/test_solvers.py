"""Tests for solver implementations with mocked dependencies."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

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
        from nemo_evaluator.solvers.nat import NatSolver

        solver = NatSolver(nat_url="http://fake-nat:8000")

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"response": "Fixed the bug", "trajectory": []}
            mock_resp.raise_for_status = MagicMock()
            mock_post.return_value = mock_resp

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
        from nemo_evaluator.solvers.openclaw import OpenClawSolver
        import inspect

        sig = inspect.signature(OpenClawSolver.solve)
        assert "sandbox" in sig.parameters
