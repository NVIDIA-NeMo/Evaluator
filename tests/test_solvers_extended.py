"""Tests for solvers not covered by test_solvers.py — Completion, trajectory_util."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch


from nemo_evaluator.environments.base import SeedResult
from nemo_evaluator.solvers.base import SolveResult


# ── CompletionSolver ─────────────────────────────────────────────────────


class TestCompletionSolver:
    @patch("nemo_evaluator.solvers.completion.ModelClient")
    async def test_solve_returns_text(self, MockClient):
        client = AsyncMock()
        client._post_with_retry.return_value = {
            "choices": [{"text": "the answer", "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "model": "test-model",
        }
        client.close = AsyncMock()
        MockClient.return_value = client

        from nemo_evaluator.solvers.completion import CompletionSolver

        solver = CompletionSolver(
            base_url="http://localhost:8000/v1",
            model="test-model",
        )
        solver._model_client = client

        task = SeedResult(prompt="What is 2+2?", expected_answer="4")
        result = await solver.solve(task)

        assert isinstance(result, SolveResult)
        assert result.response == "the answer"
        assert result.model_response is not None
        assert result.model_response.prompt_tokens == 10
        assert result.model_response.completion_tokens == 5
        assert len(result.trajectory) > 0

    @patch("nemo_evaluator.solvers.completion.ModelClient")
    async def test_solve_with_temperature(self, MockClient):
        client = AsyncMock()
        client._post_with_retry.return_value = {
            "choices": [{"text": "warm", "finish_reason": "length"}],
            "usage": {},
            "model": "m",
        }
        MockClient.return_value = client

        from nemo_evaluator.solvers.completion import CompletionSolver

        solver = CompletionSolver(
            base_url="http://localhost:8000/v1",
            model="m",
            temperature=0.8,
            max_tokens=100,
        )
        solver._model_client = client

        task = SeedResult(prompt="hi", expected_answer="")
        result = await solver.solve(task)
        assert result.response == "warm"

        payload = client._post_with_retry.call_args[0][1]
        assert payload["temperature"] == 0.8
        assert payload["max_tokens"] == 100


# ── trajectory_util ──────────────────────────────────────────────────────


class TestTrajectoryUtil:
    def test_build_single_turn_atif(self):
        from nemo_evaluator.solvers.trajectory_util import build_single_turn_atif

        traj = build_single_turn_atif(
            prompt="hello",
            response="world",
            model_name="test",
            prompt_tokens=5,
            completion_tokens=3,
        )
        assert isinstance(traj, list)
        assert len(traj) >= 1
        step = traj[0]
        assert "model_name" in step or "content" in step or "role" in step

    def test_build_atif_trajectory(self):
        from nemo_evaluator.solvers.trajectory_util import build_atif_trajectory

        events = [
            {"type": "text", "content": "thinking...", "role": "assistant"},
            {"type": "tool_call", "name": "bash", "arguments": {"command": "ls"}, "role": "assistant"},
            {"type": "tool_result", "content": "file.py", "role": "tool"},
        ]
        traj = build_atif_trajectory(events)
        assert isinstance(traj, list)
