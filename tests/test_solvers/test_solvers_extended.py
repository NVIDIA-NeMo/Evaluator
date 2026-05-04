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
"""Tests for solvers not covered by test_solvers.py — Completion, trajectory_util."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch


from nemo_evaluator.environments.base import SeedResult
from nemo_evaluator.solvers.base import SolveResult


# ── CompletionSolver ─────────────────────────────────────────────────────


class TestCompletionSolver:
    def _make_solver(self):
        """Instantiate CompletionSolver with a mocked ModelClient."""
        with patch("nemo_evaluator.engine.model_client.ModelClient"):
            from nemo_evaluator.solvers.completion import CompletionSolver

            solver = CompletionSolver(
                base_url="http://localhost:8000/v1",
                model="test-model",
            )
        client = AsyncMock()
        client.close = AsyncMock()
        solver._model_client = client
        return solver, client

    async def test_solve_returns_text(self):
        solver, client = self._make_solver()
        client._post_with_retry.return_value = {
            "choices": [{"text": "the answer", "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "model": "test-model",
        }
        task = SeedResult(prompt="What is 2+2?", expected_answer="4")
        result = await solver.solve(task)

        assert isinstance(result, SolveResult)
        assert result.response == "the answer"
        assert result.model_response is not None
        assert result.model_response.prompt_tokens == 10
        assert len(result.trajectory) > 0

    async def test_solve_with_temperature(self):
        with patch("nemo_evaluator.engine.model_client.ModelClient"):
            from nemo_evaluator.solvers.completion import CompletionSolver

            solver = CompletionSolver(
                base_url="http://localhost:8000/v1",
                model="m",
                temperature=0.8,
                max_tokens=100,
            )
        client = AsyncMock()
        client._post_with_retry.return_value = {
            "choices": [{"text": "warm", "finish_reason": "length"}],
            "usage": {},
            "model": "m",
        }
        solver._model_client = client

        task = SeedResult(prompt="hi", expected_answer="")
        result = await solver.solve(task)
        assert result.response == "warm"

        payload = client._post_with_retry.call_args[0][1]
        assert payload["temperature"] == 0.8
        assert payload["max_tokens"] == 100

    async def test_close(self):
        solver, client = self._make_solver()
        await solver.close()
        client.close.assert_called_once()


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

    def test_build_single_turn_atif_per_step_metrics(self):
        from nemo_evaluator.solvers.trajectory_util import build_single_turn_atif

        traj = build_single_turn_atif(
            prompt="hello",
            response="world",
            model_name="test",
            prompt_tokens=10,
            completion_tokens=4,
        )
        doc = traj[0]
        agent_step = next(s for s in doc["steps"] if s["source"] == "agent")
        assert "metrics" in agent_step
        assert agent_step["metrics"]["prompt_tokens"] == 10
        assert agent_step["metrics"]["completion_tokens"] == 4

    def test_build_atif_trajectory_final_metrics(self):
        from nemo_evaluator.solvers.trajectory_util import build_atif_trajectory

        steps = [
            {"source": "user", "message": "hi"},
            {"source": "agent", "message": "hello", "metrics": {"prompt_tokens": 7, "completion_tokens": 2}},
        ]
        traj = build_atif_trajectory(steps, prompt_tokens=7, completion_tokens=2)
        doc = traj[0]
        fm = doc["final_metrics"]
        assert fm["total_prompt_tokens"] == 7
        assert fm["total_completion_tokens"] == 2
        assert fm["total_steps"] == 2

    def test_build_atif_trajectory_final_metrics_always_present(self):
        from nemo_evaluator.solvers.trajectory_util import build_atif_trajectory

        traj = build_atif_trajectory([{"source": "user", "message": "ping"}])
        assert "final_metrics" in traj[0]
        assert traj[0]["final_metrics"]["total_steps"] == 1

    def test_build_atif_trajectory_derives_totals_from_per_step_metrics(self):
        from nemo_evaluator.solvers.trajectory_util import build_atif_trajectory

        # No explicit token totals — should be summed from per-step metrics.
        steps = [
            {"source": "user", "message": "go"},
            {"source": "agent", "message": "step1", "metrics": {"prompt_tokens": 30, "completion_tokens": 10}},
            {"source": "system", "message": "obs"},
            {"source": "agent", "message": "step2", "metrics": {"prompt_tokens": 50, "completion_tokens": 15}},
        ]
        traj = build_atif_trajectory(steps)
        fm = traj[0]["final_metrics"]
        assert fm["total_prompt_tokens"] == 80
        assert fm["total_completion_tokens"] == 25
        assert fm["total_steps"] == 4

    def test_build_atif_trajectory(self):
        from nemo_evaluator.solvers.trajectory_util import build_atif_trajectory

        events = [
            {"type": "text", "content": "thinking...", "role": "assistant"},
            {"type": "tool_call", "name": "bash", "arguments": {"command": "ls"}, "role": "assistant"},
            {"type": "tool_result", "content": "file.py", "role": "tool"},
        ]
        traj = build_atif_trajectory(events)
        assert isinstance(traj, list)
