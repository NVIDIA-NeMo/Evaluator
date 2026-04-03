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
"""Integration tests: run_evaluation end-to-end with mock solver."""

import asyncio


from nemo_evaluator.environments.base import EvalEnvironment, SeedResult, VerifyResult
from nemo_evaluator.engine.eval_loop import run_evaluation
from nemo_evaluator.solvers import SolveResult


class _MockEnv(EvalEnvironment):
    name = "mock_math"

    def __init__(self):
        super().__init__()
        self._dataset = [
            {"q": "1+1", "a": "2"},
            {"q": "2+3", "a": "5"},
            {"q": "10-7", "a": "3"},
        ]

    async def seed(self, idx):
        r = self._dataset[idx]
        return SeedResult(prompt=r["q"], expected_answer=r["a"], metadata={"idx": idx})

    async def verify(self, response, expected, **meta):
        correct = response.strip() == expected.strip()
        return VerifyResult(
            reward=1.0 if correct else 0.0, extracted_answer=response.strip(), scoring_details={"method": "exact"}
        )


class _MockSolver:
    """Returns the expected answer for first 2 problems, wrong for the third."""

    async def solve(self, task):
        if task.expected_answer in ("2", "5"):
            return SolveResult(response=task.expected_answer)
        return SolveResult(response="wrong")

    async def close(self):
        pass


class TestRunEvaluationIntegration:
    def test_basic_run(self):
        env = _MockEnv()
        solver = _MockSolver()
        bundle = asyncio.run(run_evaluation(env, solver, n_repeats=1))

        assert "benchmark" in bundle
        bm = bundle["benchmark"]
        assert bm["samples"] == 3
        scores = bm["scores"]
        assert "pass@1" in scores
        # 2 out of 3 correct
        assert 0.6 < scores["pass@1"]["value"] < 0.7

    def test_repeats(self):
        env = _MockEnv()
        solver = _MockSolver()
        bundle = asyncio.run(run_evaluation(env, solver, n_repeats=3))

        results = bundle["_results"]
        assert len(results) == 9  # 3 problems x 3 repeats

    def test_max_problems(self):
        env = _MockEnv()
        solver = _MockSolver()
        bundle = asyncio.run(run_evaluation(env, solver, n_repeats=1, max_problems=2))

        results = bundle["_results"]
        assert len(results) == 2

    def test_results_have_correct_fields(self):
        env = _MockEnv()
        solver = _MockSolver()
        bundle = asyncio.run(run_evaluation(env, solver, n_repeats=1))

        for r in bundle["_results"]:
            assert "problem_idx" in r
            assert "repeat" in r
            assert "reward" in r
            assert "model_response" in r
            assert "expected_answer" in r
            assert "scoring_details" in r

    def test_runtime_stats_present(self):
        env = _MockEnv()
        solver = _MockSolver()
        bundle = asyncio.run(run_evaluation(env, solver, n_repeats=1))

        scores = bundle["benchmark"]["scores"]
        assert "runtime" in scores
        assert "failures" in scores

    def test_env_close_called(self):
        env = _MockEnv()
        closed = []
        original_close = env.close

        async def tracking_close():
            closed.append(True)
            await original_close()

        env.close = tracking_close
        solver = _MockSolver()
        asyncio.run(run_evaluation(env, solver, n_repeats=1))
        assert closed, "env.close() was not called"

    def test_skip_failed_continues(self):
        """With skip_failed=True, failing tasks don't abort the run."""

        class _FailingSolver:
            async def solve(self, task):
                if task.expected_answer == "3":
                    raise RuntimeError("kaboom")
                return SolveResult(response=task.expected_answer)

            async def close(self):
                pass

        env = _MockEnv()
        solver = _FailingSolver()
        bundle = asyncio.run(run_evaluation(env, solver, n_repeats=1, skip_failed=True))
        results = bundle["_results"]
        assert len(results) == 3

    def test_problem_range(self):
        env = _MockEnv()
        solver = _MockSolver()
        bundle = asyncio.run(run_evaluation(env, solver, n_repeats=1, problem_range=(1, 3)))
        results = bundle["_results"]
        assert len(results) == 2

    def test_shard_info(self):
        env = _MockEnv()
        solver = _MockSolver()
        bundle = asyncio.run(run_evaluation(env, solver, n_repeats=1, shard_info=(0, 2)))
        results = bundle["_results"]
        assert 0 < len(results) <= 3

    def test_all_wrong_zero_score(self):
        class _WrongSolver:
            async def solve(self, task):
                return SolveResult(response="wrong-always")

            async def close(self):
                pass

        env = _MockEnv()
        solver = _WrongSolver()
        bundle = asyncio.run(run_evaluation(env, solver, n_repeats=1))
        scores = bundle["benchmark"]["scores"]
        assert scores["pass@1"]["value"] == 0.0

    def test_all_correct_perfect_score(self):
        class _PerfectSolver:
            async def solve(self, task):
                return SolveResult(response=task.expected_answer)

            async def close(self):
                pass

        env = _MockEnv()
        solver = _PerfectSolver()
        bundle = asyncio.run(run_evaluation(env, solver, n_repeats=1))
        scores = bundle["benchmark"]["scores"]
        assert scores["pass@1"]["value"] == 1.0

    def test_concurrent_execution(self):
        """Multiple concurrent tasks should produce correct result count."""
        env = _MockEnv()
        solver = _MockSolver()
        bundle = asyncio.run(run_evaluation(env, solver, n_repeats=2, max_concurrent=8))
        results = bundle["_results"]
        assert len(results) == 6
