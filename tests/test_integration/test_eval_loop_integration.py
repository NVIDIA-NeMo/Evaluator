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
from typing import Any


from nemo_evaluator.environments.base import EvalEnvironment, SeedResult, VerifyResult
from nemo_evaluator.environments.custom import BenchmarkDefinition, ByobEnvironment, scorer
from nemo_evaluator.engine.eval_loop import run_evaluation
from nemo_evaluator.observability.types import ModelResponse
from nemo_evaluator.scoring import ScorerInput
from nemo_evaluator.scoring.metric import MetricOutputSpec
from nemo_evaluator.sandbox.base import Sandbox
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

    async def verify(
        self, response: str, expected: str, sandbox: Sandbox | None = None, **meta: Any
    ) -> VerifyResult:
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

        env.close = tracking_close  # type: ignore[method-assign]  # ty: ignore[invalid-assignment]
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

    def test_typed_byob_metric_result_preserved_in_results(self):
        outputs = [
            MetricOutputSpec.continuous_score("reward"),
            MetricOutputSpec.continuous_score("format"),
            MetricOutputSpec.label("judge_label"),
            MetricOutputSpec.label("rationale"),
        ]

        @scorer(metric_type="tests.eval_loop_typed", outputs=outputs)
        def typed_scorer(sample: ScorerInput) -> dict[str, object]:
            matched = sample.response == sample.target
            return {
                "reward": 0.8 if matched else 0.0,
                "format": 1.0,
                "judge_label": "pass",
                "rationale": "answer matched",
            }

        env = ByobEnvironment(
            BenchmarkDefinition(
                name="typed_eval_loop",
                dataset=lambda: [{"question": "1+1", "answer": "2"}],
                prompt="{question}",
                target_field="answer",
                scorer_fn=typed_scorer,
            )
        )
        solver = _MockSolver()

        bundle = asyncio.run(run_evaluation(env, solver, n_repeats=1))

        result = bundle["_results"][0]
        assert result["reward"] == 0.8
        assert result["scoring_details"]["reward"] == 0.8
        assert result["scoring_details"]["format"] == 1.0
        assert result["scoring_details"]["outputs"] == {
            "reward": 0.8,
            "format": 1.0,
            "judge_label": "pass",
            "rationale": "answer matched",
        }
        assert result["scoring_details"]["judge_label"] == "pass"
        assert result["scoring_details"]["rationale"] == "answer matched"
        assert result["scoring_details"]["metric_type"] == "tests.eval_loop_typed"


class _MockSolverWithTrajectory:
    """Always correct; returns a non-empty trajectory so we can assert it survives resume."""

    async def solve(self, task):
        return SolveResult(
            response=task.expected_answer,
            trajectory=[{"step": 1, "action": "think", "output": task.expected_answer}],
        )

    async def close(self):
        pass


class _MockSolverWithModelResponse:
    """Returns a full ModelResponse with token breakdown so runtime stats can be verified."""

    PROMPT_TOKENS = 10
    COMPLETION_TOKENS = 5
    REASONING_TOKENS = 2
    TOTAL_TOKENS = 17  # 10 + 5 + 2
    LATENCY_MS = 42.0
    FINISH_REASON = "stop"
    MODEL = "mock-model"

    async def solve(self, task):
        return SolveResult(
            response=task.expected_answer,
            model_response=ModelResponse(
                content=task.expected_answer,
                model=self.MODEL,
                finish_reason=self.FINISH_REASON,
                prompt_tokens=self.PROMPT_TOKENS,
                completion_tokens=self.COMPLETION_TOKENS,
                total_tokens=self.TOTAL_TOKENS,
                latency_ms=self.LATENCY_MS,
                reasoning_tokens=self.REASONING_TOKENS,
            ),
            trajectory=[{"step": 1, "action": "answer"}],
        )

    async def close(self):
        pass


class TestResumeTrajectories:
    def test_trajectories_preserved_on_full_resume(self, tmp_path):
        """All steps in verified_cache on resume → collector never called → 0-byte trajectories.jsonl.

        Regression test: shard killed after all tasks complete (inference +
        verification) but before write_all() flushes trajectories.jsonl to disk.
        On resume every step hits the verified_cache early-return path at
        eval_loop.py:202–225 which never calls collector.record().
        """
        env = _MockEnv()
        solver = _MockSolverWithTrajectory()
        log_dir = tmp_path / "logs"

        # First run: completes fully, populates both inference_log and verified_log
        asyncio.run(run_evaluation(env, solver, n_repeats=2, step_log_dir=log_dir))

        # Second run with resume=True: all steps served from verified_cache
        bundle = asyncio.run(run_evaluation(env, solver, n_repeats=2, step_log_dir=log_dir, resume=True))

        artifacts = bundle["_artifacts"]
        assert artifacts is not None
        assert len(artifacts.steps) == 6  # 3 problems × 2 repeats
        for step in artifacts.steps:
            assert step.trajectory, f"missing trajectory p{step.problem_idx} r{step.repeat}"

    def test_trajectories_preserved_on_verify_only_resume(self, tmp_path):
        """Inference cached but verify not done → trajectory silently dropped.

        Regression test for the partial-crash case: shard killed after all
        inference but before verification. On resume, tasks hit the inferred_cache
        path at eval_loop.py:272–278 which restores response/tokens but never sets
        step.trajectory, so collector.record() records the step without trajectory.
        """
        env = _MockEnv()
        solver = _MockSolverWithTrajectory()
        log_dir = tmp_path / "logs"

        # First run: completes fully, populates both logs
        asyncio.run(run_evaluation(env, solver, n_repeats=1, step_log_dir=log_dir))
        # Simulate crash after inference but before verification by deleting verified_log
        (log_dir / "verified_log.jsonl").unlink()

        # Resume: all tasks are inference-cached, verify-only
        bundle = asyncio.run(run_evaluation(env, solver, n_repeats=1, step_log_dir=log_dir, resume=True))

        artifacts = bundle["_artifacts"]
        assert artifacts is not None
        assert len(artifacts.steps) == 3  # 3 problems × 1 repeat
        for step in artifacts.steps:
            assert step.trajectory, f"missing trajectory p{step.problem_idx} r{step.repeat}"


class TestResumeRuntimeStats:
    def test_total_tokens_complete_on_full_resume(self, tmp_path):
        """On resume with all steps in verified_cache, total_tokens must reflect ALL steps.

        Regression test: inference_log only stores total token count, not the full
        ModelResponse breakdown, so cached steps have model_response=None and contribute
        0 to total_tokens in runtime_stats.
        """
        env = _MockEnv()
        solver = _MockSolverWithModelResponse()
        log_dir = tmp_path / "logs"

        # First run: 3 problems × 1 repeat = 3 steps, each with TOTAL_TOKENS tokens
        asyncio.run(run_evaluation(env, solver, n_repeats=1, step_log_dir=log_dir))

        # Second run: all 3 steps served from verified_cache
        bundle = asyncio.run(run_evaluation(env, solver, n_repeats=1, step_log_dir=log_dir, resume=True))

        expected_total = 3 * _MockSolverWithModelResponse.TOTAL_TOKENS
        actual_total = bundle["benchmark"]["scores"]["runtime"]["total_tokens"]
        assert actual_total == expected_total, (
            f"expected total_tokens={expected_total} (3 steps × {_MockSolverWithModelResponse.TOTAL_TOKENS}), "
            f"got {actual_total} — cached steps not contributing to token stats"
        )

    def test_latency_percentiles_complete_on_full_resume(self, tmp_path):
        """On resume with all steps in verified_cache, latency percentiles must include cached steps."""
        env = _MockEnv()
        solver = _MockSolverWithModelResponse()
        log_dir = tmp_path / "logs"

        asyncio.run(run_evaluation(env, solver, n_repeats=1, step_log_dir=log_dir))

        bundle = asyncio.run(run_evaluation(env, solver, n_repeats=1, step_log_dir=log_dir, resume=True))

        runtime = bundle["benchmark"]["scores"]["runtime"]
        # All 3 cached steps have the same latency — p50 should equal that latency
        assert runtime["latency_percentiles_ms"]["p50"] == _MockSolverWithModelResponse.LATENCY_MS, (
            f"expected p50={_MockSolverWithModelResponse.LATENCY_MS}ms from cached steps, "
            f"got {runtime['latency_percentiles_ms']['p50']} — latency not restored from cache"
        )
