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
import logging

from nemo_evaluator.engine.eval_loop import run_evaluation
from nemo_evaluator.environments.base import EvalEnvironment, SeedResult, VerifyResult
from nemo_evaluator.observability.types import ModelResponse
from nemo_evaluator.solvers import SolveResult
from nemo_evaluator.solvers.base import ErrorKind

EVAL_LOOP_LOGGER = "nemo_evaluator.engine.eval_loop"


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

        Reproduces a regression where a shard killed after all tasks complete
        (inference + verification) but before write_all() flushes
        trajectories.jsonl to disk caused trajectory loss. On resume every step
        hits the verified_cache early-return path at eval_loop.py:202–225 which
        never calls collector.record().
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

        Reproduces a regression where a shard killed after all inference but
        before verification dropped trajectories on resume. On resume, tasks
        hit the inferred_cache path at eval_loop.py:272–278 which restores
        response/tokens but never sets step.trajectory, so collector.record()
        records the step without trajectory.
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

        Reproduces a regression where inference_log only stored total token
        count, not the full ModelResponse breakdown, so cached steps had
        model_response=None and contributed 0 to total_tokens in runtime_stats.
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


class _JudgeFnEnv(EvalEnvironment):
    """Env whose verify() attaches a ``_judge_fn`` closure and ``needs_judge``."""

    name = "mock_judge_fn"

    def __init__(self, judge_fn):
        super().__init__()
        self._dataset = [{"idx": 0}]
        self._judge_fn = judge_fn

    async def seed(self, idx):
        return SeedResult(prompt="q", expected_answer="a", metadata={"idx": idx})

    async def verify(self, response, expected, **meta):  # noqa: ARG002
        return VerifyResult(
            reward=0.0,
            extracted_answer=response,
            scoring_details={
                "needs_judge": True,
                "_judge_fn": self._judge_fn,
                "task_id": f"t{meta.get('idx', 0)}",
            },
        )


class _OneProblemSolver:
    async def solve(self, task):  # noqa: ARG002
        return SolveResult(response="resp")

    async def close(self):
        pass


class TestJudgeFnSerialization:
    def test_judge_fn_popped_on_happy_path(self, tmp_path):
        async def ok_fn(client):  # noqa: ARG001
            return {"reward": 0.9, "judge": {"total": 0.9}}

        env = _JudgeFnEnv(ok_fn)

        class _FakeJudgeClient:
            pass

        bundle = asyncio.run(
            run_evaluation(
                env,
                _OneProblemSolver(),
                n_repeats=1,
                judge_client=_FakeJudgeClient(),
                step_log_dir=tmp_path,
            )
        )

        (result,) = bundle["_results"]
        assert "_judge_fn" not in result["scoring_details"]
        assert result["scoring_details"]["needs_judge"] is False
        assert result["reward"] == 0.9
        assert result["scoring_details"]["judge"] == {"total": 0.9}

    def test_judge_fn_popped_when_no_judge_client(self, tmp_path, caplog):
        import logging

        async def unused_fn(client):  # noqa: ARG001
            raise AssertionError("must not be invoked without a judge_client")

        env = _JudgeFnEnv(unused_fn)
        with caplog.at_level(logging.WARNING, logger="nemo_evaluator.engine.eval_loop"):
            bundle = asyncio.run(
                run_evaluation(
                    env,
                    _OneProblemSolver(),
                    n_repeats=1,
                    judge_client=None,
                    step_log_dir=tmp_path,
                )
            )

        (result,) = bundle["_results"]
        assert "_judge_fn" not in result["scoring_details"]
        assert result["scoring_details"]["needs_judge"] is False
        assert "error" in result["scoring_details"]["judge"]
        assert any("needs_judge" in rec.message and "no judge_client" in rec.message for rec in caplog.records)

    def test_judge_closure_exception_recorded_not_swallowed(self, tmp_path):
        async def broken_fn(client):  # noqa: ARG001
            raise RuntimeError("judge exploded")

        env = _JudgeFnEnv(broken_fn)

        class _FakeJudgeClient:
            pass

        bundle = asyncio.run(
            run_evaluation(
                env,
                _OneProblemSolver(),
                n_repeats=1,
                judge_client=_FakeJudgeClient(),
                step_log_dir=tmp_path,
            )
        )

        (result,) = bundle["_results"]
        assert "_judge_fn" not in result["scoring_details"]
        judge = result["scoring_details"]["judge"]
        assert judge["total"] is None
        assert "judge exploded" in judge["error"]


BIG_TRAJECTORY = [{"step": 1, "action": "think", "output": "x" * 4096}]


class _BigTrajectorySolver:
    """Trajectory JSON far exceeds the tiny NEL_MAX_TRAJECTORY_BYTES cap set in tests."""

    def __init__(self):
        self.calls = 0

    async def solve(self, task):
        self.calls += 1
        return SolveResult(response=task.expected_answer, trajectory=BIG_TRAJECTORY)

    async def close(self):
        pass


class TestResumeOversizedTrajectory:
    """FEP-1295: trajectories over the checkpoint cap used to be rewritten as
    ``{"trajectory": None, "_truncated": True}``, so after a walltime kill the
    resumed run produced zero-step trajectories. They now spill to a gzip
    sidecar and are dereferenced on resume."""

    def _first_run(self, tmp_path, monkeypatch, solver):
        monkeypatch.setenv("NEL_MAX_TRAJECTORY_BYTES", "100")
        env = _MockEnv()
        log_dir = tmp_path / "logs"
        asyncio.run(run_evaluation(env, solver, n_repeats=1, step_log_dir=log_dir))
        assert "trajectory_ref" in (log_dir / "inference_log.jsonl").read_text()
        return env, log_dir

    def test_survives_fully_cached_resume(self, tmp_path, monkeypatch):
        solver = _BigTrajectorySolver()
        env, log_dir = self._first_run(tmp_path, monkeypatch, solver)

        bundle = asyncio.run(run_evaluation(env, solver, n_repeats=1, step_log_dir=log_dir, resume=True))

        assert solver.calls == 3
        steps = bundle["_artifacts"].steps
        assert len(steps) == 3
        for step in steps:
            assert step.trajectory == BIG_TRAJECTORY, f"lost trajectory p{step.problem_idx} r{step.repeat}"

    def test_survives_verify_only_resume(self, tmp_path, monkeypatch):
        solver = _BigTrajectorySolver()
        env, log_dir = self._first_run(tmp_path, monkeypatch, solver)
        (log_dir / "verified_log.jsonl").unlink()

        bundle = asyncio.run(run_evaluation(env, solver, n_repeats=1, step_log_dir=log_dir, resume=True))

        assert solver.calls == 3
        steps = bundle["_artifacts"].steps
        assert len(steps) == 3
        for step in steps:
            assert step.trajectory == BIG_TRAJECTORY, f"lost trajectory p{step.problem_idx} r{step.repeat}"


class _CountingEnv(_MockEnv):
    def __init__(self):
        super().__init__()
        self.verify_calls = 0

    async def verify(self, response, expected, **meta):
        self.verify_calls += 1
        return await super().verify(response, expected, **meta)


class _TimeoutShapedSolver:
    """Harbor timeout with workspace diff: error=None, error_kind=SOLVE_TIMEOUT."""

    def __init__(self):
        self.calls = 0

    async def solve(self, task):
        self.calls += 1
        return SolveResult(response="partial work", error_kind=ErrorKind.SOLVE_TIMEOUT)

    async def close(self):
        pass


class _CrashedSolver:
    """Graceful solve failure whose message classifies as turn_budget_exhausted."""

    def __init__(self):
        self.calls = 0

    async def solve(self, task):
        self.calls += 1
        return SolveResult(response="", error="Turn budget exhausted after 50 turns")

    async def close(self):
        pass


class _TurnBudgetDetailsSolver:
    """Succeeds but flags a failure label via scoring_details (Harbor workspace classification)."""

    async def solve(self, task):
        return SolveResult(
            response=task.expected_answer,
            scoring_details={"error": "turn budget exhausted", "error_category": "turn_budget_exhausted"},
        )

    async def close(self):
        pass


class _FlakyInfraSolver:
    """First call fails with an infra-kind error; subsequent calls succeed."""

    def __init__(self):
        self.calls = 0

    async def solve(self, task):
        self.calls += 1
        if self.calls == 1:
            return SolveResult(response="", error="connection refused", error_kind=ErrorKind.INFRA)
        return SolveResult(response=task.expected_answer)

    async def close(self):
        pass


class TestResumeFailureLabels:
    """FEP-1295: the inference checkpoint stored no error/error_kind/solver
    scoring_details, so verify-only resume re-verified solve failures and lost
    failure labels like solve_timeout and turn_budget_exhausted."""

    def test_solve_timeout_label_survives_verify_only_resume(self, tmp_path):
        env = _CountingEnv()
        solver = _TimeoutShapedSolver()
        log_dir = tmp_path / "logs"
        asyncio.run(run_evaluation(env, solver, n_repeats=1, max_problems=1, step_log_dir=log_dir))
        (log_dir / "verified_log.jsonl").unlink()

        bundle = asyncio.run(
            run_evaluation(env, solver, n_repeats=1, max_problems=1, step_log_dir=log_dir, resume=True)
        )

        assert solver.calls == 1, "verify-only resume must not re-solve"
        (result,) = bundle["_results"]
        assert result["scoring_details"]["error_category"] == "solve_timeout"

    def test_solve_failure_replayed_without_reverify(self, tmp_path, caplog):
        env = _CountingEnv()
        solver = _CrashedSolver()
        log_dir = tmp_path / "logs"
        asyncio.run(run_evaluation(env, solver, n_repeats=1, max_problems=1, step_log_dir=log_dir))
        assert env.verify_calls == 0
        (log_dir / "verified_log.jsonl").unlink()

        with caplog.at_level(logging.WARNING, logger=EVAL_LOOP_LOGGER):
            bundle = asyncio.run(
                run_evaluation(env, solver, n_repeats=1, max_problems=1, step_log_dir=log_dir, resume=True)
            )

        assert solver.calls == 1
        assert env.verify_calls == 0, "cached solve failure must be replayed, not re-verified"
        (result,) = bundle["_results"]
        assert result["reward"] == 0.0
        assert result["scoring_details"]["error_category"] == "turn_budget_exhausted"
        assert result["scoring_details"]["method"] == "solve_failed"
        assert any("replaying cached solve failure" in rec.message for rec in caplog.records)

    def test_solver_scoring_details_survive_verify_only_resume(self, tmp_path):
        env = _CountingEnv()
        solver = _TurnBudgetDetailsSolver()
        log_dir = tmp_path / "logs"
        asyncio.run(run_evaluation(env, solver, n_repeats=1, max_problems=1, step_log_dir=log_dir))
        (log_dir / "verified_log.jsonl").unlink()

        bundle = asyncio.run(
            run_evaluation(env, solver, n_repeats=1, max_problems=1, step_log_dir=log_dir, resume=True)
        )

        assert env.verify_calls == 2
        (result,) = bundle["_results"]
        assert result["scoring_details"]["error_category"] == "turn_budget_exhausted"
        assert result["reward"] == 1.0

    def test_unverified_infra_inference_is_retried(self, tmp_path):
        env = _CountingEnv()
        solver = _FlakyInfraSolver()
        log_dir = tmp_path / "logs"
        asyncio.run(run_evaluation(env, solver, n_repeats=1, max_problems=1, step_log_dir=log_dir))
        (log_dir / "verified_log.jsonl").unlink()

        bundle = asyncio.run(
            run_evaluation(env, solver, n_repeats=1, max_problems=1, step_log_dir=log_dir, resume=True)
        )

        assert solver.calls == 2, "infra-kind unverified inference must be re-solved on resume"
        (result,) = bundle["_results"]
        assert result["reward"] == 1.0


class _DummySandboxLifecycle:
    """Stub lifecycle handing out a bare object as the agent sandbox."""

    captures_workspace = True

    async def setup(self):
        pass

    async def get_agent_sandbox(self):
        return object()

    async def transition_to_verify(self, response_text, solver_modified):
        pass

    async def get_verify_sandbox(self):
        return None

    async def teardown(self):
        pass


class _SandboxAcceptingSolver:
    async def solve(self, task, sandbox=None):
        return SolveResult(response=task.expected_answer)

    async def close(self):
        pass


class TestResumeCaptureMarkers:
    """FEP-1295: a kill landing before/during workspace capture leaves verify-only
    resume scoring an unmodified workspace with no trace. Markers written after
    transition_to_verify() make that observable via scoring_details."""

    def _patch_lifecycle(self, monkeypatch):
        monkeypatch.setattr(
            "nemo_evaluator.engine.eval_loop.pick_lifecycle",
            lambda *args, **kwargs: _DummySandboxLifecycle(),
        )

    def test_first_run_writes_markers(self, tmp_path, monkeypatch):
        self._patch_lifecycle(monkeypatch)
        log_dir = tmp_path / "logs"
        asyncio.run(run_evaluation(_MockEnv(), _SandboxAcceptingSolver(), n_repeats=1, step_log_dir=log_dir))

        for idx in range(3):
            assert (log_dir / "capture_markers" / f"p{idx}_r0.captured").is_file()

    def test_resume_with_marker_not_flagged(self, tmp_path, monkeypatch):
        self._patch_lifecycle(monkeypatch)
        log_dir = tmp_path / "logs"
        asyncio.run(
            run_evaluation(_MockEnv(), _SandboxAcceptingSolver(), n_repeats=1, max_problems=1, step_log_dir=log_dir)
        )
        (log_dir / "verified_log.jsonl").unlink()

        bundle = asyncio.run(
            run_evaluation(
                _MockEnv(),
                _SandboxAcceptingSolver(),
                n_repeats=1,
                max_problems=1,
                step_log_dir=log_dir,
                resume=True,
            )
        )

        (result,) = bundle["_results"]
        assert "resume_workspace_capture" not in result["scoring_details"]

    def test_resume_without_marker_flagged(self, tmp_path, monkeypatch, caplog):
        self._patch_lifecycle(monkeypatch)
        log_dir = tmp_path / "logs"
        asyncio.run(
            run_evaluation(_MockEnv(), _SandboxAcceptingSolver(), n_repeats=1, max_problems=1, step_log_dir=log_dir)
        )
        (log_dir / "verified_log.jsonl").unlink()
        (log_dir / "capture_markers" / "p0_r0.captured").unlink()

        with caplog.at_level(logging.WARNING, logger=EVAL_LOOP_LOGGER):
            bundle = asyncio.run(
                run_evaluation(
                    _MockEnv(),
                    _SandboxAcceptingSolver(),
                    n_repeats=1,
                    max_problems=1,
                    step_log_dir=log_dir,
                    resume=True,
                )
            )

        (result,) = bundle["_results"]
        assert result["scoring_details"]["resume_workspace_capture"] == "missing"
        assert any("capture marker" in rec.message for rec in caplog.records)

    def test_verify_only_workspace_reexecuted_on_resume(self, tmp_path, monkeypatch, caplog):
        """A verify-only sandbox rollout re-solves on resume (proper score) instead of
        being committed as a workspace-missing zero-reward stub."""
        self._patch_lifecycle(monkeypatch)
        log_dir = tmp_path / "logs"

        solves = {"count": 0}

        class _CountingSandboxSolver(_SandboxAcceptingSolver):
            async def solve(self, task, sandbox=None):
                solves["count"] += 1
                return await super().solve(task, sandbox=sandbox)

        asyncio.run(
            run_evaluation(_MockEnv(), _CountingSandboxSolver(), n_repeats=1, max_problems=1, step_log_dir=log_dir)
        )
        assert solves["count"] == 1
        (log_dir / "verified_log.jsonl").unlink()

        with caplog.at_level(logging.INFO, logger=EVAL_LOOP_LOGGER):
            bundle = asyncio.run(
                run_evaluation(
                    _MockEnv(),
                    _CountingSandboxSolver(),
                    n_repeats=1,
                    max_problems=1,
                    step_log_dir=log_dir,
                    resume=True,
                )
            )

        (result,) = bundle["_results"]
        assert "resume_workspace_capture" not in result["scoring_details"]  # not a stub
        assert result["reward"] == 1.0  # re-verified against a real solve, not a pristine workspace
        assert solves["count"] == 2  # re-solved on resume rather than replayed from cache
        assert any("re-solved" in rec.message for rec in caplog.records)

    def test_stale_marker_invalidated_by_retry_inference(self, tmp_path, monkeypatch):
        """A verify retry re-solves and re-appends; a kill before the retry's capture
        must not be masked by the marker left over from the first attempt."""

        transitions = {"count": 0}

        class _KilledOnRetryLifecycle(_DummySandboxLifecycle):
            async def transition_to_verify(self, response_text, solver_modified):
                transitions["count"] += 1
                if transitions["count"] == 2:
                    raise RuntimeError("killed before workspace capture")

        monkeypatch.setattr(
            "nemo_evaluator.engine.eval_loop.pick_lifecycle",
            lambda *args, **kwargs: _KilledOnRetryLifecycle(),
        )

        class _FlakyVerifyEnv(_MockEnv):
            def __init__(self):
                super().__init__()
                self.verify_calls = 0

            async def verify(self, response, expected, **meta):
                self.verify_calls += 1
                if self.verify_calls == 1:
                    raise RuntimeError("verify hiccup, triggers system retry")
                return await super().verify(response, expected, **meta)

        env = _FlakyVerifyEnv()
        log_dir = tmp_path / "logs"
        asyncio.run(
            run_evaluation(
                env,
                _SandboxAcceptingSolver(),
                n_repeats=1,
                max_problems=1,
                step_log_dir=log_dir,
                max_system_retries=2,
                skip_failed=True,
            )
        )

        assert transitions["count"] == 2
        assert not (log_dir / "capture_markers" / "p0_r0.captured").exists(), (
            "marker from attempt 1 must be invalidated by attempt 2's inference append"
        )

        (log_dir / "verified_log.jsonl").unlink()
        self._patch_lifecycle(monkeypatch)
        bundle = asyncio.run(
            run_evaluation(
                env,
                _SandboxAcceptingSolver(),
                n_repeats=1,
                max_problems=1,
                step_log_dir=log_dir,
                resume=True,
            )
        )

        (result,) = bundle["_results"]
        assert result["scoring_details"]["resume_workspace_capture"] == "missing"


class _MixedPopEnv(_MockEnv):
    """Three sandbox rollouts mirroring the FEP-888 Run A population."""

    async def seed(self, idx):
        return SeedResult(prompt=f"q{idx}", expected_answer=f"a{idx}", metadata={"idx": idx})

    async def verify(self, response, expected, **meta):
        return VerifyResult(reward=1.0 if response == expected else 0.0, scoring_details={"method": "exact"})


class _MixedPopSolver:
    """Per-idx behavior: 0 = solved, 1 = diff-carrying timeout (error nulled, workspace-bound,
    the p59/p368 case), 2 = genuine solve failure (error set, graded 0 without a workspace)."""

    def __init__(self, calls):
        self._calls = calls

    async def solve(self, task, sandbox=None):
        idx = task.metadata["idx"]
        self._calls.append(idx)
        if idx == 1:
            return SolveResult(
                response=task.expected_answer,
                error_kind=ErrorKind.SOLVE_TIMEOUT,
                scoring_details={"error": "turn budget exhausted", "error_category": "turn_budget_exhausted"},
            )
        if idx == 2:
            return SolveResult(response="", error="litellm.BadRequestError: 400", error_kind=ErrorKind.NONE)
        return SolveResult(response=task.expected_answer)

    async def close(self):
        pass


class TestVerifyOnlyResumeReexecution:
    """FEP-888 regression: a synthetic checkpoint mirroring Run A (real-schema, produced by
    run_evaluation). On a kill that lands after the solve checkpoint but before verification,
    verify-only sandbox rollouts with a recoverable/successful solve must re-execute, while a
    genuine solve failure must replay graded-0 without a second attempt."""

    def _patch_lifecycle(self, monkeypatch):
        monkeypatch.setattr(
            "nemo_evaluator.engine.eval_loop.pick_lifecycle",
            lambda *args, **kwargs: _DummySandboxLifecycle(),
        )

    def test_reexecution_matrix_on_verify_only_resume(self, tmp_path, monkeypatch):
        self._patch_lifecycle(monkeypatch)
        log_dir = tmp_path / "logs"

        calls_1 = []
        asyncio.run(run_evaluation(_MixedPopEnv(), _MixedPopSolver(calls_1), n_repeats=1, step_log_dir=log_dir))
        assert sorted(calls_1) == [0, 1, 2]  # cold run solves everything

        # Simulate the crash window: solves are checkpointed, verification is not.
        (log_dir / "verified_log.jsonl").unlink()

        calls_2 = []
        bundle = asyncio.run(
            run_evaluation(_MixedPopEnv(), _MixedPopSolver(calls_2), n_repeats=1, step_log_dir=log_dir, resume=True)
        )

        # Success (0) and diff-carrying timeout (1) re-solve; genuine failure (2) does not.
        assert sorted(calls_2) == [0, 1]
        results = {r["problem_idx"]: r for r in bundle["_results"]}
        assert results[0]["reward"] == 1.0
        assert results[1]["reward"] == 1.0
        assert results[2]["reward"] == 0.0
        assert results[2]["scoring_details"]["method"] == "solve_failed"


class TestSidecarReset:
    def _plant_leftover_sidecars(self, log_dir):
        (log_dir / "trajectory_overflow").mkdir(parents=True, exist_ok=True)
        (log_dir / "trajectory_overflow" / "p0_r0.json.gz").write_bytes(b"junk")
        (log_dir / "capture_markers").mkdir(exist_ok=True)
        (log_dir / "capture_markers" / "p0_r0.captured").touch()

    def test_fresh_run_clears_leftover_sidecars(self, tmp_path):
        log_dir = tmp_path / "logs"
        self._plant_leftover_sidecars(log_dir)

        asyncio.run(run_evaluation(_MockEnv(), _MockSolver(), n_repeats=1, step_log_dir=log_dir))

        assert not (log_dir / "trajectory_overflow").exists()
        assert not (log_dir / "capture_markers").exists()

    def test_config_changed_resume_clears_leftover_sidecars(self, tmp_path):
        log_dir = tmp_path / "logs"
        asyncio.run(
            run_evaluation(_MockEnv(), _MockSolver(), n_repeats=1, step_log_dir=log_dir, config={"temperature": 0.1})
        )
        self._plant_leftover_sidecars(log_dir)

        asyncio.run(
            run_evaluation(
                _MockEnv(),
                _MockSolver(),
                n_repeats=1,
                step_log_dir=log_dir,
                resume=True,
                config={"temperature": 0.9},
            )
        )

        assert not (log_dir / "trajectory_overflow").exists()
        assert not (log_dir / "capture_markers").exists()
