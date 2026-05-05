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
"""End-to-end evaluation loop tests using cached solver + fixtured environments.

These tests exercise the full `run_evaluation` code path without any network calls
or real model inference -- every response is replayed from golden fixtures.
"""

from __future__ import annotations

import pytest

from tests.conftest import (
    AVAILABLE_FIXTURES,
    FIXTURE_DIR,
    CachedSolver,
    FixturedEnvironment,
    MockJudgeClient,
    MockSandboxManager,
)


@pytest.mark.parametrize("bench", AVAILABLE_FIXTURES)
@pytest.mark.asyncio
async def test_eval_loop_completes(bench: str):
    """Full eval loop: seed → solve (cached) → verify → metrics."""
    from nemo_evaluator.engine.eval_loop import run_evaluation

    fixture_path = FIXTURE_DIR / f"{bench}.json"
    env = FixturedEnvironment(fixture_path)
    solver = CachedSolver(fixture_path)

    bundle = await run_evaluation(
        env=env,
        solver=solver,
        n_repeats=1,
        max_problems=5,
        max_concurrent=2,
    )

    scores = bundle["benchmark"]["scores"]
    assert "pass@1" in scores
    assert "summary" in scores
    assert scores["pass@1"]["value"] >= 0.0
    assert scores["summary"]["n"] == 5

    results = bundle["_results"]
    assert len(results) == 5
    for r in results:
        assert "reward" in r
        assert "model_response" in r


@pytest.mark.asyncio
async def test_eval_loop_n_repeats():
    """n_repeats > 1 gives us len(results) == problems * repeats."""
    from nemo_evaluator.engine.eval_loop import run_evaluation

    bench = "mmlu"
    fixture_path = FIXTURE_DIR / f"{bench}.json"
    if not fixture_path.exists():
        pytest.skip("mmlu fixture not found")

    env = FixturedEnvironment(fixture_path)
    solver = CachedSolver(fixture_path)

    bundle = await run_evaluation(
        env=env,
        solver=solver,
        n_repeats=3,
        max_problems=5,
        max_concurrent=2,
    )

    results = bundle["_results"]
    assert len(results) == 15  # 5 problems * 3 repeats


@pytest.mark.asyncio
async def test_eval_loop_with_sandbox_manager():
    """Eval loop works when a sandbox manager is provided (even if unused)."""
    from nemo_evaluator.engine.eval_loop import run_evaluation

    bench = "gsm8k"
    fixture_path = FIXTURE_DIR / f"{bench}.json"
    if not fixture_path.exists():
        pytest.skip("gsm8k fixture not found")

    env = FixturedEnvironment(fixture_path)
    solver = CachedSolver(fixture_path)
    mgr = MockSandboxManager()

    bundle = await run_evaluation(
        env=env,
        solver=solver,
        n_repeats=1,
        max_problems=5,
        max_concurrent=2,
        sandbox_manager=mgr,
    )

    assert "benchmark" in bundle
    assert "scores" in bundle["benchmark"]


@pytest.mark.asyncio
async def test_eval_loop_judge_client():
    """Judge client is called when verify returns needs_judge=True."""
    from nemo_evaluator.engine.eval_loop import run_evaluation

    bench = "simpleqa"
    fixture_path = FIXTURE_DIR / f"{bench}.json"
    if not fixture_path.exists():
        pytest.skip("simpleqa fixture not found")

    env = FixturedEnvironment(fixture_path)
    solver = CachedSolver(fixture_path)
    judge = MockJudgeClient(default_score=0.75)

    bundle = await run_evaluation(
        env=env,
        solver=solver,
        n_repeats=1,
        max_problems=5,
        max_concurrent=2,
        judge_client=judge,
    )

    assert "benchmark" in bundle
    assert "scores" in bundle["benchmark"]


@pytest.mark.asyncio
async def test_eval_loop_metrics_shape():
    """Verify that all expected metric keys are present."""
    from nemo_evaluator.engine.eval_loop import run_evaluation

    bench = "drop"
    fixture_path = FIXTURE_DIR / f"{bench}.json"
    if not fixture_path.exists():
        pytest.skip("drop fixture not found")

    env = FixturedEnvironment(fixture_path)
    solver = CachedSolver(fixture_path)

    bundle = await run_evaluation(
        env=env,
        solver=solver,
        n_repeats=1,
        max_problems=5,
    )

    scores = bundle["benchmark"]["scores"]
    assert "pass@1" in scores

    pass1 = scores["pass@1"]
    assert "value" in pass1
    assert "bootstrap_ci_lower" in pass1
    assert "bootstrap_ci_upper" in pass1
    assert pass1["bootstrap_ci_lower"] <= pass1["value"] <= pass1["bootstrap_ci_upper"]
    # sample-level ci_lower/ci_upper only present when n_repeats >= 2
    assert "ci_lower" not in pass1
    assert "ci_upper" not in pass1

    summary = scores["summary"]
    assert "n" in summary
    assert "mean" in summary

    assert "runtime" in scores
    assert "failures" in scores


@pytest.mark.asyncio
async def test_eval_loop_zero_reward():
    """Eval loop handles benchmarks where all rewards are zero (simpleqa fixture)."""
    from nemo_evaluator.engine.eval_loop import run_evaluation

    bench = "simpleqa"
    fixture_path = FIXTURE_DIR / f"{bench}.json"
    if not fixture_path.exists():
        pytest.skip("simpleqa fixture not found")

    env = FixturedEnvironment(fixture_path)
    solver = CachedSolver(fixture_path)

    bundle = await run_evaluation(
        env=env,
        solver=solver,
        n_repeats=1,
        max_problems=5,
    )

    scores = bundle["benchmark"]["scores"]
    assert scores["pass@1"]["value"] == 0.0
