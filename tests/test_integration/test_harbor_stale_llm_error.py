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
"""Harbor timeout behavior when an earlier LLM retry marker is stale."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from nemo_evaluator.engine.eval_loop import run_evaluation
from nemo_evaluator.environments.base import EvalEnvironment, SeedResult, VerifyResult
from nemo_evaluator.sandbox.base import SandboxSpec
from nemo_evaluator.solvers.harbor import HarborSolver
from tests.conftest import MockSandbox, MockSandboxManager


@dataclass(frozen=True)
class _AgentOutcome:
    timed_out: bool
    prompt_tokens: int
    completion_tokens: int
    state_verified: bool
    response: str = ""
    marker_age_seconds: float | None = None
    marker_successful_tokens: int = 0


class _StateSandbox(MockSandbox):
    def __init__(self, image: str) -> None:
        super().__init__(image=image)
        self.state_verified = False


class _StateSandboxManager(MockSandboxManager):
    async def acquire(self, spec, outside_endpoints=None) -> _StateSandbox:
        sandbox = _StateSandbox(spec.image)
        await sandbox.start(outside_endpoints=outside_endpoints)
        self._acquired.append(sandbox)
        return sandbox


class _MountedAdapter:
    is_mounted = True

    def __init__(self, sandbox: _StateSandbox, *args: Any, **kwargs: Any) -> None:
        self.sandbox = sandbox


class _AgentContext:
    def __init__(self) -> None:
        self.metadata: dict[str, Any] = {}
        self.rollout_details: list[Any] = []
        self.n_input_tokens = 0
        self.n_output_tokens = 0

    def is_empty(self) -> bool:
        return not (self.metadata or self.rollout_details or self.n_input_tokens or self.n_output_tokens)


class _ControlledAgent:
    def __init__(self, outcome: _AgentOutcome, logs_dir: Path) -> None:
        self.outcome = outcome
        self.logs_dir = logs_dir
        self.ready = asyncio.Event()
        self.marker_timestamp: float | None = None

    async def setup(self, adapter: _MountedAdapter) -> None:
        if self.outcome.marker_age_seconds is not None:
            marker = self.logs_dir / "last_llm_error.json"
            self.marker_timestamp = time.time() - self.outcome.marker_age_seconds
            marker.write_text(
                json.dumps(
                    {
                        "etype": "RateLimitError",
                        "emsg": "HTTP 429 from an earlier retry",
                        "written_at": self.marker_timestamp,
                        "request_timeout_seconds": 60,
                        "retry_after_seconds": 0,
                        "successful_tokens": self.outcome.marker_successful_tokens,
                    }
                )
            )
            os.utime(marker, (self.marker_timestamp, self.marker_timestamp))

    async def run(self, prompt: str, adapter: _MountedAdapter, context: _AgentContext) -> None:
        context.n_input_tokens = self.outcome.prompt_tokens
        context.n_output_tokens = self.outcome.completion_tokens
        if self.outcome.response:
            context.metadata["response"] = self.outcome.response
        adapter.sandbox.state_verified = self.outcome.state_verified
        self.ready.set()
        if self.outcome.timed_out:
            await asyncio.Event().wait()


class _AgentScenario:
    def __init__(self, outcomes: list[_AgentOutcome]) -> None:
        self.outcomes = outcomes
        self.attempts = 0
        self.current: _ControlledAgent | None = None

    def make_agent(self, logs_dir: Path, *, model_url: str = "") -> _ControlledAgent:
        outcome = self.outcomes[self.attempts]
        self.attempts += 1
        self.current = _ControlledAgent(outcome, logs_dir)
        return self.current


class _StateVerifiedEnvironment(EvalEnvironment):
    name = "harbor://state-verified-stale-marker"

    def __init__(self) -> None:
        super().__init__()
        self.dataset = [{"task_id": "stale-marker"}]
        self.verify_calls = 0

    async def seed(self, idx: int) -> SeedResult:
        return SeedResult(
            prompt="mutate verifier-visible container state",
            expected_answer="state complete",
            metadata={"task_id": self.dataset[idx]["task_id"]},
            sandbox_spec=SandboxSpec(image="stateful-test-sandbox"),
        )

    async def verify(
        self,
        response: str,
        expected: str,
        sandbox: _StateSandbox | None = None,
        **metadata: Any,
    ) -> VerifyResult:
        self.verify_calls += 1
        return VerifyResult(
            reward=1.0 if sandbox is not None and sandbox.state_verified else 0.0,
            scoring_details={"method": "state-verifier"},
        )


def _solver_for(scenario: _AgentScenario) -> HarborSolver:
    solver = HarborSolver(
        harbor_agent="terminus-2",
        model_url="http://model.invalid/v1",
        model_id="test-model",
        api_key="test-only",
        timeout=1800,
        run_timeout=60,
    )
    solver._create_agent = scenario.make_agent

    async def compressed_wait(agent_task, sandbox, agent_started_at, effective_timeout, jitter):
        agent = scenario.current
        assert agent is not None
        if agent.outcome.timed_out:
            await asyncio.wait_for(agent.ready.wait(), timeout=1)
            agent_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await agent_task
            return True, None
        await agent_task
        return False, None

    solver._wait_for_agent = compressed_wait
    return solver


async def _run_scenario(monkeypatch: pytest.MonkeyPatch, outcomes: list[_AgentOutcome]):
    import harbor.models.agent.context as context_mod
    import nemo_evaluator.solvers.harbor as harbor_mod
    import nemo_evaluator.solvers.harbor_adapter as adapter_mod

    monkeypatch.setattr(context_mod, "AgentContext", _AgentContext)
    monkeypatch.setattr(adapter_mod, "SandboxEnvironmentAdapter", _MountedAdapter)
    monkeypatch.setattr(harbor_mod, "_capture_workspace_diff", lambda sandbox: asyncio.sleep(0, result=""))

    scenario = _AgentScenario(outcomes)
    environment = _StateVerifiedEnvironment()
    bundle = await run_evaluation(
        environment,
        _solver_for(scenario),
        n_repeats=1,
        max_concurrent=1,
        max_system_retries=2,
        sandbox_manager=_StateSandboxManager(),
    )
    return scenario, environment, bundle["_results"][0]


@pytest.mark.asyncio
async def test_stale_retry_marker_does_not_skip_state_verification(monkeypatch):
    scenario, environment, result = await _run_scenario(
        monkeypatch,
        [
            _AgentOutcome(
                timed_out=True,
                prompt_tokens=23,
                completion_tokens=7,
                state_verified=True,
                marker_age_seconds=2 * 60,
            )
        ],
    )

    assert scenario.attempts == 1
    assert environment.verify_calls == 1
    assert result["reward"] == 1.0
    assert result["scoring_details"]["method"] == "state-verifier"


@pytest.mark.asyncio
async def test_recent_retry_marker_superseded_by_success_does_not_skip_state_verification(monkeypatch):
    scenario, environment, result = await _run_scenario(
        monkeypatch,
        [
            _AgentOutcome(
                timed_out=True,
                prompt_tokens=23,
                completion_tokens=7,
                state_verified=True,
                marker_age_seconds=0,
            )
        ],
    )

    assert scenario.attempts == 1
    assert environment.verify_calls == 1
    assert result["reward"] == 1.0
    assert result["scoring_details"]["method"] == "state-verifier"


@pytest.mark.asyncio
async def test_recent_retry_marker_without_later_success_remains_current(monkeypatch):
    scenario, environment, result = await _run_scenario(
        monkeypatch,
        [
            _AgentOutcome(
                timed_out=True,
                prompt_tokens=23,
                completion_tokens=7,
                state_verified=True,
                marker_age_seconds=0,
                marker_successful_tokens=30,
            )
        ],
    )

    assert scenario.attempts == 1
    assert environment.verify_calls == 0
    assert result["reward"] == 0.0
    assert result["scoring_details"]["method"] == "solve_failed"
    assert result["scoring_details"]["error_category"] == "rate_limit"


@pytest.mark.asyncio
async def test_stale_retry_marker_does_not_block_zero_progress_infra_retry(monkeypatch):
    scenario, environment, result = await _run_scenario(
        monkeypatch,
        [
            _AgentOutcome(
                timed_out=True,
                prompt_tokens=0,
                completion_tokens=0,
                state_verified=False,
                marker_age_seconds=2 * 60,
            ),
            _AgentOutcome(
                timed_out=False,
                prompt_tokens=5,
                completion_tokens=2,
                state_verified=True,
                response="retry completed",
            ),
        ],
    )

    assert scenario.attempts == 2
    assert environment.verify_calls == 1
    assert result["reward"] == 1.0
    assert result["scoring_details"]["method"] == "state-verifier"
