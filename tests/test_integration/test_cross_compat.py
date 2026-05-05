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
"""Cross-compatibility matrix tests.

Verify that every agentic benchmark can work with every solver type,
and every solver type can drive every lifecycle strategy.
These tests use mocks -- no real Docker, no real models.
"""

from __future__ import annotations

import inspect

import pytest

from nemo_evaluator.environments.base import SeedResult
from nemo_evaluator.sandbox.base import SandboxSpec
from nemo_evaluator.sandbox.strategies import (
    LifecycleContext,
    NoSandbox,
    StatefulSandbox,
    StatelessSandbox,
    pick_lifecycle,
)
from tests.conftest import (
    FIXTURE_DIR,
    CachedSolver,
    MockSandboxManager,
)

# ---------------------------------------------------------------------------
# Environment types × Solver types (protocol-level compatibility)
# ---------------------------------------------------------------------------

SOLVER_CLASSES = []


def _load_solver_classes():
    """Dynamically import all known solver classes."""
    global SOLVER_CLASSES
    pairs = []
    try:
        from nemo_evaluator.solvers.chat import ChatSolver

        pairs.append(("ChatSolver", ChatSolver))
    except ImportError:
        pass
    try:
        from nemo_evaluator.solvers.nat import NatSolver

        pairs.append(("NatSolver", NatSolver))
    except ImportError:
        pass
    try:
        from nemo_evaluator.solvers.openclaw import OpenClawSolver

        pairs.append(("OpenClawSolver", OpenClawSolver))
    except ImportError:
        pass
    SOLVER_CLASSES = pairs


_load_solver_classes()


class TestSolverProtocol:
    @pytest.mark.parametrize("name,cls", SOLVER_CLASSES)
    def test_solve_method_exists(self, name, cls):
        sig = inspect.signature(cls.solve)
        assert "task" in sig.parameters or "self" in sig.parameters

    @pytest.mark.parametrize("name,cls", SOLVER_CLASSES)
    def test_sandbox_acceptance(self, name, cls):
        sig = inspect.signature(cls.solve)
        accepts_sandbox = "sandbox" in sig.parameters
        if name in ("OpenClawSolver",):
            assert accepts_sandbox, f"{name} should accept sandbox"
        elif name == "ChatSolver":
            assert not accepts_sandbox, "ChatSolver should not accept sandbox"


# ---------------------------------------------------------------------------
# Solver × Lifecycle strategy combinations
# ---------------------------------------------------------------------------

LIFECYCLE_NAMES = ["NoSandbox", "StatefulSandbox", "StatelessSandbox"]


class TestSolverLifecycleCombos:
    def _make_seed(self, needs_sandbox: bool, two_container: bool = False):
        kw = dict(prompt="test", expected_answer="42")
        if needs_sandbox:
            kw["sandbox_spec"] = SandboxSpec(image="python:3.12-slim")
        if two_container:
            kw["sandbox_spec"] = SandboxSpec(image="agent:latest")
            kw["verify_sandbox_spec"] = SandboxSpec(image="verify:latest")
            kw["capture_cmd"] = "cp -r /workspace /output/"
            kw["apply_cmd"] = "cp -r /input/workspace /workspace/"
        return SeedResult(**kw)

    @pytest.mark.asyncio
    async def test_chat_solver_no_sandbox(self):
        """ChatSolver + NoSandbox: common case for text-only benchmarks."""
        self._make_seed(needs_sandbox=False)
        lc = NoSandbox()
        await lc.setup()

        # ChatSolver doesn't need sandbox
        agent_sb = await lc.get_agent_sandbox()
        assert agent_sb is None

        await lc.transition_to_verify("The answer is 42", solver_modified=False)
        verify_sb = await lc.get_verify_sandbox()
        assert verify_sb is None
        await lc.teardown()

    @pytest.mark.asyncio
    async def test_chat_solver_stateful(self):
        """ChatSolver + StatefulSandbox: code execution benchmarks."""
        mgr = MockSandboxManager()
        spec = SandboxSpec(image="python:3.12-slim")
        lc = StatefulSandbox(mgr, spec)
        await lc.setup()

        # ChatSolver doesn't call get_agent_sandbox, but verify still gets one
        await lc.transition_to_verify("print('hello')", solver_modified=False)
        verify_sb = await lc.get_verify_sandbox()
        assert verify_sb is not None
        await lc.teardown()

    @pytest.mark.asyncio
    async def test_harbor_solver_stateful(self):
        """HarborSolver + StatefulSandbox: shared container."""
        mgr = MockSandboxManager()
        spec = SandboxSpec(image="python:3.12-slim")
        lc = StatefulSandbox(mgr, spec)
        await lc.setup()

        agent_sb = await lc.get_agent_sandbox()
        assert agent_sb is not None
        assert agent_sb.is_running

        await lc.transition_to_verify("patch applied", solver_modified=True)
        verify_sb = await lc.get_verify_sandbox()
        assert verify_sb is agent_sb  # Same sandbox
        await lc.teardown()

    @pytest.mark.asyncio
    async def test_harbor_solver_stateless(self):
        """HarborSolver + StatelessSandbox: two-container model."""
        mgr = MockSandboxManager()
        ctx = LifecycleContext(
            sandbox_mgr=mgr,
            agent_spec=SandboxSpec(image="agent:latest"),
            verify_spec=SandboxSpec(image="verify:latest"),
            capture_cmd="git diff > /output/patch.diff",
            apply_cmd="git apply /input/patch.diff",
        )
        from nemo_evaluator.sandbox.transfer import HostVolumeTransfer

        lc = StatelessSandbox(ctx, transfer=HostVolumeTransfer())
        await lc.setup()

        agent_sb = await lc.get_agent_sandbox()
        assert agent_sb is not None

        await lc.transition_to_verify("diff content", solver_modified=True)
        verify_sb = await lc.get_verify_sandbox()
        assert verify_sb is not None
        assert verify_sb is not agent_sb  # Different sandboxes
        await lc.teardown()

    @pytest.mark.asyncio
    async def test_pick_lifecycle_routes_correctly(self):
        """pick_lifecycle selects the right strategy based on seed fields."""
        mgr = MockSandboxManager()

        # text-only → NoSandbox
        lc1 = pick_lifecycle(self._make_seed(False), None)
        assert isinstance(lc1, NoSandbox)

        # sandbox_spec only → StatefulSandbox
        lc2 = pick_lifecycle(self._make_seed(True), mgr)
        assert isinstance(lc2, StatefulSandbox)

        # verify_sandbox_spec → StatelessSandbox
        lc3 = pick_lifecycle(self._make_seed(False, two_container=True), mgr)
        assert isinstance(lc3, StatelessSandbox)


# ---------------------------------------------------------------------------
# Full E2E: cached solver × fixture benchmarks × lifecycle strategies
# ---------------------------------------------------------------------------

_TEXT_ONLY_FIXTURES = ["mmlu", "gpqa", "gsm8k", "math500", "drop", "triviaqa", "simpleqa"]
_SANDBOX_FIXTURES = ["humaneval"]

_AVAILABLE_TEXT_FIXTURES = [b for b in _TEXT_ONLY_FIXTURES if (FIXTURE_DIR / f"{b}.json").exists()]
_AVAILABLE_SANDBOX_FIXTURES = [b for b in _SANDBOX_FIXTURES if (FIXTURE_DIR / f"{b}.json").exists()]


@pytest.mark.parametrize("bench", _AVAILABLE_TEXT_FIXTURES)
@pytest.mark.asyncio
async def test_text_benchmark_full_loop(bench: str):
    """Text-only benchmark through full eval loop with NoSandbox lifecycle."""
    from nemo_evaluator.engine.eval_loop import run_evaluation

    fp = FIXTURE_DIR / f"{bench}.json"
    from tests.conftest import FixturedEnvironment

    env = FixturedEnvironment(fp)
    solver = CachedSolver(fp)

    bundle = await run_evaluation(
        env=env,
        solver=solver,
        n_repeats=1,
        max_problems=5,
        max_concurrent=2,
    )
    assert bundle["benchmark"]["scores"]["summary"]["n"] == 5


@pytest.mark.parametrize("bench", _AVAILABLE_SANDBOX_FIXTURES)
@pytest.mark.asyncio
async def test_sandbox_benchmark_with_mock_manager(bench: str):
    """Sandbox-requiring benchmark through eval loop with mock sandbox."""
    from nemo_evaluator.engine.eval_loop import run_evaluation
    from tests.conftest import FixturedEnvironment

    fp = FIXTURE_DIR / f"{bench}.json"
    env = FixturedEnvironment(fp)
    solver = CachedSolver(fp)
    mgr = MockSandboxManager()

    bundle = await run_evaluation(
        env=env,
        solver=solver,
        n_repeats=1,
        max_problems=5,
        max_concurrent=2,
        sandbox_manager=mgr,
    )
    assert bundle["benchmark"]["scores"]["summary"]["n"] == 5
