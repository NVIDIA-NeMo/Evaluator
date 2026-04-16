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
"""Tests for sandbox lifecycle strategies: NoSandbox, StatefulSandbox, StatelessSandbox."""

from __future__ import annotations

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
from nemo_evaluator.sandbox.transfer import HostVolumeTransfer
from tests.conftest import MockSandboxManager


def _make_seed(**overrides) -> SeedResult:
    defaults = dict(prompt="test", expected_answer="42")
    defaults.update(overrides)
    return SeedResult(**defaults)


# ---------------------------------------------------------------------------
# NoSandbox
# ---------------------------------------------------------------------------


class TestNoSandbox:
    @pytest.mark.asyncio
    async def test_all_methods_return_none(self):
        lc = NoSandbox()
        await lc.setup()
        assert await lc.get_agent_sandbox() is None
        await lc.transition_to_verify("response", solver_modified=False)
        assert await lc.get_verify_sandbox() is None
        await lc.teardown()


# ---------------------------------------------------------------------------
# StatefulSandbox
# ---------------------------------------------------------------------------


class TestStatefulSandbox:
    @pytest.mark.asyncio
    async def test_lazy_acquisition(self):
        """Sandbox is only acquired on first get_agent_sandbox or get_verify_sandbox."""
        mgr = MockSandboxManager()
        spec = SandboxSpec(image="python:3.12-slim")
        lc = StatefulSandbox(mgr, spec)

        await lc.setup()
        assert len(mgr._acquired) == 0

        sb = await lc.get_agent_sandbox()
        assert sb is not None
        assert len(mgr._acquired) == 1

    @pytest.mark.asyncio
    async def test_same_sandbox_for_both_phases(self):
        """Agent and verify get the same sandbox instance."""
        mgr = MockSandboxManager()
        spec = SandboxSpec(image="python:3.12-slim")
        lc = StatefulSandbox(mgr, spec)

        await lc.setup()
        agent_sb = await lc.get_agent_sandbox()
        verify_sb = await lc.get_verify_sandbox()
        assert agent_sb is verify_sb
        assert len(mgr._acquired) == 1

    @pytest.mark.asyncio
    async def test_verify_without_agent(self):
        """ChatSolver skips get_agent_sandbox; verify still gets a sandbox."""
        mgr = MockSandboxManager()
        spec = SandboxSpec(image="python:3.12-slim")
        lc = StatefulSandbox(mgr, spec)

        await lc.setup()
        await lc.transition_to_verify("response", solver_modified=False)
        verify_sb = await lc.get_verify_sandbox()
        assert verify_sb is not None
        assert len(mgr._acquired) == 1

    @pytest.mark.asyncio
    async def test_teardown_releases(self):
        mgr = MockSandboxManager()
        spec = SandboxSpec(image="python:3.12-slim")
        lc = StatefulSandbox(mgr, spec)

        await lc.setup()
        sb = await lc.get_agent_sandbox()
        assert sb.is_running
        await lc.teardown()
        assert not sb.is_running


# ---------------------------------------------------------------------------
# StatelessSandbox
# ---------------------------------------------------------------------------


class TestStatelessSandbox:
    def _make_ctx(self, mgr=None, capture_cmd=None, apply_cmd=None):
        mgr = mgr or MockSandboxManager()
        return LifecycleContext(
            sandbox_mgr=mgr,
            agent_spec=SandboxSpec(image="agent:latest"),
            verify_spec=SandboxSpec(image="verify:latest"),
            capture_cmd=capture_cmd,
            apply_cmd=apply_cmd,
        ), mgr

    @pytest.mark.asyncio
    async def test_separate_containers(self):
        """Agent and verify get different sandbox instances."""
        ctx, mgr = self._make_ctx()
        transfer = HostVolumeTransfer()
        lc = StatelessSandbox(ctx, transfer)

        await lc.setup()
        agent_sb = await lc.get_agent_sandbox()
        await lc.transition_to_verify("diff content", solver_modified=True)
        verify_sb = await lc.get_verify_sandbox()

        assert agent_sb is not verify_sb
        assert len(mgr._acquired) == 2
        await lc.teardown()

    @pytest.mark.asyncio
    async def test_capture_cmd_executed(self):
        """capture_cmd is run in agent container during transition."""
        ctx, mgr = self._make_ctx(capture_cmd="git diff > /output/patch.diff")
        transfer = HostVolumeTransfer()
        lc = StatelessSandbox(ctx, transfer)

        await lc.setup()
        agent_sb = await lc.get_agent_sandbox()
        await lc.transition_to_verify("response", solver_modified=True)

        assert any("git diff" in cmd for cmd in agent_sb._exec_log)

    @pytest.mark.asyncio
    async def test_apply_cmd_executed(self):
        """apply_cmd is run in verify container after acquisition."""
        ctx, mgr = self._make_ctx(apply_cmd="git apply /input/patch.diff")
        transfer = HostVolumeTransfer()
        lc = StatelessSandbox(ctx, transfer)

        await lc.setup()
        await lc.get_agent_sandbox()
        await lc.transition_to_verify("response", solver_modified=False)
        verify_sb = await lc.get_verify_sandbox()

        assert any("git apply" in cmd for cmd in verify_sb._exec_log)
        await lc.teardown()

    @pytest.mark.asyncio
    async def test_teardown_idempotent(self):
        """Multiple teardown calls are safe (idempotent via _torn_down flag)."""
        ctx, mgr = self._make_ctx()
        transfer = HostVolumeTransfer()
        lc = StatelessSandbox(ctx, transfer)

        await lc.setup()
        await lc.get_agent_sandbox()
        await lc.teardown()
        await lc.teardown()  # should not raise

    @pytest.mark.asyncio
    async def test_transition_after_teardown_is_noop(self):
        """transition_to_verify after teardown is a no-op (torn_down guard)."""
        ctx, mgr = self._make_ctx(capture_cmd="echo capture")
        transfer = HostVolumeTransfer()
        lc = StatelessSandbox(ctx, transfer)

        await lc.setup()
        await lc.get_agent_sandbox()
        await lc.teardown()
        await lc.transition_to_verify("response", solver_modified=True)


# ---------------------------------------------------------------------------
# pick_lifecycle
# ---------------------------------------------------------------------------


class TestPickLifecycle:
    def test_no_sandbox_when_no_manager(self):
        seed = _make_seed()
        lc = pick_lifecycle(seed, sandbox_mgr=None)
        assert isinstance(lc, NoSandbox)

    def test_stateful_when_spec_resolvable(self):
        seed = _make_seed(sandbox_spec=SandboxSpec(image="python:3.12-slim"))
        mgr = MockSandboxManager()
        lc = pick_lifecycle(seed, sandbox_mgr=mgr)
        assert isinstance(lc, StatefulSandbox)

    def test_stateless_when_verify_spec_set(self):
        seed = _make_seed(
            sandbox_spec=SandboxSpec(image="agent:latest"),
            verify_sandbox_spec=SandboxSpec(image="verify:latest"),
            capture_cmd="git diff > /output/patch.diff",
            apply_cmd="git apply /input/patch.diff",
        )
        mgr = MockSandboxManager()
        lc = pick_lifecycle(seed, sandbox_mgr=mgr)
        assert isinstance(lc, StatelessSandbox)

    def test_no_sandbox_when_spec_unresolvable(self):
        seed = _make_seed()
        mgr = MockSandboxManager()
        lc = pick_lifecycle(seed, sandbox_mgr=mgr)
        assert isinstance(lc, NoSandbox)

    def test_force_stateful_overrides_verify_spec(self):
        """force_stateful=True routes to StatefulSandbox even when verify_sandbox_spec is set."""
        seed = _make_seed(
            sandbox_spec=SandboxSpec(image="agent:latest"),
            verify_sandbox_spec=SandboxSpec(image="verify:latest"),
            capture_cmd="git diff > /output/patch.diff",
            apply_cmd="git apply /input/patch.diff",
        )
        mgr = MockSandboxManager()
        lc = pick_lifecycle(seed, sandbox_mgr=mgr, force_stateful=True)
        assert isinstance(lc, StatefulSandbox)

    def test_force_stateful_false_keeps_stateless_with_verify_spec(self):
        """Default behaviour unchanged: verify_sandbox_spec present → StatelessSandbox."""
        seed = _make_seed(
            sandbox_spec=SandboxSpec(image="agent:latest"),
            verify_sandbox_spec=SandboxSpec(image="verify:latest"),
            capture_cmd="git diff > /output/patch.diff",
            apply_cmd="git apply /input/patch.diff",
        )
        mgr = MockSandboxManager()
        lc = pick_lifecycle(seed, sandbox_mgr=mgr, force_stateful=False)
        assert isinstance(lc, StatelessSandbox)
