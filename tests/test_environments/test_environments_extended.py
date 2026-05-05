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
"""Tests for evaluation environments — Harbor, Skills, LmEval, Composite, Container."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nemo_evaluator.environments.base import EvalEnvironment, SeedResult, VerifyResult

# ── CompositeEnvironment ─────────────────────────────────────────────────


class TestCompositeEnvironment:
    def _make_envs(self):
        seed_env = MagicMock(spec=EvalEnvironment)
        verify_env = MagicMock(spec=EvalEnvironment)
        seed_env.dataset_size = AsyncMock(return_value=3)
        seed_env.seed = AsyncMock(return_value=SeedResult(prompt="q", expected_answer="a", metadata={"idx": 0}))
        verify_env.verify = AsyncMock(return_value=VerifyResult(reward=1.0, extracted_answer="a"))
        seed_env.close = AsyncMock()
        verify_env.close = AsyncMock()
        return seed_env, verify_env

    async def test_seed_delegates(self):
        from nemo_evaluator.environments.composite import CompositeEnvironment

        seed_env, verify_env = self._make_envs()
        comp = CompositeEnvironment(seed_env, verify_env)
        result = await comp.seed(0)
        seed_env.seed.assert_called_once_with(0)
        assert result.prompt == "q"

    async def test_verify_delegates(self):
        from nemo_evaluator.environments.composite import CompositeEnvironment

        seed_env, verify_env = self._make_envs()
        comp = CompositeEnvironment(seed_env, verify_env)
        result = await comp.verify("a", "a")
        verify_env.verify.assert_called_once()
        assert result.reward == 1.0

    async def test_close_both(self):
        from nemo_evaluator.environments.composite import CompositeEnvironment

        seed_env, verify_env = self._make_envs()
        comp = CompositeEnvironment(seed_env, verify_env)
        await comp.close()
        seed_env.close.assert_called_once()
        verify_env.close.assert_called_once()


# ── ContainerEnvironment ────────────────────────────────────────────────


class TestContainerEnvironment:
    def test_seed_raises(self):
        from nemo_evaluator.environments.container import ContainerEnvironment

        env = ContainerEnvironment(image="test:latest")
        with pytest.raises(NotImplementedError):
            import asyncio

            asyncio.run(env.seed(0))

    def test_verify_raises(self):
        from nemo_evaluator.environments.container import ContainerEnvironment

        env = ContainerEnvironment(image="test:latest")
        with pytest.raises(NotImplementedError):
            import asyncio

            asyncio.run(env.verify("a", "b"))


# ── SkillsEnvironment ───────────────────────────────────────────────────


class TestSkillsEnvironment:
    @patch("nemo_evaluator.environments.skills.SkillsEnvironment.__init__", return_value=None)
    def test_instantiation(self, mock_init):
        from nemo_evaluator.environments.skills import SkillsEnvironment

        SkillsEnvironment(benchmark="gsm8k")
        mock_init.assert_called_once_with(benchmark="gsm8k")


# ── HarborEnvironment ───────────────────────────────────────────────────


class TestHarborEnvironment:
    @patch("nemo_evaluator.environments.harbor.HarborEnvironment.__init__", return_value=None)
    def test_instantiation(self, mock_init):
        from nemo_evaluator.environments.harbor import HarborEnvironment

        HarborEnvironment(dataset_path="/tmp/test.json")
        mock_init.assert_called_once()


# ── Base EvalEnvironment ────────────────────────────────────────────────


class TestEvalEnvironmentBase:
    def test_dataset_property(self):
        class ConcreteEnv(EvalEnvironment):
            async def seed(self, idx):
                return SeedResult(prompt="", expected_answer="")

            async def verify(self, response, expected, sandbox=None, **meta):
                return VerifyResult(reward=0.0)

        env = ConcreteEnv()
        assert env.dataset == []
        env.dataset = [{"a": 1}]
        assert len(env) == 1

    async def test_default_sandbox_specs_none(self):
        class ConcreteEnv(EvalEnvironment):
            async def seed(self, idx):
                return SeedResult(prompt="", expected_answer="")

            async def verify(self, response, expected, sandbox=None, **meta):
                return VerifyResult(reward=0.0)

        env = ConcreteEnv()
        assert await env.sandbox_specs() is None

    async def test_default_run_batch_none(self):
        class ConcreteEnv(EvalEnvironment):
            async def seed(self, idx):
                return SeedResult(prompt="", expected_answer="")

            async def verify(self, response, expected, sandbox=None, **meta):
                return VerifyResult(reward=0.0)

        env = ConcreteEnv()
        assert await env.run_batch() is None

    async def test_default_image_build_requests_none(self):
        class ConcreteEnv(EvalEnvironment):
            async def seed(self, idx):
                return SeedResult(prompt="", expected_answer="")

            async def verify(self, response, expected, sandbox=None, **meta):
                return VerifyResult(reward=0.0)

        env = ConcreteEnv()
        assert await env.image_build_requests() is None
