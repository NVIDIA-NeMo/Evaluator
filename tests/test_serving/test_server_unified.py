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
"""Tests for unified verify endpoint -- accepts both evaluator and gym formats."""

import pytest
from fastapi.testclient import TestClient

from nemo_evaluator.environments.base import EvalEnvironment, SeedResult, VerifyResult
from nemo_evaluator.serving.app import generate_app


class _TestBench(EvalEnvironment):
    name = "test_unified"

    def __init__(self):
        super().__init__()
        self._dataset = [{"q": "2+2", "a": "4"}]

    async def seed(self, idx):
        return SeedResult(prompt="2+2", expected_answer="4")

    async def verify(self, response, expected, **meta):
        return VerifyResult(
            reward=1.0 if response.strip() == expected.strip() else 0.0,
            extracted_answer=response.strip(),
            scoring_details={"method": "exact"},
        )


class TestUnifiedVerify:
    @pytest.fixture
    def client_eval_mode(self):
        app = generate_app(_TestBench(), gym_compat=False)
        return TestClient(app)

    @pytest.fixture
    def client_gym_mode(self):
        app = generate_app(_TestBench(), gym_compat=True)
        return TestClient(app)

    def test_evaluator_format_in_eval_mode(self, client_eval_mode):
        r = client_eval_mode.post("/verify", json={"response": "4", "expected": "4", "metadata": {}})
        assert r.status_code == 200
        assert r.json()["reward"] == 1.0
        assert r.json()["scoring_details"]["method"] == "exact"

    def test_evaluator_format_in_gym_mode(self, client_gym_mode):
        """GymEnvironment sends evaluator format -- should work in gym mode too."""
        r = client_gym_mode.post("/verify", json={"response": "4", "expected": "4", "metadata": {}})
        assert r.status_code == 200
        assert r.json()["reward"] == 1.0

    def test_gym_format_in_gym_mode(self, client_gym_mode):
        """Gym sends its native format with expected_answer."""
        r = client_gym_mode.post("/verify", json={"response": "4", "expected_answer": "4", "metadata": {}})
        assert r.status_code == 200
        assert r.json()["reward"] == 1.0

    def test_gym_format_in_eval_mode(self, client_eval_mode):
        """Gym format should also work in evaluator mode."""
        r = client_eval_mode.post("/verify", json={"response": "4", "expected_answer": "4", "metadata": {}})
        assert r.status_code == 200
        assert r.json()["reward"] == 1.0

    def test_wrong_answer(self, client_eval_mode):
        r = client_eval_mode.post("/verify", json={"response": "5", "expected": "4"})
        assert r.status_code == 200
        assert r.json()["reward"] == 0.0

    def test_seed_includes_messages_system(self, client_eval_mode):
        r = client_eval_mode.post("/seed_session", json={"idx": 0})
        assert r.status_code == 200
        d = r.json()
        assert "messages" in d
        assert "system" in d
