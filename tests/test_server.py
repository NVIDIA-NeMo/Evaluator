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
import pytest
from fastapi.testclient import TestClient

from nemo_evaluator.environments.base import EvalEnvironment, SeedResult, VerifyResult
from nemo_evaluator.serving.app import generate_app


class DummyEnv(EvalEnvironment):
    name = "dummy"

    def __init__(self):
        super().__init__()
        self._dataset = [{"q": f"Question {i}", "a": str(i)} for i in range(5)]

    async def seed(self, idx: int) -> SeedResult:
        row = self._dataset[idx]
        return SeedResult(prompt=row["q"], expected_answer=row["a"])

    async def verify(self, response: str, expected: str, **meta) -> VerifyResult:
        return VerifyResult(
            reward=1.0 if response.strip() == expected else 0.0,
            extracted_answer=response.strip(),
            scoring_details={"method": "exact"},
        )


class TestServer:
    @pytest.fixture
    def client(self):
        app = generate_app(DummyEnv())
        return TestClient(app)

    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["dataset_size"] == 5

    def test_seed_valid(self, client):
        resp = client.post("/seed_session", json={"idx": 0})
        assert resp.status_code == 200
        assert resp.json()["prompt"] == "Question 0"
        assert resp.json()["expected_answer"] == "0"

    def test_seed_out_of_bounds(self, client):
        resp = client.post("/seed_session", json={"idx": 99})
        assert resp.status_code == 400

    def test_seed_negative(self, client):
        resp = client.post("/seed_session", json={"idx": -1})
        assert resp.status_code == 400

    def test_verify(self, client):
        resp = client.post("/verify", json={"response": "0", "expected": "0"})
        assert resp.status_code == 200
        assert resp.json()["reward"] == 1.0

    def test_verify_wrong(self, client):
        resp = client.post("/verify", json={"response": "wrong", "expected": "0"})
        assert resp.status_code == 200
        assert resp.json()["reward"] == 0.0

    def test_dataset_size(self, client):
        resp = client.get("/dataset_size")
        assert resp.json()["size"] == 5


class TestGymCompatServer:
    @pytest.fixture
    def client(self):
        app = generate_app(DummyEnv(), gym_compat=True)
        return TestClient(app)

    def test_gym_verify_with_string_response(self, client):
        resp = client.post(
            "/verify",
            json={
                "response": "0",
                "expected_answer": "0",
            },
        )
        assert resp.status_code == 200
        assert resp.json()["reward"] == 1.0

    def test_gym_verify_with_structured_response(self, client):
        resp = client.post(
            "/verify",
            json={
                "response": {"output": [{"type": "message", "role": "assistant", "content": [{"text": "0"}]}]},
                "expected_answer": "0",
            },
        )
        assert resp.status_code == 200
        assert resp.json()["reward"] == 1.0
