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
"""Shared test fixtures: mock sandbox, cached solver, fixtured environment."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures"

# ---------------------------------------------------------------------------
# Sandbox mocks
# ---------------------------------------------------------------------------


@dataclass
class MockExecResult:
    stdout: str = ""
    stderr: str = ""
    return_code: int = 0


class MockSandbox:
    """In-memory sandbox implementing the Sandbox protocol."""

    def __init__(self, image: str = "python:3.12-slim", exec_results: dict[str, MockExecResult] | None = None):
        from nemo_evaluator.sandbox.base import SandboxSpec

        self._spec = SandboxSpec(image=image)
        self._running = False
        self._exec_results = exec_results or {}
        self._exec_log: list[str] = []
        self._files: dict[str, bytes] = {}
        self._container_ip = "172.17.0.2"
        self._outside_endpoints: list = []

    @property
    def spec(self):
        return self._spec

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def container_ip(self) -> str | None:
        return self._container_ip if self._running else None

    async def start(self, *, outside_endpoints=None) -> None:
        self._outside_endpoints = outside_endpoints or []
        self._running = True

    async def stop(self) -> None:
        self._running = False

    async def exec(
        self,
        command: str,
        timeout_sec: float = 180,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        user: str | int | None = None,
    ) -> MockExecResult:
        self._exec_log.append(command)
        for pattern, result in self._exec_results.items():
            if pattern in command:
                return result
        return MockExecResult()

    async def upload(self, local_path, remote_path: str) -> None:
        self._files[remote_path] = Path(local_path).read_bytes() if Path(local_path).exists() else b""

    async def download(self, remote_path: str, local_path) -> None:
        Path(local_path).write_bytes(self._files.get(remote_path, b""))

    def resolve_outside_endpoint(self, url: str) -> str:
        return url

    def resolved_endpoint_url(self, env_var: str) -> str | None:
        for ep in self._outside_endpoints:
            if ep.env_var == env_var:
                return ep.url
        return None

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, *exc):
        await self.stop()


class MockSandboxManager:
    """In-memory sandbox manager that returns MockSandbox instances."""

    def __init__(self, exec_results: dict[str, MockExecResult] | None = None):
        self._exec_results = exec_results or {}
        self._acquired: list[MockSandbox] = []
        self._pulled: set[str] = set()

    def resolve_spec(self, seed, extra_volumes=None, base_override=None):
        return base_override or seed.sandbox_spec

    async def acquire(self, spec, outside_endpoints=None) -> MockSandbox:
        sb = MockSandbox(image=spec.image, exec_results=self._exec_results)
        await sb.start()
        self._acquired.append(sb)
        return sb

    async def release(self, sandbox) -> None:
        await sandbox.stop()

    async def _ensure_pulled(self, image: str) -> None:
        self._pulled.add(image)

    async def pre_pull(self, images_or_specs) -> None:
        for item in images_or_specs:
            img = item.image if hasattr(item, "image") else str(item)
            await self._ensure_pulled(img)

    def get_transfer_strategy(self):
        from nemo_evaluator.sandbox.transfer import HostVolumeTransfer

        return HostVolumeTransfer()

    async def shutdown(self) -> None:
        for sb in self._acquired:
            if sb.is_running:
                await sb.stop()
        self._acquired.clear()


# ---------------------------------------------------------------------------
# Fixtured environment (reads from JSON fixtures, no HF download)
# ---------------------------------------------------------------------------


class FixturedEnvironment:
    """Replays seed/verify data from a fixture JSON file."""

    def __init__(self, fixture_path: Path):
        with open(fixture_path) as f:
            self._data = json.load(f)
        self.name = fixture_path.stem

    async def dataset_size(self) -> int:
        return len(self._data)

    def __len__(self) -> int:
        return len(self._data)

    async def prepare(self) -> None:
        pass

    async def image_build_requests(self) -> list:
        return []

    async def seed(self, idx: int):
        from nemo_evaluator.environments.base import SeedResult

        row = self._data[idx]
        return SeedResult(
            prompt=row["prompt"],
            expected_answer=row["expected_answer"],
            metadata=row.get("metadata", {}),
            messages=row.get("messages"),
            system=row.get("system"),
        )

    async def verify(self, response: str, expected: str, sandbox=None, **meta):
        from nemo_evaluator.environments.base import VerifyResult

        # Match by both response and expected_answer for exact replay
        for row in self._data:
            if row["expected_answer"] == expected and row["model_response"] == response:
                return VerifyResult(
                    reward=row["reward"],
                    extracted_answer=row.get("extracted_answer"),
                    scoring_details=row.get("scoring_details", {}),
                )
        # Fallback: match by expected_answer only
        for row in self._data:
            if row["expected_answer"] == expected:
                return VerifyResult(
                    reward=row["reward"],
                    extracted_answer=row.get("extracted_answer"),
                    scoring_details=row.get("scoring_details", {}),
                )
        return VerifyResult(reward=0.0)

    async def sandbox_specs(self):
        return None

    async def close(self):
        pass


# ---------------------------------------------------------------------------
# Cached solver (replays golden model responses from fixtures)
# ---------------------------------------------------------------------------


class CachedSolver:
    """Replays pre-recorded responses from a fixture file."""

    def __init__(self, fixture_path: Path):
        with open(fixture_path) as f:
            data = json.load(f)
        self._responses = {row["prompt"]: row["model_response"] for row in data}
        self._idx = 0
        self._ordered = [row["model_response"] for row in data]

    async def solve(self, task, sandbox=None):
        from nemo_evaluator.solvers.base import SolveResult

        text = self._responses.get(task.prompt) or self._ordered[self._idx % len(self._ordered)]
        self._idx += 1
        return SolveResult(response=text)

    async def close(self):
        pass


# ---------------------------------------------------------------------------
# Mock judge client
# ---------------------------------------------------------------------------


class MockJudgeClient:
    """Returns a canned judge score."""

    def __init__(self, default_score: float = 1.0):
        self.default_score = default_score

    async def chat(self, messages=None, prompt=None, system=None):
        from nemo_evaluator.observability.types import ModelResponse

        return ModelResponse(
            content=f"Score: {self.default_score}",
            total_tokens=50,
        )


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fixture_dir():
    return FIXTURE_DIR


@pytest.fixture
def mock_sandbox():
    return MockSandbox()


@pytest.fixture
def mock_sandbox_manager():
    return MockSandboxManager()


def load_fixture(name: str) -> list[dict[str, Any]]:
    """Load a named fixture file."""
    path = FIXTURE_DIR / f"{name}.json"
    if not path.exists():
        pytest.skip(f"Fixture {name}.json not found")
    with open(path) as f:
        return json.load(f)


AVAILABLE_FIXTURES = [p.stem for p in sorted(FIXTURE_DIR.glob("*.json"))] if FIXTURE_DIR.exists() else []
