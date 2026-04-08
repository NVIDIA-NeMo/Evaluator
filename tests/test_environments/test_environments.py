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
"""Tests for environment seed/verify using golden fixtures (offline)
and real HF dataset loading (network, marked)."""

from __future__ import annotations

import pytest

from tests.conftest import AVAILABLE_FIXTURES, FixturedEnvironment, load_fixture, FIXTURE_DIR


# ---------------------------------------------------------------------------
# Offline: seed/verify round-trip from fixtures
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bench", AVAILABLE_FIXTURES)
@pytest.mark.asyncio
async def test_fixture_seed_shape(bench: str):
    """seed() returns SeedResult with prompt and expected_answer."""
    env = FixturedEnvironment(FIXTURE_DIR / f"{bench}.json")
    size = await env.dataset_size()
    assert size > 0

    seed = await env.seed(0)
    assert seed.prompt, f"{bench}: empty prompt"
    assert seed.expected_answer is not None


@pytest.mark.parametrize("bench", AVAILABLE_FIXTURES)
@pytest.mark.asyncio
async def test_fixture_verify_with_golden_response(bench: str):
    """verify() with the golden model response reproduces the fixture reward."""
    data = load_fixture(bench)
    env = FixturedEnvironment(FIXTURE_DIR / f"{bench}.json")

    for row in data:
        if not row["model_response"]:
            continue
        vr = await env.verify(row["model_response"], row["expected_answer"])
        assert vr.reward == row["reward"], f"{bench}[{row['idx']}]: expected reward={row['reward']}, got {vr.reward}"


# ---------------------------------------------------------------------------
# Offline: verify scoring functions directly
# ---------------------------------------------------------------------------

SCORING_BENCHMARKS = {
    "mmlu": "nemo_evaluator.scoring.pattern",
    "gpqa": "nemo_evaluator.scoring.pattern",
    "gsm8k": "nemo_evaluator.scoring.pattern",
    "math500": "nemo_evaluator.scoring.pattern",
    "drop": "nemo_evaluator.scoring.pattern",
    "triviaqa": "nemo_evaluator.scoring.pattern",
}


@pytest.mark.network
@pytest.mark.parametrize("bench", [b for b in AVAILABLE_FIXTURES if b in SCORING_BENCHMARKS])
def test_scorer_determinism(bench: str):
    """Scoring the same response twice gives the same result."""
    from nemo_evaluator.scoring.types import ScorerInput
    import nemo_evaluator.benchmarks  # noqa: F401

    data = load_fixture(bench)
    row = data[0]
    if not row["model_response"]:
        pytest.skip("no model response")

    from nemo_evaluator.environments.registry import get_environment

    env = get_environment(bench, num_examples=5)

    sample = ScorerInput(
        response=row["model_response"],
        target=row["expected_answer"],
        metadata=row.get("metadata", {}),
    )
    scorer = env._scorer if hasattr(env, "_scorer") else None
    if scorer is None:
        pytest.skip(f"no scorer for {bench}")

    r1 = scorer(sample)
    r2 = scorer(sample)
    assert r1["correct"] == r2["correct"], f"Scorer not deterministic for {bench}"


# ---------------------------------------------------------------------------
# Network: real HF dataset loading (slow, CI-cacheable)
# ---------------------------------------------------------------------------

NETWORK_BENCHMARKS = ["mmlu", "gpqa", "gsm8k", "math500", "drop", "humaneval", "triviaqa"]


@pytest.mark.network
@pytest.mark.parametrize("bench", NETWORK_BENCHMARKS)
@pytest.mark.asyncio
async def test_real_dataset_loads(bench: str):
    """Loading from HF succeeds and returns at least 5 samples."""
    import nemo_evaluator.benchmarks  # noqa: F401
    from nemo_evaluator.environments.registry import get_environment

    env = get_environment(bench, num_examples=5)
    size = await env.dataset_size()
    assert size >= 5, f"{bench}: only {size} samples loaded"

    seed = await env.seed(0)
    assert seed.prompt
    assert seed.expected_answer is not None


@pytest.mark.network
@pytest.mark.asyncio
async def test_pinchbench_loads_from_github():
    """PinchBench downloads task definitions from GitHub."""
    import nemo_evaluator.benchmarks  # noqa: F401
    from nemo_evaluator.environments.registry import get_environment

    env = get_environment("pinchbench", num_examples=5)
    size = await env.dataset_size()
    assert size >= 5

    seed = await env.seed(0)
    assert seed.prompt
