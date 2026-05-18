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
"""Offline tests for ToolSandboxEnvironment (no Docker, no network)."""

from __future__ import annotations

import json
import pytest

from nemo_evaluator.benchmarks.toolsandbox import ToolSandboxEnvironment, _to_openai_base_url


# ---------------------------------------------------------------------------
# URL normalization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url, expected",
    [
        (
            "https://integrate.api.nvidia.com/v1/chat/completions",
            "https://integrate.api.nvidia.com/v1",
        ),
        (
            "https://integrate.api.nvidia.com/v1/completions",
            "https://integrate.api.nvidia.com/v1",
        ),
        (
            "https://integrate.api.nvidia.com/v1/responses",
            "https://integrate.api.nvidia.com/v1",
        ),
        (
            "https://integrate.api.nvidia.com/v1",
            "https://integrate.api.nvidia.com/v1",
        ),
        (
            "http://localhost:8000/v1/chat/completions",
            "http://localhost:8000/v1",
        ),
        (
            "http://localhost:8000/v1",
            "http://localhost:8000/v1",
        ),
    ],
    ids=[
        "strip_chat_completions",
        "strip_completions",
        "strip_responses",
        "no_op",
        "localhost_with_suffix",
        "localhost_base",
    ],
)
def test_to_openai_base_url(url: str, expected: str) -> None:
    assert _to_openai_base_url(url) == expected


# ---------------------------------------------------------------------------
# ToolSandboxEnvironment construction and defaults
# ---------------------------------------------------------------------------


def test_default_construction() -> None:
    env = ToolSandboxEnvironment()
    assert env._image == "toolsandbox-nel:latest"
    assert env._user_model == "meta/llama-3.1-70b-instruct"
    assert env._scenarios == []
    assert env._parallel == 4
    assert not env._test_mode


def test_custom_params() -> None:
    env = ToolSandboxEnvironment(
        image="toolsandbox-nel:v1.2",
        user_model="meta/llama-3.1-8b-instruct",
        scenarios=["wifi_off", "cellular_off"],
        parallel=8,
        test_mode=True,
    )
    assert env._image == "toolsandbox-nel:v1.2"
    assert env._scenarios == ["wifi_off", "cellular_off"]
    assert env._parallel == 8
    assert env._test_mode


# ---------------------------------------------------------------------------
# Docker command construction
# ---------------------------------------------------------------------------


def test_docker_cmd_no_scenarios() -> None:
    env = ToolSandboxEnvironment()
    cmd = env._build_docker_cmd(
        output_dir=__import__("pathlib").Path("/tmp/output"),
        base_url="https://integrate.api.nvidia.com/v1",
        model_id="nvidia/nemotron-3-super-120b-a12b",
        api_key="test-key",
    )
    assert "docker" in cmd
    assert "--agent" in cmd
    assert "Gorilla" in cmd
    assert "--user" in cmd
    assert "GPT_4_o_2024_05_13" in cmd
    assert "--scenarios" not in cmd
    assert "--test_mode" not in cmd
    assert "NVIDIA_AGENT_MODEL=nvidia/nemotron-3-super-120b-a12b" in " ".join(cmd)


def test_docker_cmd_with_scenarios() -> None:
    env = ToolSandboxEnvironment(scenarios=["wifi_off", "make_call"])
    cmd = env._build_docker_cmd(
        output_dir=__import__("pathlib").Path("/tmp/output"),
        base_url="https://integrate.api.nvidia.com/v1",
        model_id="test-model",
        api_key="key",
    )
    idx = cmd.index("--scenarios")
    assert cmd[idx + 1] == "wifi_off"
    assert cmd[idx + 2] == "make_call"


def test_docker_cmd_test_mode() -> None:
    env = ToolSandboxEnvironment(test_mode=True)
    cmd = env._build_docker_cmd(
        output_dir=__import__("pathlib").Path("/tmp/output"),
        base_url="https://integrate.api.nvidia.com/v1",
        model_id="test-model",
        api_key="key",
    )
    assert "--test_mode" in cmd
    assert "--scenarios" not in cmd


def test_docker_cmd_no_api_key() -> None:
    env = ToolSandboxEnvironment()
    cmd = env._build_docker_cmd(
        output_dir=__import__("pathlib").Path("/tmp/output"),
        base_url="https://integrate.api.nvidia.com/v1",
        model_id="test-model",
        api_key="",
    )
    cmd_str = " ".join(cmd)
    assert "NVIDIA_API_KEY" not in cmd_str


# ---------------------------------------------------------------------------
# Score extraction
# ---------------------------------------------------------------------------


def test_extract_scores_full() -> None:
    summary = {
        "similarity": 0.72,
        "turn_count": 4.3,
        "per_category": {
            "single_tool_call": {"similarity": 0.91, "count": 45},
            "multiple_tool_call": {"similarity": 0.61, "count": 20},
        },
    }
    scores = ToolSandboxEnvironment._extract_scores(summary)

    assert scores["similarity"]["value"] == pytest.approx(0.72, abs=1e-4)
    assert scores["turn_count"]["value"] == pytest.approx(4.3, abs=1e-4)
    assert "per_category/single_tool_call/similarity" in scores
    assert scores["per_category/single_tool_call/similarity"]["value"] == pytest.approx(0.91, abs=1e-4)
    assert "per_category/multiple_tool_call/similarity" in scores


def test_extract_scores_empty_summary() -> None:
    scores = ToolSandboxEnvironment._extract_scores({})
    assert scores == {}


def test_extract_scores_no_categories() -> None:
    summary = {"similarity": 0.5}
    scores = ToolSandboxEnvironment._extract_scores(summary)
    assert scores == {"similarity": {"value": 0.5}}


def test_extract_scores_category_missing_similarity() -> None:
    summary = {
        "similarity": 0.8,
        "per_category": {"single_tool_call": {"count": 10}},
    }
    scores = ToolSandboxEnvironment._extract_scores(summary)
    assert "per_category/single_tool_call/similarity" not in scores
    assert scores["similarity"]["value"] == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# Result summary loading (from temp directory)
# ---------------------------------------------------------------------------


def test_load_result_summary(tmp_path):
    env = ToolSandboxEnvironment()

    # Simulate ToolSandbox output structure
    run_dir = tmp_path / "agent_Gorilla_user_GPT_4_o_20240513_12345"
    run_dir.mkdir()
    summary_data = {"similarity": 0.65, "turn_count": 3.1}
    (run_dir / "result_summary.json").write_text(json.dumps(summary_data))

    result = env._load_result_summary(tmp_path)
    assert result == summary_data


def test_load_result_summary_missing(tmp_path) -> None:
    env = ToolSandboxEnvironment()
    result = env._load_result_summary(tmp_path)
    assert result == {}


# ---------------------------------------------------------------------------
# Bundle structure
# ---------------------------------------------------------------------------


def test_parse_results_bundle_keys(tmp_path) -> None:
    env = ToolSandboxEnvironment()

    run_dir = tmp_path / "run_1"
    run_dir.mkdir()
    (run_dir / "result_summary.json").write_text(
        json.dumps({"similarity": 0.7, "turn_count": 5.0})
    )

    bundle = env._parse_results(tmp_path, exit_code=0, model_id="my-model")

    assert "benchmark" in bundle
    assert "config" in bundle
    assert bundle["benchmark"]["name"] == "toolsandbox"
    assert bundle["benchmark"]["scores"]["similarity"]["value"] == pytest.approx(0.7)
    assert bundle["config"]["framework"] == "toolsandbox"
    assert bundle["config"]["model"] == "my-model"
    assert bundle["_container_exit_code"] == 0


# ---------------------------------------------------------------------------
# seed/verify raise NotImplementedError
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_seed_raises() -> None:
    env = ToolSandboxEnvironment()
    with pytest.raises(NotImplementedError):
        await env.seed(0)


@pytest.mark.asyncio
async def test_verify_raises() -> None:
    env = ToolSandboxEnvironment()
    with pytest.raises(NotImplementedError):
        await env.verify("response", "expected")
