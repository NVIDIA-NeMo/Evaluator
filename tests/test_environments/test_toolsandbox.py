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
    assert env._runner == "docker"
    assert env._image == "toolsandbox-nel:latest"
    assert env._user_model == "meta/llama-3.1-70b-instruct"
    assert env._scenarios == []
    assert env._parallel == 4
    assert not env._test_mode


def test_custom_params() -> None:
    env = ToolSandboxEnvironment(
        runner="subprocess",
        image="toolsandbox-nel:v1.2",
        python_exe="/opt/toolsandbox-venv/bin/python",
        user_model="meta/llama-3.1-8b-instruct",
        scenarios=["wifi_off", "cellular_off"],
        parallel=8,
        test_mode=True,
    )
    assert env._runner == "subprocess"
    assert env._python_exe == "/opt/toolsandbox-venv/bin/python"
    assert env._scenarios == ["wifi_off", "cellular_off"]
    assert env._parallel == 8
    assert env._test_mode


# ---------------------------------------------------------------------------
# Docker command construction
# ---------------------------------------------------------------------------

_P = __import__("pathlib").Path


def test_docker_cmd_no_scenarios() -> None:
    env = ToolSandboxEnvironment(runner="docker")
    cmd, _ = env._build_cmd(
        output_dir=_P("/tmp/output"),
        base_url="https://integrate.api.nvidia.com/v1",
        model_id="nvidia/nemotron-3-super-120b-a12b",
        api_key="test-key",
    )
    assert cmd[0] == "docker"
    assert "--agent" in cmd
    assert "Gorilla" in cmd
    assert "--user" in cmd
    assert "GPT_4_o_2024_05_13" in cmd
    assert "--scenarios" not in cmd
    assert "--test_mode" not in cmd
    assert "NVIDIA_AGENT_MODEL=nvidia/nemotron-3-super-120b-a12b" in " ".join(cmd)


def test_docker_cmd_with_scenarios() -> None:
    env = ToolSandboxEnvironment(runner="docker", scenarios=["wifi_off", "make_call"])
    cmd, _ = env._build_cmd(
        output_dir=_P("/tmp/output"),
        base_url="https://integrate.api.nvidia.com/v1",
        model_id="test-model",
        api_key="key",
    )
    idx = cmd.index("--scenarios")
    assert cmd[idx + 1] == "wifi_off"
    assert cmd[idx + 2] == "make_call"


def test_docker_cmd_test_mode() -> None:
    env = ToolSandboxEnvironment(runner="docker", test_mode=True)
    cmd, _ = env._build_cmd(
        output_dir=_P("/tmp/output"),
        base_url="https://integrate.api.nvidia.com/v1",
        model_id="test-model",
        api_key="key",
    )
    assert "--test_mode" in cmd
    assert "--scenarios" not in cmd


def test_docker_cmd_no_api_key() -> None:
    env = ToolSandboxEnvironment(runner="docker")
    cmd, _ = env._build_cmd(
        output_dir=_P("/tmp/output"),
        base_url="https://integrate.api.nvidia.com/v1",
        model_id="test-model",
        api_key="",
    )
    assert "NVIDIA_API_KEY" not in " ".join(cmd)


# ---------------------------------------------------------------------------
# Apptainer command construction
# ---------------------------------------------------------------------------


def test_apptainer_cmd_basics() -> None:
    env = ToolSandboxEnvironment(runner="apptainer", image="/shared/nel/toolsandbox.sif")
    cmd, _ = env._build_cmd(
        output_dir=_P("/tmp/output"),
        base_url="https://integrate.api.nvidia.com/v1",
        model_id="test-model",
        api_key="key",
    )
    assert cmd[0] == "apptainer"
    assert "run" in cmd
    assert "--bind" in cmd
    assert "/shared/nel/toolsandbox.sif" in cmd
    assert "--agent" in cmd


# ---------------------------------------------------------------------------
# Subprocess command construction
# ---------------------------------------------------------------------------


def test_subprocess_cmd_basics() -> None:
    env = ToolSandboxEnvironment(
        runner="subprocess",
        python_exe="/opt/toolsandbox-venv/bin/python",
        entrypoint="/opt/toolsandbox_entrypoint.py",
    )
    cmd, env_vars = env._build_cmd(
        output_dir=_P("/tmp/output"),
        base_url="https://integrate.api.nvidia.com/v1",
        model_id="test-model",
        api_key="test-key",
    )
    assert cmd[0] == "/opt/toolsandbox-venv/bin/python"
    assert cmd[1] == "/opt/toolsandbox_entrypoint.py"
    assert "--agent" in cmd
    assert "Gorilla" in cmd
    # API config passed via environment, not CLI flags
    assert env_vars["NVIDIA_BASE_URL"] == "https://integrate.api.nvidia.com/v1"
    assert env_vars["NVIDIA_AGENT_MODEL"] == "test-model"
    assert env_vars["NVIDIA_API_KEY"] == "test-key"
    # output_dir is a local host path, not the container-mount path /output
    assert "/tmp/output" in " ".join(cmd)
    assert cmd.count("/output") == 0 or all(c != "/output" for c in cmd)


def test_subprocess_cmd_no_api_key_does_not_override() -> None:
    """When api_key='', we must not overwrite any existing env var with empty string."""
    import os as _os
    env_env = ToolSandboxEnvironment(runner="subprocess")
    _, env_vars = env_env._build_cmd(
        output_dir=_P("/tmp/output"),
        base_url="https://integrate.api.nvidia.com/v1",
        model_id="test-model",
        api_key="",
    )
    # If NVIDIA_API_KEY already existed in os.environ, it should not be blanked.
    # If it wasn't there, it should still not be there (or be identical to original).
    original = _os.environ.get("NVIDIA_API_KEY", "")
    assert env_vars.get("NVIDIA_API_KEY", "") == original


# ---------------------------------------------------------------------------
# Score extraction
# ---------------------------------------------------------------------------


def test_extract_scores_full() -> None:
    # Real format confirmed by smoke test
    summary = {
        "per_scenario_results": [],
        "category_aggregated_results": {
            "ALL_CATEGORIES": {"similarity": 0.72, "turn_count": 4.3},
            "STATE_DEPENDENCY": {"similarity": 0.91, "turn_count": 3.1},
            "MULTIPLE_TOOL_CALL": {"similarity": 0.61, "turn_count": 5.0},
        },
    }
    scores = ToolSandboxEnvironment._extract_scores(summary)

    assert scores["similarity"]["value"] == pytest.approx(0.72, abs=1e-4)
    assert scores["turn_count"]["value"] == pytest.approx(4.3, abs=1e-2)
    assert "per_category/STATE_DEPENDENCY/similarity" in scores
    assert scores["per_category/STATE_DEPENDENCY/similarity"]["value"] == pytest.approx(0.91, abs=1e-4)
    assert "per_category/MULTIPLE_TOOL_CALL/similarity" in scores
    # ALL_CATEGORIES is not duplicated as a per_category entry
    assert "per_category/ALL_CATEGORIES/similarity" not in scores


def test_extract_scores_empty_summary() -> None:
    scores = ToolSandboxEnvironment._extract_scores({})
    assert scores == {}


def test_extract_scores_no_categories() -> None:
    summary = {"category_aggregated_results": {"ALL_CATEGORIES": {"similarity": 0.5}}}
    scores = ToolSandboxEnvironment._extract_scores(summary)
    assert scores == {"similarity": {"value": 0.5}}


def test_extract_scores_category_missing_similarity() -> None:
    summary = {
        "category_aggregated_results": {
            "ALL_CATEGORIES": {"similarity": 0.8, "turn_count": 3.0},
            "STATE_DEPENDENCY": {"turn_count": 3.0},  # no similarity
        }
    }
    scores = ToolSandboxEnvironment._extract_scores(summary)
    assert "per_category/STATE_DEPENDENCY/similarity" not in scores
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
        json.dumps({
            "per_scenario_results": [{"name": "s1"}, {"name": "s2"}],
            "category_aggregated_results": {
                "ALL_CATEGORIES": {"similarity": 0.7, "turn_count": 5.0}
            },
        })
    )

    bundle = env._parse_results(tmp_path, exit_code=0, model_id="my-model")

    assert "benchmark" in bundle
    assert "config" in bundle
    assert bundle["benchmark"]["name"] == "toolsandbox"
    assert bundle["benchmark"]["samples"] == 2
    assert bundle["benchmark"]["scores"]["similarity"]["value"] == pytest.approx(0.7)
    assert bundle["config"]["framework"] == "toolsandbox"
    assert bundle["config"]["runner"] == "docker"
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
