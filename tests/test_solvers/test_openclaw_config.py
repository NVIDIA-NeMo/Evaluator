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
"""Unit tests for the synchronous, side-effect-free helpers in ``openclaw``.

These pin the OpenClaw config builder, PinchBench/FWS detection, web-search
credential preflight, and timeout resolution. The async solver loop (which
needs a live sandbox) is covered elsewhere.
"""

from __future__ import annotations

import pytest

from nemo_evaluator.environments.base import SeedResult
from nemo_evaluator.solvers.openclaw import (
    OpenClawSolver,
    _build_fws_env_setup,
    _build_openclaw_config,
    _check_prerequisites,
    _check_web_search_env,
    _is_pinchbench_task,
    _metadata_requires_fws,
    _normalize_task_sessions,
    _web_search_exec_env,
)


class TestBuildOpenclawConfig:
    def test_derives_provider_from_model_id_prefix(self) -> None:
        config = _build_openclaw_config("http://localhost:9000", "nvidia/nemotron-super")
        providers = config["models"]["providers"]
        assert "nvidia" in providers
        assert providers["nvidia"]["baseUrl"] == "http://localhost:9000"
        # OpenClaw references the model as "<provider>/<model_id>".
        assert config["agents"]["defaults"]["model"]["primary"] == "nvidia/nvidia/nemotron-super"

    def test_provider_defaults_to_custom_without_prefix(self) -> None:
        config = _build_openclaw_config("http://localhost:9000", "bare-model")
        assert "custom" in config["models"]["providers"]

    def test_empty_api_key_omits_apikey_field(self) -> None:
        # A placeholder apiKey trips OpenClaw's auth resolver and breaks the
        # synthetic local-key path -- the field must be absent, not blank.
        config = _build_openclaw_config("http://localhost:9000", "nvidia/m", api_key="")
        assert "apiKey" not in config["models"]["providers"]["nvidia"]

    def test_api_key_included_when_provided(self) -> None:
        config = _build_openclaw_config("http://localhost:9000", "nvidia/m", api_key="sk-real")
        assert config["models"]["providers"]["nvidia"]["apiKey"] == "sk-real"

    def test_optional_token_limits_included_only_when_set(self) -> None:
        without = _build_openclaw_config("u", "nvidia/m")["models"]["providers"]["nvidia"]["models"][0]
        assert "contextWindow" not in without and "maxTokens" not in without

        with_limits = _build_openclaw_config("u", "nvidia/m", context_window=8192, max_tokens=1024)
        entry = with_limits["models"]["providers"]["nvidia"]["models"][0]
        assert entry["contextWindow"] == 8192
        assert entry["maxTokens"] == 1024

    def test_tavily_web_search_wires_tools_and_plugin(self) -> None:
        config = _build_openclaw_config("u", "nvidia/m", web_search_provider="tavily")
        assert config["tools"]["web"]["search"]["provider"] == "tavily"
        assert config["plugins"]["entries"]["tavily"]["enabled"] is True

    def test_no_plugins_key_without_web_search(self) -> None:
        assert "plugins" not in _build_openclaw_config("u", "nvidia/m")


class TestPinchbenchAndFws:
    @pytest.mark.parametrize(
        ("metadata", "expected"),
        [
            ({"source": "pinchbench"}, True),
            ({"pinchbench_source_ref": "tasks/x"}, True),
            ({"source": "harbor"}, False),
            ({}, False),
        ],
        ids=["source", "source-ref", "other-source", "empty"],
    )
    def test_is_pinchbench_task(self, metadata: dict, expected: bool) -> None:
        assert _is_pinchbench_task(metadata) is expected

    @pytest.mark.parametrize(
        ("metadata", "expected"),
        [
            ({"prerequisites": ["fws"]}, True),
            ({"prerequisites": ["server:FWS"]}, True),
            ({"prerequisites": ["cli:gh"]}, False),
            ({"prerequisites": "fws"}, False),  # not a list
            ({}, False),
        ],
        ids=["fws", "fws-case-insensitive", "no-fws", "not-a-list", "empty"],
    )
    def test_metadata_requires_fws(self, metadata: dict, expected: bool) -> None:
        assert _metadata_requires_fws(metadata) is expected

    def test_fws_env_setup_empty_when_not_required(self) -> None:
        assert _build_fws_env_setup({}) == ""

    def test_fws_env_setup_starts_server_without_gh_config(self) -> None:
        setup = _build_fws_env_setup({"prerequisites": ["fws"]})
        assert "fws server start" in setup
        assert "GH_TOKEN" not in setup

    def test_fws_env_setup_includes_gh_config_when_cli_gh_present(self) -> None:
        setup = _build_fws_env_setup({"prerequisites": ["fws", "cli:gh"]})
        assert "fws server start" in setup
        assert "GH_TOKEN" in setup
        assert "hosts.yml" in setup


class TestNormalizeTaskSessions:
    def test_defaults_to_single_session_when_absent(self) -> None:
        task = SeedResult(prompt="do the thing", expected_answer="")
        sessions = _normalize_task_sessions(task)
        assert sessions == [{"id": "default", "prompt": "do the thing", "new_session": False}]

    def test_normalizes_explicit_sessions_and_synthesizes_ids(self) -> None:
        task = SeedResult(
            prompt="ignored",
            expected_answer="",
            metadata={"sessions": [{"prompt": "p1"}, {"id": "second", "prompt": "p2", "new_session": True}]},
        )
        sessions = _normalize_task_sessions(task)
        assert sessions == [
            {"id": "session_1", "prompt": "p1", "new_session": False},
            {"id": "second", "prompt": "p2", "new_session": True},
        ]

    def test_skips_malformed_turns_and_falls_back_when_all_invalid(self) -> None:
        task = SeedResult(
            prompt="fallback",
            expected_answer="",
            metadata={"sessions": ["not-a-dict", {"prompt": "   "}]},
        )
        sessions = _normalize_task_sessions(task)
        assert sessions == [{"id": "default", "prompt": "fallback", "new_session": False}]


class TestWebSearchEnv:
    def test_exec_env_empty_without_provider(self) -> None:
        assert _web_search_exec_env(None) == {}

    def test_exec_env_forwards_tavily_key_when_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-123")
        assert _web_search_exec_env("tavily") == {"TAVILY_API_KEY": "tvly-123"}

    def test_exec_env_empty_when_tavily_key_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        assert _web_search_exec_env("tavily") == {}

    def test_check_env_noop_without_provider(self) -> None:
        _check_web_search_env(None)  # must not raise

    def test_check_env_raises_when_tavily_key_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="TAVILY_API_KEY"):
            _check_web_search_env("tavily")

    def test_check_env_passes_when_tavily_key_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-123")
        _check_web_search_env("tavily")  # must not raise


class TestCheckPrerequisites:
    # ``_check_prerequisites`` imports shutil/subprocess locally, so patch the
    # real modules rather than attributes on the openclaw module.
    def test_raises_when_node_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("shutil.which", lambda _: None)
        with pytest.raises(RuntimeError, match="Node.js"):
            _check_prerequisites("openclaw")

    def test_raises_on_old_node(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/node")
        monkeypatch.setattr("subprocess.check_output", lambda *a, **k: "v18.19.0")
        with pytest.raises(RuntimeError, match=">= 22"):
            _check_prerequisites("openclaw")

    def test_raises_when_openclaw_binary_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/node" if name == "node" else None)
        monkeypatch.setattr("subprocess.check_output", lambda *a, **k: "v22.3.0")
        with pytest.raises(RuntimeError, match="not found in PATH"):
            _check_prerequisites("openclaw")

    def test_passes_when_node_and_binary_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/" + name)
        monkeypatch.setattr("subprocess.check_output", lambda *a, **k: "v22.3.0")
        _check_prerequisites("openclaw")  # must not raise


class TestEffectiveTimeout:
    def _solver(self, timeout: float = 100.0) -> OpenClawSolver:
        return OpenClawSolver(timeout=timeout, skip_preflight=True)

    def test_uses_solver_default_without_task_override(self) -> None:
        task = SeedResult(prompt="p", expected_answer="")
        assert self._solver(timeout=100.0)._effective_timeout(task) == 100.0

    def test_task_timeout_caps_solver_default(self) -> None:
        task = SeedResult(prompt="p", expected_answer="", metadata={"timeout_seconds": 10})
        assert self._solver(timeout=100.0)._effective_timeout(task) == 10.0

    def test_falls_back_to_agent_timeout_sec_key(self) -> None:
        task = SeedResult(prompt="p", expected_answer="", metadata={"agent_timeout_sec": 5})
        assert self._solver(timeout=100.0)._effective_timeout(task) == 5.0

    def test_never_exceeds_solver_default(self) -> None:
        task = SeedResult(prompt="p", expected_answer="", metadata={"timeout_seconds": 9999})
        assert self._solver(timeout=100.0)._effective_timeout(task) == 100.0

    def test_override_ignores_task_timeout(self) -> None:
        task = SeedResult(prompt="p", expected_answer="", metadata={"timeout_seconds": 10})
        solver = OpenClawSolver(timeout=100.0, timeout_strategy="override", skip_preflight=True)
        assert solver._effective_timeout(task) == 100.0

    def test_override_uses_run_timeout_over_benchmark(self) -> None:
        task = SeedResult(prompt="p", expected_answer="", metadata={"timeout_seconds": 10})
        solver = OpenClawSolver(timeout=100.0, run_timeout=500.0, timeout_strategy="override", skip_preflight=True)
        assert solver._effective_timeout(task) == 500.0

    def test_max_takes_larger_of_nel_and_task(self) -> None:
        task = SeedResult(prompt="p", expected_answer="", metadata={"timeout_seconds": 9999})
        solver = OpenClawSolver(timeout=100.0, timeout_strategy="max", skip_preflight=True)
        assert solver._effective_timeout(task) == 9999.0

    def test_max_agent_timeout_caps_result(self) -> None:
        task = SeedResult(prompt="p", expected_answer="", metadata={"timeout_seconds": 9999})
        solver = OpenClawSolver(timeout=100.0, timeout_strategy="max", max_agent_timeout=500.0, skip_preflight=True)
        assert solver._effective_timeout(task) == 500.0
