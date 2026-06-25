# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Tests for nemo_evaluator.solvers.harbor — all mocked, no Docker/Harbor."""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

import nemo_evaluator.solvers.harbor as harbor_mod
from nemo_evaluator.environments.base import SeedResult
from nemo_evaluator.errors import InfraError
from nemo_evaluator.solvers.harbor import (
    _ensure_claude_host_env,
    _ensure_host_env,
    _error_from_crash_marker,
    _extract_response,
    _model_id_for_openai,
    _resolve_agent_timeout,
    _resolve_api_key,
)

# ── Pure helpers ─────────────────────────────────────────────────────────


class TestExtractResponse:
    def test_metadata_response(self):
        ctx = MagicMock()
        ctx.metadata = {"response": "hello from metadata"}
        ctx.rollout_details = []
        assert _extract_response(ctx) == "hello from metadata"

    def test_rollout_fallback(self):
        ctx = MagicMock()
        ctx.metadata = {}
        ctx.rollout_details = [{"content": "from rollout"}]
        assert _extract_response(ctx) == "from rollout"

    def test_empty_when_nothing(self):
        ctx = MagicMock()
        ctx.metadata = {}
        ctx.rollout_details = []
        assert _extract_response(ctx) == ""

    def test_metadata_wins_over_rollout(self):
        ctx = MagicMock()
        ctx.metadata = {"response": "meta"}
        ctx.rollout_details = [{"content": "roll"}]
        assert _extract_response(ctx) == "meta"

    def test_rollout_with_object_content(self):
        detail = MagicMock()
        detail.content = "obj content"
        ctx = MagicMock()
        ctx.metadata = {}
        ctx.rollout_details = [detail]
        assert _extract_response(ctx) == "obj content"


class TestResolveApiKey:
    def test_explicit_key(self):
        assert _resolve_api_key("sk-123") == "sk-123"

    def test_env_fallback(self, monkeypatch):
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "from-env")
        assert _resolve_api_key(None) == "from-env"

    def test_none_when_nothing(self, monkeypatch):
        for var in ("LLM_API_KEY", "NVIDIA_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
            monkeypatch.delenv(var, raising=False)
        assert _resolve_api_key(None) is None


class TestEnsureHostEnv:
    def test_sets_env_vars(self, monkeypatch):
        for v in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL", "LITELLM_LOG"):
            monkeypatch.delenv(v, raising=False)
        _ensure_host_env("key-1", "my-model", has_custom_url=True)
        assert os.environ["LLM_API_KEY"] == "key-1"
        assert "LLM_BASE_URL" not in os.environ
        assert os.environ["LLM_MODEL"] == "openai/my-model"

    def test_setdefault_does_not_clobber(self, monkeypatch):
        """A second call must not overwrite values set by the first."""
        monkeypatch.setenv("LLM_API_KEY", "first-key")
        monkeypatch.setenv("LLM_MODEL", "first-model")
        _ensure_host_env("second-key", "second-model", has_custom_url=True)
        assert os.environ["LLM_API_KEY"] == "first-key"
        assert os.environ["LLM_MODEL"] == "first-model"

    def test_no_key_uses_dummy(self, monkeypatch):
        for v in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL"):
            monkeypatch.delenv(v, raising=False)
        _ensure_host_env("no-key-needed", None, has_custom_url=False)
        assert os.environ["LLM_API_KEY"] == "no-key-needed"

    def test_model_no_prefix_without_url(self, monkeypatch):
        monkeypatch.delenv("LLM_MODEL", raising=False)
        _ensure_host_env("k", "bare-model", has_custom_url=False)
        assert os.environ["LLM_MODEL"] == "bare-model"

    def test_already_prefixed_model(self, monkeypatch):
        monkeypatch.delenv("LLM_MODEL", raising=False)
        _ensure_host_env("k", "openai/already", has_custom_url=True)
        assert os.environ["LLM_MODEL"] == "openai/already"


class TestClaudeCodeAgent:
    """claude-code talks directly to the Anthropic API, not via LiteLLM."""

    def test_model_id_not_prefixed_for_claude_code(self):
        assert (
            _model_id_for_openai(
                "aws/anthropic/bedrock-claude-sonnet-4-5-v1",
                has_custom_url=True,
                agent="claude-code",
            )
            == "aws/anthropic/bedrock-claude-sonnet-4-5-v1"
        )

    def test_model_id_still_prefixed_for_openhands(self):
        """Default agent path is unchanged — openai/ prefix still applied."""
        assert _model_id_for_openai("my-model", has_custom_url=True) == "openai/my-model"

    def test_ensure_claude_host_env_sets_anthropic_vars(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
        _ensure_claude_host_env("sk-abc", "https://inference-api.nvidia.com")
        assert os.environ["ANTHROPIC_API_KEY"] == "sk-abc"
        assert os.environ["ANTHROPIC_BASE_URL"] == "https://inference-api.nvidia.com"

    def test_ensure_claude_host_env_setdefault(self, monkeypatch):
        """Existing values must not be overwritten."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "user-exported")
        _ensure_claude_host_env("sk-override", "https://example.com")
        assert os.environ["ANTHROPIC_API_KEY"] == "user-exported"

    def test_ensure_claude_host_env_skips_empty_url(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
        _ensure_claude_host_env("sk-abc", "")
        assert "ANTHROPIC_BASE_URL" not in os.environ


# ── Download agent logs ──────────────────────────────────────────────────


class TestDownloadAgentLogs:
    async def test_no_files_warns(self):
        from nemo_evaluator.solvers.harbor import _download_agent_logs

        sandbox = AsyncMock()
        sandbox.exec.return_value = MagicMock(stdout="", return_code=0)
        await _download_agent_logs(sandbox, Path("/tmp/test-dest"))
        sandbox.exec.assert_called()

    async def test_download_retries_on_failure(self, tmp_path):
        from nemo_evaluator.solvers.harbor import _download_agent_logs_inner

        sandbox = AsyncMock()
        ls_result = MagicMock(stdout="/logs/agent/log.json\n", return_code=0)
        tar_result = MagicMock(return_code=0)
        sandbox.exec.side_effect = [ls_result, tar_result]
        sandbox.download.side_effect = RuntimeError("network error")

        await _download_agent_logs_inner(sandbox, tmp_path, max_retries=2)
        assert sandbox.download.call_count == 2


class TestCrashMarker:
    def test_non_infra_marker_returns_agent_crash_error(self, tmp_path):
        (tmp_path / "nel_agent_error.json").write_text(json.dumps({"etype": "ValueError", "emsg": "bad input"}))

        assert _error_from_crash_marker(tmp_path) == "Agent crashed: ValueError: bad input"

    def test_timeout_signal_marker_is_not_agent_crash(self, tmp_path):
        (tmp_path / "nel_agent_error.json").write_text(
            json.dumps({"etype": "KeyboardInterrupt", "emsg": "NEL timeout signal 15"})
        )

        assert _error_from_crash_marker(tmp_path) is None

    def test_infra_marker_raises_infra_error(self, tmp_path):
        (tmp_path / "nel_agent_error.json").write_text(
            json.dumps({"etype": "APIConnectionError", "emsg": "connection failed"})
        )

        with pytest.raises(InfraError, match="Agent infrastructure failure: APIConnectionError: connection failed"):
            _error_from_crash_marker(tmp_path)


# ── Patch functions ──────────────────────────────────────────────────────


class TestPatchOpenhandsSDK:
    async def test_patches_applied(self):
        from nemo_evaluator.solvers.harbor import _patch_openhands_sdk

        sandbox = AsyncMock()
        sandbox.exec.return_value = MagicMock(stdout="stuck_detection disabled", stderr="", return_code=0)
        await _patch_openhands_sdk(sandbox)
        assert sandbox.exec.call_count >= 4

    async def test_runner_patch_disables_visualizer(self):
        """Patch 0 must inject both stuck_detection=False AND visualizer=None.

        The visualizer disable is critical: rich's grapheme splitter hangs in a
        CPU-bound loop on prompts containing U+200B adjacent to URLs (seen on 26
        django SWE-bench tasks), deadlocking conversation.send_message() before
        a single LLM request is made. If this assertion ever fails because the
        patch was refactored, the django timeouts will regress.
        """
        import base64

        from nemo_evaluator.solvers.harbor import _patch_openhands_sdk

        sandbox = AsyncMock()
        sandbox.exec.return_value = MagicMock(
            stdout="stuck_detection disabled, visualizer disabled", stderr="", return_code=0
        )
        await _patch_openhands_sdk(sandbox)

        decoded_scripts: list[str] = []
        for call in sandbox.exec.call_args_list:
            cmd = call.args[0] if call.args else call.kwargs.get("cmd", "")
            if "base64 -d" not in cmd:
                continue
            encoded = cmd.split("echo ", 1)[1].split(" ", 1)[0]
            try:
                decoded_scripts.append(base64.b64decode(encoded).decode())
            except Exception:
                pass

        runner_patch = next(
            (s for s in decoded_scripts if "/installed-agent/run_agent.py" in s and "stuck_detection" in s),
            None,
        )
        assert runner_patch is not None, "runner patch script not emitted"
        assert 'conv_kwargs["visualizer"] = None' in runner_patch, (
            f"runner patch must disable the default visualizer to avoid rich grapheme hang; got:\n{runner_patch}"
        )

    async def test_runner_patch_injects_llm_timeout(self, tmp_path):
        import base64

        from nemo_evaluator.solvers.harbor import _patch_openhands_sdk

        runner = tmp_path / "run_agent.py"
        runner.write_text("import os\nllm_kwargs = {'model': 'm'}\n    llm = LLM(**llm_kwargs)\n")

        sandbox = AsyncMock()
        sandbox.exec.return_value = MagicMock(stdout="ok", stderr="", return_code=0)
        await _patch_openhands_sdk(sandbox)

        decoded_scripts = []
        for call in sandbox.exec.call_args_list:
            cmd = call.args[0] if call.args else call.kwargs.get("cmd", "")
            if "base64 -d" in cmd:
                encoded = cmd.split("echo ", 1)[1].split(" ", 1)[0]
                decoded_scripts.append(base64.b64decode(encoded).decode())

        timeout_patch = next(
            (s for s in decoded_scripts if "llm_timeout" in s and "LLM_TIMEOUT" in s),
            None,
        )
        assert timeout_patch is not None, "LLM timeout patch script not emitted"

        timeout_patch = timeout_patch.replace(
            "p = '/installed-agent/run_agent.py'",
            f"p = {str(runner)!r}",
        )
        exec(compile(timeout_patch, "llm_timeout_patch.py", "exec"), {})  # noqa: S102

        patched = runner.read_text()
        assert "import os" in patched
        assert 'os.environ.get("LLM_TIMEOUT")' in patched
        assert 'llm_kwargs["timeout"] = int(timeout_raw)' in patched

    async def test_runner_patch_preserves_tool_timing(self, tmp_path):
        import base64

        from nemo_evaluator.solvers.harbor import _patch_openhands_sdk

        runner = tmp_path / "run_agent.py"
        runner.write_text(
            """\
import os


def build_trajectory(events, llm_metrics, model_name):
    steps = []
    step_id = 1

    for event in events:
        event_type = event.get("type", "")

        if event_type == "assistant_message":
            step = {
                "step_id": step_id,
                "timestamp": event.get("timestamp"),
                "source": "agent",
                "message": event.get("content", ""),
                "model_name": model_name,
            }
            tool_calls = event.get("tool_calls", [])
            if tool_calls:
                step["tool_calls"] = [
                    {
                        "tool_call_id": tc.get("id", ""),
                        "function_name": tc.get("name", ""),
                        "arguments": tc.get("arguments", {}),
                    }
                    for tc in tool_calls
                ]
            steps.append(step)
            step_id += 1

        elif event_type == "tool_result":
            # Find the previous step and add observation
            if steps and steps[-1].get("source") == "agent":
                steps[-1]["observation"] = {
                    "results": [
                        {
                            "source_call_id": event.get("tool_call_id"),
                            "content": event.get("content", ""),
                        }
                    ]
                }

    trajectory = {
        "schema_version": "ATIF-v1.5",
        "session_id": os.environ.get("SESSION_ID", "harbor-session"),
        "agent": {"name": "openhands-sdk", "version": "unknown"},
        "steps": steps,
        "final_metrics": {},
    }
    return trajectory
"""
        )

        sandbox = AsyncMock()
        sandbox.exec.return_value = MagicMock(stdout="ok", stderr="", return_code=0)
        await _patch_openhands_sdk(sandbox)

        decoded_scripts = []
        for call in sandbox.exec.call_args_list:
            cmd = call.args[0] if call.args else call.kwargs.get("cmd", "")
            if "base64 -d" in cmd:
                encoded = cmd.split("echo ", 1)[1].split(" ", 1)[0]
                decoded_scripts.append(base64.b64decode(encoded).decode())

        trajectory_patch = next(s for s in decoded_scripts if "reasoning+metrics" in s)
        trajectory_patch = trajectory_patch.replace(
            "p = '/installed-agent/run_agent.py'",
            f"p = {str(runner)!r}",
        )
        exec(compile(trajectory_patch, "trajectory_patch.py", "exec"), {})

        namespace = {}
        exec(compile(runner.read_text(), str(runner), "exec"), namespace)
        trajectory = namespace["build_trajectory"](
            [
                {
                    "type": "assistant_message",
                    "timestamp": "2026-06-05T12:00:00.000Z",
                    "tool_calls": [{"id": "call_1", "name": "terminal", "arguments": {"command": "pwd"}}],
                },
                {
                    "type": "tool_result",
                    "tool_call_id": "call_1",
                    "content": "/testbed",
                    "timestamp": "2026-06-05T12:00:01.250Z",
                },
            ],
            {},
            "model",
        )

        assert trajectory["schema_version"] == "ATIF-v1.7"
        result = trajectory["steps"][0]["observation"]["results"][0]
        assert result["source_call_id"] == "call_1"
        assert result["content"] == "/testbed"
        assert result["extra"] == {
            "started_at": "2026-06-05T12:00:00.000Z",
            "completed_at": "2026-06-05T12:00:01.250Z",
            "duration_ms": 1250.0,
        }


class TestSandboxEnvironmentAdapter:
    def test_exposes_all_base_environment_public_attrs(self, tmp_path):
        """Adapter must expose every public attr set by BaseEnvironment.__init__.

        Derived by introspecting BaseEnvironment so the test catches future
        Harbor upgrades that add new required attributes.
        """
        import inspect
        from unittest.mock import MagicMock

        from harbor.environments.base import BaseEnvironment

        from nemo_evaluator.solvers.harbor_adapter import SandboxEnvironmentAdapter

        base_src = inspect.getsource(BaseEnvironment.__init__)
        base_attrs = {
            line.strip().split("=")[0].strip().removeprefix("self.").strip()
            for line in base_src.splitlines()
            if line.strip().startswith("self.") and not line.strip().startswith("self._") and "=" in line
        }
        # environment_dir / environment_name are only meaningful for container-spawning
        # environments (Docker, Daytona, …) that read Dockerfiles from a local directory.
        # terminus-2 never accesses them, so the adapter intentionally omits them.
        container_only_attrs = {"environment_dir", "environment_name"}
        base_attrs -= container_only_attrs

        adapter = SandboxEnvironmentAdapter(MagicMock(), session_id="test-session", logs_dir=tmp_path)
        missing = {a for a in base_attrs if not hasattr(adapter, a)}
        assert not missing, f"SandboxEnvironmentAdapter missing public attrs: {missing}"


class TestConfigureTerminusTokenizer:
    """Tokenizer is fetched/registered only for terminus-2; failures fail fast."""

    @pytest.fixture(autouse=True)
    def _reset_state(self, monkeypatch):
        monkeypatch.setattr(harbor_mod, "_TOKENIZER_REGISTRY", {})
        monkeypatch.setattr(harbor_mod, "_MISSING_TOKENIZER_WARNED", set())
        monkeypatch.setattr(harbor_mod, "_IGNORED_TOKENIZER_LOGGED", set())

    def test_non_terminus_with_tokenizer_does_not_fetch(self, monkeypatch, caplog):
        import logging

        build = MagicMock()
        monkeypatch.setattr(harbor_mod, "_build_custom_tokenizer", build)
        with caplog.at_level(logging.INFO):
            harbor_mod._configure_terminus_tokenizer(is_terminus2=False, model_name="my-model", tokenizer="Qwen/Qwen3")
        build.assert_not_called()
        assert harbor_mod._TOKENIZER_REGISTRY == {}
        assert "ignoring it for the current agent" in caplog.text

    def test_non_terminus_info_logged_once(self, monkeypatch, caplog):
        import logging

        monkeypatch.setattr(harbor_mod, "_build_custom_tokenizer", MagicMock())
        with caplog.at_level(logging.INFO):
            for _ in range(3):
                harbor_mod._configure_terminus_tokenizer(
                    is_terminus2=False, model_name="my-model", tokenizer="Qwen/Qwen3"
                )
        assert caplog.text.count("ignoring it for the current agent") == 1

    def test_terminus_without_tokenizer_warns_once(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING):
            for _ in range(3):
                harbor_mod._configure_terminus_tokenizer(is_terminus2=True, model_name="my-model", tokenizer=None)
        assert caplog.text.count("No tokenizer configured") == 1

    def test_terminus_with_tokenizer_registers(self, monkeypatch):
        monkeypatch.setattr(harbor_mod, "_build_custom_tokenizer", lambda spec: {"spec": spec})
        monkeypatch.setattr(harbor_mod, "_install_token_counter_patch", MagicMock())
        harbor_mod._configure_terminus_tokenizer(is_terminus2=True, model_name="my-model", tokenizer="Qwen/Qwen3")
        assert harbor_mod._TOKENIZER_REGISTRY["my-model"] == {"spec": "Qwen/Qwen3"}

    def test_terminus_tokenizer_fetch_failure_raises(self, monkeypatch):
        def boom(spec):
            raise OSError("offline: cannot reach huggingface")

        monkeypatch.setattr(harbor_mod, "_build_custom_tokenizer", boom)
        with pytest.raises(RuntimeError, match="Failed to load tokenizer"):
            harbor_mod._configure_terminus_tokenizer(is_terminus2=True, model_name="my-model", tokenizer="Qwen/Qwen3")

    def test_terminus_tokenizer_built_once_per_model(self, monkeypatch):
        build = MagicMock(return_value={"tok": 1})
        monkeypatch.setattr(harbor_mod, "_build_custom_tokenizer", build)
        monkeypatch.setattr(harbor_mod, "_install_token_counter_patch", MagicMock())
        for _ in range(3):
            harbor_mod._configure_terminus_tokenizer(is_terminus2=True, model_name="my-model", tokenizer="Qwen/Qwen3")
        build.assert_called_once()


class TestBuildCustomTokenizer:
    """Local tokenizer.json paths use create_tokenizer; repo ids use create_pretrained_tokenizer."""

    def test_local_file_uses_create_tokenizer(self, monkeypatch, tmp_path):
        token_file = tmp_path / "tokenizer.json"
        token_file.write_text('{"fake": "tokenizer"}')
        created = MagicMock(return_value={"type": "local"})
        pretrained = MagicMock()
        monkeypatch.setattr("litellm.utils.create_tokenizer", created)
        monkeypatch.setattr("litellm.utils.create_pretrained_tokenizer", pretrained)

        result = harbor_mod._build_custom_tokenizer(str(token_file))

        created.assert_called_once_with('{"fake": "tokenizer"}')
        pretrained.assert_not_called()
        assert result == {"type": "local"}

    def test_repo_id_uses_create_pretrained_tokenizer(self, monkeypatch):
        created = MagicMock()
        pretrained = MagicMock(return_value={"type": "hf"})
        monkeypatch.setattr("litellm.utils.create_tokenizer", created)
        monkeypatch.setattr("litellm.utils.create_pretrained_tokenizer", pretrained)

        result = harbor_mod._build_custom_tokenizer("Qwen/Qwen3-8B")

        pretrained.assert_called_once_with("Qwen/Qwen3-8B")
        created.assert_not_called()
        assert result == {"type": "hf"}


class TestPatchedTokenCounter:
    """litellm.utils.token_counter is wrapped to inject the registered custom tokenizer."""

    @pytest.fixture(autouse=True)
    def _reset_state(self, monkeypatch):
        monkeypatch.setattr(harbor_mod, "_TOKEN_COUNTER_PATCHED", False)
        monkeypatch.setattr(harbor_mod, "_TOKENIZER_REGISTRY", {})

    @staticmethod
    def _install_recorder(monkeypatch):
        calls = {}

        def fake_original(*args, **kwargs):
            calls["custom_tokenizer"] = kwargs.get("custom_tokenizer")
            calls["args"] = args
            return 0

        monkeypatch.setattr("litellm.utils.token_counter", fake_original)
        harbor_mod._install_token_counter_patch()
        import litellm.utils as litellm_utils

        return litellm_utils.token_counter, calls

    def test_rebinds_litellm_token_counter(self, monkeypatch):
        monkeypatch.setattr("litellm.utils.token_counter", lambda **kwargs: 0)
        import litellm.utils as litellm_utils

        before = litellm_utils.token_counter
        harbor_mod._install_token_counter_patch()
        assert litellm_utils.token_counter is not before
        assert harbor_mod._TOKEN_COUNTER_PATCHED is True

    def test_injects_custom_tokenizer_for_registered_model(self, monkeypatch):
        patched, calls = self._install_recorder(monkeypatch)
        harbor_mod._TOKENIZER_REGISTRY["M"] = {"sentinel": 1}
        patched(model="M", messages=[{"role": "user", "content": "x"}])
        assert calls["custom_tokenizer"] == {"sentinel": 1}

    def test_does_not_inject_for_unregistered_model(self, monkeypatch):
        patched, calls = self._install_recorder(monkeypatch)
        patched(model="other", messages=[{"role": "user", "content": "x"}])
        assert calls["custom_tokenizer"] is None

    def test_respects_explicit_custom_tokenizer_kwarg(self, monkeypatch):
        patched, calls = self._install_recorder(monkeypatch)
        harbor_mod._TOKENIZER_REGISTRY["M"] = {"sentinel": 1}
        patched(model="M", messages=[], custom_tokenizer={"explicit": 2})
        assert calls["custom_tokenizer"] == {"explicit": 2}

    def test_respects_positional_custom_tokenizer(self, monkeypatch):
        patched, calls = self._install_recorder(monkeypatch)
        harbor_mod._TOKENIZER_REGISTRY["M"] = {"sentinel": 1}
        patched("M", {"explicit": 2})
        assert calls["custom_tokenizer"] is None
        assert calls["args"] == ("M", {"explicit": 2})

    def test_idempotent_install(self, monkeypatch):
        monkeypatch.setattr("litellm.utils.token_counter", lambda **kwargs: 0)
        harbor_mod._install_token_counter_patch()
        import litellm.utils as litellm_utils

        first = litellm_utils.token_counter
        harbor_mod._install_token_counter_patch()
        assert litellm_utils.token_counter is first


class TestResolveAgentTimeout:
    """Tests for per-task timeout resolution."""

    @pytest.mark.parametrize(
        ("strategy", "config", "task", "cap", "expected"),
        [
            ("override", 3600, 900, None, 3600),
            ("override", 3600, None, None, 3600),
            ("task", 3600, 900, None, 900),
            ("task", 3600, None, None, 3600),
            ("task", 3600, 12000, 7200, 7200),
            ("max", 3600, 900, None, 3600),
            ("max", 3600, 12000, None, 12000),
            ("max", 3600, 12000, 7200, 7200),
            ("max", 3600, None, None, 3600),
            ("bogus", 3600, 900, None, 3600),
        ],
        ids=[
            "override-ignores-task",
            "override-no-task",
            "task-uses-task",
            "task-fallback-to-nel",
            "task-capped",
            "max-nel-larger",
            "max-task-larger",
            "max-capped",
            "max-no-task-fallback",
            "unknown-fallback",
        ],
    )
    def test_effective_timeout(self, strategy, config, task, cap, expected):
        assert _resolve_agent_timeout(strategy, config, task, cap) == expected


class TestHarborSolverLlmTimeout:
    def test_llm_kwargs_timeout_maps_to_container_env(self):
        from unittest.mock import patch

        with patch("nemo_evaluator.solvers.harbor._check_harbor_installed"):
            from nemo_evaluator.solvers.harbor import HarborSolver

            solver = HarborSolver(
                harbor_agent="openhands-sdk",
                model_url="http://localhost:8000",
                model_id="test-model",
                api_key="test-key",
                harbor_agent_kwargs={"llm_kwargs": {"timeout": 3600}},
            )
            assert solver._container_env["LLM_TIMEOUT"] == "3600"

    def test_explicit_container_env_overrides_llm_kwargs_timeout(self):
        from unittest.mock import patch

        with patch("nemo_evaluator.solvers.harbor._check_harbor_installed"):
            from nemo_evaluator.solvers.harbor import HarborSolver

            solver = HarborSolver(
                harbor_agent="openhands-sdk",
                model_url="http://localhost:8000",
                model_id="test-model",
                api_key="test-key",
                harbor_agent_kwargs={"llm_kwargs": {"timeout": 3600}},
                container_env={"LLM_TIMEOUT": "7200"},
            )
            assert solver._container_env["LLM_TIMEOUT"] == "7200"


class _TimeoutFlushSandbox:
    is_running = True

    def __init__(
        self,
        flush_event: asyncio.Event | None = None,
        *,
        signal_error: Exception | None = None,
    ) -> None:
        self.flush_event = flush_event
        self.signal_error = signal_error
        self.exec_calls: list[tuple[str, float | None]] = []

    async def exec(self, command: str, timeout_sec: float | None = None):
        self.exec_calls.append((command, timeout_sec))
        if "find /logs/agent" in command:
            return MagicMock(stdout="1\n", stderr="", return_code=0)
        if "/installed-agent/run_agent.py" in command:
            if self.signal_error is not None:
                raise self.signal_error
            if self.flush_event is not None:
                self.flush_event.set()
            return MagicMock(stdout="", stderr="", return_code=0)
        return MagicMock(stdout="", stderr="", return_code=0)


class TestWaitForAgentTimeout:
    def _solver(self):
        from nemo_evaluator.solvers.harbor import HarborSolver

        solver = HarborSolver.__new__(HarborSolver)
        solver._run_timeout = 0.05
        solver._timeout_strategy = "override"
        return solver

    @pytest.mark.asyncio
    async def test_timeout_sends_sigterm_and_waits_for_trajectory_flush(self):
        solver = self._solver()
        flush_event = asyncio.Event()
        sandbox = _TimeoutFlushSandbox(flush_event)

        async def agent_run():
            await flush_event.wait()

        agent_task = asyncio.create_task(agent_run())

        timed_out, agent_error = await solver._wait_for_agent(
            agent_task,
            sandbox,
            time.monotonic(),
            effective_timeout=0.05,
            jitter=0.0,
        )

        assert timed_out is True
        assert agent_error is None
        assert agent_task.done()
        assert not agent_task.cancelled()
        assert any("/installed-agent/run_agent.py" in command for command, _ in sandbox.exec_calls)

    @pytest.mark.asyncio
    async def test_timeout_cancels_agent_if_sigterm_flush_fails(self):
        solver = self._solver()
        sandbox = _TimeoutFlushSandbox(signal_error=RuntimeError("signal failed"))

        async def agent_run():
            await asyncio.Event().wait()

        agent_task = asyncio.create_task(agent_run())

        timed_out, agent_error = await solver._wait_for_agent(
            agent_task,
            sandbox,
            time.monotonic(),
            effective_timeout=0.05,
            jitter=0.0,
        )

        assert timed_out is True
        assert agent_error is None
        assert agent_task.cancelled()
        assert any("/installed-agent/run_agent.py" in command for command, _ in sandbox.exec_calls)


class TestSolveTimeoutPlumbing:
    @pytest.mark.asyncio
    async def test_solve_starts_timeout_clock_at_agent_run(self, monkeypatch):
        import nemo_evaluator.solvers.harbor as harbor_mod
        from nemo_evaluator.solvers.harbor import HarborSolver

        class FakeContext:
            def __init__(self) -> None:
                self.metadata = {}
                self.rollout_details = []
                self.n_input_tokens = 0
                self.n_output_tokens = 0

            def is_empty(self) -> bool:
                return False

        class FakeAdapter:
            is_mounted = True

            def __init__(self, *args, **kwargs) -> None:
                pass

        class FakeAgent:
            async def setup(self, adapter) -> None:
                pass

            async def run(self, prompt, adapter, context) -> None:
                context.metadata["response"] = "agent answer"
                context.n_input_tokens = 3
                context.n_output_tokens = 4

        class FakeSandbox:
            is_running = True

            def resolved_endpoint_url(self, name):
                return None

            def resolve_outside_endpoint(self, url):
                return url

            async def exec(self, command: str, timeout_sec: float | None = None):
                return MagicMock(stdout="", stderr="", return_code=0)

        import harbor.models.agent.context as context_mod

        import nemo_evaluator.solvers.harbor_adapter as harbor_adapter_mod

        monkeypatch.setattr(context_mod, "AgentContext", FakeContext)
        monkeypatch.setattr(harbor_adapter_mod, "SandboxEnvironmentAdapter", FakeAdapter)
        monkeypatch.setattr(harbor_mod, "_capture_workspace_diff", AsyncMock(return_value=""))
        monkeypatch.setattr(
            harbor_mod,
            "_recover_from_logs",
            lambda logs_dir: {"trajectory": [], "prompt_tokens": 0, "completion_tokens": 0, "response": ""},
        )
        monkeypatch.setattr(harbor_mod.random, "uniform", lambda low, high: 0.0)

        solver = HarborSolver.__new__(HarborSolver)
        solver._model_url = "http://model"
        solver._model_id = "model"
        solver._timeout = 60.0
        solver._run_timeout = 60.0
        solver._timeout_strategy = "override"
        solver._max_agent_timeout = None
        solver._harbor_agent = "terminus-2"
        solver._container_env = {}
        solver._skill = None
        solver._skill_dir = None
        solver._create_agent = lambda logs_dir, model_url="": FakeAgent()

        wait_args: dict[str, float] = {}

        async def fake_wait_for_agent(agent_task, sandbox, agent_started_at, effective_timeout, jitter):
            wait_args["agent_started_at"] = agent_started_at
            wait_args["effective_timeout"] = effective_timeout
            wait_args["jitter"] = jitter
            await agent_task
            return False, None

        solver._wait_for_agent = fake_wait_for_agent

        result = await solver.solve(
            SeedResult(prompt="do task", expected_answer="", metadata={"task_id": "task-1"}),
            sandbox=FakeSandbox(),
        )

        assert result.response == "agent answer"
        assert result.model_response.total_tokens == 7
        assert wait_args["agent_started_at"] > 0
        assert wait_args["effective_timeout"] == 60.0
        assert wait_args["jitter"] == 0.0


class TestInjectSkill:
    """Tests for HarborSolver._inject_skill — mocked sandbox, no Docker."""

    def _make_solver(self, skill=None, skill_dir=None):
        from unittest.mock import patch

        with patch("nemo_evaluator.solvers.harbor._check_harbor_installed"):
            from nemo_evaluator.solvers.harbor import HarborSolver

            return HarborSolver(
                harbor_agent="terminus-2",
                model_url="http://localhost:8000",
                model_id="glm5",
                api_key="test-key",
                skill=skill,
                skill_dir=skill_dir,
            )

    @pytest.mark.asyncio
    async def test_no_skill_returns_prompt_unchanged(self):
        solver = self._make_solver()
        sandbox = AsyncMock()
        result = await solver._inject_skill(sandbox, "original prompt")
        assert result == "original prompt"
        sandbox.exec.assert_not_called()

    @pytest.mark.asyncio
    async def test_skill_without_dir_prepends_trigger(self):
        solver = self._make_solver(skill="caveman")
        sandbox = AsyncMock()
        result = await solver._inject_skill(sandbox, "do the task")
        assert result.startswith("Before working on this task")
        assert "/skills/caveman/SKILL.md" in result
        assert result.endswith("do the task")
        # mkdir only (no upload, no verify)
        sandbox.exec.assert_called_once()

    @pytest.mark.asyncio
    async def test_skill_with_dir_uploads_files(self, tmp_path):
        skill_dir = tmp_path / "caveman"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("# Caveman")
        (skill_dir / "extra.md").write_text("extra")

        solver = self._make_solver(skill="caveman", skill_dir=str(skill_dir))
        sandbox = AsyncMock()
        sandbox.exec.return_value.return_code = 0
        sandbox.exec.return_value.stdout = "-rw-r--r-- 1 root root 9 SKILL.md\n# Caveman"
        result = await solver._inject_skill(sandbox, "do the task")

        assert "/skills/caveman/SKILL.md" in result
        # mkdir + verify = 2 exec calls (no nested dirs in flat layout)
        assert sandbox.exec.call_count == 2
        assert sandbox.upload.call_count == 2

    @pytest.mark.asyncio
    async def test_skill_with_nested_dirs_creates_parents(self, tmp_path):
        skill_dir = tmp_path / "caveman"
        (skill_dir / "references").mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("# Caveman")
        (skill_dir / "references" / "patterns.md").write_text("patterns")

        solver = self._make_solver(skill="caveman", skill_dir=str(skill_dir))
        sandbox = AsyncMock()
        sandbox.exec.return_value.return_code = 0
        sandbox.exec.return_value.stdout = "-rw-r--r-- 1 root root 9 SKILL.md\n# Caveman"
        await solver._inject_skill(sandbox, "do the task")

        # mkdir top + mkdir references subdir + verify = 3 exec calls
        assert sandbox.exec.call_count == 3
        assert sandbox.upload.call_count == 2

    @pytest.mark.asyncio
    async def test_nonexistent_skill_dir_warns_but_continues(self, caplog):
        import logging

        solver = self._make_solver(skill="caveman", skill_dir="/nonexistent/path")
        sandbox = AsyncMock()
        with caplog.at_level(logging.WARNING):
            result = await solver._inject_skill(sandbox, "do the task")

        assert "does not exist" in caplog.text
        assert "/skills/caveman/SKILL.md" in result
        sandbox.upload.assert_not_called()
