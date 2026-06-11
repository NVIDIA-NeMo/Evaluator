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

import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_evaluator.solvers.harbor import (
    _ensure_claude_host_env,
    _ensure_host_env,
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


# ── Patch functions ──────────────────────────────────────────────────────


class TestPatchOpenhandsSDK:
    async def test_patches_applied(self):
        from nemo_evaluator.solvers.harbor import _patch_openhands_sdk

        sandbox = AsyncMock()
        sandbox.exec.return_value = MagicMock(stdout="stuck_detection disabled", stderr="", return_code=0)
        await _patch_openhands_sdk(sandbox)
        assert sandbox.exec.call_count >= 3

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
