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
    _count_zero_token_agent_turns,
    _ensure_claude_host_env,
    _ensure_host_env,
    _error_from_crash_marker,
    _extract_response,
    _model_id_for_openai,
    _resolve_agent_timeout,
    _resolve_api_key,
    _sandbox_agent_exec_timeout,
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
        (tmp_path / "agent_error.json").write_text(json.dumps({"etype": "ValueError", "emsg": "bad input"}))

        assert _error_from_crash_marker(tmp_path) == "Agent crashed: ValueError: bad input"

    def test_litellm_timeout_marker_treated_as_crash_not_infra(self, tmp_path):
        # TODO: litellm request timeouts (etype "Timeout") are currently surfaced as a
        # generic crash, not infra. Whether they should be retryable infra is deferred —
        # see the TODO on _INFRA_ERROR_NAMES. This test pins the current behavior.
        (tmp_path / "agent_error.json").write_text(json.dumps({"etype": "Timeout", "emsg": "request timed out"}))

        assert _error_from_crash_marker(tmp_path) == "Agent crashed: Timeout: request timed out"

    def test_infra_marker_raises_infra_error(self, tmp_path):
        (tmp_path / "agent_error.json").write_text(
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
        assert "import os as _llm_timeout_os" in patched
        assert '_llm_timeout_os.environ.get("LLM_TIMEOUT")' in patched
        assert 'llm_kwargs["timeout"] = int(timeout_raw)' in patched

    async def test_runner_timeout_patch_does_not_shadow_os(self, tmp_path):
        import base64

        from nemo_evaluator.solvers.harbor import _patch_openhands_sdk

        runner = tmp_path / "run_agent.py"
        runner.write_text(
            """\
import os


def main():
    model = os.environ.get("LLM_MODEL", "m")
    llm_kwargs = {"model": model}
    llm = LLM(**llm_kwargs)
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
        namespace = {"LLM": lambda **kwargs: kwargs}
        exec(compile(patched, str(runner), "exec"), namespace)  # noqa: S102
        namespace["main"]()

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

    _EVENTS_MSG_AND_ACTION = (
        'MessageEvent("agent", "partial reasoning", 1.0), ActionEvent(2.0, "call-1", "bash", \'{"command": "ls"}\')'
    )
    _EVENTS_TWO_MESSAGES = (
        'MessageEvent("agent", "partial reasoning", 1.0), MessageEvent("agent", "more reasoning", 2.0)'
    )

    @staticmethod
    def _stub_runner(module_imports: str, run_body: str, events: str) -> str:
        """Build a minimal run_agent.py stub for the flush patch to wrap.

        ``module_imports`` controls which names exist at module scope (used to
        prove ``_write_partial_trajectory`` self-imports its dependencies).
        ``run_body`` is the indented body of ``_Conversation.run`` — raise to
        exercise the crash path, ``raise_signal`` to exercise the timeout path.
        ``events`` is the comma-separated event constructor list; use message-
        only events to keep the flush off the ``_trajectory_tool_args`` sibling
        helper (which is not in change #5's self-sufficiency scope).
        """
        return (
            module_imports
            + """


class MessageEvent:
    def __init__(self, source, content, timestamp):
        self.source = source
        self.timestamp = timestamp
        self.llm_message = type("LM", (), {"content": content})()


class ActionEvent:
    def __init__(self, timestamp, tool_call_id, tool_name, arguments):
        self.timestamp = timestamp
        self.tool_call_id = tool_call_id
        self.tool_name = tool_name
        fn = type("Fn", (), {"arguments": arguments})()
        self.tool_call = type("TC", (), {"function": fn})()


def build_trajectory(events, metrics, model):
    return {"steps": events, "final_metrics": metrics, "model_name": model}


class _Args:
    instruction = "solve it"


class _Conversation:
    def __init__(self, events):
        self.state = type("S", (), {"events": events})()

    def send_message(self, instruction):
        self._sent = instruction

    def run(self):
"""
            + run_body
            + """

def main():
    args = _Args()
    args.trajectory_path = os.environ["NEL_TEST_TRAJECTORY_PATH"]
    model = "test-model"
    llm = type("LLM", (), {"metrics": None})()
    conversation = _Conversation([__EVENTS__])

    # Send instruction and run
    conversation.send_message(args.instruction)
    conversation.run()

    print("Total cost: 0.0")
""".replace("__EVENTS__", events)
        )

    @staticmethod
    async def _apply_flush_patch(runner: Path) -> str:
        """Emit the trajectory-flush patch and apply it to *runner* in place."""
        import base64

        from nemo_evaluator.solvers.harbor import _patch_openhands_sdk

        sandbox = AsyncMock()
        sandbox.exec.return_value = MagicMock(stdout="budget_flush=True", stderr="", return_code=0)
        await _patch_openhands_sdk(sandbox)

        decoded_scripts = []
        for call in sandbox.exec.call_args_list:
            cmd = call.args[0] if call.args else call.kwargs.get("cmd", "")
            if "base64 -d" in cmd:
                encoded = cmd.split("echo ", 1)[1].split(" ", 1)[0]
                decoded_scripts.append(base64.b64decode(encoded).decode())

        flush_patch = next(
            (s for s in decoded_scripts if "_trajectory_flush_exc" in s and "budget_flush" in s),
            None,
        )
        assert flush_patch is not None, "trajectory-flush patch script not emitted"

        flush_patch = flush_patch.replace(
            "p = '/installed-agent/run_agent.py'",
            f"p = {str(runner)!r}",
        )
        exec(compile(flush_patch, "flush_patch.py", "exec"), {})  # noqa: S102
        return runner.read_text()

    @staticmethod
    def _run_patched_main(patched: str, runner: Path, monkeypatch) -> list[str]:
        """Execute the patched ``main()``, returning the marker dir creations.

        The exit block's ``/logs/agent`` marker write is not writable under
        test, so ``os.makedirs`` is spied to both record whether the marker
        branch was taken and short-circuit the unwritable path.  Signal
        handlers installed by the wrap are saved and restored.
        """
        import contextlib
        import signal

        makedirs_calls: list[str] = []

        def _spy_makedirs(path, *a, **k):
            makedirs_calls.append(path)
            raise OSError("marker dir not writable under test")

        monkeypatch.setattr(os, "makedirs", _spy_makedirs)

        namespace: dict = {}
        exec(compile(patched, str(runner), "exec"), namespace)  # noqa: S102
        old_term = signal.getsignal(signal.SIGTERM)
        old_int = signal.getsignal(signal.SIGINT)
        try:
            with contextlib.suppress(SystemExit, OSError):
                namespace["main"]()
        finally:
            signal.signal(signal.SIGTERM, old_term)
            signal.signal(signal.SIGINT, old_int)
        return makedirs_calls

    async def test_flush_patch_crash_writes_trajectory_and_marker(self, tmp_path, monkeypatch):
        """Genuine crash → flush with reason=etype AND the marker branch is taken.

        The injected block is the largest patch (helpers + signal handlers +
        two anchors), so a stub run is the only thing that catches an
        indentation slip or a closure-scope mistake before a live eval.
        """
        traj_path = tmp_path / "trajectory" / "out.json"
        monkeypatch.setenv("NEL_TEST_TRAJECTORY_PATH", str(traj_path))

        runner = tmp_path / "run_agent.py"
        runner.write_text(
            self._stub_runner(
                "import json\nimport os\nimport sys\nfrom pathlib import Path",
                '        raise RuntimeError("simulated crash")\n',
                self._EVENTS_MSG_AND_ACTION,
            )
        )
        patched = await self._apply_flush_patch(runner)

        assert "_trajectory_flush_exc = None" in patched
        assert "def _write_partial_trajectory" in patched
        assert "class TrajectoryFlushRequested" in patched
        # Crash marker is written atomically (tmp + os.replace), matching the
        # partial-trajectory write, so a kill mid-write can't leave a truncated
        # marker that _error_from_crash_marker would silently drop.
        assert '_marker = "/logs/agent/agent_error.json"' in patched
        assert "_os.replace(_marker_tmp, _marker)" in patched
        assert "sys.exit(0)" in patched

        makedirs_calls = self._run_patched_main(patched, runner, monkeypatch)

        assert traj_path.is_file(), "partial trajectory was not flushed"
        data = json.loads(traj_path.read_text())
        assert data["extra"]["partial_trajectory"]["reason"] == "RuntimeError"
        assert data["extra"]["partial_trajectory"]["events"] == 2
        assert len(data["steps"]) == 2
        assert "/logs/agent" in makedirs_calls, "a genuine crash must take the marker-write branch"

    async def test_flush_patch_timeout_writes_trajectory_no_marker(self, tmp_path, monkeypatch):
        """NEL timeout signal → flush with reason='timeout' and NO crash marker.

        Raising SIGINT inside ``run()`` fires the installed handler, which
        raises ``TrajectoryFlushRequested``; the guarded exit block must skip
        the marker write so the host never sees a bogus crash.
        """
        traj_path = tmp_path / "trajectory" / "out.json"
        monkeypatch.setenv("NEL_TEST_TRAJECTORY_PATH", str(traj_path))

        runner = tmp_path / "run_agent.py"
        runner.write_text(
            self._stub_runner(
                "import json\nimport os\nimport sys\nfrom pathlib import Path",
                "        import signal as _sig\n        _sig.raise_signal(_sig.SIGINT)\n",
                self._EVENTS_MSG_AND_ACTION,
            )
        )
        patched = await self._apply_flush_patch(runner)

        makedirs_calls = self._run_patched_main(patched, runner, monkeypatch)

        assert traj_path.is_file(), "partial trajectory was not flushed on timeout"
        data = json.loads(traj_path.read_text())
        assert data["extra"]["partial_trajectory"]["reason"] == "timeout"
        assert data["extra"]["partial_trajectory"]["events"] == 2
        assert "/logs/agent" not in makedirs_calls, "the timeout path must not write a crash marker"

    async def test_flush_patch_self_sufficient_without_module_imports(self, tmp_path, monkeypatch):
        """_write_partial_trajectory flushes even without module-scope json/Path.

        Proves change #5: the function self-imports its dependencies.  ``os``
        and ``sys`` remain at module scope because the wrap/exit blocks (out of
        scope for #5) still reference them there.
        """
        traj_path = tmp_path / "trajectory" / "out.json"
        monkeypatch.setenv("NEL_TEST_TRAJECTORY_PATH", str(traj_path))

        runner = tmp_path / "run_agent.py"
        runner.write_text(
            self._stub_runner(
                "import os\nimport sys",
                '        raise RuntimeError("simulated crash")\n',
                self._EVENTS_TWO_MESSAGES,
            )
        )
        patched = await self._apply_flush_patch(runner)

        self._run_patched_main(patched, runner, monkeypatch)

        assert traj_path.is_file(), "flush must not depend on module-scope json/Path"
        data = json.loads(traj_path.read_text())
        assert data["extra"]["partial_trajectory"]["events"] == 2
        assert len(data["steps"]) == 2


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

    def test_sandbox_agent_exec_timeout_adds_flush_grace(self):
        assert _sandbox_agent_exec_timeout(10800.0) > 10800.0


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

    @pytest.mark.asyncio
    async def test_inner_command_timeout_finishes_first_and_skips_nel_sigterm(self):
        solver = self._solver()
        sandbox = _TimeoutFlushSandbox()

        async def agent_run():
            await asyncio.sleep(0.001)
            raise RuntimeError("Command failed (exit 124): stderr: timed out after 10800s")

        agent_task = asyncio.create_task(agent_run())

        timed_out, agent_error = await solver._wait_for_agent(
            agent_task,
            sandbox,
            time.monotonic(),
            effective_timeout=0.05,
            jitter=0.0,
        )

        assert timed_out is False
        assert agent_error is not None
        assert "timed out after 10800s" in str(agent_error)
        assert not any("/installed-agent/run_agent.py" in command for command, _ in sandbox.exec_calls)


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

        adapter_kwargs: dict[str, object] = {}

        class FakeAdapter:
            is_mounted = True

            def __init__(self, *args, **kwargs) -> None:
                adapter_kwargs.update(kwargs)

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
        assert adapter_kwargs["default_timeout"] == _sandbox_agent_exec_timeout(60.0)

    @pytest.mark.asyncio
    async def test_solve_sets_command_timeout_after_10800s_effective_timeout_with_jitter(self, monkeypatch):
        import nemo_evaluator.solvers.harbor as harbor_mod
        from nemo_evaluator.solvers.harbor import HarborSolver

        class FakeContext:
            def __init__(self) -> None:
                self.metadata = {"response": "agent answer"}
                self.rollout_details = []
                self.n_input_tokens = 3
                self.n_output_tokens = 4

            def is_empty(self) -> bool:
                return False

        adapter_kwargs: dict[str, object] = {}

        class FakeAdapter:
            is_mounted = True

            def __init__(self, *args, **kwargs) -> None:
                adapter_kwargs.update(kwargs)

        class FakeAgent:
            async def setup(self, adapter) -> None:
                pass

            async def run(self, prompt, adapter, context) -> None:
                pass

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
        monkeypatch.setattr(harbor_mod.random, "uniform", lambda low, high: high)

        solver = HarborSolver.__new__(HarborSolver)
        solver._model_url = "http://model"
        solver._model_id = "model"
        solver._timeout = 10800.0
        solver._run_timeout = 10800.0
        solver._timeout_strategy = "override"
        solver._max_agent_timeout = None
        solver._harbor_agent = "terminus-2"
        solver._container_env = {}
        solver._skill = None
        solver._skill_dir = None
        solver._create_agent = lambda logs_dir, model_url="": FakeAgent()

        wait_args: dict[str, float] = {}

        async def fake_wait_for_agent(agent_task, sandbox, agent_started_at, effective_timeout, jitter):
            wait_args["effective_timeout"] = effective_timeout
            wait_args["jitter"] = jitter
            await agent_task
            return False, None

        solver._wait_for_agent = fake_wait_for_agent

        await solver.solve(
            SeedResult(prompt="do task", expected_answer="", metadata={"task_id": "task-1"}),
            sandbox=FakeSandbox(),
        )

        assert wait_args["jitter"] == 120.0
        assert wait_args["effective_timeout"] == 10920.0
        assert adapter_kwargs["default_timeout"] == _sandbox_agent_exec_timeout(10920.0)
        assert adapter_kwargs["default_timeout"] == 10950.0


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


# ── Zero-token agent-turn accounting ─────────────────────────────────────


class TestCountZeroTokenAgentTurns:
    def test_empty_list(self):
        assert _count_zero_token_agent_turns([]) == 0

    def test_single_doc_not_wrapped_in_list(self):
        doc = {"steps": [{"source": "agent", "metrics": {"prompt_tokens": 0, "completion_tokens": 0}}]}
        assert _count_zero_token_agent_turns(doc) == 1

    def test_counts_only_zero_token_agent_steps(self):
        doc = {
            "steps": [
                {"source": "agent", "metrics": {"prompt_tokens": 0, "completion_tokens": 0}},
                {"source": "agent", "metrics": {"prompt_tokens": 10, "completion_tokens": 5}},
                {"source": "agent", "metrics": {"prompt_tokens": 0, "completion_tokens": 0}},
            ]
        }
        assert _count_zero_token_agent_turns([doc]) == 2

    def test_non_agent_zero_token_step_ignored(self):
        doc = {
            "steps": [
                {"source": "environment", "metrics": {"prompt_tokens": 0, "completion_tokens": 0}},
                {"source": "user", "metrics": {"prompt_tokens": 0, "completion_tokens": 0}},
            ]
        }
        assert _count_zero_token_agent_turns([doc]) == 0

    def test_agent_step_without_metrics_ignored(self):
        doc = {"steps": [{"source": "agent"}, {"source": "agent", "metrics": None}]}
        assert _count_zero_token_agent_turns([doc]) == 0

    def test_partial_zero_not_counted(self):
        # Only prompt_tokens == 0 (completion nonzero) must not count.
        doc = {"steps": [{"source": "agent", "metrics": {"prompt_tokens": 0, "completion_tokens": 3}}]}
        assert _count_zero_token_agent_turns([doc]) == 0

    def test_non_dict_docs_and_steps_skipped(self):
        assert _count_zero_token_agent_turns(["not-a-dict", 42, None]) == 0
        assert _count_zero_token_agent_turns([{"steps": ["not-a-dict", None]}]) == 0

    def test_missing_or_none_steps(self):
        assert _count_zero_token_agent_turns([{}]) == 0
        assert _count_zero_token_agent_turns([{"steps": None}]) == 0

    def test_aggregates_across_multiple_docs(self):
        docs = [
            {"steps": [{"source": "agent", "metrics": {"prompt_tokens": 0, "completion_tokens": 0}}]},
            {"steps": [{"source": "agent", "metrics": {"prompt_tokens": 0, "completion_tokens": 0}}]},
        ]
        assert _count_zero_token_agent_turns(docs) == 2


# ── OpenHands condenser config plumbing ──────────────────────────────────


class TestHarborSolverCondenser:
    def _make(self, agent="openhands-sdk", agent_kwargs=None, container_env=None):
        from unittest.mock import patch

        with patch("nemo_evaluator.solvers.harbor._check_harbor_installed"):
            from nemo_evaluator.solvers.harbor import HarborSolver

            return HarborSolver(
                harbor_agent=agent,
                model_url="http://localhost:8000",
                model_id="test-model",
                api_key="test-key",
                harbor_agent_kwargs=agent_kwargs,
                container_env=container_env,
            )

    def test_condenser_max_size_maps_to_env(self):
        solver = self._make(agent_kwargs={"condenser_max_size": 80})
        assert solver._container_env["OH_CONDENSER_MAX_SIZE"] == "80"
        assert "OH_CONDENSER_MAX_TOKENS" not in solver._container_env

    def test_condenser_max_tokens_maps_to_env(self):
        solver = self._make(agent_kwargs={"condenser_max_size": 80, "condenser_max_tokens": 4096})
        assert solver._container_env["OH_CONDENSER_MAX_SIZE"] == "80"
        assert solver._container_env["OH_CONDENSER_MAX_TOKENS"] == "4096"

    def test_float_values_coerced_to_int_string(self):
        solver = self._make(agent_kwargs={"condenser_max_size": 80.0, "condenser_max_tokens": 4096.7})
        assert solver._container_env["OH_CONDENSER_MAX_SIZE"] == "80"
        assert solver._container_env["OH_CONDENSER_MAX_TOKENS"] == "4096"

    def test_explicit_container_env_overrides(self):
        solver = self._make(
            agent_kwargs={"condenser_max_size": 80},
            container_env={"OH_CONDENSER_MAX_SIZE": "12"},
        )
        assert solver._container_env["OH_CONDENSER_MAX_SIZE"] == "12"

    def test_no_condenser_env_without_kwargs(self):
        solver = self._make(agent_kwargs={})
        assert "OH_CONDENSER_MAX_SIZE" not in solver._container_env
        assert "OH_CONDENSER_MAX_TOKENS" not in solver._container_env

    def test_non_openhands_agent_ignores_condenser_kwargs(self):
        solver = self._make(
            agent="terminus-2",
            agent_kwargs={"condenser_max_size": 80, "condenser_max_tokens": 4096},
        )
        assert "OH_CONDENSER_MAX_SIZE" not in solver._container_env
        assert "OH_CONDENSER_MAX_TOKENS" not in solver._container_env


class TestCreateAgentDropsCondenserKwargs:
    def test_condenser_kwargs_not_forwarded_to_factory(self, monkeypatch, tmp_path):
        from nemo_evaluator.solvers.harbor import HarborSolver

        solver = HarborSolver.__new__(HarborSolver)
        solver._harbor_agent = "openhands-sdk"
        solver._harbor_agent_kwargs = {
            "condenser_max_size": 80,
            "condenser_max_tokens": 4096,
            "model_name": "test-model",
        }
        solver._model_url = "http://model"
        solver._model_id = "test-model"
        solver._api_key = "test-key"
        solver._tokenizer = None
        solver._max_input_tokens = None
        solver._max_output_tokens = None

        monkeypatch.setattr(harbor_mod, "_patch_terminus_tmux_send_keys", lambda: None)
        monkeypatch.setattr(harbor_mod, "_configure_terminus_tokenizer", lambda **kw: None)

        import harbor.agents.factory as factory_mod

        captured: dict = {}

        def fake_create(name, logs_dir=None, **kwargs):
            captured["name"] = name
            captured["kwargs"] = kwargs
            return MagicMock()

        monkeypatch.setattr(factory_mod.AgentFactory, "create_agent_from_name", fake_create)

        solver._create_agent(tmp_path)

        assert captured["name"] == "openhands-sdk"
        assert "condenser_max_size" not in captured["kwargs"]
        assert "condenser_max_tokens" not in captured["kwargs"]


# ── solve(): zero-token turn flagging ────────────────────────────────────


class _FakeZeroTokenContext:
    def __init__(self) -> None:
        self.metadata = {"response": "agent answer"}
        self.rollout_details = []
        self.n_input_tokens = 3
        self.n_output_tokens = 4

    def is_empty(self) -> bool:
        return False


class _FakeZeroTokenAgent:
    async def setup(self, adapter) -> None:
        pass

    async def run(self, prompt, adapter, context) -> None:
        pass


class _FakeZeroTokenSandbox:
    is_running = True

    def resolved_endpoint_url(self, name):
        return None

    def resolve_outside_endpoint(self, url):
        return url

    async def exec(self, command: str, timeout_sec: float | None = None):
        return MagicMock(stdout="", stderr="", return_code=0)


class TestSolveZeroTokenFlag:
    def _build_solver(self, monkeypatch, trajectory):
        class FakeAdapter:
            is_mounted = True

            def __init__(self, *args, **kwargs) -> None:
                pass

        import harbor.models.agent.context as context_mod

        import nemo_evaluator.solvers.harbor_adapter as harbor_adapter_mod

        monkeypatch.setattr(context_mod, "AgentContext", _FakeZeroTokenContext)
        monkeypatch.setattr(harbor_adapter_mod, "SandboxEnvironmentAdapter", FakeAdapter)
        monkeypatch.setattr(harbor_mod, "_patch_openhands_sdk", AsyncMock(return_value=None))
        monkeypatch.setattr(harbor_mod, "_capture_workspace_diff", AsyncMock(return_value=""))
        monkeypatch.setattr(
            harbor_mod,
            "_recover_from_logs",
            lambda logs_dir: {
                "trajectory": trajectory,
                "prompt_tokens": 3,
                "completion_tokens": 4,
                "response": "agent answer",
            },
        )
        monkeypatch.setattr(harbor_mod.random, "uniform", lambda low, high: 0.0)

        from nemo_evaluator.solvers.harbor import HarborSolver

        solver = HarborSolver.__new__(HarborSolver)
        solver._model_url = "http://model"
        solver._model_id = "model"
        solver._timeout = 60.0
        solver._run_timeout = 60.0
        solver._cmd_timeout = None
        solver._timeout_strategy = "override"
        solver._max_agent_timeout = None
        solver._harbor_agent = "openhands-sdk"
        solver._container_env = {}
        solver._skill = None
        solver._skill_dir = None
        solver._create_agent = lambda logs_dir, model_url="": _FakeZeroTokenAgent()

        async def fake_wait_for_agent(agent_task, sandbox, agent_started_at, effective_timeout, jitter):
            await agent_task
            return False, None

        solver._wait_for_agent = fake_wait_for_agent
        return solver

    @pytest.mark.asyncio
    async def test_zero_token_turns_flagged_in_trajectory(self, monkeypatch, caplog):
        import logging

        trajectory = [
            {
                "steps": [
                    {"source": "agent", "metrics": {"prompt_tokens": 0, "completion_tokens": 0}},
                    {"source": "agent", "metrics": {"prompt_tokens": 12, "completion_tokens": 3}},
                    {"source": "agent", "metrics": {"prompt_tokens": 0, "completion_tokens": 0}},
                ]
            }
        ]
        solver = self._build_solver(monkeypatch, trajectory)

        with caplog.at_level(logging.WARNING):
            result = await solver.solve(
                SeedResult(prompt="do task", expected_answer="", metadata={"task_id": "task-1"}),
                sandbox=_FakeZeroTokenSandbox(),
            )

        assert result.trajectory[0]["final_metrics"]["zero_token_turns"] == 2
        assert "zero-token agent turn" in caplog.text

    @pytest.mark.asyncio
    async def test_no_flag_when_all_turns_have_tokens(self, monkeypatch):
        trajectory = [
            {
                "steps": [
                    {"source": "agent", "metrics": {"prompt_tokens": 12, "completion_tokens": 3}},
                ]
            }
        ]
        solver = self._build_solver(monkeypatch, trajectory)

        result = await solver.solve(
            SeedResult(prompt="do task", expected_answer="", metadata={"task_id": "task-1"}),
            sandbox=_FakeZeroTokenSandbox(),
        )

        assert "final_metrics" not in result.trajectory[0]
