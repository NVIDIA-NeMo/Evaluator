"""Tests for nemo_evaluator.solvers.harbor — all mocked, no Docker/Harbor."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock


from nemo_evaluator.solvers.harbor import (
    _ensure_env,
    _extract_response,
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


class TestEnsureEnv:
    def test_sets_env_vars(self, monkeypatch):
        for v in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL", "LITELLM_LOG"):
            monkeypatch.delenv(v, raising=False)
        _ensure_env("key-1", "http://localhost:5000/v1", "my-model")
        assert os.environ["LLM_API_KEY"] == "key-1"
        assert os.environ["LLM_BASE_URL"] == "http://localhost:5000/v1"
        assert os.environ["LLM_MODEL"] == "openai/my-model"

    def test_no_key_uses_dummy(self, monkeypatch):
        for v in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL"):
            monkeypatch.delenv(v, raising=False)
        _ensure_env(None, None, None)
        assert os.environ["LLM_API_KEY"] == "no-key-needed"

    def test_model_no_prefix_without_url(self, monkeypatch):
        monkeypatch.delenv("LLM_MODEL", raising=False)
        _ensure_env("k", None, "bare-model")
        assert os.environ["LLM_MODEL"] == "bare-model"

    def test_already_prefixed_model(self, monkeypatch):
        monkeypatch.delenv("LLM_MODEL", raising=False)
        _ensure_env("k", "http://x", "openai/already")
        assert os.environ["LLM_MODEL"] == "openai/already"


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
        assert sandbox.exec.call_count >= 4
