# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Tests for the terminus-2 runtime monkeypatches in nemo_evaluator.solvers.harbor.

All patches mutate the globally-imported harbor ``Terminus2``/``Chat`` classes,
so :func:`patch_sandbox` snapshots and restores them (and the module-level
idempotency guards) around every test.
"""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import pytest

import nemo_evaluator.solvers.harbor as harbor
from nemo_evaluator.solvers.harbor import (
    _patch_chat_token_anchor,
    _patch_terminus_api_token_anchor,
    _patch_terminus_cle_reset,
    _patch_terminus_unwind_min_pairs,
    _terminus2_build_local_fallback_llm_content,
    _terminus2_count_total_tokens,
)

# Module-level stand-ins whose *source* the patch functions inspect. Defining
# them here (not inline) guarantees ``inspect.getsource`` can read them.


def _query_llm_with_marker(self, *args, **kwargs):
    full_summarize_failed_with_cle = False
    return full_summarize_failed_with_cle


def _unwind_with_marker(self, chat, target_free_tokens=4000):
    min_pairs_to_remove = 10
    return min_pairs_to_remove


def _diverged_method(self, *args, **kwargs):
    return None


_FLAGS = (
    "_TERMINUS_CLE_RESET_PATCHED",
    "_TERMINUS_UNWIND_PATCHED",
    "_TERMINUS_API_ANCHOR_PATCHED",
    "_CHAT_TOKEN_ANCHOR_PATCHED",
)


@pytest.fixture
def patch_sandbox():
    """Snapshot terminus-2/Chat patch targets and guard flags; restore after."""
    from harbor.agents.terminus_2 import terminus_2 as t2
    from harbor.llms.chat import Chat

    terminus = t2.Terminus2
    saved_methods = {
        "_query_llm": terminus._query_llm,
        "_unwind": terminus._unwind_messages_to_free_tokens,
        "_count": terminus._count_total_tokens,
        "chat_chat": Chat.chat,
    }
    had_fallback = "_build_local_fallback_llm_content" in terminus.__dict__
    fallback_val = terminus.__dict__.get("_build_local_fallback_llm_content")
    had_records = "_records_api_token_anchor" in Chat.__dict__
    records_val = Chat.__dict__.get("_records_api_token_anchor")
    saved_flags = {name: getattr(harbor, name) for name in _FLAGS}

    for name in _FLAGS:
        setattr(harbor, name, False)

    yield SimpleNamespace(harbor=harbor, t2=t2, terminus=terminus, Chat=Chat)

    terminus._query_llm = saved_methods["_query_llm"]
    terminus._unwind_messages_to_free_tokens = saved_methods["_unwind"]
    terminus._count_total_tokens = saved_methods["_count"]
    Chat.chat = saved_methods["chat_chat"]
    if had_fallback:
        terminus._build_local_fallback_llm_content = fallback_val
    elif "_build_local_fallback_llm_content" in terminus.__dict__:
        delattr(terminus, "_build_local_fallback_llm_content")
    if had_records:
        Chat._records_api_token_anchor = records_val
    elif "_records_api_token_anchor" in Chat.__dict__:
        delattr(Chat, "_records_api_token_anchor")
    for name, value in saved_flags.items():
        setattr(harbor, name, value)


# ── _terminus2_count_total_tokens ────────────────────────────────────────


class TestCountTotalTokens:
    @staticmethod
    def _agent():
        return SimpleNamespace(_model_name="gpt-4o")

    @staticmethod
    def _messages(n):
        return [{"role": "user", "content": "x"} for _ in range(n)]

    def test_no_anchor_warns_once_and_falls_back(self, monkeypatch):
        monkeypatch.setattr("litellm.utils.token_counter", lambda model, messages: len(messages) * 10)
        chat = SimpleNamespace(messages=self._messages(3))
        result = _terminus2_count_total_tokens(self._agent(), chat)
        assert result == 30
        assert chat._api_anchor_warned is True

    def test_no_anchor_already_warned_does_not_rewarn(self, monkeypatch, caplog):
        monkeypatch.setattr("litellm.utils.token_counter", lambda model, messages: len(messages) * 10)
        chat = SimpleNamespace(messages=self._messages(2), _api_anchor_warned=True)
        with caplog.at_level(logging.WARNING):
            result = _terminus2_count_total_tokens(self._agent(), chat)
        assert result == 20
        assert "No API token-usage anchor" not in caplog.text

    def test_more_messages_than_anchor_adds_delta(self, monkeypatch):
        monkeypatch.setattr("litellm.utils.token_counter", lambda model, messages: len(messages) * 10)
        chat = SimpleNamespace(messages=self._messages(5), _api_token_anchor=(2, 1000))
        result = _terminus2_count_total_tokens(self._agent(), chat)
        assert result == 1000 + 3 * 10

    def test_exactly_anchor_returns_api_total(self, monkeypatch):
        monkeypatch.setattr("litellm.utils.token_counter", lambda model, messages: len(messages) * 10)
        chat = SimpleNamespace(messages=self._messages(5), _api_token_anchor=(5, 1234))
        result = _terminus2_count_total_tokens(self._agent(), chat)
        assert result == 1234

    def test_fewer_than_anchor_uses_plain_estimate_and_logs_once(self, monkeypatch, caplog):
        monkeypatch.setattr("litellm.utils.token_counter", lambda model, messages: len(messages) * 10)
        chat = SimpleNamespace(messages=self._messages(5), _api_token_anchor=(8, 1000))
        with caplog.at_level(logging.DEBUG):
            result = _terminus2_count_total_tokens(self._agent(), chat)
        assert result == 50
        assert chat._api_anchor_below_warned is True
        assert "shrank below the API anchor" in caplog.text

    def test_fewer_than_anchor_does_not_relog(self, monkeypatch, caplog):
        monkeypatch.setattr("litellm.utils.token_counter", lambda model, messages: len(messages) * 10)
        chat = SimpleNamespace(messages=self._messages(5), _api_token_anchor=(8, 1000), _api_anchor_below_warned=True)
        with caplog.at_level(logging.DEBUG):
            result = _terminus2_count_total_tokens(self._agent(), chat)
        assert result == 50
        assert "shrank below the API anchor" not in caplog.text


# ── _terminus2_build_local_fallback_llm_content ──────────────────────────


class TestBuildLocalFallback:
    def test_xml_parser_emits_xml(self):
        out = _terminus2_build_local_fallback_llm_content(SimpleNamespace(_parser_name="xml"))
        assert "<response>" in out
        assert "<task_complete>false</task_complete>" in out
        assert "<commands>" in out

    def test_non_xml_parser_emits_json(self):
        out = _terminus2_build_local_fallback_llm_content(SimpleNamespace(_parser_name="json"))
        data = json.loads(out)
        assert data["task_complete"] is False
        assert data["commands"] == []
        assert "analysis" in data and "plan" in data


# ── _patch_terminus_cle_reset ────────────────────────────────────────────


class TestPatchCleReset:
    def test_apply(self, patch_sandbox):
        terminus = patch_sandbox.terminus
        original = terminus._query_llm
        _patch_terminus_cle_reset()
        assert harbor._TERMINUS_CLE_RESET_PATCHED is True
        assert terminus._query_llm is not original
        assert hasattr(terminus, "_build_local_fallback_llm_content")

    def test_idempotent_when_flag_set(self, patch_sandbox):
        terminus = patch_sandbox.terminus
        harbor._TERMINUS_CLE_RESET_PATCHED = True
        original = terminus._query_llm
        _patch_terminus_cle_reset()
        assert terminus._query_llm is original

    def test_skips_when_source_already_marked(self, patch_sandbox):
        terminus = patch_sandbox.terminus
        terminus._query_llm = _query_llm_with_marker
        _patch_terminus_cle_reset()
        assert harbor._TERMINUS_CLE_RESET_PATCHED is True
        assert terminus._query_llm is _query_llm_with_marker

    def test_diverged_source_raises(self, patch_sandbox):
        patch_sandbox.terminus._query_llm = _diverged_method
        with pytest.raises(RuntimeError, match="diverged"):
            _patch_terminus_cle_reset()


# ── _patch_terminus_unwind_min_pairs ─────────────────────────────────────


class _FakeChat:
    def __init__(self, messages):
        self._messages = list(messages)

    @property
    def messages(self):
        return self._messages

    def reset_response_chain(self):
        pass


class TestPatchUnwindMinPairs:
    def test_apply(self, patch_sandbox):
        terminus = patch_sandbox.terminus
        original = terminus._unwind_messages_to_free_tokens
        _patch_terminus_unwind_min_pairs()
        assert harbor._TERMINUS_UNWIND_PATCHED is True
        assert terminus._unwind_messages_to_free_tokens is not original

    def test_removes_at_least_min_pairs_even_when_target_met(self, patch_sandbox):
        terminus = patch_sandbox.terminus
        _patch_terminus_unwind_min_pairs()
        unwind = terminus._unwind_messages_to_free_tokens

        fake_self = SimpleNamespace(
            _llm=SimpleNamespace(get_model_context_limit=lambda: 100_000),
            logger=SimpleNamespace(debug=lambda *a, **k: None),
            _count_total_tokens=lambda chat: 10,
        )
        min_pairs = harbor._TERMINUS_UNWIND_MIN_PAIRS
        remainder = 4
        chat = _FakeChat(["system"] + [f"m{i}" for i in range(2 * min_pairs + remainder)])

        unwind(fake_self, chat, target_free_tokens=4000)

        # Free-token target is met from the start, but the minimum of min_pairs
        # pairs (2 * min_pairs messages) must still be removed.
        assert len(chat.messages) == 1 + remainder

    def test_idempotent_when_flag_set(self, patch_sandbox):
        terminus = patch_sandbox.terminus
        harbor._TERMINUS_UNWIND_PATCHED = True
        original = terminus._unwind_messages_to_free_tokens
        _patch_terminus_unwind_min_pairs()
        assert terminus._unwind_messages_to_free_tokens is original

    def test_skips_when_source_already_marked(self, patch_sandbox):
        terminus = patch_sandbox.terminus
        terminus._unwind_messages_to_free_tokens = _unwind_with_marker
        _patch_terminus_unwind_min_pairs()
        assert harbor._TERMINUS_UNWIND_PATCHED is True
        assert terminus._unwind_messages_to_free_tokens is _unwind_with_marker

    def test_diverged_source_raises(self, patch_sandbox):
        patch_sandbox.terminus._unwind_messages_to_free_tokens = _diverged_method
        with pytest.raises(RuntimeError, match="diverged"):
            _patch_terminus_unwind_min_pairs()


# ── _patch_chat_token_anchor ─────────────────────────────────────────────


class _FakeUsage:
    prompt_tokens = 100
    completion_tokens = 20
    cache_tokens = 0
    cost_usd = 0.0


class _FakeModel:
    async def call(self, prompt, message_history, logging_path=None, previous_response_id=None, **kwargs):
        return SimpleNamespace(
            usage=_FakeUsage(),
            content="ok",
            response_id=None,
            prompt_token_ids=None,
            completion_token_ids=None,
            logprobs=None,
            extra=None,
            reasoning_content=None,
        )


class TestPatchChatTokenAnchor:
    async def test_records_anchor_after_chat(self, patch_sandbox):
        Chat = patch_sandbox.Chat
        _patch_chat_token_anchor()
        assert harbor._CHAT_TOKEN_ANCHOR_PATCHED is True

        chat = Chat(_FakeModel())
        await chat.chat("hello")

        # Two messages appended (user + assistant); anchor = prompt + completion.
        assert chat._api_token_anchor == (2, 120)

    async def test_partial_usage_does_not_record_anchor(self, patch_sandbox):
        Chat = patch_sandbox.Chat

        async def _stub_chat(self, *args, **kwargs):
            return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=100))

        Chat.chat = _stub_chat
        _patch_chat_token_anchor()

        holder = SimpleNamespace(_messages=["user", "assistant"])
        await Chat.chat(holder, "hi")

        assert getattr(holder, "_api_token_anchor", None) is None

    def test_idempotent_when_flag_set(self, patch_sandbox):
        Chat = patch_sandbox.Chat
        harbor._CHAT_TOKEN_ANCHOR_PATCHED = True
        original = Chat.chat
        _patch_chat_token_anchor()
        assert Chat.chat is original

    def test_skips_when_chat_already_records(self, patch_sandbox):
        Chat = patch_sandbox.Chat
        Chat._records_api_token_anchor = True
        original = Chat.chat
        _patch_chat_token_anchor()
        assert harbor._CHAT_TOKEN_ANCHOR_PATCHED is True
        assert Chat.chat is original


# ── _patch_terminus_api_token_anchor ─────────────────────────────────────


class TestPatchApiTokenAnchor:
    def test_apply(self, patch_sandbox):
        terminus = patch_sandbox.terminus
        Chat = patch_sandbox.Chat
        _patch_terminus_api_token_anchor()
        assert harbor._TERMINUS_API_ANCHOR_PATCHED is True
        assert terminus._count_total_tokens is _terminus2_count_total_tokens
        assert harbor._CHAT_TOKEN_ANCHOR_PATCHED is True
        assert getattr(Chat, "_records_api_token_anchor", False) is True

    def test_idempotent_when_flag_set(self, patch_sandbox):
        terminus = patch_sandbox.terminus
        harbor._TERMINUS_API_ANCHOR_PATCHED = True
        original = terminus._count_total_tokens
        _patch_terminus_api_token_anchor()
        assert terminus._count_total_tokens is original

    def test_diverged_source_raises(self, patch_sandbox):
        patch_sandbox.terminus._count_total_tokens = _diverged_method
        with pytest.raises(RuntimeError, match="diverged"):
            _patch_terminus_api_token_anchor()


# ── HarborSolver._create_agent gating ────────────────────────────────────


class TestCreateAgentGating:
    @staticmethod
    def _solver(agent):
        from unittest.mock import patch

        with patch("nemo_evaluator.solvers.harbor._check_harbor_installed"):
            from nemo_evaluator.solvers.harbor import HarborSolver

            return HarborSolver(
                harbor_agent=agent,
                model_url="http://localhost:8000",
                model_id="glm5",
                api_key="test-key",
            )

    def test_terminus2_triggers_all_patches(self, tmp_path, monkeypatch):
        from unittest.mock import MagicMock, patch

        solver = self._solver("terminus-2")
        cle, unwind, anchor = MagicMock(), MagicMock(), MagicMock()
        monkeypatch.setattr("nemo_evaluator.solvers.harbor._patch_terminus_cle_reset", cle)
        monkeypatch.setattr("nemo_evaluator.solvers.harbor._patch_terminus_unwind_min_pairs", unwind)
        monkeypatch.setattr("nemo_evaluator.solvers.harbor._patch_terminus_api_token_anchor", anchor)

        sentinel = object()
        with patch("harbor.agents.factory.AgentFactory.create_agent_from_name", return_value=sentinel):
            result = solver._create_agent(tmp_path)

        assert result is sentinel
        cle.assert_called_once()
        unwind.assert_called_once()
        anchor.assert_called_once()

    def test_non_terminus_agent_skips_patches(self, tmp_path, monkeypatch):
        from unittest.mock import MagicMock, patch

        solver = self._solver("openhands")
        cle, unwind, anchor = MagicMock(), MagicMock(), MagicMock()
        monkeypatch.setattr("nemo_evaluator.solvers.harbor._patch_terminus_cle_reset", cle)
        monkeypatch.setattr("nemo_evaluator.solvers.harbor._patch_terminus_unwind_min_pairs", unwind)
        monkeypatch.setattr("nemo_evaluator.solvers.harbor._patch_terminus_api_token_anchor", anchor)

        with patch("harbor.agents.factory.AgentFactory.create_agent_from_name", return_value=object()):
            solver._create_agent(tmp_path)

        cle.assert_not_called()
        unwind.assert_not_called()
        anchor.assert_not_called()
