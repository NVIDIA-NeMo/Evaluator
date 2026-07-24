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
    _patch_harbor_lite_llm_context_length_matcher,
    _patch_terminus_api_token_anchor,
    _patch_terminus_cle_reset,
    _patch_terminus_compaction_logging,
    _patch_terminus_compaction_turn_marker,
    _patch_terminus_unwind_min_pairs,
    _terminus2_build_local_fallback_llm_content,
    _terminus2_count_total_tokens,
    _terminus2_nel_dump_trajectory_with_continuation_index,
    _terminus2_nel_flush_pending_compaction,
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
    "_TERMINUS_COMPACTION_PATCHED",
    "_TERMINUS_COMPACTION_TURN_MARKER_PATCHED",
)

_NEL_METHODS = (
    "_nel_ensure_compaction_state",
    "_nel_stash_compaction_token_snapshots",
    "_nel_token_snapshot",
    "_nel_compaction_mechanism",
    "_nel_make_compaction_event",
    "_nel_record_failed_compaction_attempt",
    "_nel_set_proactive_pending_compaction",
    "_nel_set_cle_pending_compaction",
    "_nel_flush_pending_compaction",
    "_nel_compaction_llm_call_kwargs",
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
        "_run_agent_loop": terminus._run_agent_loop,
        "_check_proactive": terminus._check_proactive_summarization,
        "_dump": terminus._dump_trajectory_with_continuation_index,
        "_run_subagent": terminus._run_subagent,
        "_summarize": terminus._summarize,
        "chat_chat": Chat.chat,
    }
    saved_nel = {name: terminus.__dict__.get(name) for name in _NEL_METHODS}
    had_fallback = "_build_local_fallback_llm_content" in terminus.__dict__
    fallback_val = terminus.__dict__.get("_build_local_fallback_llm_content")
    had_records = "_records_api_token_anchor" in Chat.__dict__
    records_val = Chat.__dict__.get("_records_api_token_anchor")
    saved_flags = {name: getattr(harbor, name) for name in _FLAGS}
    saved_query_src = harbor._TERMINUS_QUERY_LLM_SRC

    for name in _FLAGS:
        setattr(harbor, name, False)

    yield SimpleNamespace(harbor=harbor, t2=t2, terminus=terminus, Chat=Chat)

    terminus._query_llm = saved_methods["_query_llm"]
    terminus._unwind_messages_to_free_tokens = saved_methods["_unwind"]
    terminus._count_total_tokens = saved_methods["_count"]
    terminus._run_agent_loop = saved_methods["_run_agent_loop"]
    terminus._check_proactive_summarization = saved_methods["_check_proactive"]
    terminus._dump_trajectory_with_continuation_index = saved_methods["_dump"]
    terminus._run_subagent = saved_methods["_run_subagent"]
    terminus._summarize = saved_methods["_summarize"]
    Chat.chat = saved_methods["chat_chat"]
    for name, value in saved_nel.items():
        if value is not None:
            setattr(terminus, name, value)
        elif name in terminus.__dict__:
            delattr(terminus, name)
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
    harbor._TERMINUS_QUERY_LLM_SRC = saved_query_src


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


# ── anchored estimate + custom tokenizer integration ─────────────────────


class TestAnchoredEstimateUsesCustomTokenizer:
    """The anchored estimate routes its token_counter calls through the custom tokenizer."""

    @pytest.fixture(autouse=True)
    def _reset_counter_patch(self, monkeypatch):
        monkeypatch.setattr(harbor, "_TOKEN_COUNTER_PATCHED", False)
        monkeypatch.setattr(harbor, "_TOKENIZER_REGISTRY", {})

    @staticmethod
    def _install_with_fake_original(monkeypatch):
        def fake_original(*args, model=None, messages=None, custom_tokenizer=None, **kwargs):
            per_msg = 100 if custom_tokenizer is not None else 10
            return len(messages) * per_msg

        monkeypatch.setattr("litellm.utils.token_counter", fake_original)
        harbor._install_token_counter_patch()

    @staticmethod
    def _messages(n):
        return [{"role": "user", "content": "x"} for _ in range(n)]

    def test_remainder_estimate_uses_custom_tokenizer(self, monkeypatch):
        self._install_with_fake_original(monkeypatch)
        harbor._TOKENIZER_REGISTRY["M"] = {"sentinel": True}

        agent = SimpleNamespace(_model_name="M")
        chat = SimpleNamespace(messages=self._messages(5), _api_token_anchor=(2, 1000))
        result = _terminus2_count_total_tokens(agent, chat)
        assert result == 1000 + 3 * 100

    def test_full_fallback_uses_custom_tokenizer(self, monkeypatch):
        self._install_with_fake_original(monkeypatch)
        harbor._TOKENIZER_REGISTRY["M"] = {"sentinel": True}

        agent = SimpleNamespace(_model_name="M")
        chat = SimpleNamespace(messages=self._messages(4))
        result = _terminus2_count_total_tokens(agent, chat)
        assert result == 4 * 100

    def test_unregistered_model_uses_default_tokenizer(self, monkeypatch):
        self._install_with_fake_original(monkeypatch)

        agent = SimpleNamespace(_model_name="unregistered")
        chat = SimpleNamespace(messages=self._messages(5), _api_token_anchor=(2, 1000))
        result = _terminus2_count_total_tokens(agent, chat)
        assert result == 1000 + 3 * 10


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
        cle, unwind, anchor, ctxlim, compaction, turn_marker = (
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
        )
        monkeypatch.setattr("nemo_evaluator.solvers.harbor._patch_terminus_cle_reset", cle)
        monkeypatch.setattr("nemo_evaluator.solvers.harbor._patch_terminus_unwind_min_pairs", unwind)
        monkeypatch.setattr("nemo_evaluator.solvers.harbor._patch_terminus_api_token_anchor", anchor)
        monkeypatch.setattr(
            "nemo_evaluator.solvers.harbor._patch_harbor_lite_llm_context_length_matcher",
            ctxlim,
        )
        monkeypatch.setattr("nemo_evaluator.solvers.harbor._patch_terminus_compaction_logging", compaction)
        monkeypatch.setattr(
            "nemo_evaluator.solvers.harbor._patch_terminus_compaction_turn_marker",
            turn_marker,
        )

        sentinel = object()
        with patch("harbor.agents.factory.AgentFactory.create_agent_from_name", return_value=sentinel):
            result = solver._create_agent(tmp_path)

        assert result is sentinel
        cle.assert_called_once()
        unwind.assert_called_once()
        anchor.assert_called_once()
        ctxlim.assert_called_once()
        compaction.assert_called_once()
        turn_marker.assert_called_once()

    def test_non_terminus_agent_skips_patches(self, tmp_path, monkeypatch):
        from unittest.mock import MagicMock, patch

        solver = self._solver("openhands")
        cle, unwind, anchor, ctxlim, compaction, turn_marker = (
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
        )
        monkeypatch.setattr("nemo_evaluator.solvers.harbor._patch_terminus_cle_reset", cle)
        monkeypatch.setattr("nemo_evaluator.solvers.harbor._patch_terminus_unwind_min_pairs", unwind)
        monkeypatch.setattr("nemo_evaluator.solvers.harbor._patch_terminus_api_token_anchor", anchor)
        monkeypatch.setattr(
            "nemo_evaluator.solvers.harbor._patch_harbor_lite_llm_context_length_matcher",
            ctxlim,
        )
        monkeypatch.setattr("nemo_evaluator.solvers.harbor._patch_terminus_compaction_logging", compaction)
        monkeypatch.setattr(
            "nemo_evaluator.solvers.harbor._patch_terminus_compaction_turn_marker",
            turn_marker,
        )

        with patch("harbor.agents.factory.AgentFactory.create_agent_from_name", return_value=object()):
            solver._create_agent(tmp_path)

        cle.assert_not_called()
        unwind.assert_not_called()
        anchor.assert_not_called()
        ctxlim.assert_not_called()
        compaction.assert_not_called()
        turn_marker.assert_not_called()


# ── _patch_harbor_lite_llm_context_length_matcher ────────────────────────


@pytest.fixture
def ctxlim_patch_sandbox():
    """Snapshot harbor.llms.lite_llm.LiteLLM._is_context_length_error and the
    module-level idempotency guard; restore after the test."""
    from harbor.llms import lite_llm as harbor_litellm

    saved_method = harbor_litellm.LiteLLM._is_context_length_error
    saved_flag = harbor._HARBOR_LITELLM_CTXLIM_PATCHED
    harbor._HARBOR_LITELLM_CTXLIM_PATCHED = False
    try:
        yield harbor_litellm
    finally:
        harbor_litellm.LiteLLM._is_context_length_error = saved_method
        harbor._HARBOR_LITELLM_CTXLIM_PATCHED = saved_flag


_VLLM_ACTUAL_BODY = (
    "You passed 262145 input tokens and requested 0 output tokens. "
    "However, the model's context length is only 262144 tokens, "
    "resulting in a maximum input length of 262144 tokens. Please "
    "reduce the length of the input prompt. "
    "(parameter=input_tokens, value=262145)"
)


class _CtxlimFakeError(Exception):
    """Exception subclass whose ``str(err)`` carries the phrase under test."""


class TestHarborLiteLLMContextLengthMatcher:
    """The patch widens Harbor's substring matcher to recognize vLLM's actual
    ``VLLMValidationError`` phrasing (``"You passed N input tokens... However,
    the model's context length is only M tokens..."``) raised by self-hosted
    vLLM deployments when a request exceeds the configured context length.
    Without this, the wire error stays a plain ``BadRequestError`` and
    Terminus-2's reactive summarization (``terminus_2.py::_query_llm``
    ``except ContextLengthExceededError`` block) never triggers."""

    @staticmethod
    def _install():
        _patch_harbor_lite_llm_context_length_matcher()
        from harbor.llms.lite_llm import LiteLLM

        return LiteLLM.__new__(LiteLLM)

    @pytest.mark.parametrize(
        ("error", "expected"),
        [
            pytest.param(
                SimpleNamespace(body="", message=_VLLM_ACTUAL_BODY, error=""),
                True,
                id="vllm_body_in_message_attr",
            ),
            pytest.param(
                _CtxlimFakeError("litellm.BadRequestError: " + _VLLM_ACTUAL_BODY),
                True,
                id="vllm_body_only_in_str_of_exception",
            ),
            pytest.param(
                SimpleNamespace(body=_VLLM_ACTUAL_BODY, message="", error=""),
                True,
                id="vllm_body_in_body_attr",
            ),
            pytest.param(
                SimpleNamespace(
                    body="",
                    message="Error code: 400 - context length exceeded (see docs)",
                    error="",
                ),
                True,
                id="original_openai_phrasing_still_matches",
            ),
            pytest.param(
                SimpleNamespace(
                    body="",
                    message="Invalid tool call format: expected JSON object",
                    error="",
                ),
                False,
                id="unrelated_error_does_not_match",
            ),
            pytest.param(
                SimpleNamespace(
                    body="",
                    message="You asked about the context limit; here is the answer.",
                    error="",
                ),
                False,
                id="similar_but_non_overflow_phrase_does_not_match",
            ),
        ],
    )
    def test_is_context_length_error(self, ctxlim_patch_sandbox, error, expected):
        instance = self._install()
        assert instance._is_context_length_error(error) is expected

    def test_idempotent(self, ctxlim_patch_sandbox):
        _patch_harbor_lite_llm_context_length_matcher()
        first = ctxlim_patch_sandbox.LiteLLM._is_context_length_error
        _patch_harbor_lite_llm_context_length_matcher()
        assert ctxlim_patch_sandbox.LiteLLM._is_context_length_error is first

    def test_drift_marker_present_but_phrase_missing_logs_warning(self, ctxlim_patch_sandbox, caplog):
        """If vLLM's `parameter=input_tokens` marker is in the body but no
        classifier phrase matches, the patched matcher must still return False
        (agent still crashes) but log a WARNING so the drift is visible."""
        instance = self._install()
        drifted = SimpleNamespace(
            body="SomeFutureVLLMPhrasing: 262145 tokens is over the limit. (parameter=input_tokens, value=262145)",
            message="",
            error="",
        )
        with caplog.at_level(logging.WARNING, logger="nemo_evaluator.solvers.harbor"):
            result = instance._is_context_length_error(drifted)
        assert result is False, "drift alarm must NOT reclassify — fix the classifier instead"
        assert any("stable ctx-overflow marker" in rec.getMessage() for rec in caplog.records), (
            "expected drift-alarm WARNING when the marker is present but no phrase matches"
        )

    def test_no_drift_warning_when_phrase_matches(self, ctxlim_patch_sandbox, caplog):
        """When a classifier phrase DOES match, the drift alarm must stay quiet
        even if the marker is also present — otherwise every real overflow
        would trigger a spurious warning."""
        instance = self._install()
        real_body = SimpleNamespace(
            body=_VLLM_ACTUAL_BODY,  # contains BOTH "model's context length" AND parameter=input_tokens
            message="",
            error="",
        )
        with caplog.at_level(logging.WARNING, logger="nemo_evaluator.solvers.harbor"):
            result = instance._is_context_length_error(real_body)
        assert result is True
        assert not any("stable ctx-overflow marker" in rec.getMessage() for rec in caplog.records), (
            "drift alarm fired even though the classifier caught the overflow"
        )

    def test_no_drift_warning_when_neither_marker_nor_phrase(self, ctxlim_patch_sandbox, caplog):
        """Unrelated BadRequestErrors must not trigger the drift alarm."""
        instance = self._install()
        err = SimpleNamespace(
            body="",
            message="Invalid tool call format: expected JSON object",
            error="",
        )
        with caplog.at_level(logging.WARNING, logger="nemo_evaluator.solvers.harbor"):
            result = instance._is_context_length_error(err)
        assert result is False
        assert not any("stable ctx-overflow marker" in rec.getMessage() for rec in caplog.records)


# ── _patch_terminus_compaction_logging ───────────────────────────────────


class TestPatchCompactionLogging:
    def test_apply_patches_run_loop_and_query_llm(self, patch_sandbox):
        terminus = patch_sandbox.terminus
        _patch_terminus_cle_reset()
        _patch_terminus_unwind_min_pairs()
        _patch_terminus_compaction_logging()
        assert harbor._TERMINUS_COMPACTION_PATCHED is True
        assert harbor._TERMINUS_QUERY_LLM_SRC is not None
        assert "_nel_set_cle_pending_compaction" in harbor._TERMINUS_QUERY_LLM_SRC
        assert hasattr(terminus, "_nel_flush_pending_compaction")

    def test_flush_appends_compaction_step_without_handoff_user_step(self, patch_sandbox, tmp_path):
        import json

        from nemo_evaluator.solvers.compaction_logging import CompactionTokens

        _patch_terminus_compaction_logging()
        terminus = patch_sandbox.terminus
        agent = terminus(
            logs_dir=tmp_path,
            model_name="test-model",
            model_info={"max_input_tokens": 1000, "max_output_tokens": 1000},
        )
        agent._trajectory_steps = []
        agent._linear_history = False
        agent._summarization_count = 1
        agent._pending_handoff_prompt = "handoff"
        agent._nel_stashed_tokens_before = CompactionTokens(
            prompt_tokens_approx=190_000,
            context_limit=200_000,
            free_tokens=10_000,
        )
        agent._nel_stashed_tokens_after = CompactionTokens(
            prompt_tokens_approx=8_000,
            context_limit=200_000,
            free_tokens=192_000,
        )
        agent._nel_stashed_tokens_intermediate = None
        agent._nel_pending_compaction = None
        chat = SimpleNamespace(
            messages=[
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": "question prompt"},
                {"role": "assistant", "content": "model questions"},
            ]
        )
        from nemo_evaluator.solvers.harbor import _terminus2_nel_set_proactive_pending_compaction

        _terminus2_nel_set_proactive_pending_compaction(agent, chat, "handoff", None)
        _terminus2_nel_flush_pending_compaction(agent, chat)
        assert len(agent._trajectory_steps) == 1
        step = agent._trajectory_steps[0]
        assert step.source == "system"
        assert step.extra is not None
        assert step.extra["context_management"]["type"] == "compaction"
        assert step.extra["compaction"]["llm_call_count"] == 3
        content = json.loads(step.observation.results[0].content)
        assert content == [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "question prompt"},
            {"role": "assistant", "content": "model questions"},
            {"role": "user", "content": "handoff"},
        ]
        assert agent._pending_handoff_prompt is None

    def test_cle_fallback_does_not_mutate_summarization_count(self, patch_sandbox, tmp_path):
        """Regression: NEL owns its own compaction counter and must not touch
        Terminus's ``_summarization_count`` (which drives linear_history
        continuation file naming). ``_summarize`` bumps that counter before it
        can fail, so on the CLE short/ultimate fallback NEL previously
        double-incremented it and corrupted the continuation chain."""
        from nemo_evaluator.solvers.compaction_logging import CompactionTokens
        from nemo_evaluator.solvers.harbor import _terminus2_nel_set_cle_pending_compaction

        _patch_terminus_compaction_logging()
        terminus = patch_sandbox.terminus
        agent = terminus(
            logs_dir=tmp_path,
            model_name="test-model",
            model_info={"max_input_tokens": 1000, "max_output_tokens": 1000},
        )
        agent._trajectory_steps = []
        agent._linear_history = False
        chat = SimpleNamespace(messages=[{"role": "system", "content": "s"}])

        def _snap():
            return CompactionTokens(prompt_tokens_approx=8_000, context_limit=200_000, free_tokens=192_000)

        agent._summarization_count = 5

        for expected_index in (1, 2):
            _terminus2_nel_set_cle_pending_compaction(
                agent,
                chat,
                "handoff",
                "terminus_short_fallback",
                [],
                None,
                _snap(),
                _snap(),
                None,
                None,
            )
            assert agent._nel_pending_compaction.compaction_index == expected_index

        assert agent._summarization_count == 5

    def test_idempotent_when_marker_present(self, patch_sandbox):
        terminus = patch_sandbox.terminus

        async def _run_agent_loop_with_marker(self, *args, **kwargs):
            self._nel_flush_pending_compaction(None)

        terminus._run_agent_loop = _run_agent_loop_with_marker
        original = terminus._run_agent_loop
        _patch_terminus_compaction_logging()
        assert harbor._TERMINUS_COMPACTION_PATCHED is True
        assert terminus._run_agent_loop is original


class TestPatchTerminusCompactionTurnMarker:
    def test_marks_summarize_subagent_calls_not_generic_subagent(self, patch_sandbox):
        terminus = patch_sandbox.terminus
        _patch_terminus_compaction_logging()
        _patch_terminus_compaction_turn_marker()
        assert harbor._TERMINUS_COMPACTION_TURN_MARKER_PATCHED is True
        assert hasattr(terminus, "_nel_compaction_llm_call_kwargs")
        assert harbor._TERMINUS_QUERY_LLM_SRC is not None
        assert "_nel_compaction_llm_call_kwargs()" in harbor._TERMINUS_QUERY_LLM_SRC
        summarize_src = terminus._summarize.__code__.co_consts
        assert (
            any(isinstance(c, str) and "_nel_compaction_llm_call_kwargs" in c for c in summarize_src)
            or "_nel_compaction_llm_call_kwargs" in terminus._summarize.__code__.co_names
        )
        assert "llm_call_kwargs" in terminus._run_subagent.__code__.co_varnames

    def test_idempotent(self, patch_sandbox):
        terminus = patch_sandbox.terminus
        _patch_terminus_compaction_logging()
        _patch_terminus_compaction_turn_marker()
        first_subagent = terminus._run_subagent
        first_summarize = terminus._summarize
        _patch_terminus_compaction_turn_marker()
        assert terminus._run_subagent is first_subagent
        assert terminus._summarize is first_summarize

    def test_compaction_kwargs_carry_header_default_subagent_does_not(self, patch_sandbox, tmp_path):
        from nemo_evaluator.adapters.call_kind import NEL_CALL_KIND_COMPACTION, NEL_CALL_KIND_HEADER

        terminus = patch_sandbox.terminus
        _patch_terminus_compaction_logging()
        _patch_terminus_compaction_turn_marker()
        agent = terminus(
            logs_dir=tmp_path,
            model_name="test-model",
            model_info={"max_input_tokens": 1000, "max_output_tokens": 1000},
        )
        agent._llm_call_kwargs = {"temperature": 0.2}
        marked = agent._nel_compaction_llm_call_kwargs()
        assert marked["extra_headers"][NEL_CALL_KIND_HEADER] == NEL_CALL_KIND_COMPACTION
        assert agent._llm_call_kwargs == {"temperature": 0.2}
        assert "extra_headers" not in agent._llm_call_kwargs


# ── _terminus2_nel_dump_trajectory_with_continuation_index ───────────────


def _make_agent_for_dump(terminus_cls, tmp_path, *, linear_history=False, summarization_count=0):
    from datetime import datetime, timezone
    from types import SimpleNamespace

    from harbor.models.trajectories.step import Step

    agent = terminus_cls(
        logs_dir=tmp_path,
        model_name="test-model",
        model_info={"max_input_tokens": 1000, "max_output_tokens": 1000},
    )
    agent._trajectory_steps = [
        Step(
            step_id=1,
            timestamp=datetime.now(timezone.utc).isoformat(),
            source="user",
            message="start",
        )
    ]
    agent._linear_history = linear_history
    agent._summarization_count = summarization_count
    agent._session_id = "test-session"
    agent._parser_name = "default"
    agent._temperature = 0.0
    agent._llm_kwargs = {}
    agent._context = SimpleNamespace(
        n_input_tokens=100,
        n_output_tokens=50,
        n_cache_tokens=0,
        cost_usd=0.01,
    )
    return agent


class TestDumpTrajectoryWithContinuationIndex:
    def test_no_context_skips_write(self, patch_sandbox, tmp_path):
        terminus = patch_sandbox.terminus
        agent = terminus(
            logs_dir=tmp_path,
            model_name="test-model",
            model_info={"max_input_tokens": 1000, "max_output_tokens": 1000},
        )
        agent._context = None
        agent._trajectory_steps = []
        agent._linear_history = False
        agent._summarization_count = 0
        _terminus2_nel_dump_trajectory_with_continuation_index(agent, 0)
        assert not (tmp_path / "trajectory.json").exists()

    def test_non_linear_writes_trajectory_json(self, patch_sandbox, tmp_path):
        terminus = patch_sandbox.terminus
        agent = _make_agent_for_dump(terminus, tmp_path, linear_history=False)
        _terminus2_nel_dump_trajectory_with_continuation_index(agent, 0)
        traj_path = tmp_path / "trajectory.json"
        assert traj_path.exists()
        data = json.loads(traj_path.read_text())
        assert data["session_id"] == "test-session"
        # Non-linear should not set continuation_index in agent extra
        assert "continuation_index" not in (data.get("agent") or {}).get("extra", {})

    def test_linear_continuation_index_gt_0_writes_cont_file(self, patch_sandbox, tmp_path):
        terminus = patch_sandbox.terminus
        agent = _make_agent_for_dump(terminus, tmp_path, linear_history=True, summarization_count=2)
        _terminus2_nel_dump_trajectory_with_continuation_index(agent, 1)
        cont_path = tmp_path / "trajectory.cont-1.json"
        assert cont_path.exists()
        data = json.loads(cont_path.read_text())
        assert data["agent"]["extra"]["continuation_index"] == 1

    def test_linear_points_to_next_continuation_when_not_last(self, patch_sandbox, tmp_path):
        terminus = patch_sandbox.terminus
        agent = _make_agent_for_dump(terminus, tmp_path, linear_history=True, summarization_count=3)
        _terminus2_nel_dump_trajectory_with_continuation_index(agent, 1)
        data = json.loads((tmp_path / "trajectory.cont-1.json").read_text())
        assert data.get("continued_trajectory_ref") == "trajectory.cont-2.json"

    def test_linear_last_segment_has_no_continued_ref(self, patch_sandbox, tmp_path):
        terminus = patch_sandbox.terminus
        agent = _make_agent_for_dump(terminus, tmp_path, linear_history=True, summarization_count=2)
        _terminus2_nel_dump_trajectory_with_continuation_index(agent, 2)
        data = json.loads((tmp_path / "trajectory.cont-2.json").read_text())
        assert not data.get("continued_trajectory_ref")

    def test_root_in_linear_chain_points_to_first_continuation(self, patch_sandbox, tmp_path):
        terminus = patch_sandbox.terminus
        agent = _make_agent_for_dump(terminus, tmp_path, linear_history=True, summarization_count=1)
        _terminus2_nel_dump_trajectory_with_continuation_index(agent, 0)
        data = json.loads((tmp_path / "trajectory.json").read_text())
        assert data.get("continued_trajectory_ref") == "trajectory.cont-1.json"
        # Root segment with index=0 must NOT carry continuation_index in agent extra
        assert "continuation_index" not in data["agent"]["extra"]
