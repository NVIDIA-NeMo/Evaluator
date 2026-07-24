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
    _patch_terminus_unwind_min_pairs,
    _terminus2_build_local_fallback_llm_content,
    _terminus2_count_total_tokens,
)

# Module-level stand-ins whose *source* the patch functions inspect. Defining
# them here (not inline) guarantees ``inspect.getsource`` can read them.


def _query_llm_with_marker(self, *args, **kwargs):
    full_summarize_failed_with_cle = False
    marker = "EVALUATOR_LLM_ERROR_TRAJECTORY_STEP"
    usage_marker = "self._apply_failed_llm_usage(chat, e)"
    return full_summarize_failed_with_cle, marker, usage_marker


def _unwind_with_marker(self, chat, target_free_tokens=4000):
    min_pairs_to_remove = 10
    return min_pairs_to_remove


def _diverged_method(self, *args, **kwargs):
    return None


async def _litellm_call_length_anchor_missing_locals(self, *args, **kwargs):
    from harbor.llms.base import OutputLengthExceededError

    content = "missing local guard"
    if True:
        if True:
            exc = OutputLengthExceededError(
                "length",
                truncated_response=content,
            )
            raise exc


async def _responses_api_length_anchor_missing_locals(self, *args, **kwargs):
    from harbor.llms.base import OutputLengthExceededError

    content = "missing responses local guard"
    reason = "max_output_tokens"
    if True:
        if True:
            if reason == "max_output_tokens":
                raise OutputLengthExceededError(
                    f"Model {self._model_name} hit max_tokens limit. Response was truncated.",
                    truncated_response=content,
                )


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
    from harbor.llms.lite_llm import LiteLLM

    terminus = t2.Terminus2
    saved_methods = {
        "_query_llm": terminus._query_llm,
        "_run_agent_loop": terminus._run_agent_loop,
        "_unwind": terminus._unwind_messages_to_free_tokens,
        "_count": terminus._count_total_tokens,
        "chat_chat": Chat.chat,
        "litellm_call": LiteLLM.call,
        "litellm_call_responses": LiteLLM._call_responses,
    }
    had_fallback = "_build_local_fallback_llm_content" in terminus.__dict__
    fallback_val = terminus.__dict__.get("_build_local_fallback_llm_content")
    had_failed_llm_usage = "_apply_failed_llm_usage" in terminus.__dict__
    failed_llm_usage_val = terminus.__dict__.get("_apply_failed_llm_usage")
    had_failed_llm_saver = "_save_failed_llm_response" in terminus.__dict__
    failed_llm_saver_val = terminus.__dict__.get("_save_failed_llm_response")
    had_failed_llm_appender = "_append_pending_failed_llm_response_steps" in terminus.__dict__
    failed_llm_appender_val = terminus.__dict__.get("_append_pending_failed_llm_response_steps")
    had_records = "_records_api_token_anchor" in Chat.__dict__
    records_val = Chat.__dict__.get("_records_api_token_anchor")
    saved_flags = {name: getattr(harbor, name) for name in _FLAGS}

    for name in _FLAGS:
        setattr(harbor, name, False)

    yield SimpleNamespace(harbor=harbor, t2=t2, terminus=terminus, Chat=Chat)

    terminus._query_llm = saved_methods["_query_llm"]
    terminus._run_agent_loop = saved_methods["_run_agent_loop"]
    terminus._unwind_messages_to_free_tokens = saved_methods["_unwind"]
    terminus._count_total_tokens = saved_methods["_count"]
    Chat.chat = saved_methods["chat_chat"]
    LiteLLM.call = saved_methods["litellm_call"]
    LiteLLM._call_responses = saved_methods["litellm_call_responses"]
    if had_fallback:
        terminus._build_local_fallback_llm_content = fallback_val
    elif "_build_local_fallback_llm_content" in terminus.__dict__:
        delattr(terminus, "_build_local_fallback_llm_content")
    if had_failed_llm_usage:
        terminus._apply_failed_llm_usage = failed_llm_usage_val
    elif "_apply_failed_llm_usage" in terminus.__dict__:
        delattr(terminus, "_apply_failed_llm_usage")
    if had_failed_llm_saver:
        terminus._save_failed_llm_response = failed_llm_saver_val
    elif "_save_failed_llm_response" in terminus.__dict__:
        delattr(terminus, "_save_failed_llm_response")
    if had_failed_llm_appender:
        terminus._append_pending_failed_llm_response_steps = failed_llm_appender_val
    elif "_append_pending_failed_llm_response_steps" in terminus.__dict__:
        delattr(terminus, "_append_pending_failed_llm_response_steps")
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
        assert hasattr(terminus, "_apply_failed_llm_usage")
        assert hasattr(terminus, "_save_failed_llm_response")
        assert hasattr(terminus, "_append_pending_failed_llm_response_steps")

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

    def test_litellm_length_patch_requires_expected_locals_before_mutating(self, patch_sandbox):
        from harbor.llms.lite_llm import LiteLLM

        LiteLLM.call = _litellm_call_length_anchor_missing_locals
        original_responses = LiteLLM._call_responses
        original_query = patch_sandbox.terminus._query_llm
        with pytest.raises(RuntimeError, match="required source pattern"):
            _patch_terminus_cle_reset()

        assert LiteLLM.call is _litellm_call_length_anchor_missing_locals
        assert LiteLLM._call_responses is original_responses
        assert patch_sandbox.terminus._query_llm is original_query
        assert "_apply_failed_llm_usage" not in patch_sandbox.terminus.__dict__

    def test_responses_length_patch_requires_expected_locals_before_mutating(self, patch_sandbox):
        from harbor.llms.lite_llm import LiteLLM

        original_call = LiteLLM.call
        original_query = patch_sandbox.terminus._query_llm
        LiteLLM._call_responses = _responses_api_length_anchor_missing_locals
        with pytest.raises(RuntimeError, match="required source pattern"):
            _patch_terminus_cle_reset()

        assert LiteLLM.call is original_call
        assert LiteLLM._call_responses is _responses_api_length_anchor_missing_locals
        assert patch_sandbox.terminus._query_llm is original_query
        assert "_apply_failed_llm_usage" not in patch_sandbox.terminus.__dict__


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


# ── failed LLM response accounting ───────────────────────────────────────


class _FakeLiteLLMChoice(dict):
    def __init__(self, data, token_ids):
        super().__init__(data)
        self.provider_specific_fields = {"token_ids": token_ids}


class _FakeLiteLLMResponse(dict):
    def __init__(
        self,
        *,
        content,
        finish_reason,
        prompt_tokens,
        completion_tokens,
        model,
        reasoning,
        reasoning_content=None,
    ):
        message = {"content": content, "reasoning": reasoning}
        if reasoning_content is not None:
            message["reasoning_content"] = reasoning_content
        choice = _FakeLiteLLMChoice(
            {
                "message": message,
                "finish_reason": finish_reason,
                "logprobs": {"content": [{"logprob": -0.1}] * completion_tokens},
            },
            list(range(completion_tokens)),
        )
        super().__init__({"model": model, "choices": [choice]})
        self.prompt_token_ids = list(range(prompt_tokens))
        self.usage = SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            prompt_tokens_details=SimpleNamespace(cached_tokens=0),
        )
        self._hidden_params = {"response_cost": 0.0}


class _FakeResponsesAPIResponse:
    def __init__(self):
        self.status = "incomplete"
        self.incomplete_details = SimpleNamespace(reason="max_output_tokens")
        self.model = "responses-model"
        self.id = "resp-1"
        self.usage = SimpleNamespace(input_tokens=13, output_tokens=5)
        self.output = [
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text="TRUNCATED_RESPONSES_OUTPUT")],
            )
        ]
        self._hidden_params = {"response_cost": 0.0}


class _FakeLengthChat:
    def __init__(self, error):
        self._error = error
        self._messages = []
        self._cumulative_input_tokens = 0
        self._cumulative_output_tokens = 0
        self._cumulative_cache_tokens = 0
        self._cumulative_cost = 0.0

    @property
    def messages(self):
        return self._messages

    @property
    def total_input_tokens(self):
        return self._cumulative_input_tokens

    @property
    def total_output_tokens(self):
        return self._cumulative_output_tokens

    @property
    def total_cache_tokens(self):
        return self._cumulative_cache_tokens

    @property
    def total_cost(self):
        return self._cumulative_cost

    async def chat(self, *_args, **_kwargs):
        raise self._error

    def reset_response_chain(self):
        pass


class TestPatchFailedLLMResponseAccounting:
    async def test_chat_completion_length_error_keeps_usage_model_and_reasoning(self, patch_sandbox, monkeypatch):
        import litellm
        from harbor.llms.base import OutputLengthExceededError
        from harbor.llms.lite_llm import LiteLLM

        _patch_terminus_cle_reset()

        async def fake_acompletion(**_kwargs):
            return _FakeLiteLLMResponse(
                content="TRUNCATED_OUTPUT",
                finish_reason="length",
                prompt_tokens=23,
                completion_tokens=9,
                model="wire-model",
                reasoning="fallback-reasoning",
                reasoning_content="preferred-reasoning",
            )

        monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

        llm = LiteLLM(
            model_name="openai/nano-v3.5",
            api_base="http://model.test",
            api_key="test-key",
        )
        with pytest.raises(OutputLengthExceededError) as excinfo:
            await llm.call("prompt")

        error = excinfo.value
        assert error.truncated_response == "TRUNCATED_OUTPUT"
        assert error.llm_model_name == "wire-model"
        assert error.llm_reasoning_content == "preferred-reasoning"
        assert error.llm_usage.prompt_tokens == 23
        assert error.llm_usage.completion_tokens == 9

    async def test_chat_completion_length_error_uses_reasoning_fallback(self, patch_sandbox, monkeypatch):
        import litellm
        from harbor.llms.base import OutputLengthExceededError
        from harbor.llms.lite_llm import LiteLLM

        _patch_terminus_cle_reset()

        async def fake_acompletion(**_kwargs):
            return _FakeLiteLLMResponse(
                content="TRUNCATED_OUTPUT",
                finish_reason="length",
                prompt_tokens=5,
                completion_tokens=4,
                model="wire-model",
                reasoning="fallback-reasoning",
            )

        monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

        llm = LiteLLM(
            model_name="openai/nano-v3.5",
            api_base="http://model.test",
            api_key="test-key",
        )
        with pytest.raises(OutputLengthExceededError) as excinfo:
            await llm.call("prompt")

        assert excinfo.value.llm_reasoning_content == "fallback-reasoning"

    async def test_responses_api_length_error_keeps_usage_details(self, patch_sandbox, monkeypatch):
        import litellm
        from harbor.llms.base import OutputLengthExceededError
        from harbor.llms.lite_llm import LiteLLM

        _patch_terminus_cle_reset()

        async def fake_aresponses(**_kwargs):
            return _FakeResponsesAPIResponse()

        monkeypatch.setattr(litellm, "aresponses", fake_aresponses)

        llm = LiteLLM(
            model_name="openai/nano-v3.5",
            api_base="http://model.test",
            api_key="test-key",
            use_responses_api=True,
        )
        with pytest.raises(OutputLengthExceededError) as excinfo:
            await llm._call_responses("prompt")

        error = excinfo.value
        assert error.truncated_response == "TRUNCATED_RESPONSES_OUTPUT"
        assert error.llm_model_name == "responses-model"
        assert error.llm_reasoning_content is None
        assert error.llm_usage.prompt_tokens == 13
        assert error.llm_usage.completion_tokens == 5

    async def test_responses_api_length_error_keeps_details_via_call(self, patch_sandbox, monkeypatch):
        import litellm
        from harbor.llms.base import OutputLengthExceededError
        from harbor.llms.lite_llm import LiteLLM

        _patch_terminus_cle_reset()

        async def fake_aresponses(**kwargs):
            assert kwargs["input"] == [{"role": "user", "content": "prompt"}]
            return _FakeResponsesAPIResponse()

        monkeypatch.setattr(litellm, "aresponses", fake_aresponses)

        llm = LiteLLM(
            model_name="openai/nano-v3.5",
            api_base="http://model.test",
            api_key="test-key",
            use_responses_api=True,
        )
        with pytest.raises(OutputLengthExceededError) as excinfo:
            await llm.call("prompt")

        assert excinfo.value.truncated_response == "TRUNCATED_RESPONSES_OUTPUT"
        assert excinfo.value.llm_model_name == "responses-model"
        assert excinfo.value.llm_usage.prompt_tokens == 13
        assert excinfo.value.llm_usage.completion_tokens == 5

    async def test_salvaged_length_response_counts_usage_without_failed_step(self, patch_sandbox, tmp_path):
        from harbor.agents.terminus_2 import terminus_2
        from harbor.llms.base import OutputLengthExceededError
        from harbor.models.metric.usage_info import UsageInfo

        _patch_terminus_cle_reset()

        salvaged = (
            "<response><analysis>a</analysis><plan>p</plan><commands></commands>"
            "<task_complete>false</task_complete></response>"
        )
        error = OutputLengthExceededError("length", truncated_response=salvaged + " trailing tokens")
        error.llm_usage = UsageInfo(prompt_tokens=19, completion_tokens=7, cache_tokens=3, cost_usd=0.25)

        chat = _FakeLengthChat(error)
        agent = terminus_2.Terminus2(
            tmp_path,
            model_name="openai/nano-v3.5",
            parser_name="xml",
            max_turns=1,
            api_base="http://model.test",
            suppress_max_turns_warning=True,
        )
        agent._parser = SimpleNamespace(salvage_truncated_response=lambda _content: (salvaged, False))

        result = await agent._query_llm(chat, "prompt", (None, None, None))

        assert result.content == salvaged
        assert result.model_name == "openai/nano-v3.5"
        assert chat.total_input_tokens == 19
        assert chat.total_output_tokens == 7
        assert chat.total_cache_tokens == 3
        assert chat.total_cost == 0.25
        assert chat._api_token_anchor == (0, 26)
        assert getattr(agent, "_pending_failed_llm_response_steps", []) == []

    async def test_salvaged_length_response_is_recorded_as_normal_step(self, patch_sandbox, monkeypatch, tmp_path):
        import litellm
        from harbor.agents.terminus_2 import terminus_2
        from harbor.llms.chat import Chat
        from harbor.models.trajectories import Step

        _patch_terminus_cle_reset()

        salvaged = (
            "<response>\n"
            "<analysis>salvaged analysis</analysis>\n"
            "<plan>salvaged plan</plan>\n"
            "<commands>\n"
            "</commands>\n"
            "<task_complete>false</task_complete>\n"
            "</response>"
        )

        async def fake_acompletion(**_kwargs):
            return _FakeLiteLLMResponse(
                content=salvaged + "\nTRAILING_TRUNCATED_TEXT",
                finish_reason="length",
                prompt_tokens=13,
                completion_tokens=5,
                model="wire-model",
                reasoning="salvaged-reasoning",
            )

        monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

        agent = terminus_2.Terminus2(
            tmp_path,
            model_name="openai/nano-v3.5",
            parser_name="xml",
            max_turns=1,
            api_base="http://model.test",
            collect_rollout_details=True,
            enable_summarize=False,
            record_terminal_session=False,
            suppress_max_turns_warning=True,
            llm_call_kwargs={"max_tokens": 5},
        )
        agent._context = SimpleNamespace(n_input_tokens=None, n_output_tokens=None, n_cache_tokens=None, cost_usd=None)
        agent._session = SimpleNamespace(is_session_alive=lambda: _return_async(True))
        agent._trajectory_steps = [Step(step_id=1, source="user", message="initial prompt")]
        agent._dump_trajectory = lambda: None
        agent._record_asciinema_marker = lambda *_args, **_kwargs: None

        async def execute_commands(_commands, _session):
            return False, "observation"

        chat = Chat(agent._llm)
        agent._execute_commands = execute_commands
        await agent._run_agent_loop("initial prompt", chat)

        assert [step.source for step in agent._trajectory_steps] == ["user", "agent"]
        step = agent._trajectory_steps[1]
        assert step.model_name == "wire-model"
        assert step.reasoning_content == "salvaged-reasoning"
        assert step.message == "Analysis: salvaged analysis\nPlan: salvaged plan"
        assert step.metrics.prompt_tokens == 13
        assert step.metrics.completion_tokens == 5
        assert getattr(agent, "_pending_failed_llm_response_steps", []) == []

    async def test_length_response_becomes_ordered_atif_step_without_double_count(
        self, patch_sandbox, monkeypatch, tmp_path
    ):
        import litellm
        from harbor.agents.terminus_2 import terminus_2
        from harbor.llms.chat import Chat
        from harbor.models.trajectories import Agent, FinalMetrics, Step, Trajectory

        from nemo_evaluator.reports.trajectories import generate_trajectories_report

        _patch_terminus_cle_reset()

        wire_rows = []

        async def fake_acompletion(**_kwargs):
            if not wire_rows:
                wire_rows.append(
                    {
                        "problem_idx": 0,
                        "repeat": 0,
                        "session_id": "sess-length-retry",
                        "status_code": 200,
                        "finish_reason": "length",
                        "timestamp": "2026-07-24T00:00:00.000Z",
                        "model": "wire-model",
                        "request_hash": "length-call",
                        "usage": {"prompt_tokens": 7, "completion_tokens": 4, "total_tokens": 11},
                    }
                )
                return _FakeLiteLLMResponse(
                    content="TRUNCATED_OUTPUT",
                    finish_reason="length",
                    prompt_tokens=7,
                    completion_tokens=4,
                    model="wire-model",
                    reasoning="raw-reasoning-field",
                )
            wire_rows.append(
                {
                    "problem_idx": 0,
                    "repeat": 0,
                    "session_id": "sess-length-retry",
                    "status_code": 200,
                    "finish_reason": "stop",
                    "timestamp": "2026-07-24T00:00:01.000Z",
                    "model": "retry-model",
                    "request_hash": "retry-call",
                    "usage": {"prompt_tokens": 6, "completion_tokens": 2, "total_tokens": 8},
                }
            )
            return _FakeLiteLLMResponse(
                content=json.dumps(
                    {
                        "analysis": "retry analysis",
                        "plan": "retry plan",
                        "commands": [],
                        "task_complete": False,
                    }
                ),
                finish_reason="stop",
                prompt_tokens=6,
                completion_tokens=2,
                model="retry-model",
                reasoning="retry-reasoning",
            )

        monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

        agent = terminus_2.Terminus2(
            tmp_path,
            model_name="openai/nano-v3.5",
            parser_name="json",
            max_turns=1,
            api_base="http://model.test",
            collect_rollout_details=True,
            enable_summarize=False,
            record_terminal_session=False,
            suppress_max_turns_warning=True,
            llm_call_kwargs={"max_tokens": 4},
        )
        agent._context = SimpleNamespace(n_input_tokens=None, n_output_tokens=None, n_cache_tokens=None, cost_usd=None)
        agent._session = SimpleNamespace(is_session_alive=lambda: _return_async(True))
        agent._trajectory_steps = [Step(step_id=1, source="user", message="initial prompt")]
        agent._llm.get_model_output_limit = lambda: 4
        dumped_step_counts = []
        agent._dump_trajectory = lambda: dumped_step_counts.append(len(agent._trajectory_steps))
        agent._record_asciinema_marker = lambda *_args, **_kwargs: None

        async def execute_commands(_commands, _session):
            return False, "observation"

        agent._execute_commands = execute_commands
        await agent._run_agent_loop("initial prompt", Chat(agent._llm))

        assert dumped_step_counts == [3]
        assert [step.source for step in agent._trajectory_steps] == ["user", "agent", "agent"]
        assert [step.message for step in agent._trajectory_steps] == [
            "initial prompt",
            "TRUNCATED_OUTPUT",
            "Analysis: retry analysis\nPlan: retry plan",
        ]
        failed_step, retry_step = agent._trajectory_steps[1:]
        assert failed_step.reasoning_content == "raw-reasoning-field"
        assert failed_step.metrics.prompt_tokens + failed_step.metrics.completion_tokens == 11
        assert retry_step.metrics.prompt_tokens + retry_step.metrics.completion_tokens == 8
        assert sum(row["usage"]["total_tokens"] for row in wire_rows) == sum(
            step.metrics.prompt_tokens + step.metrics.completion_tokens for step in agent._trajectory_steps[1:]
        )

        bench_dir = tmp_path / "pb"
        bench_dir.mkdir()
        agent_steps = [step for step in agent._trajectory_steps if step.source == "agent"]
        prompt_total = sum(step.metrics.prompt_tokens or 0 for step in agent_steps if step.metrics)
        completion_total = sum(step.metrics.completion_tokens or 0 for step in agent_steps if step.metrics)
        trajectory = Trajectory(
            session_id="sess-length-retry",
            agent=Agent(name="terminus-2", version="test", model_name="openai/nano-v3.5"),
            steps=agent._trajectory_steps,
            final_metrics=FinalMetrics(
                total_prompt_tokens=prompt_total,
                total_completion_tokens=completion_total,
                total_steps=len(agent._trajectory_steps),
            ),
        )
        trajectory_row = {
            "problem_idx": 0,
            "repeat": 0,
            "reward": 0.0,
            "trajectory": [trajectory.model_dump(mode="json", exclude_none=True)],
        }
        (bench_dir / "trajectories.jsonl").write_text(json.dumps(trajectory_row) + "\n", encoding="utf-8")
        (bench_dir / "model_traffic.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in wire_rows),
            encoding="utf-8",
        )

        report_path = generate_trajectories_report(tmp_path, enrich=True)
        report = json.loads(report_path.read_text(encoding="utf-8"))["benchmarks"][0]
        tokens = report["tokens_stats"]
        assert tokens["per_step_sum"] == 19
        assert tokens["wire_total"] == 19
        assert tokens["wire_total_for_trajectory_comparison"] == 19
        assert tokens["final_metrics_total"] == 19
        assert tokens["problems_with_per_step_vs_final_metrics_mismatch"] == 0
        assert tokens["problems_with_wire_vs_final_metrics_mismatch"] == 0
        assert tokens["all_sources_match"] is True
        assert report["wire_calls"]["problems_with_more_wire_than_steps"] == 0
        assert report["wire_calls"]["problems_with_fewer_wire_than_steps"] == 0

        enriched = json.loads((bench_dir / "trajectories_enriched.jsonl").read_text(encoding="utf-8"))
        enriched_agent_steps = [step for step in enriched["trajectory"][0]["steps"] if step["source"] == "agent"]
        assert [step["message"] for step in enriched_agent_steps] == [
            "TRUNCATED_OUTPUT",
            "Analysis: retry analysis\nPlan: retry plan",
        ]
        assert [step["metrics"]["total_tokens"] for step in enriched_agent_steps] == [11, 8]
        assert [step["metrics"]["extra"]["finish_reason"] for step in enriched_agent_steps] == ["length", "stop"]

    def test_terminus_run_loop_diverged_source_raises(self, patch_sandbox):
        from harbor.llms.lite_llm import LiteLLM

        original_call = LiteLLM.call
        original_responses = LiteLLM._call_responses
        patch_sandbox.terminus._run_agent_loop = _diverged_method
        with pytest.raises(RuntimeError, match="diverged"):
            _patch_terminus_cle_reset()
        assert LiteLLM.call is original_call
        assert LiteLLM._call_responses is original_responses
        assert "_apply_failed_llm_usage" not in patch_sandbox.terminus.__dict__


async def _return_async(value):
    return value


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
        cle, unwind, anchor, ctxlim = MagicMock(), MagicMock(), MagicMock(), MagicMock()
        monkeypatch.setattr("nemo_evaluator.solvers.harbor._patch_terminus_cle_reset", cle)
        monkeypatch.setattr("nemo_evaluator.solvers.harbor._patch_terminus_unwind_min_pairs", unwind)
        monkeypatch.setattr("nemo_evaluator.solvers.harbor._patch_terminus_api_token_anchor", anchor)
        monkeypatch.setattr(
            "nemo_evaluator.solvers.harbor._patch_harbor_lite_llm_context_length_matcher",
            ctxlim,
        )

        sentinel = object()
        with patch("harbor.agents.factory.AgentFactory.create_agent_from_name", return_value=sentinel):
            result = solver._create_agent(tmp_path)

        assert result is sentinel
        cle.assert_called_once()
        unwind.assert_called_once()
        anchor.assert_called_once()
        ctxlim.assert_called_once()

    def test_non_terminus_agent_skips_patches(self, tmp_path, monkeypatch):
        from unittest.mock import MagicMock, patch

        solver = self._solver("openhands")
        cle, unwind, anchor, ctxlim = MagicMock(), MagicMock(), MagicMock(), MagicMock()
        monkeypatch.setattr("nemo_evaluator.solvers.harbor._patch_terminus_cle_reset", cle)
        monkeypatch.setattr("nemo_evaluator.solvers.harbor._patch_terminus_unwind_min_pairs", unwind)
        monkeypatch.setattr("nemo_evaluator.solvers.harbor._patch_terminus_api_token_anchor", anchor)
        monkeypatch.setattr(
            "nemo_evaluator.solvers.harbor._patch_harbor_lite_llm_context_length_matcher",
            ctxlim,
        )

        with patch("harbor.agents.factory.AgentFactory.create_agent_from_name", return_value=object()):
            solver._create_agent(tmp_path)

        cle.assert_not_called()
        unwind.assert_not_called()
        anchor.assert_not_called()
        ctxlim.assert_not_called()


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
