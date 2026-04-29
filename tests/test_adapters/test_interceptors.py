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
import pytest

from nemo_evaluator.adapters.registry import InterceptorRegistry
from nemo_evaluator.adapters.types import AdapterRequest, AdapterResponse, InterceptorContext
from nemo_evaluator.errors import GracefulError


def _req(body=None, **kw):
    return AdapterRequest(
        method="POST",
        path="/v1/chat/completions",
        headers={"content-type": "application/json"},
        body=body or {"model": "test", "messages": [{"role": "user", "content": "hi"}]},
        ctx=InterceptorContext(),
    )


def _resp(body=None, status_code=200):
    return AdapterResponse(
        status_code=status_code,
        headers={},
        body=body or {},
        ctx=InterceptorContext(),
    )


async def test_drop_params():
    ic = InterceptorRegistry.create("drop_params", {"params": ["temperature", "top_p"]})
    req = _req({"model": "test", "messages": [], "temperature": 0.7, "top_p": 0.9, "max_tokens": 100})
    result = await ic.intercept_request(req)
    assert "temperature" not in result.body
    assert "top_p" not in result.body
    assert result.body["max_tokens"] == 100


async def test_modify_tools():
    ic = InterceptorRegistry.create("modify_tools", {"strip_properties": ["x"]})
    req = _req(
        {
            "model": "test",
            "messages": [],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "f",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "x": {"type": "string"},
                                "y": {"type": "integer"},
                            },
                            "required": ["x", "y"],
                        },
                    },
                }
            ],
        }
    )
    result = await ic.intercept_request(req)
    props = result.body["tools"][0]["function"]["parameters"]["properties"]
    assert "x" not in props
    assert "y" in props
    assert "x" not in result.body["tools"][0]["function"]["parameters"]["required"]


async def test_system_message_prepend():
    ic = InterceptorRegistry.create(
        "system_message",
        {
            "system_message": "Be helpful",
            "strategy": "prepend",
        },
    )
    req = _req({"model": "test", "messages": [{"role": "user", "content": "hi"}]})
    result = await ic.intercept_request(req)
    assert result.body["messages"][0] == {"role": "system", "content": "Be helpful"}
    assert result.body["messages"][1] == {"role": "user", "content": "hi"}


async def test_system_message_replace():
    ic = InterceptorRegistry.create(
        "system_message",
        {
            "system_message": "New system",
            "strategy": "replace",
        },
    )
    req = _req(
        {
            "model": "test",
            "messages": [
                {"role": "system", "content": "Old system"},
                {"role": "user", "content": "hi"},
            ],
        }
    )
    result = await ic.intercept_request(req)
    sys_msgs = [m for m in result.body["messages"] if m["role"] == "system"]
    assert len(sys_msgs) == 1
    assert sys_msgs[0]["content"] == "New system"


async def test_payload_modifier_remove():
    ic = InterceptorRegistry.create("payload_modifier", {"params_to_remove": ["stream"]})
    req = _req({"model": "test", "messages": [], "stream": True})
    result = await ic.intercept_request(req)
    assert "stream" not in result.body


async def test_payload_modifier_add():
    ic = InterceptorRegistry.create("payload_modifier", {"params_to_add": {"temperature": 0.5}})
    req = _req({"model": "test", "messages": []})
    result = await ic.intercept_request(req)
    assert result.body["temperature"] == 0.5


async def test_turn_counter_basic():
    ic = InterceptorRegistry.create("turn_counter", {"max_turns": 5})
    for _ in range(5):
        req = _req({"model": "test", "messages": [{"role": "user", "content": "hi"}]})
        await ic.intercept_request(req)

    with pytest.raises(GracefulError, match="Turn budget exhausted"):
        await ic.intercept_request(_req({"model": "test", "messages": [{"role": "user", "content": "hi"}]}))


async def test_turn_counter_session_isolation():
    """Repeats of the same problem get independent turn budgets when
    the proxy injects distinct session_id values."""
    ic = InterceptorRegistry.create("turn_counter", {"max_turns": 3})
    body = {"model": "test", "messages": [{"role": "user", "content": "same prompt"}]}

    for session_id in ("aaa", "bbb"):
        for _ in range(3):
            r = _req(body)
            r.ctx.extra["session_id"] = session_id
            await ic.intercept_request(r)

    r = _req(body)
    r.ctx.extra["session_id"] = "aaa"
    with pytest.raises(GracefulError, match="Turn budget exhausted"):
        await ic.intercept_request(r)

    r = _req(body)
    r.ctx.extra["session_id"] = "ccc"
    await ic.intercept_request(r)


async def test_turn_counter_default_does_not_touch_user_message_below_threshold():
    """Defaults reproduce the prior threshold+system_message behavior — user msg untouched at low turns."""
    ic = InterceptorRegistry.create("turn_counter", {"max_turns": 100})
    body = {"model": "test", "messages": [{"role": "user", "content": "hi"}]}
    r = _req(body)
    r.ctx.extra["session_id"] = "default-mode"
    result = await ic.intercept_request(r)
    assert result.body["messages"][-1] == {"role": "user", "content": "hi"}


async def test_turn_counter_periodic_user_appends_notice_every_step():
    ic = InterceptorRegistry.create(
        "turn_counter",
        {"max_turns": 10, "position": "user_message", "trigger": "periodic"},
    )
    for expected_remaining in (9, 8, 7):
        body = {"model": "test", "messages": [{"role": "user", "content": "hi"}]}
        r = _req(body)
        r.ctx.extra["session_id"] = "every-step"
        result = await ic.intercept_request(r)
        last = result.body["messages"][-1]
        assert last["role"] == "user"
        assert (
            last["content"]
            == f"hi\n\nENVIRONMENT REMINDER: You have {expected_remaining} turns left to complete the task."
        )


async def test_turn_counter_periodic_user_respects_interval():
    ic = InterceptorRegistry.create(
        "turn_counter",
        {"max_turns": 100, "position": "user_message", "trigger": "periodic", "interval": 3},
    )
    expected_modified = {3, 6}
    for turn in range(1, 7):
        body = {"model": "test", "messages": [{"role": "user", "content": "hi"}]}
        r = _req(body)
        r.ctx.extra["session_id"] = "interval-test"
        result = await ic.intercept_request(r)
        last_content = result.body["messages"][-1]["content"]
        if turn in expected_modified:
            assert "ENVIRONMENT REMINDER" in last_content, f"turn {turn} should be modified"
        else:
            assert last_content == "hi", f"turn {turn} should be untouched"


async def test_turn_counter_periodic_user_appends_to_last_user_only():
    ic = InterceptorRegistry.create(
        "turn_counter",
        {"max_turns": 10, "position": "user_message", "trigger": "periodic"},
    )
    body = {
        "model": "test",
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "first user"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "latest user"},
        ],
    }
    r = _req(body)
    r.ctx.extra["session_id"] = "multi-msg"
    result = await ic.intercept_request(r)
    assert result.body["messages"][1]["content"] == "first user"
    assert result.body["messages"][3]["content"].startswith(
        "latest user\n\nENVIRONMENT REMINDER: You have 9 turns left"
    )


async def test_turn_counter_periodic_user_extends_existing_last_text_block():
    """When the last user content block is text, append into it (no two adjacent text blocks)."""
    ic = InterceptorRegistry.create(
        "turn_counter",
        {"max_turns": 10, "position": "user_message", "trigger": "periodic"},
    )
    body = {
        "model": "test",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "https://example.com/x.png"}},
                    {"type": "text", "text": "describe this"},
                ],
            }
        ],
    }
    r = _req(body)
    r.ctx.extra["session_id"] = "list-text-tail"
    result = await ic.intercept_request(r)
    blocks = result.body["messages"][-1]["content"]
    assert len(blocks) == 2
    assert blocks[0]["type"] == "image_url"
    assert blocks[1] == {
        "type": "text",
        "text": "describe this\n\nENVIRONMENT REMINDER: You have 9 turns left to complete the task.",
    }


async def test_turn_counter_periodic_user_appends_text_block_when_tail_is_nontext():
    """When the last user content block is non-text, append a new text block at the end."""
    ic = InterceptorRegistry.create(
        "turn_counter",
        {"max_turns": 10, "position": "user_message", "trigger": "periodic"},
    )
    body = {
        "model": "test",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe this"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/x.png"}},
                ],
            }
        ],
    }
    r = _req(body)
    r.ctx.extra["session_id"] = "list-image-tail"
    result = await ic.intercept_request(r)
    blocks = result.body["messages"][-1]["content"]
    assert len(blocks) == 3
    assert blocks[0] == {"type": "text", "text": "describe this"}
    assert blocks[1]["type"] == "image_url"
    assert blocks[2] == {
        "type": "text",
        "text": "ENVIRONMENT REMINDER: You have 9 turns left to complete the task.",
    }


async def test_turn_counter_periodic_user_still_enforces_max_turns():
    ic = InterceptorRegistry.create(
        "turn_counter",
        {"max_turns": 2, "position": "user_message", "trigger": "periodic"},
    )
    for _ in range(2):
        body = {"model": "test", "messages": [{"role": "user", "content": "hi"}]}
        r = _req(body)
        r.ctx.extra["session_id"] = "budget-test"
        await ic.intercept_request(r)

    body = {"model": "test", "messages": [{"role": "user", "content": "hi"}]}
    r = _req(body)
    r.ctx.extra["session_id"] = "budget-test"
    with pytest.raises(GracefulError, match="Turn budget exhausted"):
        await ic.intercept_request(r)


async def test_turn_counter_periodic_system_emits_template_as_system_message():
    ic = InterceptorRegistry.create(
        "turn_counter",
        {"max_turns": 10, "position": "system_message", "trigger": "periodic", "interval": 2},
    )
    for turn in range(1, 5):
        body = {"model": "test", "messages": [{"role": "user", "content": "hi"}]}
        r = _req(body)
        r.ctx.extra["session_id"] = "periodic-system"
        result = await ic.intercept_request(r)
        if turn % 2 == 0:
            last = result.body["messages"][-1]
            assert last["role"] == "system"
            assert last["content"] == f"ENVIRONMENT REMINDER: You have {10 - turn} turns left to complete the task."
        else:
            assert result.body["messages"][-1]["role"] == "user"


async def test_turn_counter_threshold_user_uses_threshold_body_without_system_prefix():
    """threshold+user_message reuses the threshold WARN/URGENT body (without [SYSTEM] prefix)."""
    ic = InterceptorRegistry.create(
        "turn_counter",
        {"max_turns": 10, "position": "user_message", "trigger": "threshold"},
    )
    for turn in range(1, 11):
        body = {"model": "test", "messages": [{"role": "user", "content": "hi"}]}
        r = _req(body)
        r.ctx.extra["session_id"] = "threshold-user"
        result = await ic.intercept_request(r)
        last_content = result.body["messages"][-1]["content"]
        if turn < 8:
            assert last_content == "hi", f"turn {turn} (ratio {turn / 10}) should be untouched"
        else:
            assert "[SYSTEM]" not in last_content, f"turn {turn}: user-position must omit [SYSTEM] prefix"
            if turn >= 10:  # ratio >= URGENT_THRESHOLD (0.95)
                assert "URGENT" in last_content
                assert "MUST provide your final answer NOW" in last_content
            else:  # 0.80 <= ratio < 0.95
                assert "URGENT" not in last_content
                assert "Begin wrapping up" in last_content


async def test_turn_counter_invalid_position_raises():
    with pytest.raises(ValueError, match="not a valid InjectionPosition"):
        InterceptorRegistry.create(
            "turn_counter",
            {"max_turns": 10, "position": "bogus"},
        )


async def test_turn_counter_invalid_trigger_raises():
    with pytest.raises(ValueError, match="not a valid InjectionTrigger"):
        InterceptorRegistry.create(
            "turn_counter",
            {"max_turns": 10, "trigger": "bogus"},
        )


async def test_turn_counter_invalid_interval_raises():
    with pytest.raises(ValueError, match="interval"):
        InterceptorRegistry.create(
            "turn_counter",
            {"max_turns": 10, "position": "user_message", "trigger": "periodic", "interval": 0},
        )


async def test_turn_counter_warns_when_max_turns_is_unset(caplog):
    """Any turn_counter without max_turns logs a warning that injection is disabled."""
    import logging

    for cfg in ({}, {"position": "user_message"}, {"trigger": "periodic"}):
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="nemo_evaluator.adapters.interceptors.turn_counter"):
            InterceptorRegistry.create("turn_counter", cfg)
        assert any("max_turns" in record.message for record in caplog.records), (
            f"config {cfg} should log a max_turns warning"
        )


async def test_turn_counter_with_max_turns_does_not_warn(caplog):
    import logging

    with caplog.at_level(logging.WARNING, logger="nemo_evaluator.adapters.interceptors.turn_counter"):
        InterceptorRegistry.create("turn_counter", {"max_turns": 10})
    assert not any("max_turns" in record.message for record in caplog.records)


async def test_turn_counter_without_max_turns_is_noop_for_any_combo():
    """Without max_turns the interceptor counts and logs but never injects, regardless of position/trigger."""
    for cfg in (
        {},
        {"position": "user_message"},
        {"trigger": "periodic"},
        {"position": "user_message", "trigger": "periodic"},
    ):
        ic = InterceptorRegistry.create("turn_counter", cfg)
        body = {"model": "test", "messages": [{"role": "user", "content": "hi"}]}
        r = _req(body)
        r.ctx.extra["session_id"] = f"noop-{cfg!r}"
        result = await ic.intercept_request(r)
        assert result.body["messages"] == [{"role": "user", "content": "hi"}], (
            f"config {cfg} should no-op without max_turns"
        )


async def test_raise_client_errors_4xx():
    ic = InterceptorRegistry.create("raise_client_errors")
    resp = _resp({"error": "bad request"}, status_code=400)
    with pytest.raises(RuntimeError, match="Upstream returned 400"):
        await ic.intercept_response(resp)


async def test_raise_client_errors_429_passes():
    ic = InterceptorRegistry.create("raise_client_errors")
    resp = _resp({"error": "rate limited"}, status_code=429)
    result = await ic.intercept_response(resp)
    assert result.status_code == 429


async def test_reasoning_normalize():
    ic = InterceptorRegistry.create("reasoning")
    resp = _resp(
        {
            "choices": [
                {
                    "message": {
                        "content": "<think>reason</think>answer",
                    },
                }
            ],
        }
    )
    result = await ic.intercept_response(resp)
    msg = result.body["choices"][0]["message"]
    assert msg["reasoning_content"] == "reason"
    assert msg["content"] == "answer"
