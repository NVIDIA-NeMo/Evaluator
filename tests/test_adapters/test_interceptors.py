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
import logging

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


async def test_payload_modifier_headers_add():
    ic = InterceptorRegistry.create(
        "payload_modifier",
        {
            "headers_to_add": {
                "X-NMP-Principal-Id": "service:evaluator",
                "X-Inference-Priority": "batch",
            }
        },
    )
    req = _req()
    result = await ic.intercept_request(req)
    assert result.headers["X-NMP-Principal-Id"] == "service:evaluator"
    assert result.headers["X-Inference-Priority"] == "batch"
    assert result.headers["content-type"] == "application/json"


async def test_payload_modifier_headers_add_overrides_case_insensitive():
    ic = InterceptorRegistry.create("payload_modifier", {"headers_to_add": {"Authorization": "Bearer new"}})
    req = _req()
    req.headers["authorization"] = "Bearer old"
    result = await ic.intercept_request(req)
    auth_values = [v for k, v in result.headers.items() if k.lower() == "authorization"]
    assert auth_values == ["Bearer new"]


async def test_payload_modifier_headers_remove_case_insensitive():
    ic = InterceptorRegistry.create("payload_modifier", {"headers_to_remove": ["X-Trace-Id", "AUTHORIZATION"]})
    req = _req()
    req.headers.update({"X-Trace-Id": "abc", "Authorization": "Bearer t", "X-Keep": "1"})
    result = await ic.intercept_request(req)
    assert "X-Trace-Id" not in result.headers
    assert "Authorization" not in result.headers
    assert result.headers["X-Keep"] == "1"


async def test_payload_modifier_headers_rename_case_insensitive():
    ic = InterceptorRegistry.create("payload_modifier", {"headers_to_rename": {"X-Old-Auth": "X-New-Auth"}})
    req = _req()
    req.headers["x-old-auth"] = "token123"
    result = await ic.intercept_request(req)
    assert "x-old-auth" not in {k.lower() for k in result.headers}
    assert result.headers["X-New-Auth"] == "token123"


async def test_payload_modifier_headers_drop_hop_by_hop(caplog):
    caplog.set_level(logging.WARNING)
    ic = InterceptorRegistry.create(
        "payload_modifier",
        {
            "headers_to_add": {
                "Host": "evil.example.com",
                "Content-Length": "9999",
                "Connection": "close",
                "Transfer-Encoding": "chunked",
                "X-Inference-Priority": "batch",
            }
        },
    )
    req = _req()
    result = await ic.intercept_request(req)
    assert result.headers["X-Inference-Priority"] == "batch"
    assert result.headers.get("Host") != "evil.example.com"
    assert result.headers.get("Connection") != "close"
    assert result.headers.get("Transfer-Encoding") != "chunked"


async def test_payload_modifier_headers_rename_drops_hop_by_hop(caplog):
    """Renaming any header *to* a hop-by-hop name must be rejected at init —
    otherwise it would be a backdoor around the headers_to_add guard."""
    caplog.set_level(logging.WARNING)
    ic = InterceptorRegistry.create(
        "payload_modifier",
        {
            "headers_to_rename": {
                "X-Foo": "Host",
                "X-Bar": "Content-Length",
                "X-Keep": "X-Allowed",
            }
        },
    )
    req = _req()
    req.headers.update({"X-Foo": "evil", "X-Bar": "9999", "X-Keep": "yes"})
    result = await ic.intercept_request(req)

    # Hop-by-hop rename targets are dropped — original keys remain unrenamed.
    assert "Host" not in result.headers
    assert "Content-Length" not in result.headers
    assert result.headers["X-Foo"] == "evil"
    assert result.headers["X-Bar"] == "9999"
    # Non-hop-by-hop rename still works.
    assert "X-Keep" not in result.headers
    assert result.headers["X-Allowed"] == "yes"

    # Each disallowed rename target gets its own warning.
    warnings = [r for r in caplog.records if "headers_to_rename" in r.message.lower()]
    assert len(warnings) >= 2
    assert any("hop-by-hop" in r.message.lower() for r in caplog.records)


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


def _resp_with_tool_call(reasoning: str, call_id: str = "call_1", text: str = ""):
    return _resp(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": text,
                        "reasoning_content": reasoning,
                        "tool_calls": [
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {"name": "f", "arguments": "{}"},
                            }
                        ],
                    },
                }
            ],
        }
    )


def _assistant_with_tool_call(call_id: str = "call_1", content=None, reasoning=None):
    msg = {
        "role": "assistant",
        "content": content,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": "f", "arguments": "{}"},
            }
        ],
    }
    if reasoning is not None:
        msg["reasoning_content"] = reasoning
    return msg


async def test_reasoning_replay_invalid_mode():
    with pytest.raises(ValueError, match="mode must be one of"):
        InterceptorRegistry.create("reasoning_replay", {"mode": "bogus"})


async def test_reasoning_replay_rejects_empty_tags():
    with pytest.raises(ValueError, match="tag_open and tag_close"):
        InterceptorRegistry.create("reasoning_replay", {"tag_open": "", "tag_close": "</think>"})


async def test_reasoning_replay_custom_tag():
    ic = InterceptorRegistry.create(
        "reasoning_replay",
        {"mode": "think_tags", "tag_open": "<thought>", "tag_close": "</thought>"},
    )
    await ic.intercept_response(_resp_with_tool_call("hidden chain", "call_abc"))

    req = _req(
        {
            "messages": [
                _assistant_with_tool_call("call_abc", content="visible answer"),
            ],
        }
    )
    result = await ic.intercept_request(req)
    assert result.body["messages"][0]["content"] == "<thought>hidden chain</thought>\nvisible answer"


async def test_reasoning_replay_custom_tag_idempotent():
    ic = InterceptorRegistry.create(
        "reasoning_replay",
        {"mode": "think_tags", "tag_open": "<thought>", "tag_close": "</thought>"},
    )
    await ic.intercept_response(_resp_with_tool_call("cached", "call_x"))

    req = _req(
        {
            "messages": [
                _assistant_with_tool_call(
                    "call_x",
                    content="<thought>existing</thought>\nrest",
                ),
            ],
        }
    )
    result = await ic.intercept_request(req)
    assert result.body["messages"][0]["content"] == "<thought>existing</thought>\nrest"


async def test_reasoning_replay_caches_response():
    ic = InterceptorRegistry.create("reasoning_replay")
    await ic.intercept_response(_resp_with_tool_call("hidden chain", "call_abc"))
    assert ic._get("|c:call_abc") == "hidden chain"


async def test_reasoning_replay_think_tags_round_trip():
    ic = InterceptorRegistry.create("reasoning_replay", {"mode": "think_tags"})
    await ic.intercept_response(_resp_with_tool_call("hidden chain", "call_abc"))

    req = _req(
        {
            "messages": [
                {"role": "user", "content": "go"},
                _assistant_with_tool_call("call_abc", content="visible answer"),
                {"role": "tool", "tool_call_id": "call_abc", "content": "ok"},
                {"role": "user", "content": "next"},
            ],
        }
    )
    result = await ic.intercept_request(req)
    assistant = result.body["messages"][1]
    assert assistant["content"] == "<think>hidden chain</think>\nvisible answer"
    assert "reasoning_content" not in assistant
    assert req.ctx.extra["reasoning_replay_hits"] == 1


async def test_reasoning_replay_native_round_trip():
    ic = InterceptorRegistry.create("reasoning_replay", {"mode": "native"})
    await ic.intercept_response(_resp_with_tool_call("hidden chain", "call_abc"))

    req = _req(
        {
            "messages": [
                _assistant_with_tool_call("call_abc", content="visible answer"),
            ],
        }
    )
    result = await ic.intercept_request(req)
    assistant = result.body["messages"][0]
    assert assistant["reasoning_content"] == "hidden chain"
    assert assistant["content"] == "visible answer"


async def test_reasoning_replay_both_mode():
    ic = InterceptorRegistry.create("reasoning_replay", {"mode": "both"})
    await ic.intercept_response(_resp_with_tool_call("hidden", "call_abc"))

    req = _req(
        {"messages": [_assistant_with_tool_call("call_abc", content="text")]},
    )
    result = await ic.intercept_request(req)
    assistant = result.body["messages"][0]
    assert assistant["reasoning_content"] == "hidden"
    assert assistant["content"] == "<think>hidden</think>\ntext"


async def test_reasoning_replay_native_skips_when_already_set():
    ic = InterceptorRegistry.create("reasoning_replay", {"mode": "native"})
    await ic.intercept_response(_resp_with_tool_call("from-cache", "call_abc"))

    req = _req(
        {
            "messages": [
                _assistant_with_tool_call("call_abc", content="x", reasoning="agent-set"),
            ],
        }
    )
    result = await ic.intercept_request(req)
    assert result.body["messages"][0]["reasoning_content"] == "agent-set"
    assert "reasoning_replay_hits" not in req.ctx.extra


async def test_reasoning_replay_both_preserves_existing_native():
    ic = InterceptorRegistry.create("reasoning_replay", {"mode": "both"})
    await ic.intercept_response(_resp_with_tool_call("from-cache", "call_abc"))

    req = _req(
        {
            "messages": [
                _assistant_with_tool_call("call_abc", content="x", reasoning="agent-set"),
            ],
        }
    )
    result = await ic.intercept_request(req)
    assistant = result.body["messages"][0]
    assert assistant["reasoning_content"] == "agent-set"
    assert assistant["content"] == "<think>from-cache</think>\nx"


async def test_reasoning_replay_content_hash_fallback():
    ic = InterceptorRegistry.create("reasoning_replay", {"mode": "think_tags"})
    resp = _resp(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "the visible answer",
                        "reasoning_content": "thought",
                    },
                }
            ],
        }
    )
    await ic.intercept_response(resp)

    req = _req(
        {
            "messages": [
                {"role": "assistant", "content": "the visible answer"},
            ],
        }
    )
    result = await ic.intercept_request(req)
    assert result.body["messages"][0]["content"] == "<think>thought</think>\nthe visible answer"


async def test_reasoning_replay_multiple_tool_calls_share_reasoning():
    ic = InterceptorRegistry.create("reasoning_replay", {"mode": "native"})
    resp = _resp(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "reasoning_content": "shared",
                        "tool_calls": [
                            {"id": "a", "type": "function", "function": {"name": "f", "arguments": "{}"}},
                            {"id": "b", "type": "function", "function": {"name": "g", "arguments": "{}"}},
                        ],
                    },
                }
            ],
        }
    )
    await ic.intercept_response(resp)
    assert ic._get("|c:a") == "shared"
    assert ic._get("|c:b") == "shared"


async def test_reasoning_replay_lru_eviction():
    ic = InterceptorRegistry.create("reasoning_replay", {"cache_max_entries": 2})
    await ic.intercept_response(_resp_with_tool_call("r1", "call_1"))
    await ic.intercept_response(_resp_with_tool_call("r2", "call_2"))
    await ic.intercept_response(_resp_with_tool_call("r3", "call_3"))
    assert ic._get("|c:call_1") is None
    assert ic._get("|c:call_2") == "r2"
    assert ic._get("|c:call_3") == "r3"


async def test_reasoning_replay_list_content():
    ic = InterceptorRegistry.create("reasoning_replay", {"mode": "think_tags"})
    await ic.intercept_response(_resp_with_tool_call("hidden", "call_abc"))

    req = _req(
        {
            "messages": [
                _assistant_with_tool_call(
                    "call_abc",
                    content=[{"type": "text", "text": "answer"}],
                ),
            ],
        }
    )
    result = await ic.intercept_request(req)
    content = result.body["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "<think>hidden</think>\n"}
    assert content[1] == {"type": "text", "text": "answer"}


async def test_reasoning_replay_no_match_is_noop():
    ic = InterceptorRegistry.create("reasoning_replay", {"mode": "think_tags"})
    req = _req(
        {
            "messages": [
                _assistant_with_tool_call("call_unknown", content="text"),
            ],
        }
    )
    result = await ic.intercept_request(req)
    assert result.body["messages"][0]["content"] == "text"
    assert "reasoning_replay_hits" not in req.ctx.extra


async def test_reasoning_replay_handles_malformed_body():
    ic = InterceptorRegistry.create("reasoning_replay")
    req = _req({"messages": "not-a-list"})
    result = await ic.intercept_request(req)
    assert result.body["messages"] == "not-a-list"

    resp = _resp(b"raw bytes")
    out = await ic.intercept_response(resp)
    assert out.body == b"raw bytes"


async def test_reasoning_replay_session_isolation():
    """Two concurrent sessions with the same hash-key text must not cross-talk."""
    ic = InterceptorRegistry.create("reasoning_replay", {"mode": "think_tags"})

    resp_a = _resp(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "the answer",
                        "reasoning_content": "session-A-thought",
                    },
                }
            ],
        }
    )
    resp_a.ctx.extra["session_id"] = "A"
    await ic.intercept_response(resp_a)

    resp_b = _resp(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "the answer",
                        "reasoning_content": "session-B-thought",
                    },
                }
            ],
        }
    )
    resp_b.ctx.extra["session_id"] = "B"
    await ic.intercept_response(resp_b)

    req_a = _req({"messages": [{"role": "assistant", "content": "the answer"}]})
    req_a.ctx.extra["session_id"] = "A"
    out_a = await ic.intercept_request(req_a)
    assert out_a.body["messages"][0]["content"] == "<think>session-A-thought</think>\nthe answer"

    req_b = _req({"messages": [{"role": "assistant", "content": "the answer"}]})
    req_b.ctx.extra["session_id"] = "B"
    out_b = await ic.intercept_request(req_b)
    assert out_b.body["messages"][0]["content"] == "<think>session-B-thought</think>\nthe answer"


async def test_reasoning_replay_idempotent_in_think_tags():
    """If content already begins with <think>, do not double-wrap."""
    ic = InterceptorRegistry.create("reasoning_replay", {"mode": "think_tags"})
    await ic.intercept_response(_resp_with_tool_call("cached", "call_x"))

    req = _req(
        {
            "messages": [
                _assistant_with_tool_call(
                    "call_x",
                    content="<think>already-there</think>\nvisible",
                ),
            ],
        }
    )
    result = await ic.intercept_request(req)
    assert result.body["messages"][0]["content"] == "<think>already-there</think>\nvisible"
    assert "reasoning_replay_hits" not in req.ctx.extra


async def test_reasoning_replay_idempotent_list_content():
    ic = InterceptorRegistry.create("reasoning_replay", {"mode": "think_tags"})
    await ic.intercept_response(_resp_with_tool_call("cached", "call_x"))

    req = _req(
        {
            "messages": [
                _assistant_with_tool_call(
                    "call_x",
                    content=[
                        {"type": "text", "text": "<think>existing</think>"},
                        {"type": "text", "text": "rest"},
                    ],
                ),
            ],
        }
    )
    result = await ic.intercept_request(req)
    content = result.body["messages"][0]["content"]
    assert content[0]["text"] == "<think>existing</think>"
    assert len(content) == 2


async def test_reasoning_replay_ignores_whitespace_only_reasoning():
    """Whitespace-only reasoning_content must not poison the cache."""
    ic = InterceptorRegistry.create("reasoning_replay")
    resp = _resp(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "answer",
                        "reasoning_content": "   \n\t  ",
                        "tool_calls": [
                            {"id": "call_x", "type": "function", "function": {"name": "f", "arguments": "{}"}},
                        ],
                    },
                }
            ],
        }
    )
    await ic.intercept_response(resp)
    assert ic._get("|c:call_x") is None


async def test_reasoning_replay_ignores_empty_tool_call_id():
    """Empty-string tool_call.id must fall through to content-hash, not collide."""
    ic = InterceptorRegistry.create("reasoning_replay")
    resp = _resp(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "the visible text",
                        "reasoning_content": "thought",
                        "tool_calls": [
                            {"id": "", "type": "function", "function": {"name": "f", "arguments": "{}"}},
                        ],
                    },
                }
            ],
        }
    )
    await ic.intercept_response(resp)
    assert ic._get("|c:") is None


async def test_reasoning_replay_pipeline_dual_stage():
    """Through AdapterPipeline: same instance must run on request AND response side."""
    from nemo_evaluator.adapters.pipeline import AdapterPipeline
    from nemo_evaluator.adapters.types import AdapterResponse, RequestToResponseInterceptor

    class _FakeEndpoint(RequestToResponseInterceptor):
        def __init__(self, body: dict):
            self._body = body
            self.last_req: AdapterRequest | None = None

        async def intercept_request(self, req: AdapterRequest):
            self.last_req = req
            return AdapterResponse(status_code=200, headers={}, body=self._body, ctx=req.ctx)

    replay = InterceptorRegistry.create("reasoning_replay", {"mode": "think_tags"})
    endpoint = _FakeEndpoint(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "answer",
                        "reasoning_content": "captured",
                        "tool_calls": [
                            {"id": "call_z", "type": "function", "function": {"name": "f", "arguments": "{}"}},
                        ],
                    },
                }
            ],
        }
    )
    pipeline = AdapterPipeline([replay, endpoint])

    resp1 = await pipeline.process(_req({"messages": [{"role": "user", "content": "go"}]}))
    assert resp1.status_code == 200
    assert replay._get("|c:call_z") == "captured"

    req2 = _req(
        {
            "messages": [
                _assistant_with_tool_call("call_z", content="answer"),
                {"role": "user", "content": "next"},
            ],
        }
    )
    await pipeline.process(req2)
    forwarded = endpoint.last_req.body["messages"][0]
    assert forwarded["content"] == "<think>captured</think>\nanswer"


async def test_reasoning_replay_composes_with_normalize():
    norm = InterceptorRegistry.create("reasoning")
    replay = InterceptorRegistry.create("reasoning_replay", {"mode": "think_tags"})
    resp = _resp(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "<think>cot</think>visible",
                        "tool_calls": [
                            {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}},
                        ],
                    },
                }
            ],
        }
    )
    await norm.intercept_response(resp)
    await replay.intercept_response(resp)
    assert replay._get("|c:c1") == "cot"
