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
from unittest.mock import AsyncMock, patch

import pytest

from nemo_evaluator.engine.model_client import ModelClient
from nemo_evaluator.schemas import RetryConfig

MOCK_CHAT_RESPONSE = {
    "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    "model": "test",
}

MOCK_TOOL_RESPONSE = {
    "choices": [{"message": {"content": "ok", "tool_calls": None}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    "model": "test",
}


class TestModelClient:
    def test_strips_chat_completions_from_url(self):
        c = ModelClient(base_url="https://api.example.com/v1/chat/completions", model="test")
        assert c.base_url == "https://api.example.com/v1"

    def test_strips_trailing_slash(self):
        c = ModelClient(base_url="https://api.example.com/v1/", model="test")
        assert c.base_url == "https://api.example.com/v1"

    @pytest.mark.asyncio
    async def test_connection_pooling_reuses_client(self):
        c = ModelClient(base_url="https://api.example.com/v1", model="test")
        client1 = c._get_client()
        client2 = c._get_client()
        assert client1 is client2
        await c.close()

    @pytest.mark.asyncio
    async def test_close_releases_client(self):
        c = ModelClient(base_url="https://api.example.com/v1", model="test")
        c._get_client()
        await c.close()
        assert c._http is None

    def test_exponential_backoff(self):
        c = ModelClient(
            base_url="https://api.example.com/v1",
            model="test",
            retry=RetryConfig(base_delay=1.0, max_delay=60.0, jitter=False),
        )
        assert c._backoff_delay(0) == 1.0
        assert c._backoff_delay(1) == 2.0
        assert c._backoff_delay(2) == 4.0
        assert c._backoff_delay(10) == 60.0

    def test_backoff_jitter_stays_within_bounds(self):
        c = ModelClient(
            base_url="https://api.example.com/v1",
            model="test",
            retry=RetryConfig(base_delay=1.0, jitter=True),
        )
        delays = [c._backoff_delay(0) for _ in range(100)]
        assert min(delays) >= 0.5
        assert max(delays) <= 1.5

    def test_parse_response_extracts_content_and_tokens(self):
        c = ModelClient(base_url="https://api.example.com/v1", model="test")
        data = {
            "choices": [{"message": {"content": "42"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "model": "test",
        }
        resp = c._parse_response(data, 100.0, "What is 6*7?", None)
        assert resp.content == "42"
        assert resp.total_tokens == 15
        assert resp.request_prompt == "What is 6*7?"

    def test_parse_response_rejects_empty_choices(self):
        from nemo_evaluator.errors import InfraError

        c = ModelClient(base_url="https://api.example.com/v1", model="test")
        with pytest.raises(InfraError, match="empty choices"):
            c._parse_response({"choices": []}, 100.0, "test", None)

    @pytest.mark.asyncio
    async def test_chat_payload_includes_all_generation_fields(self):
        c = ModelClient(
            base_url="https://api.example.com/v1",
            model="test",
            temperature=0.7,
            max_tokens=1024,
            top_p=0.9,
            seed=42,
            stop=["END"],
            frequency_penalty=0.5,
            presence_penalty=-0.5,
        )
        with patch.object(c, "_post_with_retry", new_callable=AsyncMock, return_value=MOCK_CHAT_RESPONSE) as mock_post:
            await c.chat(prompt="hello")
            payload = mock_post.call_args[0][1]
            assert payload["temperature"] == 0.7
            assert payload["max_tokens"] == 1024
            assert payload["top_p"] == 0.9
            assert payload["seed"] == 42
            assert payload["stop"] == ["END"]
            assert payload["frequency_penalty"] == 0.5
            assert payload["presence_penalty"] == -0.5
        await c.close()

    @pytest.mark.asyncio
    async def test_chat_payload_omits_none_fields(self):
        c = ModelClient(base_url="https://api.example.com/v1", model="test")
        with patch.object(c, "_post_with_retry", new_callable=AsyncMock, return_value=MOCK_CHAT_RESPONSE) as mock_post:
            await c.chat(prompt="hello")
            payload = mock_post.call_args[0][1]
            assert "temperature" not in payload
            assert "max_tokens" not in payload
            assert "top_p" not in payload
            assert "stop" not in payload
            assert "frequency_penalty" not in payload
            assert "presence_penalty" not in payload
        await c.close()

    @pytest.mark.asyncio
    async def test_vlm_chat_payload_includes_generation_fields(self):
        c = ModelClient(
            base_url="https://api.example.com/v1",
            model="test",
            temperature=0.5,
            stop=["DONE"],
            frequency_penalty=0.3,
            presence_penalty=0.1,
        )
        with patch.object(c, "_post_with_retry", new_callable=AsyncMock, return_value=MOCK_CHAT_RESPONSE) as mock_post:
            await c.vlm_chat(prompt="describe", images=["data:image/png;base64,abc"])
            payload = mock_post.call_args[0][1]
            assert payload["temperature"] == 0.5
            assert payload["stop"] == ["DONE"]
            assert payload["frequency_penalty"] == 0.3
            assert payload["presence_penalty"] == 0.1
        await c.close()

    @pytest.mark.asyncio
    async def test_chat_with_tools_payload_includes_generation_fields(self):
        c = ModelClient(
            base_url="https://api.example.com/v1",
            model="test",
            temperature=0.8,
            max_tokens=512,
            top_p=0.95,
            stop=["<END>"],
            frequency_penalty=1.0,
            presence_penalty=-1.0,
        )
        tools = [{"type": "function", "function": {"name": "test_fn", "parameters": {}}}]
        with patch.object(c, "_post_with_retry", new_callable=AsyncMock, return_value=MOCK_TOOL_RESPONSE) as mock_post:
            await c.chat_with_tools(messages=[{"role": "user", "content": "hi"}], tools=tools)
            payload = mock_post.call_args[0][1]
            assert payload["temperature"] == 0.8
            assert payload["max_tokens"] == 512
            assert payload["top_p"] == 0.95
            assert payload["stop"] == ["<END>"]
            assert payload["frequency_penalty"] == 1.0
            assert payload["presence_penalty"] == -1.0
        await c.close()

    @pytest.mark.asyncio
    async def test_chat_with_tools_overrides_take_precedence(self):
        c = ModelClient(
            base_url="https://api.example.com/v1",
            model="test",
            temperature=0.8,
            stop=["<END>"],
            frequency_penalty=1.0,
        )
        tools = [{"type": "function", "function": {"name": "test_fn", "parameters": {}}}]
        with patch.object(c, "_post_with_retry", new_callable=AsyncMock, return_value=MOCK_TOOL_RESPONSE) as mock_post:
            await c.chat_with_tools(
                messages=[{"role": "user", "content": "hi"}],
                tools=tools,
                temperature=0.1,
                stop=["OVERRIDE"],
            )
            payload = mock_post.call_args[0][1]
            assert payload["temperature"] == 0.1
            assert payload["stop"] == ["OVERRIDE"]
        await c.close()

    def test_all_generation_fields_stored(self):
        c = ModelClient(
            base_url="https://api.example.com/v1",
            model="test",
            temperature=0.7,
            max_tokens=1024,
            top_p=0.9,
            seed=42,
            stop=["END", "STOP"],
            frequency_penalty=0.5,
            presence_penalty=-0.5,
        )
        assert c.temperature == 0.7
        assert c.max_tokens == 1024
        assert c.top_p == 0.9
        assert c.seed == 42
        assert c.stop == ["END", "STOP"]
        assert c.frequency_penalty == 0.5
        assert c.presence_penalty == -0.5

    def test_generation_fields_default_none(self):
        c = ModelClient(base_url="https://api.example.com/v1", model="test")
        assert c.temperature is None
        assert c.max_tokens is None
        assert c.top_p is None
        assert c.seed is None
        assert c.stop is None
        assert c.frequency_penalty is None
        assert c.presence_penalty is None


class TestBuildGenerationPayload:
    """Tests for ModelClient._build_generation_payload."""

    def _client(self, **kwargs):
        return ModelClient(base_url="https://api.example.com/v1", model="test", **kwargs)

    def test_returns_empty_when_all_none(self):
        assert self._client()._build_generation_payload() == {}

    def test_includes_set_defaults(self):
        c = self._client(temperature=0.7, max_tokens=1024, seed=42)
        result = c._build_generation_payload()
        assert result == {"temperature": 0.7, "max_tokens": 1024, "seed": 42}

    def test_omits_none_defaults(self):
        c = self._client(temperature=0.5)
        result = c._build_generation_payload()
        assert "top_p" not in result
        assert "seed" not in result
        assert result == {"temperature": 0.5}

    def test_overrides_replace_defaults(self):
        c = self._client(temperature=0.7, max_tokens=1024)
        result = c._build_generation_payload(temperature=0.1, max_tokens=512)
        assert result == {"temperature": 0.1, "max_tokens": 512}

    def test_overrides_add_new_fields(self):
        c = self._client(temperature=0.5)
        result = c._build_generation_payload(seed=99)
        assert result == {"temperature": 0.5, "seed": 99}

    def test_extra_overrides_passed_through(self):
        c = self._client(temperature=0.5)
        result = c._build_generation_payload(guided_json={"type": "object"})
        assert result == {"temperature": 0.5, "guided_json": {"type": "object"}}

    def test_all_generation_keys_supported(self):
        c = self._client(
            temperature=0.7,
            max_tokens=1024,
            top_p=0.9,
            seed=42,
            stop=["END"],
            frequency_penalty=0.5,
            presence_penalty=-0.5,
        )
        result = c._build_generation_payload()
        assert result == {
            "temperature": 0.7,
            "max_tokens": 1024,
            "top_p": 0.9,
            "seed": 42,
            "stop": ["END"],
            "frequency_penalty": 0.5,
            "presence_penalty": -0.5,
        }

    def test_override_with_none_default_uses_override(self):
        c = self._client()
        result = c._build_generation_payload(temperature=0.3)
        assert result == {"temperature": 0.3}
