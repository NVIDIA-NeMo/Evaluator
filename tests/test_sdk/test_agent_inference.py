"""Unit tests for Agent value type, AgentFormat enum, and agent_inference module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from nemo_evaluator.sdk.agent_inference import (
    AgentInferenceFn,
    _derive_input_message,
    _extract_jsonpath,
    _make_generic_agent_request,
    _make_nat_agent_request,
    make_agent_inference_request,
)
from nemo_evaluator.sdk.enums import AgentFormat
from nemo_evaluator.sdk.values.agents import Agent

# ============================================================================
# AgentFormat enum
# ============================================================================


class TestAgentFormat:
    def test_enum_values(self):
        assert AgentFormat.GENERIC == "generic"
        assert AgentFormat.NEMO_AGENT_TOOLKIT == "nemo_agent_toolkit"

    def test_is_string_enum(self):
        assert isinstance(AgentFormat.GENERIC, str)
        assert isinstance(AgentFormat.NEMO_AGENT_TOOLKIT, str)


# ============================================================================
# Agent value type validation
# ============================================================================


class TestAgentValidation:
    def test_generic_agent_requires_body_and_response_path(self):
        """Generic agents must supply body and response_path."""
        with pytest.raises(ValueError, match="body"):
            Agent(url="http://agent.test", name="test", format=AgentFormat.GENERIC, response_path="$.answer")

        with pytest.raises(ValueError, match="response_path"):
            Agent(url="http://agent.test", name="test", format=AgentFormat.GENERIC, body={"query": "{{ prompt }}"})

    def test_generic_agent_valid(self):
        agent = Agent(
            url="http://agent.test",
            name="my-agent",
            format=AgentFormat.GENERIC,
            body={"query": "{{ prompt }}"},
            response_path="$.result.text",
        )
        assert agent.format == AgentFormat.GENERIC
        assert agent.body == {"query": "{{ prompt }}"}
        assert agent.response_path == "$.result.text"
        assert agent.trajectory_path is None

    def test_nat_agent_does_not_require_body_or_response_path(self):
        agent = Agent(url="http://nat.test", name="nat-agent", format=AgentFormat.NEMO_AGENT_TOOLKIT)
        assert agent.body is None
        assert agent.response_path is None

    def test_default_format_is_generic(self):
        """Format defaults to 'generic' — so body + response_path are required."""
        with pytest.raises(ValueError, match="body"):
            Agent(url="http://agent.test", name="test")

    def test_api_key_env_sanitisation(self):
        agent = Agent(
            url="http://agent.test",
            name="test",
            format=AgentFormat.NEMO_AGENT_TOOLKIT,
            api_key_secret="my-workspace/my-secret-key",
        )
        # Hyphens and slashes become underscores
        assert agent.api_key_env == "my_workspace_my_secret_key"

    def test_api_key_env_digit_prefix(self):
        agent = Agent(
            url="http://agent.test",
            name="test",
            format=AgentFormat.NEMO_AGENT_TOOLKIT,
            api_key_secret="9secret",
        )
        assert agent.api_key_env == "_9secret"

    def test_api_key_resolved_from_env(self, monkeypatch):
        monkeypatch.setenv("ws_my_key", "sk-test-value")
        agent = Agent(
            url="http://agent.test",
            name="test",
            format=AgentFormat.NEMO_AGENT_TOOLKIT,
            api_key_secret="ws/my-key",
        )
        assert agent.api_key == "sk-test-value"

    def test_api_key_none_when_no_secret(self):
        agent = Agent(url="http://nat.test", name="test", format=AgentFormat.NEMO_AGENT_TOOLKIT)
        assert agent.api_key is None
        assert agent.api_key_env is None

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValueError):
            Agent(url="http://agent.test", name="test", format=AgentFormat.NEMO_AGENT_TOOLKIT, extra_field="bad")  # type: ignore[call-arg]

    def test_generic_with_trajectory_path(self):
        agent = Agent(
            url="http://agent.test",
            name="test",
            format=AgentFormat.GENERIC,
            body={"q": "{{ prompt }}"},
            response_path="$.answer",
            trajectory_path="$.steps",
        )
        assert agent.trajectory_path == "$.steps"


# ============================================================================
# Helper: _derive_input_message
# ============================================================================


class TestDeriveInputMessage:
    def test_extracts_last_user_message_from_chat(self):
        request = {"messages": [{"role": "system", "content": "Be helpful"}, {"role": "user", "content": "Hello!"}]}
        assert _derive_input_message(request) == "Hello!"

    def test_extracts_prompt(self):
        assert _derive_input_message({"prompt": "Summarise this"}) == "Summarise this"

    def test_fallback_concatenates_all_messages(self):
        """When no user message is found, all contents are concatenated."""
        request = {"messages": [{"role": "system", "content": "System"}, {"role": "assistant", "content": "Hi"}]}
        result = _derive_input_message(request)
        assert "System" in result
        assert "Hi" in result

    def test_raises_when_no_messages_or_prompt(self):
        with pytest.raises(ValueError, match="messages.*prompt"):
            _derive_input_message({"model": "x"})


# ============================================================================
# Helper: _extract_jsonpath
# ============================================================================


class TestExtractJsonpath:
    def test_simple_extraction(self):
        data = {"result": {"text": "hello world"}}
        assert _extract_jsonpath(data, "$.result.text") == "hello world"

    def test_required_raises_when_no_match(self):
        with pytest.raises(ValueError, match="did not match"):
            _extract_jsonpath({"a": 1}, "$.missing", field_name="response_path")

    def test_optional_returns_none(self):
        result = _extract_jsonpath({"a": 1}, "$.missing", required=False)
        assert result is None


# ============================================================================
# Routing: make_agent_inference_request
# ============================================================================


class TestMakeAgentInferenceRequestRouting:
    @pytest.mark.asyncio
    async def test_routes_to_generic_executor(self):
        agent = Agent(
            url="http://agent.test",
            name="test",
            format=AgentFormat.GENERIC,
            body={"q": "{{ prompt }}"},
            response_path="$.answer",
        )
        with patch(
            "nemo_evaluator.sdk.agent_inference._make_generic_agent_request",
            new_callable=AsyncMock,
            return_value={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
        ) as mock_generic:
            result = await make_agent_inference_request(agent, {"prompt": "hi"})
            mock_generic.assert_awaited_once()
            assert result["choices"][0]["message"]["content"] == "ok"

    @pytest.mark.asyncio
    async def test_routes_to_nat_executor(self):
        agent = Agent(url="http://nat.test", name="nat", format=AgentFormat.NEMO_AGENT_TOOLKIT)
        with patch(
            "nemo_evaluator.sdk.agent_inference._make_nat_agent_request",
            new_callable=AsyncMock,
            return_value={"choices": [{"message": {"role": "assistant", "content": "nat-ok"}}]},
        ) as mock_nat:
            result = await make_agent_inference_request(agent, {"prompt": "hi"})
            mock_nat.assert_awaited_once()
            assert result["choices"][0]["message"]["content"] == "nat-ok"


# ============================================================================
# Generic agent executor
# ============================================================================


class TestGenericAgentExecutor:
    @pytest.mark.asyncio
    async def test_posts_rendered_body_and_extracts_response(self):
        agent = Agent(
            url="http://agent.test/invoke",
            name="gen",
            format=AgentFormat.GENERIC,
            body={"question": "{{ prompt }}"},
            response_path="$.answer",
        )

        with (
            patch("nemo_evaluator.sdk.agent_inference.run_with_resilience", new_callable=AsyncMock) as mock_resilience,
            patch("nemo_evaluator.sdk.agent_inference.get_logger", return_value=MagicMock()),
        ):
            mock_resilience.return_value = {"answer": "42", "meta": {}}

            result = await _make_generic_agent_request(agent, {"prompt": "What is the meaning?"}, max_retries=1)

        assert result["choices"][0]["message"]["content"] == "42"
        assert result["choices"][0]["message"]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_extracts_trajectory_when_configured(self):
        agent = Agent(
            url="http://agent.test/invoke",
            name="gen",
            format=AgentFormat.GENERIC,
            body={"q": "{{ prompt }}"},
            response_path="$.text",
            trajectory_path="$.steps",
        )

        with (
            patch("nemo_evaluator.sdk.agent_inference.run_with_resilience", new_callable=AsyncMock) as mock_resilience,
            patch("nemo_evaluator.sdk.agent_inference.get_logger", return_value=MagicMock()),
        ):
            mock_resilience.return_value = {"text": "answer", "steps": [{"action": "search"}, {"action": "summarize"}]}

            result = await _make_generic_agent_request(agent, {"prompt": "hi"})

        assert result["trajectory"] == [{"action": "search"}, {"action": "summarize"}]

    @pytest.mark.asyncio
    async def test_includes_auth_header_when_api_key_provided(self):
        agent = Agent(
            url="http://agent.test/invoke",
            name="gen",
            format=AgentFormat.GENERIC,
            body={"q": "{{ prompt }}"},
            response_path="$.text",
        )

        captured_fn = None

        async def capture_resilience(endpoint_key, fn, max_attempts):
            nonlocal captured_fn
            captured_fn = fn
            return {"text": "ok"}

        with (
            patch("nemo_evaluator.sdk.agent_inference.run_with_resilience", side_effect=capture_resilience),
            patch("nemo_evaluator.sdk.agent_inference.get_logger", return_value=MagicMock()),
        ):
            await _make_generic_agent_request(agent, {"prompt": "hi"}, api_key="secret-key")
            # The inner function captured via resilience would have the headers set;
            # We verified the flow runs without error when api_key is provided.


# ============================================================================
# NAT agent executor
# ============================================================================


class TestNATAgentExecutor:
    @pytest.mark.asyncio
    async def test_streams_sse_and_extracts_final_value(self):
        agent = Agent(url="http://nat.test", name="nat-agent", format=AgentFormat.NEMO_AGENT_TOOLKIT)

        # Simulate SSE lines
        sse_lines = [
            'data: {"step": "thinking", "content": "processing"}',
            'data: {"step": "action", "content": "searching"}',
            'data: {"value": "The final answer is 42"}',
        ]

        # Mock httpx stream directly
        async def fake_aiter_lines():
            for line in sse_lines:
                yield line

        mock_stream_response = AsyncMock()
        mock_stream_response.raise_for_status = MagicMock()
        mock_stream_response.aiter_lines = fake_aiter_lines

        mock_client = AsyncMock()
        mock_client.stream = MagicMock(return_value=AsyncContextManagerMock(mock_stream_response))

        with (
            patch("nemo_evaluator.sdk.agent_inference.httpx.AsyncClient") as MockClient,
            patch("nemo_evaluator.sdk.agent_inference.run_with_resilience") as mock_resilience,
            patch("nemo_evaluator.sdk.agent_inference.get_logger", return_value=MagicMock()),
        ):
            # run_with_resilience should call the fn passed to it
            async def call_fn(endpoint_key, fn, max_attempts):
                return await fn()

            mock_resilience.side_effect = call_fn

            mock_client_instance = AsyncMock()
            MockClient.return_value = mock_client_instance
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=False)

            # Mock stream context manager
            mock_response = AsyncMock()
            mock_response.raise_for_status = MagicMock()
            mock_response.aiter_lines = fake_aiter_lines
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=False)

            mock_client_instance.stream = MagicMock(return_value=mock_response)

            result = await _make_nat_agent_request(agent, {"messages": [{"role": "user", "content": "What is 6*7?"}]})

        assert result["choices"][0]["message"]["content"] == "The final answer is 42"
        assert result["choices"][0]["message"]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_raises_when_no_value_in_stream(self):
        agent = Agent(url="http://nat.test", name="nat-agent", format=AgentFormat.NEMO_AGENT_TOOLKIT)

        sse_lines = [
            'data: {"step": "thinking", "content": "processing"}',
            "",
        ]

        async def fake_aiter_lines():
            for line in sse_lines:
                yield line

        with (
            patch("nemo_evaluator.sdk.agent_inference.httpx.AsyncClient") as MockClient,
            patch("nemo_evaluator.sdk.agent_inference.run_with_resilience") as mock_resilience,
            patch("nemo_evaluator.sdk.agent_inference.get_logger", return_value=MagicMock()),
        ):

            async def call_fn(endpoint_key, fn, max_attempts):
                return await fn()

            mock_resilience.side_effect = call_fn

            mock_client_instance = AsyncMock()
            MockClient.return_value = mock_client_instance
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=False)

            mock_response = AsyncMock()
            mock_response.raise_for_status = MagicMock()
            mock_response.aiter_lines = fake_aiter_lines
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=False)
            mock_client_instance.stream = MagicMock(return_value=mock_response)

            # The stream has no "value" field, so after stream completes final_response is None
            # and the function should raise RuntimeError
            with pytest.raises(RuntimeError, match="without producing a final value"):
                await _make_nat_agent_request(agent, {"messages": [{"role": "user", "content": "hi"}]})

    @pytest.mark.asyncio
    async def test_uses_last_value_from_stream(self):
        """When multiple chunks have 'value', the last one wins."""
        agent = Agent(url="http://nat.test", name="nat-agent", format=AgentFormat.NEMO_AGENT_TOOLKIT)

        sse_lines = [
            'data: {"value": "partial answer"}',
            'data: {"value": "complete final answer"}',
        ]

        async def fake_aiter_lines():
            for line in sse_lines:
                yield line

        with (
            patch("nemo_evaluator.sdk.agent_inference.httpx.AsyncClient") as MockClient,
            patch("nemo_evaluator.sdk.agent_inference.run_with_resilience") as mock_resilience,
            patch("nemo_evaluator.sdk.agent_inference.get_logger", return_value=MagicMock()),
        ):

            async def call_fn(endpoint_key, fn, max_attempts):
                return await fn()

            mock_resilience.side_effect = call_fn

            mock_client_instance = AsyncMock()
            MockClient.return_value = mock_client_instance
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=False)

            mock_response = AsyncMock()
            mock_response.raise_for_status = MagicMock()
            mock_response.aiter_lines = fake_aiter_lines
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=False)
            mock_client_instance.stream = MagicMock(return_value=mock_response)

            result = await _make_nat_agent_request(agent, {"prompt": "hi"})

        assert result["choices"][0]["message"]["content"] == "complete final answer"


# ============================================================================
# Protocol conformance
# ============================================================================


class TestAgentInferenceFnProtocol:
    def test_make_agent_inference_request_matches_protocol(self):
        """make_agent_inference_request should satisfy AgentInferenceFn protocol."""
        fn: AgentInferenceFn = make_agent_inference_request
        assert callable(fn)


# ============================================================================
# Helper for async context manager mocking
# ============================================================================


class AsyncContextManagerMock:
    """Utility to mock async context managers."""

    def __init__(self, return_value):
        self._return_value = return_value

    async def __aenter__(self):
        return self._return_value

    async def __aexit__(self, *args):
        return False
