# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nemo_evaluator.api.api_dataclasses import EndpointModelConfig
from nemo_evaluator.client.client import NeMoEvaluatorClient


@pytest.mark.asyncio
async def test_client_passes_stream_and_headers_to_openai(tmp_path):
    mock_http_client = MagicMock()
    mock_transport = MagicMock()
    mock_create = AsyncMock(
        return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
        )
    )
    mock_async_openai = MagicMock()
    mock_async_openai.chat.completions.create = mock_create
    mock_async_openai.close = AsyncMock()

    with (
        patch(
            "nemo_evaluator.client.client.create_async_adapter_http_client",
            return_value=(mock_http_client, mock_transport),
        ),
        patch(
            "nemo_evaluator.client.client.AsyncOpenAI",
            return_value=mock_async_openai,
        ),
    ):
        client = NeMoEvaluatorClient(
            EndpointModelConfig(
                model_id="test-model",
                url="https://integrate.api.nvidia.com/v1",
                stream=False,
                headers={"X-Test": "1"},
                request_timeout=30,
            ),
            output_dir=str(tmp_path),
        )
        result = await client.chat_completion([{"role": "user", "content": "hi"}])
        await client.aclose()

    assert result == "ok"
    kwargs = mock_create.call_args.kwargs
    assert kwargs["stream"] is False
    assert kwargs["extra_headers"]["X-Test"] == "1"
    assert kwargs["extra_headers"]["NVCF-POLL-SECONDS"] == "1800"


@pytest.mark.asyncio
async def test_client_does_not_retry_nvcf_504(tmp_path):
    class FakeStatusError(Exception):
        def __init__(self, status_code):
            super().__init__(f"status={status_code}")
            self.status_code = status_code

    mock_http_client = MagicMock()
    mock_transport = MagicMock()
    mock_create = AsyncMock(side_effect=FakeStatusError(504))
    mock_async_openai = MagicMock()
    mock_async_openai.chat.completions.create = mock_create
    mock_async_openai.close = AsyncMock()

    with (
        patch(
            "nemo_evaluator.client.client.create_async_adapter_http_client",
            return_value=(mock_http_client, mock_transport),
        ),
        patch(
            "nemo_evaluator.client.client.AsyncOpenAI",
            return_value=mock_async_openai,
        ),
    ):
        client = NeMoEvaluatorClient(
            EndpointModelConfig(
                model_id="test-model",
                url="https://integrate.api.nvidia.com/v1",
                request_timeout=30,
                max_retries=5,
            ),
            output_dir=str(tmp_path),
        )
        with pytest.raises(FakeStatusError):
            await client.chat_completion([{"role": "user", "content": "hi"}])
        await client.aclose()

    assert mock_create.await_count == 1


@pytest.mark.asyncio
async def test_client_respects_explicit_zero_max_retries(tmp_path):
    mock_http_client = MagicMock()
    mock_transport = MagicMock()
    mock_create = AsyncMock(side_effect=RuntimeError("boom"))
    mock_async_openai = MagicMock()
    mock_async_openai.chat.completions.create = mock_create
    mock_async_openai.close = AsyncMock()

    with (
        patch(
            "nemo_evaluator.client.client.create_async_adapter_http_client",
            return_value=(mock_http_client, mock_transport),
        ),
        patch(
            "nemo_evaluator.client.client.AsyncOpenAI",
            return_value=mock_async_openai,
        ),
    ):
        client = NeMoEvaluatorClient(
            EndpointModelConfig(
                model_id="test-model",
                url="https://integrate.api.nvidia.com/v1",
                request_timeout=30,
                max_retries=0,
            ),
            output_dir=str(tmp_path),
        )
        with pytest.raises(RuntimeError):
            await client.chat_completion([{"role": "user", "content": "hi"}])
        await client.aclose()

    assert mock_create.await_count == 1


@pytest.mark.asyncio
async def test_client_accumulates_chunks_when_streaming(tmp_path):
    async def fake_stream():
        yield SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content="Hello"))]
        )
        yield SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content=" world"))]
        )
        yield SimpleNamespace(choices=[])

    mock_http_client = MagicMock()
    mock_transport = MagicMock()
    mock_create = AsyncMock(return_value=fake_stream())
    mock_async_openai = MagicMock()
    mock_async_openai.chat.completions.create = mock_create
    mock_async_openai.close = AsyncMock()

    with (
        patch(
            "nemo_evaluator.client.client.create_async_adapter_http_client",
            return_value=(mock_http_client, mock_transport),
        ),
        patch(
            "nemo_evaluator.client.client.AsyncOpenAI",
            return_value=mock_async_openai,
        ),
    ):
        client = NeMoEvaluatorClient(
            EndpointModelConfig(
                model_id="test-model",
                url="https://integrate.api.nvidia.com/v1",
                stream=True,
                request_timeout=30,
            ),
            output_dir=str(tmp_path),
        )
        result = await client.chat_completion([{"role": "user", "content": "hi"}])
        await client.aclose()

    assert result == "Hello world"
    kwargs = mock_create.call_args.kwargs
    assert kwargs["stream"] is True
    # NVCF + stream must NOT inject NVCF-POLL-SECONDS (streaming bypasses polling)
    assert "extra_headers" not in kwargs or "NVCF-POLL-SECONDS" not in kwargs.get(
        "extra_headers", {}
    )
