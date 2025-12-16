# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Tests for endpoint interceptor content handling."""

import json
from unittest.mock import Mock, patch

import pytest
import requests

from nemo_evaluator.adapters.interceptors.endpoint_interceptor import (
    EndpointInterceptor,
)
from nemo_evaluator.adapters.types import (
    AdapterGlobalContext,
    AdapterRequest,
    AdapterRequestContext,
)


@pytest.fixture
def endpoint_interceptor():
    """Create an endpoint interceptor for testing."""
    return EndpointInterceptor(params=EndpointInterceptor.Params())


@pytest.fixture
def mock_global_context():
    """Create a mock global context."""
    return AdapterGlobalContext(
        output_dir="/tmp/test_output", url="http://localhost:3300/v1/chat/completions"
    )


@pytest.fixture
def mock_flask_request():
    """Create a mock Flask request."""
    mock_req = Mock()
    mock_req.method = "POST"
    mock_req.headers = [("Content-Type", "application/json")]
    mock_req.json = {"messages": [{"role": "user", "content": "test"}]}
    mock_req.cookies = {}
    return mock_req


@pytest.fixture
def mock_adapter_request(mock_flask_request):
    """Create a mock adapter request."""
    return AdapterRequest(r=mock_flask_request, rctx=AdapterRequestContext())


def create_mock_response(status_code: int, content_dict: dict) -> requests.Response:
    """Helper function to create a mock requests.Response object."""
    response = requests.Response()
    response.status_code = status_code
    response._content = json.dumps(content_dict).encode("utf-8")
    response.headers = requests.utils.CaseInsensitiveDict(
        {"Content-Type": "application/json"}
    )
    return response


def test_chat_completion_with_none_content_multiple_choices(
    endpoint_interceptor, mock_adapter_request, mock_global_context, caplog
):
    """Test that None content in multiple choices is replaced with empty string."""
    # Given: A response with multiple choices, some with None content
    response_data = {
        "object": "chat.completion",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Valid content",
                }
            },
            {
                "message": {
                    "role": "assistant",
                    "content": None,  # This should be replaced
                }
            },
            {
                "message": {
                    "role": "assistant",
                    "content": None,  # This should be replaced
                }
            },
            {
                "message": {
                    "role": "assistant",
                    "content": "Another valid response",
                }
            },
            {
                "message": {
                    "role": "assistant",
                    "content": "",  # This should be empty string, with no warning
                }
            },
        ],
    }

    mock_response = create_mock_response(200, response_data)

    # When: The interceptor processes the request
    with patch("requests.request", return_value=mock_response):
        result = endpoint_interceptor.intercept_request(
            mock_adapter_request, mock_global_context
        )

    # Then: All None content should be replaced with empty string
    result_json = json.loads(result.r.content)
    assert result_json["choices"][0]["message"]["content"] == "Valid content"
    assert result_json["choices"][1]["message"]["content"] == ""
    assert result_json["choices"][2]["message"]["content"] == ""
    assert result_json["choices"][3]["message"]["content"] == "Another valid response"
    assert result_json["choices"][4]["message"]["content"] == ""
    # Check that warnings were logged for the None values
    warning_logs = [
        record
        for record in caplog.records
        if "message.content is None, replacing with empty string" in record.message
    ]
    assert len(warning_logs) == 2  # Two warnings for indices 1 and 2


def test_completions_endpoint_unprocessed(
    endpoint_interceptor, mock_adapter_request, mock_global_context
):
    """Test that completions endpoint (non-chat) returns unprocessed response."""
    # Given: A completions endpoint response (not chat.completion)
    response_data = {
        "object": "text_completion",
        "choices": [
            {
                "text": None,  # Different structure, should not be processed
                "index": 0,
            }
        ],
    }

    mock_response = create_mock_response(200, response_data)

    # When: The interceptor processes the request
    with patch("requests.request", return_value=mock_response):
        result = endpoint_interceptor.intercept_request(
            mock_adapter_request, mock_global_context
        )

    # Then: The response should remain unchanged (no message.content structure)
    result_json = json.loads(result.r.content)
    assert result_json["choices"][0]["text"] is None  # Should not be modified


def test_response_without_choices_unprocessed(
    endpoint_interceptor, mock_adapter_request, mock_global_context
):
    """Test that responses without choices field are not modified."""
    # Given: A response without choices field
    response_data = {
        "object": "model",
        "id": "my-model",
        "data": "some data",
    }

    mock_response = create_mock_response(200, response_data)

    # When: The interceptor processes the request
    with patch("requests.request", return_value=mock_response):
        result = endpoint_interceptor.intercept_request(
            mock_adapter_request, mock_global_context
        )

    # Then: The response should remain unchanged
    result_json = json.loads(result.r.content)
    assert result_json == response_data


def test_response_with_empty_choices_array_unprocessed(
    endpoint_interceptor, mock_adapter_request, mock_global_context
):
    """Test that responses with empty choices array are not modified."""
    # Given: A response with empty choices array
    response_data = {"object": "chat.completion", "choices": []}

    mock_response = create_mock_response(200, response_data)

    # When: The interceptor processes the request
    with patch("requests.request", return_value=mock_response):
        result = endpoint_interceptor.intercept_request(
            mock_adapter_request, mock_global_context
        )

    # Then: The response should remain unchanged
    result_json = json.loads(result.r.content)
    assert result_json == response_data


def test_response_with_none_content_preserves_other_fields(
    endpoint_interceptor, mock_adapter_request, mock_global_context
):
    """Test that replacing None content preserves other fields in the response."""
    # Given: A response with None content and other fields
    response_data = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "my-model",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        },
    }

    mock_response = create_mock_response(200, response_data)

    # When: The interceptor processes the request
    with patch("requests.request", return_value=mock_response):
        result = endpoint_interceptor.intercept_request(
            mock_adapter_request, mock_global_context
        )

    # Then: Only the content should be changed, all other fields preserved
    result_json = json.loads(result.r.content)
    assert result_json["id"] == "chatcmpl-123"
    assert result_json["model"] == "my-model"
    assert result_json["choices"][0]["index"] == 0
    assert result_json["choices"][0]["message"]["role"] == "assistant"
    assert result_json["choices"][0]["message"]["content"] == ""  # Changed from None
    assert result_json["choices"][0]["finish_reason"] == "stop"
    assert result_json["usage"]["total_tokens"] == 30
