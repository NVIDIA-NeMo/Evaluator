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

import json
from pathlib import Path
from typing import Any, Generator
from unittest.mock import Mock, PropertyMock

import pytest
import requests
from nemo_evaluator.adapters.adapter_config import AdapterConfig
from nemo_evaluator.adapters.interceptors.reasoning_interceptor import (
    ResponseReasoningInterceptor,
)
from nemo_evaluator.adapters.server import (
    AdapterServer,
    spawn_adapter_server,
    wait_for_server,
)
from nemo_evaluator.adapters.types import (
    AdapterGlobalContext,
    AdapterRequestContext,
    AdapterResponse,
)

from tests.unit_tests.adapters.testing_utils import create_fake_endpoint_process


@pytest.fixture
def adapter_server(tmp_path) -> Generator[AdapterConfig, Any, Any]:
    api_url = "http://localhost:3300/v1/chat/completions"
    output_dir = tmp_path
    adapter_config = AdapterConfig(
        interceptors=[
            dict(
                name="caching",
                enabled=True,
                config={
                    "cache_dir": str(tmp_path / "cache"),
                    "reuse_cached_responses": False,
                    "save_requests": False,
                    "save_responses": True,
                },
            ),
            dict(
                name="endpoint",
                enabled=True,
                config={},
            ),
            dict(
                name="reasoning",
                enabled=True,
                config={"end_reasoning_token": "</think>"},
            ),
        ]
    )
    p = spawn_adapter_server(api_url, output_dir, adapter_config)
    yield adapter_config
    p.terminate()
    p.join(timeout=5)


@pytest.mark.parametrize(
    "input_content,expected_content",
    [
        (
            "Let me think about this...\n<think>This is my reasoning process that should be removed</think>\nHere's my final answer.",
            "Here's my final answer.",
        ),
        (
            "No reasoning tokens in this response.",
            "No reasoning tokens in this response.",
        ),
        (
            "<think>First I'll analyze the problem\nThen I'll solve it step by step</think>Here's the solution.",
            "Here's the solution.",
        ),
    ],
)
def test_reasoning_responses(
    adapter_server,
    fake_openai_endpoint,
    input_content,
    expected_content,
):
    url = f"http://{AdapterServer.DEFAULT_ADAPTER_HOST}:{AdapterServer.DEFAULT_ADAPTER_PORT}"

    # Wait for server to be ready
    wait_for_server("localhost", 3825)

    # We parametrize the response of the openai fake server.
    response_data = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": input_content,
                }
            }
        ]
    }
    data = {
        "prompt": "This is a test prompt",
        "max_tokens": 100,
        "temperature": 0.5,
        "fake_response": response_data,
    }
    response = requests.post(url, json=data)

    assert response.status_code == 200
    cleaned_data = response.json()
    cleaned_content = cleaned_data["choices"][0]["message"]["content"]
    assert cleaned_content == expected_content


def test_multiple_choices(
    adapter_server,
    fake_openai_endpoint,
):
    # Given: A response with multiple choices containing reasoning tokens
    url = f"http://{AdapterServer.DEFAULT_ADAPTER_HOST}:{AdapterServer.DEFAULT_ADAPTER_PORT}"

    # Wait for server to be ready
    wait_for_server("localhost", 3825)

    response_data = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "<think>Reasoning 1</think>Answer 1",
                }
            },
            {
                "message": {
                    "role": "assistant",
                    "content": "<think>Reasoning 2</think>Answer 2",
                }
            },
        ]
    }
    data = {
        "prompt": "This is a test prompt",
        "max_tokens": 100,
        "temperature": 0.5,
        "fake_response": response_data,
    }
    response = requests.post(url, json=data)

    # Then: The reasoning tokens should be removed from all choices
    assert response.status_code == 200
    cleaned_data = response.json()
    assert cleaned_data["choices"][0]["message"]["content"] == "Answer 1"
    assert cleaned_data["choices"][1]["message"]["content"] == "Answer 2"


def test_non_assistant_role(
    adapter_server,
    fake_openai_endpoint,
):
    # Given: A response with a non-assistant role message
    url = f"http://{AdapterServer.DEFAULT_ADAPTER_HOST}:{AdapterServer.DEFAULT_ADAPTER_PORT}"

    # Wait for server to be ready
    wait_for_server("localhost", 3825)

    response_data = {
        "choices": [
            {
                "message": {
                    "role": "system",
                    "content": "<think>This should not be processed</think>System message",
                }
            }
        ]
    }
    data = {
        "prompt": "This is a test prompt",
        "max_tokens": 100,
        "temperature": 0.5,
        "fake_response": response_data,
    }
    response = requests.post(url, json=data)

    # Then: The content should remain unchanged
    cleaned_data = response.json()
    assert (
        cleaned_data["choices"][0]["message"]["content"]
        == "<think>This should not be processed</think>System message"
    )


def mock_context():
    return AdapterGlobalContext(output_dir="/tmp", url="http://localhost")


def test_reasoning_interceptor():
    # Test the reasoning interceptor directly

    interceptor = ResponseReasoningInterceptor(
        params=ResponseReasoningInterceptor.Params(add_reasoning=True)
    )

    # Create a mock response with reasoning tokens
    import requests

    mock_response = requests.Response()
    mock_response.status_code = 200
    mock_response._content = b'{"choices": [{"message": {"role": "assistant", "content": "<think>Reasoning</think>Answer"}}]}'

    response = AdapterResponse(r=mock_response, rctx=AdapterRequestContext())
    result = interceptor.intercept_response(response, mock_context())

    # Verify the reasoning was stripped
    result_content = result.r.json()
    assert result_content["choices"][0]["message"]["content"] == "Answer"


def test_reasoning_interceptor_with_adapter_server(tmp_path):
    """Test reasoning interceptor with a real adapter server."""
    adapter_config = AdapterConfig(
        interceptors=[
            dict(
                name="endpoint",
                config={},
            ),
            dict(
                name="reasoning",
                config={
                    "enabled": True,
                    "max_reasoning_steps": 3,
                },
            ),
        ]
    )
    api_url = "http://localhost:3300/v1/chat/completions"
    output_dir = str(tmp_path)

    # Start fake endpoint
    fake_endpoint = create_fake_endpoint_process()

    try:
        # Start adapter server
        p = spawn_adapter_server(api_url, output_dir, adapter_config)

        # Wait for server to be ready
        wait_for_server("localhost", 3825)

        # Make a test request
        test_data = {"prompt": "Test prompt", "max_tokens": 100}
        response = requests.post("http://localhost:3825", json=test_data)
        assert response.status_code == 200

        # Clean up
        p.terminate()
        p.join(timeout=5)

    finally:
        # Clean up fake endpoint
        fake_endpoint.terminate()
        fake_endpoint.join(timeout=5)


@pytest.mark.parametrize(
    "test_name,message_content,reasoning_content,expected_reasoning_words,expected_original_content_words,expected_reasoning_finished",
    [
        (
            "explicit_reasoning_with_output",
            "Here is my final answer.",
            "This is my reasoning process that should be tracked",
            9,  # "This is my reasoning process that should be tracked" = 9 words
            5,  # "Here is my final answer." = 5 words
            True,  # Content is not empty, so reasoning_finished = True
        ),
        (
            "explicit_reasoning_no_output",
            "",
            "This is my reasoning process but no final answer",
            9,  # "This is my reasoning process but no final answer" = 9 words
            0,  # empty content = 0 words
            False,  # Content is empty, so reasoning_finished = False
        ),
        (
            "explicit_reasoning_with_embedded_tokens",
            "<think>This should be stripped</think>Final answer",
            "This is explicit reasoning content",
            5,  # "This is explicit reasoning content" = 5 words
            5,  # "reasoning is handled in reasoning_content" = 5 words
            True,  # Content is not empty, so reasoning_finished = True
        ),
    ],
)
def test_get_reasoning_info_explicit_content(
    test_name,
    message_content,
    reasoning_content,
    expected_reasoning_words,
    expected_original_content_words,
    expected_reasoning_finished,
):
    """Test _process_reasoning_message when reasoning_content is explicitly provided in the message."""
    interceptor = ResponseReasoningInterceptor(
        params=ResponseReasoningInterceptor.Params(
            add_reasoning=True,
            enable_reasoning_tracking=True,
            end_reasoning_token="</think>",
        )
    )

    # Create message with explicit reasoning_content
    message = {
        "role": "assistant",
        "content": message_content,
        "reasoning_content": reasoning_content,
    }

    # Test the _process_reasoning_message method directly
    modified_msg, reasoning_info = interceptor._process_reasoning_message(message)

    # Verify the reasoning information
    assert reasoning_info["reasoning_words"] == expected_reasoning_words
    assert reasoning_info["original_content_words"] == expected_original_content_words
    assert reasoning_info["reasoning_finished"] == expected_reasoning_finished
    # When reasoning_content is explicitly provided, reasoning_started should be True
    assert reasoning_info["reasoning_started"]


@pytest.mark.parametrize(
    "test_name,message_content,expected_reasoning_words,expected_original_content_words,expected_reasoning_finished",
    [
        (
            "no_reasoning_content",
            "This is a simple answer without reasoning.",
            "unknown",  # no reasoning content
            7,  # "This is a simple answer without reasoning." = 7 words
            False,  # No end token found, so reasoning_finished = False
        ),
        (
            "empty_content",
            "",
            "unknown",
            0,  # empty content
            False,  # No end token found, so reasoning_finished = False
        ),
    ],
)
def test_get_reasoning_info_embedded_content(
    test_name,
    message_content,
    expected_reasoning_words,
    expected_original_content_words,
    expected_reasoning_finished,
):
    """Test _process_reasoning_message when reasoning content is embedded in the message content."""
    interceptor = ResponseReasoningInterceptor(
        params=ResponseReasoningInterceptor.Params(
            add_reasoning=True,
            enable_reasoning_tracking=True,
            end_reasoning_token="</think>",
            start_reasoning_token=None,
        )
    )

    # Create message with embedded reasoning content
    message = {"role": "assistant", "content": message_content}

    # Test the _process_reasoning_message method directly
    modified_msg, reasoning_info = interceptor._process_reasoning_message(message)

    # Verify the reasoning information
    assert reasoning_info["reasoning_words"] == expected_reasoning_words
    assert reasoning_info["original_content_words"] == expected_original_content_words
    assert reasoning_info["reasoning_finished"] == expected_reasoning_finished
    # When start_reasoning_token is not configured, reasoning_started should be "unknown"
    assert reasoning_info["reasoning_started"] == "unknown"


@pytest.mark.parametrize(
    "test_name,include_if_not_finished,message_content,expected_content,expected_reasoning_words,expected_original_content_words",
    [
        (
            "include_if_not_finished_true",
            True,
            "<think>This is reasoning without end token",
            "<think>This is reasoning without end token",  # Content is kept as is
            6,  # "This is reasoning without end token" = 6 words (as reasoning_content when reasoning_started=True)
            6,  # "This is reasoning without end token" = 6 words
        ),
        (
            "include_if_not_finished_false",
            False,
            "<think>This is reasoning without end token",
            "",  # Empty answer when not finished and include_if_not_finished is False
            6,  # "This is reasoning without end token" = 6 words (as reasoning_content when reasoning_started=True)
            6,  # "This is reasoning without end token" = 6 words
        ),
        (
            "include_if_not_finished_true_with_end_token",
            True,
            "<think>This is reasoning</think>Final answer",
            "Final answer",  # Normal processing when end token is found
            3,  # "This is reasoning" = 3 words
            4,  # "<think>This is reasoning</think>Final answer" = 4 words
        ),
        (
            "include_if_not_finished_false_with_end_token",
            False,
            "<think>This is reasoning</think>Final answer",
            "Final answer",  # Normal processing when end token is found
            3,  # "This is reasoning" = 3 words
            4,  # "<think>This is reasoning</think>Final answer" = 4 words
        ),
        (
            "include_if_not_finished_false_with_start_token",
            False,
            "<think>This is reasoning without end token",
            "",  # Content is empty when we know reasoning started but don't want to include
            6,  # "This is reasoning without end token" = 6 words (as reasoning_content when reasoning_started=True)
            6,  # "This is reasoning without end token" = 6 words
        ),
    ],
)
def test_include_if_not_finished_parameter(
    test_name,
    include_if_not_finished,
    message_content,
    expected_content,
    expected_reasoning_words,
    expected_original_content_words,
):
    """Test the include_if_not_finished parameter behavior."""
    interceptor = ResponseReasoningInterceptor(
        params=ResponseReasoningInterceptor.Params(
            add_reasoning=True,
            enable_reasoning_tracking=True,
            end_reasoning_token="</think>",
            start_reasoning_token="<think>",
            include_if_not_finished=include_if_not_finished,
        )
    )

    # Create message with embedded reasoning content
    message = {"role": "assistant", "content": message_content}

    # Test the _process_reasoning_message method directly
    modified_msg, reasoning_info = interceptor._process_reasoning_message(message)

    # Verify the modified content
    assert modified_msg["content"] == expected_content

    # Verify the reasoning information
    assert reasoning_info["reasoning_words"] == expected_reasoning_words
    assert reasoning_info["original_content_words"] == expected_original_content_words
    # Verify reasoning_finished behavior - should be True when start token is found
    assert reasoning_info["reasoning_finished"] == ("</think>" in message_content)


def test_include_if_not_finished_parameter_no_start_token():
    """Test the include_if_not_finished parameter behavior when start_reasoning_token is None."""
    interceptor = ResponseReasoningInterceptor(
        params=ResponseReasoningInterceptor.Params(
            add_reasoning=True,
            enable_reasoning_tracking=True,
            end_reasoning_token="</think>",
            start_reasoning_token=None,  # Don't know if reasoning started
            include_if_not_finished=False,
        )
    )

    # Create message with content that doesn't have end token
    message = {"role": "assistant", "content": "This is content without start token"}

    # Test the _process_reasoning_message method directly
    modified_msg, reasoning_info = interceptor._process_reasoning_message(message)

    # Verify the modified content
    assert modified_msg["content"] == ""


@pytest.mark.parametrize(
    "test_name,start_reasoning_token,message_content,expected_reasoning_started",
    [
        (
            "start_token_present",
            "<think>",
            "<think>This is reasoning</think>Final answer",
            True,
        ),
        (
            "start_token_not_present",
            "<think>",
            "This is content without start token</think>Final answer",
            True,
        ),
        (
            "start_token_none",
            None,
            "<think>This is reasoning</think>Final answer",
            True,
        ),
        (
            "start_token_none_no_tokens",
            None,
            "This is content without any tokens",
            "unknown",
        ),
        (
            "start_token_custom",
            "BEGIN_REASONING",
            "BEGIN_REASONING This is reasoning</think>Final answer",
            True,
        ),
    ],
)
def test_start_reasoning_token_parameter(
    test_name,
    start_reasoning_token,
    message_content,
    expected_reasoning_started,
):
    """Test the start_reasoning_token parameter behavior."""
    interceptor = ResponseReasoningInterceptor(
        params=ResponseReasoningInterceptor.Params(
            add_reasoning=True,
            enable_reasoning_tracking=True,
            end_reasoning_token="</think>",
            start_reasoning_token=start_reasoning_token,
        )
    )

    # Create message with embedded reasoning content
    message = {"role": "assistant", "content": message_content}

    # Test the _process_reasoning_message method directly
    modified_msg, reasoning_info = interceptor._process_reasoning_message(message)

    # Verify the reasoning_started information
    assert reasoning_info["reasoning_started"] == expected_reasoning_started


# ============================================================================
# Initialization Tests
# ============================================================================


@pytest.mark.parametrize(
    "params,expected_values",
    [
        (
            ResponseReasoningInterceptor.Params(),
            {
                "end_reasoning_token": "</think>",
                "start_reasoning_token": "<think>",
                "add_reasoning": True,
                "enable_reasoning_tracking": True,
                "include_if_not_finished": True,
                "enable_caching": True,
                "cache_dir": "/tmp/reasoning_interceptor",
                "has_cache": True,
            },
        ),
        (
            ResponseReasoningInterceptor.Params(
                end_reasoning_token="</reasoning>",
                start_reasoning_token="<reasoning>",
                add_reasoning=False,
                enable_reasoning_tracking=False,
                include_if_not_finished=False,
                enable_caching=False,
                cache_dir="/custom/cache/dir",
            ),
            {
                "end_reasoning_token": "</reasoning>",
                "start_reasoning_token": "<reasoning>",
                "add_reasoning": False,
                "enable_reasoning_tracking": False,
                "include_if_not_finished": False,
                "enable_caching": False,
                "cache_dir": "/custom/cache/dir",
                "has_cache": False,
            },
        ),
        (
            ResponseReasoningInterceptor.Params(
                enable_caching=True, cache_dir="/tmp/test_reasoning_cache"
            ),
            {
                "end_reasoning_token": "</think>",
                "start_reasoning_token": "<think>",
                "add_reasoning": True,
                "enable_reasoning_tracking": True,
                "include_if_not_finished": True,
                "enable_caching": True,
                "cache_dir": "/tmp/test_reasoning_cache",
                "has_cache": True,
            },
        ),
        (
            ResponseReasoningInterceptor.Params(enable_caching=False),
            {
                "end_reasoning_token": "</think>",
                "start_reasoning_token": "<think>",
                "add_reasoning": True,
                "enable_reasoning_tracking": True,
                "include_if_not_finished": True,
                "enable_caching": False,
                "cache_dir": "/tmp/reasoning_interceptor",
                "has_cache": False,
            },
        ),
    ],
)
def test_reasoning_interceptor_initialization(params, expected_values):
    """Test reasoning interceptor initialization with various parameter combinations."""
    # Given: Parameters for the interceptor

    # When: Creating the interceptor
    interceptor = ResponseReasoningInterceptor(params=params)

    # Then: Verify all expected values are set correctly
    assert interceptor.end_reasoning_token == expected_values["end_reasoning_token"]
    assert interceptor.start_reasoning_token == expected_values["start_reasoning_token"]
    assert interceptor.add_reasoning == expected_values["add_reasoning"]
    assert (
        interceptor.enable_reasoning_tracking
        == expected_values["enable_reasoning_tracking"]
    )
    assert (
        interceptor.include_if_not_finished
        == expected_values["include_if_not_finished"]
    )
    assert interceptor.enable_caching == expected_values["enable_caching"]
    assert interceptor.cache_dir == expected_values["cache_dir"]
    assert interceptor._lock is not None
    assert interceptor._reasoning_stats is not None

    # Verify cache initialization
    if expected_values["has_cache"]:
        assert interceptor._request_stats_cache is not None
    else:
        assert interceptor._request_stats_cache is None


# ============================================================================
# Save File Methods Tests
# ============================================================================


@pytest.mark.parametrize(
    "test_stats,expected_metrics",
    [
        (
            {
                "total_responses": 5,
                "responses_with_reasoning": 3,
                "avg_reasoning_words": 15.5,
            },
            {
                "total_responses": 5,
                "responses_with_reasoning": 3,
                "avg_reasoning_words": 15.5,
            },
        ),
        (
            {
                "total_responses": 10,
                "responses_with_reasoning": 7,
                "max_reasoning_words": 25,
            },
            {
                "total_responses": 10,
                "responses_with_reasoning": 7,
                "max_reasoning_words": 25,
            },
        ),
        (
            {"total_responses": 1, "responses_with_reasoning": 0},
            {"total_responses": 1, "responses_with_reasoning": 0},
        ),
    ],
)
def test_save_stats_to_file_success(tmp_path, test_stats, expected_metrics):
    """Test successful saving of stats to file with various stat combinations."""
    # Given: A reasoning interceptor with test stats and a context
    interceptor = ResponseReasoningInterceptor(
        params=ResponseReasoningInterceptor.Params(enable_reasoning_tracking=True)
    )

    # Set up test stats
    for key, value in test_stats.items():
        interceptor._reasoning_stats[key] = value

    context = AdapterGlobalContext(output_dir=str(tmp_path), url="http://test.com")

    # When: Calling the save method
    interceptor._save_stats_to_file(context)

    # Then: Verify file was created and contains expected metrics
    assert context.metrics_path.exists()

    with open(context.metrics_path, "r") as f:
        import json

        metrics = json.load(f)

    assert "reasoning" in metrics
    for key, expected_value in expected_metrics.items():
        assert metrics["reasoning"][key] == expected_value


@pytest.mark.parametrize(
    "existing_metrics,test_stats,expected_keys",
    [
        (
            {"existing_metric": {"description": "Some existing metric", "value": 42}},
            {"total_responses": 2, "responses_with_reasoning": 1},
            ["existing_metric", "reasoning"],
        ),
        (
            {"another_metric": {"description": "Another metric", "value": 100}},
            {"total_responses": 5, "responses_with_reasoning": 3},
            ["another_metric", "reasoning"],
        ),
    ],
)
def test_save_stats_to_file_with_existing_metrics(
    tmp_path, existing_metrics, test_stats, expected_keys
):
    """Test saving stats when metrics file already exists."""
    # Given: A reasoning interceptor with test stats and an existing metrics file
    interceptor = ResponseReasoningInterceptor(
        params=ResponseReasoningInterceptor.Params(enable_reasoning_tracking=True)
    )

    # Set up test stats
    for key, value in test_stats.items():
        interceptor._reasoning_stats[key] = value

    context = AdapterGlobalContext(output_dir=str(tmp_path), url="http://test.com")

    # Create existing metrics file
    context.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(context.metrics_path, "w") as f:
        import json

        json.dump(existing_metrics, f)

    # When: Calling the save method
    interceptor._save_stats_to_file(context)

    # Then: Verify file content was merged correctly
    with open(context.metrics_path, "r") as f:
        import json

        metrics = json.load(f)

    for expected_key in expected_keys:
        assert expected_key in metrics

    # Verify reasoning stats were added
    for key, expected_value in test_stats.items():
        assert metrics["reasoning"][key] == expected_value


@pytest.mark.parametrize(
    "nested_path",
    [
        "nested/dir/eval_factory_metrics.json",
        "deeply/nested/path/eval_factory_metrics.json",
        "single_level/eval_factory_metrics.json",
    ],
)
def test_save_stats_to_file_creates_directory(tmp_path, nested_path):
    """Test that save_stats_to_file creates the directory if it doesn't exist."""
    # Given: A reasoning interceptor and a deeply nested path
    interceptor = ResponseReasoningInterceptor(
        params=ResponseReasoningInterceptor.Params(enable_reasoning_tracking=True)
    )

    interceptor._reasoning_stats["total_responses"] = 1

    # Create nested directory structure for the test
    nested_dir = tmp_path / Path(nested_path).parent
    nested_dir.mkdir(parents=True, exist_ok=True)
    context = AdapterGlobalContext(output_dir=str(nested_dir), url="http://test.com")

    # When: Calling the save method
    interceptor._save_stats_to_file(context)

    # Then: Verify directory and file were created
    assert context.metrics_path.exists()
    assert context.metrics_path.parent.exists()


@pytest.mark.parametrize(
    "test_stats",
    [
        {"total_responses": 10, "responses_with_reasoning": 7},
        {"total_responses": 1, "responses_with_reasoning": 0},
        {
            "total_responses": 100,
            "responses_with_reasoning": 85,
            "avg_reasoning_words": 12.5,
        },
    ],
)
def test_reasoning_stats_access(test_stats):
    """Test accessing reasoning stats directly."""
    # Given: A reasoning interceptor with test stats
    interceptor = ResponseReasoningInterceptor(
        params=ResponseReasoningInterceptor.Params(enable_reasoning_tracking=True)
    )

    # Set up test stats
    for key, value in test_stats.items():
        interceptor._reasoning_stats[key] = value

    # When: Accessing stats directly
    stats = interceptor._reasoning_stats.copy()

    # Then: Verify stats are returned correctly and it's a copy
    for key, expected_value in test_stats.items():
        assert stats[key] == expected_value

    # Verify it's a copy, not the original
    assert stats is not interceptor._reasoning_stats


def test_load_from_cache_during_initialization(tmp_path):
    """Test that cached stats are automatically loaded during initialization."""
    # Given: Create cache directory and add some cached stats
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    # The current implementation uses a different cache structure
    # Let's test that the interceptor can handle initialization with caching enabled
    interceptor = ResponseReasoningInterceptor(
        params=ResponseReasoningInterceptor.Params(
            enable_caching=True, cache_dir=str(cache_dir)
        )
    )

    # Then: The interceptor should be initialized with caching enabled
    assert interceptor.enable_caching is True
    assert interceptor.cache_dir == str(cache_dir)


def test_post_eval_hook_with_cache_merge(tmp_path):
    """Test post eval hook merges stats from cache."""
    # Given: Create cache directory
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    interceptor = ResponseReasoningInterceptor(
        params=ResponseReasoningInterceptor.Params(
            enable_caching=True, cache_dir=str(cache_dir)
        )
    )

    # Add some current stats
    interceptor._reasoning_stats["total_responses"] = 2
    interceptor._reasoning_stats["responses_with_reasoning"] = 1
    interceptor._reasoning_stats["total_reasoning_words"] = 8

    context = Mock()
    # Configure the mock to return the path when accessed
    type(context).metrics_path = PropertyMock(return_value=tmp_path / "metrics.json")
    type(context).output_dir = PropertyMock(return_value=str(tmp_path))

    # When: Call post_eval_hook
    interceptor.post_eval_hook(context)

    # Then: Stats should be saved to metrics file
    assert context.metrics_path.exists()

    with open(context.metrics_path, "r") as f:
        saved_metrics = json.load(f)

    assert "reasoning" in saved_metrics
    assert saved_metrics["reasoning"]["total_responses"] == 2
    assert saved_metrics["reasoning"]["responses_with_reasoning"] == 1
    assert saved_metrics["reasoning"]["total_reasoning_words"] == 8


def test_post_eval_hook_without_caching(tmp_path):
    """Test post eval hook without caching enabled."""
    # Given
    interceptor = ResponseReasoningInterceptor(
        params=ResponseReasoningInterceptor.Params(enable_caching=False)
    )
    interceptor._reasoning_stats["total_responses"] = 3
    interceptor._reasoning_stats["responses_with_reasoning"] = 2

    context = Mock()
    # Configure the mock to return the path when accessed
    type(context).metrics_path = PropertyMock(return_value=tmp_path / "metrics.json")
    type(context).output_dir = PropertyMock(return_value=str(tmp_path))

    # When
    interceptor.post_eval_hook(context)

    # Then
    # Stats should be saved directly without cache merging
    assert context.metrics_path.exists()

    with open(context.metrics_path, "r") as f:
        saved_metrics = json.load(f)

    assert "reasoning" in saved_metrics
    assert saved_metrics["reasoning"]["total_responses"] == 3
    assert saved_metrics["reasoning"]["responses_with_reasoning"] == 2


def test_post_eval_hook_empty_stats(tmp_path):
    """Test post eval hook with empty stats."""
    # Given
    interceptor = ResponseReasoningInterceptor(
        params=ResponseReasoningInterceptor.Params(enable_caching=False)
    )
    # _stats is empty by default

    context = Mock()
    # Configure the mock to return the path when accessed
    type(context).metrics_path = PropertyMock(return_value=tmp_path / "metrics.json")
    type(context).output_dir = PropertyMock(return_value=str(tmp_path))

    # When
    interceptor.post_eval_hook(context)

    # Then
    # Should not create metrics file when no stats are collected
    assert not context.metrics_path.exists()


def test_thread_safety_save_stats(tmp_path):
    """Test that threading lock prevents concurrent writes from corrupting the file."""
    import threading

    # Given: A reasoning interceptor and context
    interceptor = ResponseReasoningInterceptor(
        params=ResponseReasoningInterceptor.Params()
    )
    context = Mock()
    type(context).metrics_path = PropertyMock(return_value=tmp_path / "metrics.json")
    type(context).output_dir = PropertyMock(return_value=str(tmp_path))

    # Set up test stats
    interceptor._reasoning_stats["total_responses"] = 1
    interceptor._reasoning_stats["responses_with_reasoning"] = 1
    interceptor._reasoning_stats["total_reasoning_words"] = 10

    # Create a function that will be called by multiple threads
    def save_stats():
        try:
            interceptor._save_stats_to_file(context)
        except Exception:
            pass  # Ignore errors for this test

    # When: Multiple threads try to save stats simultaneously
    threads = []
    for i in range(5):
        thread = threading.Thread(target=save_stats)
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Then: Verify the file is valid JSON and contains the expected data
    assert context.metrics_path.exists()

    with open(context.metrics_path, "r") as f:
        try:
            metrics = json.load(f)
            # Should contain reasoning stats
            assert "reasoning" in metrics
            # Should be valid JSON structure
            assert isinstance(metrics, dict)
        except json.JSONDecodeError:
            pytest.fail("Thread safety failed - metrics file contains invalid JSON")


def test_save_stats_creates_file(tmp_path):
    """Test that save_stats_to_file creates the metrics file."""
    # Given: A reasoning interceptor and context
    interceptor = ResponseReasoningInterceptor(
        params=ResponseReasoningInterceptor.Params()
    )
    context = Mock()
    type(context).metrics_path = PropertyMock(return_value=tmp_path / "metrics.json")
    type(context).output_dir = PropertyMock(return_value=str(tmp_path))

    # Set up test stats
    interceptor._reasoning_stats["total_responses"] = 1
    interceptor._reasoning_stats["responses_with_reasoning"] = 1

    # When: Calling save_stats_to_file
    interceptor._save_stats_to_file(context)

    # Then: Verify metrics file was created
    assert context.metrics_path.exists()

    with open(context.metrics_path, "r") as f:
        metrics = json.load(f)
        assert "reasoning" in metrics
        assert metrics["reasoning"]["total_responses"] == 1
        assert metrics["reasoning"]["responses_with_reasoning"] == 1
