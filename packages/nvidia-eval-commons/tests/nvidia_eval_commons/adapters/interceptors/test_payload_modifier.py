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

import pytest
from flask import Request
from nvidia_eval_commons.adapters.interceptors.payload_modifier_interceptor import (
    PayloadParamsModifierInterceptor,
)
from nvidia_eval_commons.adapters.types import (
    AdapterGlobalContext,
    AdapterRequest,
    AdapterRequestContext,
)


@pytest.fixture
def payload_modifier():
    return PayloadParamsModifierInterceptor(
        params=PayloadParamsModifierInterceptor.Params(
            params_to_remove=["remove_me", "also_remove"],
            params_to_add={"new_param": "new_value", "another_param": 123},
            params_to_rename={"old_name": "new_name", "old_key": "new_key"},
        )
    )


@pytest.mark.parametrize(
    "original_data,expected_removals,expected_additions,expected_renames",
    [
        # Test case 1: Remove parameters
        (
            {
                "remove_me": "should be removed",
                "also_remove": "should also be removed",
                "keep_me": "should stay",
                "messages": [{"role": "user", "content": "test"}],
            },
            ["remove_me", "also_remove"],
            {},
            {},
        ),
        # Test case 2: Add parameters
        (
            {"messages": [{"role": "user", "content": "test"}]},
            [],
            {"new_param": "new_value", "another_param": 123},
            {},
        ),
        # Test case 3: Rename parameters
        (
            {
                "old_name": "should be renamed",
                "old_key": "should also be renamed",
                "messages": [{"role": "user", "content": "test"}],
            },
            [],
            {},
            {"old_name": "new_name", "old_key": "new_key"},
        ),
        # Test case 4: Combined modifications
        (
            {
                "remove_me": "should be removed",
                "old_name": "should be renamed",
                "keep_me": "should stay",
                "messages": [{"role": "user", "content": "test"}],
            },
            ["remove_me"],
            {"new_param": "new_value", "another_param": 123},
            {"old_name": "new_name"},
        ),
    ],
)
def test_payload_modifications(
    payload_modifier,
    original_data,
    expected_removals,
    expected_additions,
    expected_renames,
    tmpdir,
):
    # Given
    request = Request.from_values(
        data=json.dumps(original_data), content_type="application/json"
    )
    adapter_request = AdapterRequest(r=request, rctx=AdapterRequestContext())

    # Then
    modified_request = payload_modifier.intercept_request(
        adapter_request, mock_context(tmpdir)
    )
    modified_data = json.loads(modified_request.r.get_data())

    # When
    # Verify removals
    for param in expected_removals:
        assert param not in modified_data

    # Verify additions
    for key, value in expected_additions.items():
        assert key in modified_data
        assert modified_data[key] == value

    # Verify renames
    for old_key, new_key in expected_renames.items():
        assert old_key not in modified_data
        assert new_key in modified_data
        assert modified_data[new_key] == original_data[old_key]


def mock_context(tmpdir):
    return AdapterGlobalContext(output_dir=str(tmpdir), url="http://localhost")


def test_payload_modifier(payload_modifier, tmpdir):
    data = {
        "remove_me": "value1",
        "also_remove": "value2",
        "old_name": "should_rename",
        "old_key": "should_rename2",
        "keep": "keep_value",
    }
    request = Request.from_values(
        method="POST",
        headers={"Content-Type": "application/json"},
        data=json.dumps(data),
    )
    adapter_request = AdapterRequest(r=request, rctx=AdapterRequestContext())
    modified_request = payload_modifier.intercept_request(
        adapter_request, mock_context(tmpdir)
    )
    json_output = modified_request.r.get_json()
    assert "remove_me" not in json_output
    assert "also_remove" not in json_output
    assert json_output["new_param"] == "new_value"
    assert json_output["another_param"] == 123
    assert json_output["new_name"] == "should_rename"
    assert json_output["new_key"] == "should_rename2"
    assert json_output["keep"] == "keep_value"


def test_chat_template_kwargs_addition(tmpdir):
    """Test adding chat_template_kwargs with enable_thinking: false to payload."""
    # Create payload modifier specifically for chat_template_kwargs
    chat_template_modifier = PayloadParamsModifierInterceptor(
        params=PayloadParamsModifierInterceptor.Params(
            params_to_add={"chat_template_kwargs": {"enable_thinking": False}}
        )
    )

    # Original payload with messages
    original_data = {
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
        "temperature": 0.7,
        "max_tokens": 100,
    }

    # Create request
    request = Request.from_values(
        method="POST",
        headers={"Content-Type": "application/json"},
        data=json.dumps(original_data),
    )
    adapter_request = AdapterRequest(r=request, rctx=AdapterRequestContext())

    # Process through interceptor
    modified_request = chat_template_modifier.intercept_request(
        adapter_request, mock_context(tmpdir)
    )
    modified_data = json.loads(modified_request.r.get_data())

    # Verify original data is preserved
    assert modified_data["messages"] == original_data["messages"]
    assert modified_data["temperature"] == original_data["temperature"]
    assert modified_data["max_tokens"] == original_data["max_tokens"]

    # Verify chat_template_kwargs was added correctly
    assert "chat_template_kwargs" in modified_data
    assert modified_data["chat_template_kwargs"] == {"enable_thinking": False}


def test_chat_template_kwargs_with_existing_payload(tmpdir):
    """Test modifying existing chat_template_kwargs in payload."""
    # Create payload modifier that overrides chat_template_kwargs
    chat_template_modifier = PayloadParamsModifierInterceptor(
        params=PayloadParamsModifierInterceptor.Params(
            params_to_add={"chat_template_kwargs": {"enable_thinking": False}}
        )
    )

    # Original payload with existing chat_template_kwargs
    original_data = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"},
        ],
        "temperature": 0.5,
        "top_p": 0.9,
        "stream": True,
        "model": "gpt-3.5-turbo",
        "chat_template_kwargs": {"enable_thinking": True, "other_param": "other_value"},
    }

    # Create request
    request = Request.from_values(
        method="POST",
        headers={"Content-Type": "application/json"},
        data=json.dumps(original_data),
    )
    adapter_request = AdapterRequest(r=request, rctx=AdapterRequestContext())

    # Process through interceptor
    modified_request = chat_template_modifier.intercept_request(
        adapter_request, mock_context(tmpdir)
    )
    modified_data = json.loads(modified_request.r.get_data())

    # Verify all original parameters are preserved (except chat_template_kwargs)
    for key, value in original_data.items():
        if key != "chat_template_kwargs":
            assert key in modified_data
            assert modified_data[key] == value

    # Verify chat_template_kwargs was overridden (not merged)
    assert "chat_template_kwargs" in modified_data
    assert modified_data["chat_template_kwargs"] == {"enable_thinking": False}
    # Verify the original other_param was lost (not merged)
    assert "other_param" not in modified_data["chat_template_kwargs"]

    # Verify the complete modified payload structure
    expected_keys = set(original_data.keys())
    assert set(modified_data.keys()) == expected_keys
