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

from flask import Request

from nemo_evaluator.adapters.interceptors.system_message_interceptor import (
    SystemMessageInterceptor,
)
from nemo_evaluator.adapters.types import (
    AdapterGlobalContext,
    AdapterRequest,
    AdapterRequestContext,
)


def mock_context(tmpdir):
    return AdapterGlobalContext(output_dir=str(tmpdir), url="http://localhost")


def test_system_message_prepend_strategy_with_existing(tmpdir):
    """Test prepend strategy (default) with existing system message."""
    system_message_interceptor = SystemMessageInterceptor(
        params=SystemMessageInterceptor.Params(
            system_message="detailed thinking on", strategy="prepend"
        )
    )
    data = {
        "model": "gpt-4.1",
        "messages": [
            {"role": "system", "content": "you are a helpful assistant"},
            {"role": "user", "content": "Are semicolons optional in JavaScript?"},
        ],
        "max_tokens": 100,
        "temperature": 0.5,
    }
    request = Request.from_values(
        method="POST",
        headers={"Content-Type": "application/json"},
        data=json.dumps(data),
    )
    adapter_request = AdapterRequest(
        r=request,
        rctx=AdapterRequestContext(),
    )
    adapter_response = system_message_interceptor.intercept_request(
        adapter_request, mock_context(tmpdir)
    )
    json_output = adapter_response.r.get_json()
    # Prepend should add new message before existing one
    assert (
        json_output["messages"][0]["content"]
        == "detailed thinking on\nyou are a helpful assistant"
    )
    assert json_output["messages"][0]["role"] == "system"
    assert len(json_output["messages"]) == 2


def test_system_message_prepend_strategy_default(tmpdir):
    """Test that prepend is the default strategy."""
    system_message_interceptor = SystemMessageInterceptor(
        params=SystemMessageInterceptor.Params(system_message="detailed thinking on")
    )
    data = {
        "model": "gpt-4.1",
        "messages": [
            {"role": "system", "content": "you are a helpful assistant"},
            {"role": "user", "content": "Are semicolons optional in JavaScript?"},
        ],
        "max_tokens": 100,
        "temperature": 0.5,
    }
    request = Request.from_values(
        method="POST",
        headers={"Content-Type": "application/json"},
        data=json.dumps(data),
    )
    adapter_request = AdapterRequest(
        r=request,
        rctx=AdapterRequestContext(),
    )
    adapter_response = system_message_interceptor.intercept_request(
        adapter_request, mock_context(tmpdir)
    )
    json_output = adapter_response.r.get_json()
    # Default strategy should prepend
    assert (
        json_output["messages"][0]["content"]
        == "detailed thinking on\nyou are a helpful assistant"
    )


def test_system_message_replace_strategy(tmpdir):
    """Test replace strategy replaces existing system message."""
    system_message_interceptor = SystemMessageInterceptor(
        params=SystemMessageInterceptor.Params(
            system_message="detailed thinking on", strategy="replace"
        )
    )
    data = {
        "model": "gpt-4.1",
        "messages": [
            {"role": "system", "content": "you are a helpful assistant"},
            {"role": "user", "content": "Are semicolons optional in JavaScript?"},
        ],
        "max_tokens": 100,
        "temperature": 0.5,
    }
    request = Request.from_values(
        method="POST",
        headers={"Content-Type": "application/json"},
        data=json.dumps(data),
    )
    adapter_request = AdapterRequest(
        r=request,
        rctx=AdapterRequestContext(),
    )
    adapter_response = system_message_interceptor.intercept_request(
        adapter_request, mock_context(tmpdir)
    )
    json_output = adapter_response.r.get_json()
    # Replace should only contain the new message
    assert json_output["messages"][0]["content"] == "detailed thinking on"
    assert json_output["messages"][0]["role"] == "system"


def test_system_message_append_strategy(tmpdir):
    """Test append strategy adds to end of existing system message."""
    system_message_interceptor = SystemMessageInterceptor(
        params=SystemMessageInterceptor.Params(
            system_message="detailed thinking on", strategy="append"
        )
    )
    data = {
        "model": "gpt-4.1",
        "messages": [
            {"role": "system", "content": "you are a helpful assistant"},
            {"role": "user", "content": "Are semicolons optional in JavaScript?"},
        ],
        "max_tokens": 100,
        "temperature": 0.5,
    }
    request = Request.from_values(
        method="POST",
        headers={"Content-Type": "application/json"},
        data=json.dumps(data),
    )
    adapter_request = AdapterRequest(
        r=request,
        rctx=AdapterRequestContext(),
    )
    adapter_response = system_message_interceptor.intercept_request(
        adapter_request, mock_context(tmpdir)
    )
    json_output = adapter_response.r.get_json()
    # Append should add new message after existing one
    assert (
        json_output["messages"][0]["content"]
        == "you are a helpful assistant\ndetailed thinking on"
    )
    assert json_output["messages"][0]["role"] == "system"


def test_system_message_no_existing_system_message(tmpdir):
    """Test all strategies work when no existing system message."""
    for strategy in ["replace", "prepend", "append"]:
        system_message_interceptor = SystemMessageInterceptor(
            params=SystemMessageInterceptor.Params(
                system_message="you are a helpful assistant", strategy=strategy
            )
        )
        data = {
            "model": "gpt-4.1",
            "messages": [
                {"role": "user", "content": "Are semicolons optional in JavaScript?"},
            ],
            "max_tokens": 100,
            "temperature": 0.5,
        }
        request = Request.from_values(
            method="POST",
            headers={"Content-Type": "application/json"},
            data=json.dumps(data),
        )
        adapter_request = AdapterRequest(
            r=request,
            rctx=AdapterRequestContext(),
        )
        adapter_response = system_message_interceptor.intercept_request(
            adapter_request, mock_context(tmpdir)
        )
        json_output = adapter_response.r.get_json()
        # All strategies should add the system message when none exists
        assert json_output["messages"][0]["content"] == "you are a helpful assistant"
        assert json_output["messages"][0]["role"] == "system"
        assert len(json_output["messages"]) == 2


def test_system_message_multiple_existing_messages(tmpdir):
    """Test that multiple existing system messages are joined together."""
    system_message_interceptor = SystemMessageInterceptor(
        params=SystemMessageInterceptor.Params(
            system_message="new message", strategy="prepend"
        )
    )
    data = {
        "model": "gpt-4.1",
        "messages": [
            {"role": "system", "content": "first system message"},
            {"role": "system", "content": "second system message"},
            {"role": "user", "content": "What is 2+2?"},
        ],
        "max_tokens": 100,
        "temperature": 0.5,
    }
    request = Request.from_values(
        method="POST",
        headers={"Content-Type": "application/json"},
        data=json.dumps(data),
    )
    adapter_request = AdapterRequest(
        r=request,
        rctx=AdapterRequestContext(),
    )
    adapter_response = system_message_interceptor.intercept_request(
        adapter_request, mock_context(tmpdir)
    )
    json_output = adapter_response.r.get_json()
    # Multiple system messages should be joined
    assert (
        json_output["messages"][0]["content"]
        == "new message\nfirst system message\nsecond system message"
    )
    assert json_output["messages"][0]["role"] == "system"
    # Should only have 2 messages total (1 system, 1 user)
    assert len(json_output["messages"]) == 2


def test_system_message_preserves_other_parameters(tmpdir):
    """Test that other request parameters are preserved."""
    system_message_interceptor = SystemMessageInterceptor(
        params=SystemMessageInterceptor.Params(
            system_message="test message", strategy="replace"
        )
    )
    data = {
        "model": "gpt-4.1",
        "messages": [
            {"role": "user", "content": "Test question"},
        ],
        "max_tokens": 100,
        "temperature": 0.5,
        "top_p": 0.9,
        "custom_param": "custom_value",
    }
    request = Request.from_values(
        method="POST",
        headers={"Content-Type": "application/json"},
        data=json.dumps(data),
    )
    adapter_request = AdapterRequest(
        r=request,
        rctx=AdapterRequestContext(),
    )
    adapter_response = system_message_interceptor.intercept_request(
        adapter_request, mock_context(tmpdir)
    )
    json_output = adapter_response.r.get_json()
    # All other parameters should be preserved
    assert json_output["model"] == "gpt-4.1"
    assert json_output["max_tokens"] == 100
    assert json_output["temperature"] == 0.5
    assert json_output["top_p"] == 0.9
    assert json_output["custom_param"] == "custom_value"
