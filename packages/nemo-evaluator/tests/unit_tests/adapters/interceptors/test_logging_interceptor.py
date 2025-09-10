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

import re

import pytest
import requests

from nemo_evaluator.adapters.adapter_config import AdapterConfig
from nemo_evaluator.adapters.interceptors.logging_interceptor import (
    ResponseLoggingInterceptor,
)
from nemo_evaluator.adapters.server import spawn_adapter_server, wait_for_server
from nemo_evaluator.adapters.types import (
    AdapterGlobalContext,
    AdapterRequestContext,
    AdapterResponse,
)
from tests.unit_tests.adapters.testing_utils import (
    create_fake_endpoint_process,
)


@pytest.fixture
def create_response():
    def _create_response(status_code: int, content: bytes, headers: dict):
        response = requests.Response()
        response.status_code = status_code
        response._content = content
        response.headers = requests.utils.CaseInsensitiveDict(headers)
        return response

    return _create_response


def test_response_logging_json(create_response, capfd, tmpdir):
    # Given: A response with JSON content and cache hit metadata
    interceptor = ResponseLoggingInterceptor(
        params=ResponseLoggingInterceptor.Params(max_responses=1)
    )
    response = create_response(
        200, b'{"result": "test result"}', {"Content-Type": "application/json"}
    )
    adapter_response = AdapterResponse(
        r=response, rctx=AdapterRequestContext(cache_hit=True)
    )

    # When: The response is intercepted
    result = interceptor.intercept_response(adapter_response, mock_context(tmpdir))

    # Then: The response should be logged with JSON content and cache hit status
    assert result == adapter_response
    out, err = capfd.readouterr()
    assert re.findall(r"[^_]Outgoing response[^_]", out)
    assert "200" in out
    assert "'result': 'test result'" in out


def mock_context(tmpdir):
    return AdapterGlobalContext(output_dir=str(tmpdir), url="http://localhost")


def test_response_logging_max_responses(create_response, capfd, tmpdir):
    # Given: A response and interceptor with max_responses=1
    interceptor = ResponseLoggingInterceptor(
        params=ResponseLoggingInterceptor.Params(max_responses=1)
    )
    response = create_response(
        200, b'{"result": "test result"}', {"Content-Type": "application/json"}
    )
    adapter_response = AdapterResponse(r=response, rctx=AdapterRequestContext())

    # When: Two responses are intercepted
    result1 = interceptor.intercept_response(adapter_response, mock_context(tmpdir))
    result2 = interceptor.intercept_response(adapter_response, mock_context(tmpdir))

    # Then: Only the first response should be logged
    assert result1 == adapter_response
    assert result2 == adapter_response
    out, err = capfd.readouterr()
    assert len(re.findall(r"[^_]Outgoing response[^_]", out)) == 1


def test_response_logging_unlimited(create_response, capfd, tmpdir):
    # Given: A response and interceptor with max_responses=None
    interceptor = ResponseLoggingInterceptor(
        params=ResponseLoggingInterceptor.Params(max_responses=None)
    )
    response = create_response(
        200, b'{"result": "test result"}', {"Content-Type": "application/json"}
    )
    adapter_response = AdapterResponse(r=response, rctx=AdapterRequestContext())

    # When: Multiple responses are intercepted
    for _ in range(3):
        result = interceptor.intercept_response(adapter_response, mock_context(tmpdir))

    # Then: All responses should be logged
    assert result == adapter_response
    out, err = capfd.readouterr()
    print(repr(out))
    assert len(re.findall(r"[^_]Outgoing response[^_]", out)) == 3


def test_logging_interceptor_with_adapter_server(capfd, tmp_path):
    """Test logging interceptor with a real adapter server."""
    adapter_config = AdapterConfig(
        interceptors=[
            dict(
                name="request_logging",
                config={
                    "log_requests": True,
                },
            ),
            dict(
                name="endpoint",
                config={},
            ),
            dict(
                name="response_logging",
                config={
                    "log_responses": True,
                    "max_responses": 1,
                },
            ),
        ]
    )
    api_url = "http://localhost:3300/v1/chat/completions"
    output_dir = str(tmp_path)

    try:
        # Start fake endpoint
        fake_endpoint = create_fake_endpoint_process()
        # Start adapter server
        p = spawn_adapter_server(api_url, output_dir, adapter_config)

        # Wait for server to be ready
        wait_for_server("localhost", 3825)

        # Make a test request
        test_data = {"prompt": "Test prompt", "max_tokens": 100}
        response = requests.post("http://localhost:3825", json=test_data)
        assert response.status_code == 200

        # Second request (should not be logged due to max_responses=1)
        response2 = requests.post("http://localhost:3825", json=test_data)
        assert response2.status_code == 200

        # Then: Only the first response should be logged
        out, _ = capfd.readouterr()
        assert len(re.findall(r"[^_]Outgoing response[^_]", out)) == 1

    finally:
        # Clean up
        p.terminate()
        p.join(timeout=5)

        # Clean up fake endpoint
        fake_endpoint.terminate()
        fake_endpoint.join(timeout=5)


def test_adapter_port_environment_variable(tmp_path):
    """Test that ADAPTER_PORT environment variable is respected."""
    import os

    # Save original environment variable
    original_port = os.environ.get("ADAPTER_PORT")

    try:
        # Set a custom port
        test_port = 9999
        os.environ["ADAPTER_PORT"] = str(test_port)

        adapter_config = AdapterConfig(
            interceptors=[
                dict(
                    name="endpoint",
                    config={},
                ),
            ]
        )
        api_url = "http://localhost:3300/v1/chat/completions"
        output_dir = str(tmp_path)

        # Start fake endpoint
        fake_endpoint = create_fake_endpoint_process()

        # Start adapter server
        p = None
        try:
            p = spawn_adapter_server(api_url, output_dir, adapter_config)

            # Wait for server to be ready on the custom port
            wait_for_server("localhost", test_port)

            # Make a test request to verify it's working on the custom port
            test_data = {"prompt": "Test prompt", "max_tokens": 100}
            response = requests.post(f"http://localhost:{test_port}", json=test_data)
            assert response.status_code == 200

            # Verify the server is actually running on the custom port, not default
            from nemo_evaluator.adapters.server import AdapterServer

            assert test_port != AdapterServer.DEFAULT_ADAPTER_PORT, (
                f"Server should use custom port {test_port}, not default {AdapterServer.DEFAULT_ADAPTER_PORT}"
            )

        finally:
            if p is not None:
                p.terminate()
                p.join(timeout=5)

            # Clean up fake endpoint
            fake_endpoint.terminate()
            fake_endpoint.join(timeout=5)

    finally:
        # Restore original environment variable
        if original_port is not None:
            os.environ["ADAPTER_PORT"] = original_port
        else:
            os.environ.pop("ADAPTER_PORT", None)
