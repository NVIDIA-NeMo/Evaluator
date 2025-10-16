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
from nemo_evaluator.adapters.server import AdapterServerProcess
from nemo_evaluator.adapters.types import (
    AdapterGlobalContext,
    AdapterRequestContext,
    AdapterResponse,
)
from nemo_evaluator.api.api_dataclasses import (
    ApiEndpoint,
    Evaluation,
    EvaluationConfig,
    EvaluationTarget,
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


def test_response_logging_json(create_response, caplog, tmpdir):
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
    # Use caplog to capture log messages
    assert any("Outgoing response" in record.message for record in caplog.records)
    assert any("200" in record.message for record in caplog.records)
    assert any("result" in record.message for record in caplog.records)


def mock_context(tmpdir):
    return AdapterGlobalContext(output_dir=str(tmpdir), url="http://localhost")


def test_response_logging_max_responses(create_response, caplog, tmpdir):
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
    # Use caplog to capture log messages
    assert len([r for r in caplog.records if "Outgoing response" in r.message]) == 1


def test_response_logging_unlimited(create_response, caplog, tmpdir):
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
    # Use caplog to capture log messages
    assert len([r for r in caplog.records if "Outgoing response" in r.message]) == 3


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

        # Get the actual port from environment variable or default
        import os

        adapter_port = int(os.environ.get("ADAPTER_PORT", 3825))

        evaluation = Evaluation(
            command="",
            framework_name="",
            pkg_name="",
            config=EvaluationConfig(),
            target=EvaluationTarget(
                api_endpoint=ApiEndpoint(url=api_url, adapter_config=adapter_config)
            ),
        )
        with AdapterServerProcess(evaluation) as adapter_server_process:
            # Make a test request
            test_data = {"prompt": "Test prompt", "max_tokens": 100}
            response = requests.post(
                f"http://localhost:{adapter_server_process.port}", json=test_data
            )
            assert response.status_code == 200

            # Second request (should not be logged due to max_responses=1)
            response2 = requests.post(
                f"http://localhost:{adapter_server_process.port}", json=test_data
            )
            assert response2.status_code == 200

            # Then: Only the first response should be logged
            # Check both stdout and stderr since our logging goes to stderr
            out, err = capfd.readouterr()
            assert (
                len(
                    re.findall(r"Outgoing response", err)
                    or re.findall(r"Outgoing response", out)
                )
                == 1
            )

    finally:
        # Clean up fake endpoint
        if "fake_endpoint" in locals():
            fake_endpoint.terminate()
            fake_endpoint.join(timeout=5)
