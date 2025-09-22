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

from typing import Any, Generator

import pytest
import requests

from nemo_evaluator.adapters.types import AdapterGlobalContext, AdapterRequestContext
from tests.unit_tests.adapters.testing_utils import create_fake_endpoint_process


@pytest.fixture
def fake_openai_endpoint() -> Generator[Any, Any, Any]:
    """Fixture to create a Flask app with an OpenAI response.

    Being a "proper" fake endpoint, it responds with a payload which can be
    set via app.config.response.
    """
    # Create and run the fake endpoint server
    p = create_fake_endpoint_process()

    yield p  # We only need the process reference for cleanup

    p.terminate()
    p.join(timeout=5)  # Wait for process to actually terminate


@pytest.fixture
def create_response():
    """Fixture to create parameterized response objects for testing."""

    def _create_response(status_code: int, content: bytes = None, headers: dict = None):
        response = requests.Response()
        response.status_code = status_code
        response.url = "http://test.com"
        response._content = content or b'{"error": "test error"}'
        if headers:
            response.headers = requests.utils.CaseInsensitiveDict(headers)
        return response

    return _create_response


@pytest.fixture
def adapter_global_context():
    """Fixture to create a mock adapter global context for testing."""
    return AdapterGlobalContext(output_dir="/tmp", url="http://test.com")


@pytest.fixture
def adapter_request_context():
    """Fixture to create a mock adapter request context for testing."""
    return AdapterRequestContext()
