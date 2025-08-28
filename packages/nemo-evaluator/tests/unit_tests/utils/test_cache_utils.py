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

import sqlite3
from pathlib import Path
from typing import Any, Generator

import pytest
import requests
from nemo_evaluator.adapters.adapter_config import AdapterConfig
from nemo_evaluator.adapters.server import (
    AdapterServer,
    spawn_adapter_server,
    wait_for_server,
)
from nemo_evaluator.core.resources import get_token_usage_from_cache_db

from tests.nemo_evaluator.adapters.testing_utils import (
    DEFAULT_FAKE_RESPONSE,
    create_fake_endpoint_process,
)


@pytest.fixture
def fake_endpoint():
    """Create and run a fake OpenAI API endpoint."""
    p = create_fake_endpoint_process()
    yield p
    p.terminate()
    p.join(timeout=5)  # Wait for process to actually terminate


@pytest.fixture
def adapter_server_with_cache(
    tmp_path, fake_endpoint
) -> Generator[AdapterConfig, Any, Any]:
    """Create an adapter server with caching enabled."""
    api_url = "http://localhost:3300/v1/chat/completions"
    output_dir = tmp_path
    adapter_config = AdapterConfig(
        interceptors=[
            dict(
                name="caching",
                config={
                    "cache_dir": str(tmp_path / "cache"),
                    "reuse_cached_responses": True,
                    "save_requests": False,
                    "save_responses": True,
                },
            ),
            dict(
                name="endpoint",
                config={},
            ),
            dict(
                name="response_logging",
            ),
            dict(
                name="reasoning",
            ),
        ]
    )

    # Create server process
    p = spawn_adapter_server(api_url, output_dir, adapter_config)

    yield adapter_config

    p.terminate()
    p.join(timeout=5)  # Wait for process to actually terminate


def test_get_token_usage_from_cache_db_empty(tmp_path):
    """Test token usage extraction from an empty cache database."""
    # Create an empty cache database
    cache_dir = tmp_path / "cache" / "responses"
    cache_dir.mkdir(parents=True)
    cache_db_path = cache_dir / "cache.db"

    # Create empty database with correct schema
    with sqlite3.connect(cache_db_path) as conn:
        conn.execute("CREATE TABLE cache (key TEXT PRIMARY KEY, value TEXT)")

    # Test extraction from empty cache
    usage = get_token_usage_from_cache_db(cache_db_path)
    assert usage == {}


def test_get_token_usage_from_cache_db_single_request(
    adapter_server_with_cache, fake_endpoint, tmp_path
):
    """Test token usage extraction with a single cached request."""
    url = f"http://{AdapterServer.DEFAULT_ADAPTER_HOST}:{AdapterServer.DEFAULT_ADAPTER_PORT}"

    # Wait for server to be ready
    wait_for_server("localhost", 3825)

    # Test data
    test_data = {
        "prompt": "This is a test prompt",
        "max_tokens": 100,
        "temperature": 0.5,
        "fake_response": {  # Use fake_response to customize the response
            **DEFAULT_FAKE_RESPONSE,
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        },
    }

    # Make request to populate cache
    response = requests.post(url, json=test_data)
    assert response.status_code == 200

    # Verify cache database exists
    cache_db_path = Path(tmp_path) / "cache" / "responses" / "cache.db"
    assert cache_db_path.exists(), "Cache database was not created"

    # Test token usage extraction
    usage = get_token_usage_from_cache_db(cache_db_path)
    assert usage == {
        "total_tokens": 30,
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "total_cached_requests": 1,
    }


def test_get_token_usage_from_cache_db_multiple_requests(
    adapter_server_with_cache, fake_endpoint, tmp_path
):
    """Test token usage extraction with multiple cached requests."""
    url = f"http://{AdapterServer.DEFAULT_ADAPTER_HOST}:{AdapterServer.DEFAULT_ADAPTER_PORT}"

    # Wait for server to be ready
    wait_for_server("localhost", 3825)

    # Different test prompts with different token usage
    test_cases = [
        {
            "data": {
                "prompt": "First test prompt",
                "max_tokens": 100,
                "fake_response": {
                    **DEFAULT_FAKE_RESPONSE,
                    "usage": {
                        "prompt_tokens": 5,
                        "completion_tokens": 15,
                        "total_tokens": 20,
                    },
                },
            }
        },
        {
            "data": {
                "prompt": "Second test prompt",
                "max_tokens": 200,
                "fake_response": {
                    **DEFAULT_FAKE_RESPONSE,
                    "usage": {
                        "prompt_tokens": 8,
                        "completion_tokens": 22,
                        "total_tokens": 30,
                    },
                },
            }
        },
        {
            "data": {
                "prompt": "Third test prompt",
                "max_tokens": 150,
                "fake_response": {
                    **DEFAULT_FAKE_RESPONSE,
                    "usage": {
                        "prompt_tokens": 6,
                        "completion_tokens": 18,
                        "total_tokens": 24,
                    },
                },
            }
        },
    ]

    # Make requests to populate cache
    for case in test_cases:
        response = requests.post(url, json=case["data"])
        assert response.status_code == 200

    # Verify cache database exists
    cache_db_path = Path(tmp_path) / "cache" / "responses" / "cache.db"
    assert cache_db_path.exists(), "Cache database was not created"

    # Test token usage extraction
    usage = get_token_usage_from_cache_db(cache_db_path)
    assert usage == {
        "total_tokens": 74,  # 20 + 30 + 24
        "prompt_tokens": 19,  # 5 + 8 + 6
        "completion_tokens": 55,  # 15 + 22 + 18
        "total_cached_requests": 3,
    }


def test_get_token_usage_from_cache_db_invalid_path(tmp_path):
    """Test token usage extraction with invalid database path."""
    # Test with non-existent database
    usage = get_token_usage_from_cache_db(tmp_path / "nonexistent.db")
    assert usage == {}


def test_get_token_usage_from_cache_db_invalid_schema(tmp_path):
    """Test token usage extraction with invalid database schema."""
    # Create database with invalid schema
    cache_db_path = tmp_path / "invalid_cache.db"
    with sqlite3.connect(cache_db_path) as conn:
        conn.execute("CREATE TABLE invalid_table (id INTEGER)")

    # Test extraction from invalid database
    usage = get_token_usage_from_cache_db(cache_db_path)
    assert usage == {}
