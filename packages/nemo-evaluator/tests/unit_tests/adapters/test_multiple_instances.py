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

"""Tests for multiple instances of the same interceptor with different configurations."""

import pytest
from nemo_evaluator.adapters.adapter_config import AdapterConfig, InterceptorConfig

# Import interceptors to register them
from nemo_evaluator.adapters.server import AdapterServer


class TestMultipleInstances:
    """Test that AdapterServer can handle multiple instances of the same interceptor."""

    def test_multiple_caching_instances(self):
        """Test that we can have multiple caching interceptors with different configs."""
        config = AdapterConfig(
            interceptors=[
                InterceptorConfig(
                    name="caching",
                    enabled=True,
                    config={
                        "cache_dir": "/tmp/cache1",
                        "reuse_cached_responses": True,
                        "save_responses": True,
                    },
                ),
                InterceptorConfig(
                    name="caching",
                    enabled=True,
                    config={
                        "cache_dir": "/tmp/cache2",
                        "reuse_cached_responses": False,
                        "save_responses": False,
                    },
                ),
            ]
        )

        # This should not raise an error
        server = AdapterServer(
            api_url="https://api.example.com",
            output_dir="/tmp/output",
            adapter_config=config,
        )

        # Verify that both interceptors are in the chain
        assert len(server.interceptor_chain) == 2
        assert all(
            "caching" in type(interceptor).__name__.lower()
            for interceptor in server.interceptor_chain
        )

    def test_multiple_logging_instances(self):
        """Test that we can have multiple logging interceptors with different configs."""
        config = AdapterConfig(
            interceptors=[
                InterceptorConfig(
                    name="request_logging",
                    enabled=True,
                    config={"output_dir": "/tmp/logs1", "max_requests": 5},
                ),
                InterceptorConfig(
                    name="request_logging",
                    enabled=True,
                    config={"output_dir": "/tmp/logs2", "max_requests": 10},
                ),
                InterceptorConfig(
                    name="response_logging",
                    enabled=True,
                    config={"output_dir": "/tmp/logs3", "max_responses": 3},
                ),
            ]
        )

        # This should not raise an error
        server = AdapterServer(
            api_url="https://api.example.com",
            output_dir="/tmp/output",
            adapter_config=config,
        )

        # Verify that all interceptors are in the chain
        assert len(server.interceptor_chain) == 3

    def test_disabled_interceptors_are_skipped(self):
        """Test that disabled interceptors are not included in the chain."""
        config = AdapterConfig(
            interceptors=[
                InterceptorConfig(
                    name="request_logging",
                    enabled=True,
                    config={"output_dir": "/tmp/logs"},
                ),
                InterceptorConfig(
                    name="caching",
                    enabled=False,  # This should be skipped
                    config={"cache_dir": "/tmp/cache"},
                ),
                InterceptorConfig(
                    name="response_logging",
                    enabled=True,
                    config={"output_dir": "/tmp/logs"},
                ),
            ]
        )

        server = AdapterServer(
            api_url="https://api.example.com",
            output_dir="/tmp/output",
            adapter_config=config,
        )

        # Only enabled interceptors should be in the chain
        assert len(server.interceptor_chain) == 2

    def test_mixed_enabled_disabled_instances(self):
        """Test mixing enabled and disabled instances of the same interceptor."""
        config = AdapterConfig(
            interceptors=[
                InterceptorConfig(
                    name="caching",
                    enabled=True,
                    config={"cache_dir": "/tmp/cache1"},
                ),
                InterceptorConfig(
                    name="caching",
                    enabled=False,  # This should be skipped
                    config={"cache_dir": "/tmp/cache2"},
                ),
                InterceptorConfig(
                    name="caching",
                    enabled=True,
                    config={"cache_dir": "/tmp/cache3"},
                ),
            ]
        )

        server = AdapterServer(
            api_url="https://api.example.com",
            output_dir="/tmp/output",
            adapter_config=config,
        )

        # Only the enabled instances should be in the chain
        assert len(server.interceptor_chain) == 2

    def test_validation_still_works(self):
        """Test that validation still works for unknown interceptors."""
        config = AdapterConfig(
            interceptors=[
                InterceptorConfig(
                    name="nonexistent_interceptor",
                    enabled=True,
                    config={},
                ),
            ]
        )

        with pytest.raises(ValueError, match="Unknown interceptor"):
            AdapterServer(
                api_url="https://api.example.com",
                output_dir="/tmp/output",
                adapter_config=config,
            )

    def test_stage_order_validation_still_works(self):
        """Test that stage order validation still works with multiple instances."""
        config = AdapterConfig(
            interceptors=[
                InterceptorConfig(
                    name="response_logging",  # Response interceptor first
                    enabled=True,
                    config={"output_dir": "/tmp/logs"},
                ),
                InterceptorConfig(
                    name="request_logging",  # Request interceptor after response
                    enabled=True,
                    config={"output_dir": "/tmp/logs"},
                ),
            ]
        )

        with pytest.raises(ValueError, match="Invalid stage order"):
            AdapterServer(
                api_url="https://api.example.com",
                output_dir="/tmp/output",
                adapter_config=config,
            )
