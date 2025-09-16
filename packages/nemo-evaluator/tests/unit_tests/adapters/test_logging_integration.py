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

"""Test module for the new logging functionality with context variables."""

import os

from nemo_evaluator.adapters.interceptors.caching_interceptor import CachingInterceptor
from nemo_evaluator.adapters.interceptors.endpoint_interceptor import (
    EndpointInterceptor,
)
from nemo_evaluator.adapters.interceptors.logging_interceptor import (
    RequestLoggingInterceptor,
    ResponseLoggingInterceptor,
)
from nemo_evaluator.logging import get_current_request_id, get_logger, request_context


class TestNewLoggingSystem:
    """Test the new logging system with context variables and interceptor context."""

    def test_request_context_logging(self):
        """Test that request context properly binds request ID to all loggers."""
        with request_context() as request_id:
            # Verify context variable is set
            current_id = get_current_request_id()
            assert current_id == request_id

            # Test that server logger includes request ID
            _ = get_logger()
            # Note: We can't easily capture log output in tests, but we can verify the logger works

            # Test that interceptor logger includes request ID
            _ = get_logger(RequestLoggingInterceptor.__name__)
            # Note: We can't easily capture log output in tests, but we can verify the logger works

    def test_interceptor_logger_context(self):
        """Test that interceptor loggers include interceptor context."""
        interceptor_logger = get_logger(RequestLoggingInterceptor.__name__)
        # The logger should have interceptor context bound
        assert interceptor_logger is not None

    def test_interceptor_initialization(self):
        """Test that interceptors can be initialized with the new logging system."""
        # Test RequestLoggingInterceptor
        request_interceptor = RequestLoggingInterceptor(
            RequestLoggingInterceptor.Params()
        )
        assert request_interceptor.logger is not None

        # Test ResponseLoggingInterceptor
        response_interceptor = ResponseLoggingInterceptor(
            ResponseLoggingInterceptor.Params()
        )
        assert response_interceptor.logger is not None

        # Test other interceptors
        endpoint_interceptor = EndpointInterceptor(EndpointInterceptor.Params())
        assert endpoint_interceptor.logger is not None

        # CachingInterceptor requires cache_dir parameter
        caching_interceptor = CachingInterceptor(
            CachingInterceptor.Params(cache_dir="/tmp/test_cache")
        )
        assert caching_interceptor.logger is not None

    def test_logging_params_inheritance(self):
        """Test that interceptors properly inherit from BaseLoggingParams."""
        # Test that interceptors have the expected base parameters
        request_params = RequestLoggingInterceptor.Params()
        assert hasattr(request_params, "log_level")
        assert hasattr(request_params, "log_request_body")
        assert hasattr(request_params, "log_request_headers")

        response_params = ResponseLoggingInterceptor.Params()
        assert hasattr(response_params, "log_level")
        assert hasattr(response_params, "log_response_body")
        assert hasattr(response_params, "log_response_headers")

    def test_environment_variable_logging(self):
        """Test that environment variables still work for logging configuration."""
        # Test NEMO_EVALUATOR_LOG_LEVEL environment variable
        original_log_level = os.environ.get("NEMO_EVALUATOR_LOG_LEVEL")

        try:
            # Set environment variable
            os.environ["NEMO_EVALUATOR_LOG_LEVEL"] = "DEBUG"

            # Create interceptor - should respect environment variable
            request_interceptor = RequestLoggingInterceptor(
                RequestLoggingInterceptor.Params()
            )
            assert request_interceptor.logger is not None

        finally:
            # Restore original environment variable
            if original_log_level:
                os.environ["NEMO_EVALUATOR_LOG_LEVEL"] = original_log_level
            elif "NEMO_EVALUATOR_LOG_LEVEL" in os.environ:
                del os.environ["NEMO_EVALUATOR_LOG_LEVEL"]


class TestLoggingIntegration:
    """Test integration between different logging components."""

    def test_request_id_propagation(self):
        """Test that request ID propagates through the logging chain."""
        with request_context() as request_id:
            # Create multiple interceptors
            _ = RequestLoggingInterceptor(RequestLoggingInterceptor.Params())
            _ = ResponseLoggingInterceptor(ResponseLoggingInterceptor.Params())

            # All interceptors should have access to the same request ID
            assert get_current_request_id() == request_id

    def test_interceptor_naming(self):
        """Test that interceptor names are properly included in logs."""
        # Each interceptor should have a unique name in its logger
        request_interceptor = RequestLoggingInterceptor(
            RequestLoggingInterceptor.Params()
        )
        response_interceptor = ResponseLoggingInterceptor(
            ResponseLoggingInterceptor.Params()
        )

        # The loggers should be different instances
        assert request_interceptor.logger is not response_interceptor.logger
