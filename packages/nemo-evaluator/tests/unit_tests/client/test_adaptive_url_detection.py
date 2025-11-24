# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for adaptive URL detection in client mode.

These tests verify that the AdapterTransport automatically detects whether to use:
- Base URL mode: base_url + path (e.g., http://host/v1 + /chat/completions)
- Passthrough mode: use endpoint_url directly (e.g., http://custom:2137/submit)

Detection is done at runtime: if first request fails with 404/405, retry with endpoint_url directly.
"""

import tempfile
from unittest.mock import Mock, patch

import httpx
import pytest

from nemo_evaluator.adapters.adapter_config import AdapterConfig, InterceptorConfig
from nemo_evaluator.client.adapter_transport import (
    AdapterTransport,
    create_adapter_http_client,
)


class TestAdaptiveURLDetection:
    """Test adaptive URL detection in AdapterTransport."""

    def setup_method(self):
        """Set up test fixtures."""
        self.adapter_config = AdapterConfig(
            interceptors=[
                InterceptorConfig(name="endpoint", enabled=True),
            ]
        )
        self.tmpdir = tempfile.mkdtemp()

    def test_initial_state_no_detection(self):
        """Test that transport starts with no detected mode."""
        transport = AdapterTransport(
            adapter_config=self.adapter_config,
            output_dir=self.tmpdir,
        )

        assert transport._detected_mode is None, "Should start with no detected mode"

    @patch(
        "nemo_evaluator.client.adapter_transport.AdapterTransport._process_request_through_pipeline"
    )
    def test_base_url_mode_detection_on_success(self, mock_process):
        """Test that successful first request detects base URL mode."""
        transport = AdapterTransport(
            adapter_config=self.adapter_config,
            output_dir=self.tmpdir,
        )

        # Mock successful response
        mock_response = Mock(spec=httpx.Response)
        mock_process.return_value = mock_response

        # Create a mock request
        mock_request = Mock(spec=httpx.Request)
        mock_request.url = httpx.URL("http://localhost:8000/v1/chat/completions")
        mock_request.method = "POST"

        # Handle the request
        transport.handle_request(mock_request)

        # Verify base URL mode was detected
        assert transport._detected_mode is False, (
            "Should detect base URL mode on first success"
        )

    @patch(
        "nemo_evaluator.client.adapter_transport.AdapterTransport._process_request_through_pipeline"
    )
    def test_passthrough_mode_detection_on_404_retry(self, mock_process):
        """Test that 404 on first request triggers retry with endpoint_url and detects passthrough mode."""
        endpoint_url = "http://custom:2137/submit"
        transport = AdapterTransport(
            adapter_config=self.adapter_config,
            output_dir=self.tmpdir,
            endpoint_url=endpoint_url,
        )

        # Mock 404 error on first call, success on second
        mock_response_404 = Mock()
        mock_response_404.status_code = 404
        error_404 = httpx.HTTPStatusError(
            "Not Found", request=Mock(), response=mock_response_404
        )

        mock_response_success = Mock(spec=httpx.Response)

        # First call raises 404, second call succeeds
        mock_process.side_effect = [error_404, mock_response_success]

        # Create a mock request
        mock_request = Mock(spec=httpx.Request)
        mock_request.url = httpx.URL("http://custom:2137/submit/chat/completions")
        mock_request.method = "POST"

        # Handle the request
        transport.handle_request(mock_request)

        # Verify passthrough mode was detected
        assert transport._detected_mode is True, (
            "Should detect passthrough mode after successful retry"
        )

        # Verify process was called twice (original + retry)
        assert mock_process.call_count == 2

        # Verify second call used endpoint_url directly
        second_call_args = mock_process.call_args_list[1]
        assert endpoint_url in str(second_call_args), (
            "Should have used endpoint_url for retry"
        )

    @patch(
        "nemo_evaluator.client.adapter_transport.AdapterTransport._process_request_through_pipeline"
    )
    def test_subsequent_requests_use_detected_mode(self, mock_process):
        """Test that subsequent requests use the detected mode without retrying."""
        endpoint_url = "http://custom:2137/submit"
        transport = AdapterTransport(
            adapter_config=self.adapter_config,
            output_dir=self.tmpdir,
            endpoint_url=endpoint_url,
        )

        # Manually set detected mode to passthrough
        transport._detected_mode = True

        mock_response = Mock(spec=httpx.Response)
        mock_process.return_value = mock_response

        # Create a mock request
        mock_request = Mock(spec=httpx.Request)
        mock_request.url = httpx.URL("http://custom:2137/submit/chat/completions")
        mock_request.method = "POST"

        # Handle the request
        transport.handle_request(mock_request)

        # Verify process was only called once (no retry since mode is known)
        assert mock_process.call_count == 1

        # Verify endpoint_url was used directly
        call_args = mock_process.call_args
        assert endpoint_url in str(call_args), (
            "Should have used endpoint_url for passthrough mode"
        )

    def test_create_adapter_http_client_factory(self):
        """Test the factory function creates client and transport correctly."""
        http_client, transport = create_adapter_http_client(
            adapter_config=self.adapter_config,
            output_dir=self.tmpdir,
            endpoint_url="http://localhost:8000/v1",
        )

        assert isinstance(http_client, httpx.Client)
        assert isinstance(transport, AdapterTransport)
        assert transport._detected_mode is None, "Should start with no detected mode"

    def test_custom_endpoint_urls(self):
        """Test that custom endpoint URLs work with passthrough mode."""
        custom_endpoints = [
            "http://my-server:2137/submit",
            "http://api.example.com/custom/path",
            "http://localhost:9000/inference",
        ]

        for endpoint_url in custom_endpoints:
            transport = AdapterTransport(
                adapter_config=self.adapter_config,
                output_dir=self.tmpdir,
                endpoint_url=endpoint_url,
            )

            # Set passthrough mode
            transport._detected_mode = True

            # Verify endpoint_url is stored
            assert transport.endpoint_url == endpoint_url


class TestClientModeIntegration:
    """Integration tests for client mode with adaptive detection."""

    def setup_method(self):
        """Set up test fixtures."""
        self.adapter_config = AdapterConfig(
            interceptors=[
                InterceptorConfig(name="endpoint", enabled=True),
            ]
        )
        self.tmpdir = tempfile.mkdtemp()

    def test_client_can_be_created_with_various_urls(self):
        """Test that client can be created with different URL formats."""
        # Base URL format
        http_client1, transport1 = create_adapter_http_client(
            adapter_config=self.adapter_config,
            output_dir=self.tmpdir,
            endpoint_url="http://localhost:8000/v1",
        )
        assert transport1._detected_mode is None

        # Full endpoint URL format
        http_client2, transport2 = create_adapter_http_client(
            adapter_config=self.adapter_config,
            output_dir=self.tmpdir,
            endpoint_url="http://localhost:8000/v1/chat/completions",
        )
        assert transport2._detected_mode is None

        # Both should work - detection happens at runtime
        http_client1.close()
        http_client2.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
