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

import asyncio
import os
import threading
from unittest.mock import patch

import pytest
import requests
from pydantic_core import ValidationError

from nemo_evaluator.adapters.interceptors.progress_tracking_interceptor import (
    ProgressTrackingInterceptor,
)
from nemo_evaluator.adapters.types import (
    AdapterGlobalContext,
    AdapterRequestContext,
    AdapterResponse,
)
from tests.unit_tests.adapters.testing_utils import FakeProgressTrackingServer


class TestProgressTrackingInterceptor:
    """Test the progress tracking interceptor."""

    def test_init_with_default_params(self):
        """Test initialization with default parameters."""
        interceptor = ProgressTrackingInterceptor(ProgressTrackingInterceptor.Params())
        assert interceptor.progress_tracking_url == "http://localhost:8000"
        assert interceptor.progress_tracking_interval == 1

    def test_init_with_custom_params(self):
        """Test initialization with custom parameters."""
        params = ProgressTrackingInterceptor.Params(
            progress_tracking_url="http://test-server:9000",
            progress_tracking_interval=5,
        )
        interceptor = ProgressTrackingInterceptor(params)
        assert interceptor.progress_tracking_url == "http://test-server:9000"
        assert interceptor.progress_tracking_interval == 5

    @patch.dict(os.environ, {"NEMO_JOB_ID": "job-1234"})
    def test_init_url_with_env_expansion(self):
        """Test initialization with URL with env variable is expanded."""
        params = ProgressTrackingInterceptor.Params(
            progress_tracking_url="http://test-server:8000/jobs/${NEMO_JOB_ID}/status-details"
        )
        interceptor = ProgressTrackingInterceptor(params)
        assert (
            interceptor.progress_tracking_url
            == "http://test-server:8000/jobs/job-1234/status-details"
        )

    def test_intercept_response_sends_progress(self):
        """Test that intercept_response sends progress updates."""
        # Start test server
        server = FakeProgressTrackingServer(port=8001)
        server.start()

        try:
            # Create interceptor pointing to our test server
            params = ProgressTrackingInterceptor.Params(
                progress_tracking_url="http://localhost:8001",
                progress_tracking_interval=2,
            )
            interceptor = ProgressTrackingInterceptor(params)

            # Create mock response and context
            mock_response = AdapterResponse(
                r=requests.Response(),
                rctx=AdapterRequestContext(),
            )
            context = AdapterGlobalContext(output_dir="/tmp", url="http://test")

            # Process 5 samples
            for i in range(5):
                interceptor.intercept_response(mock_response, context)

            # Check that we received updates
            updates = server.get_updates()
            assert len(updates) == 2  # Should send updates at samples 2 and 4
            assert updates[0]["samples_processed"] == 2
            assert updates[1]["samples_processed"] == 4

        finally:
            server.stop()

    def test_post_eval_hook_sends_final_progress(self):
        """Test that post_eval_hook sends final progress update."""
        # Start test server
        server = FakeProgressTrackingServer(port=8002)
        server.start()

        try:
            # Create interceptor
            params = ProgressTrackingInterceptor.Params(
                progress_tracking_url="http://localhost:8002",
                progress_tracking_interval=3,
                output_dir=None,
            )
            interceptor = ProgressTrackingInterceptor(params)

            # Process some samples first
            mock_response = AdapterResponse(
                r=requests.Response(),
                rctx=AdapterRequestContext(),
            )
            context = AdapterGlobalContext(output_dir="/tmp", url="http://test")

            # Process 4 samples
            for i in range(4):
                interceptor.intercept_response(mock_response, context)

            # Clear previous updates
            server.clear_updates()

            # Call post_eval_hook
            interceptor.post_eval_hook(context)

            # Check that we received the final update
            updates = server.get_updates()
            assert len(updates) == 1
            assert updates[0]["samples_processed"] == 4

        finally:
            server.stop()

    def test_thread_safety(self):
        """Test that the interceptor is thread-safe."""
        # Start test server
        server = FakeProgressTrackingServer(port=8003)
        server.start()

        try:
            # Create interceptor
            params = ProgressTrackingInterceptor.Params(
                progress_tracking_url="http://localhost:8003",
                progress_tracking_interval=1,
            )
            interceptor = ProgressTrackingInterceptor(params)

            # Create mock response and context
            mock_response = AdapterResponse(
                r=requests.Response(),
                rctx=AdapterRequestContext(),
            )
            context = AdapterGlobalContext(output_dir="/tmp", url="http://test")

            # Process samples from multiple threads
            def process_samples():
                for i in range(10):
                    interceptor.intercept_response(mock_response, context)

            threads = []
            for _ in range(3):
                thread = threading.Thread(target=process_samples)
                threads.append(thread)
                thread.start()

            # Wait for all threads to complete
            for thread in threads:
                thread.join()

            # Check that we received the expected number of updates
            updates = server.get_updates()
            assert len(updates) == 30  # 3 threads * 10 samples each
            # Threading is undeterministic, thus we need to sort that list
            updates = sorted(updates, key=lambda x: x["samples_processed"])
            assert updates[-1]["samples_processed"] == 30

        finally:
            server.stop()

    @patch("requests.request")
    def test_network_error_handling(self, mock_request):
        """Test that network errors are handled gracefully."""
        # Mock requests.patch to raise an exception
        mock_request.side_effect = requests.exceptions.RequestException(
            "Connection failed"
        )

        # Create interceptor
        params = ProgressTrackingInterceptor.Params(
            progress_tracking_url="http://invalid-server:9999",
            progress_tracking_interval=1,
        )
        interceptor = ProgressTrackingInterceptor(params)

        # Create mock response and context
        mock_response = AdapterResponse(
            r=requests.Response(),
            rctx=AdapterRequestContext(),
        )
        context = AdapterGlobalContext(output_dir="/tmp", url="http://test")

        # This should not raise an exception
        result = interceptor.intercept_response(mock_response, context)
        assert result == mock_response

        # Verify that the request was attempted
        mock_request.assert_called_once()

    def test_interval_configuration_validation(self):
        with pytest.raises(ValidationError):
            ProgressTrackingInterceptor.Params(
                progress_tracking_url="http://test",
                progress_tracking_interval=0,
            )

        with pytest.raises(ValidationError):
            ProgressTrackingInterceptor.Params(
                progress_tracking_url="http://test",
                progress_tracking_interval=-2,
            )

    def test_interval_configuration(self):
        """Test different interval configurations."""
        # Start test server
        server = FakeProgressTrackingServer(port=8004)
        server.start()

        try:
            # Test with interval of 3
            params = ProgressTrackingInterceptor.Params(
                progress_tracking_url="http://localhost:8004",
                progress_tracking_interval=3,
            )
            interceptor = ProgressTrackingInterceptor(params)

            mock_response = AdapterResponse(
                r=requests.Response(),
                rctx=AdapterRequestContext(),
            )
            context = AdapterGlobalContext(output_dir="/tmp", url="http://test")

            # Process 7 samples
            for i in range(7):
                interceptor.intercept_response(mock_response, context)

            # Check updates
            updates = server.get_updates()
            assert len(updates) == 2  # Should send updates at samples 3 and 6
            assert updates[0]["samples_processed"] == 3
            assert updates[1]["samples_processed"] == 6

        finally:
            server.stop()

    def test_json_payload_format(self):
        """Test that the JSON payload has the correct format."""
        # Start test server
        server = FakeProgressTrackingServer(port=8005)
        server.start()

        try:
            # Create interceptor
            params = ProgressTrackingInterceptor.Params(
                progress_tracking_url="http://localhost:8005",
                progress_tracking_interval=1,
            )
            interceptor = ProgressTrackingInterceptor(params)

            mock_response = AdapterResponse(
                r=requests.Response(),
                rctx=AdapterRequestContext(),
            )
            context = AdapterGlobalContext(output_dir="/tmp", url="http://test")

            # Process one sample
            interceptor.intercept_response(mock_response, context)

            # Check the payload format
            updates = server.get_updates()
            assert len(updates) == 1
            assert "samples_processed" in updates[0]
            assert updates[0]["samples_processed"] == 1
            assert isinstance(updates[0]["samples_processed"], int)

        finally:
            server.stop()

    def test_configured_method(self):
        """Test that the JSON payload has the correct format."""
        # Start test server
        server = FakeProgressTrackingServer(port=8006, request_method="POST")
        server.start()

        try:
            # Create interceptor
            params = ProgressTrackingInterceptor.Params(
                progress_tracking_url="http://localhost:8006",
                progress_tracking_interval=1,
                request_method="POST",
            )
            interceptor = ProgressTrackingInterceptor(params)

            mock_response = AdapterResponse(
                r=requests.Response(),
                rctx=AdapterRequestContext(),
            )
            context = AdapterGlobalContext(output_dir="/tmp", url="http://test")

            # Process one sample
            interceptor.intercept_response(mock_response, context)

            # Check the payload format
            updates = server.get_updates()
            assert len(updates) == 1
            assert "samples_processed" in updates[0]
            assert updates[0]["samples_processed"] == 1
            assert isinstance(updates[0]["samples_processed"], int)

            # Verify PATCH does not update the server
            params = ProgressTrackingInterceptor.Params(
                progress_tracking_url="http://localhost:8006",
                progress_tracking_interval=1,
                request_method="PATCH",
            )
            interceptor = ProgressTrackingInterceptor(params)
            interceptor.intercept_response(mock_response, context)
            assert updates == server.get_updates(), (
                "server should not update with misconfigured method"
            )

        finally:
            server.stop()

    def test_interval_timer_validation(self):
        with pytest.raises(ValidationError):
            ProgressTrackingInterceptor.Params(
                progress_tracking_interval_seconds=-1,
            )

    @pytest.mark.asyncio
    async def test_interval_timer(self):
        # Start test server
        server = FakeProgressTrackingServer(port=8007)
        server.start()

        try:
            params = ProgressTrackingInterceptor.Params(
                progress_tracking_url="http://localhost:8007",
                progress_tracking_interval=50,
                progress_tracking_interval_seconds=0.2,
            )
            interceptor = ProgressTrackingInterceptor(params)
            assert interceptor.progress_tracking_url == "http://localhost:8007"
            assert interceptor.progress_tracking_interval == 50
            assert interceptor.progress_tracking_interval_seconds == 0.2

            # Create mock response and context
            mock_response = AdapterResponse(
                r=requests.Response(),
                rctx=AdapterRequestContext(),
            )
            context = AdapterGlobalContext(output_dir="/tmp", url="http://test")

            # Verify no update until timer interval
            interceptor.intercept_response(mock_response, context)
            interceptor.intercept_response(mock_response, context)
            updates = server.get_updates()
            assert len(updates) == 0, "no updates until timer interval"

            # Verify first timer interval calls update
            await asyncio.sleep(0.5)
            updates = server.get_updates()
            assert len(updates) == 1, "only expected one update"
            assert updates[0]["samples_processed"] == 2

            # Verify subsequent timer interval calls update
            interceptor.intercept_response(mock_response, context)
            await asyncio.sleep(0.5)
            updates = server.get_updates()
            assert len(updates) == 2, "expected second update"
            assert updates[1]["samples_processed"] == 3

            # No calls to update after timer is stopped
            interceptor.post_eval_hook(context)
            interceptor.intercept_response(mock_response, context)
            assert interceptor._samples_processed == 4
            await asyncio.sleep(0.5)
            updates = server.get_updates()
            assert len(updates) == 2, (
                "expected post_eval_hook to skip posting update on no change and no updates after post_eval_hook cancels timed updates"
            )

        finally:
            server.stop()


    def test_skip_initial_requests_skips_sanity_check(self):
        """Test that skip_initial_requests=1 skips the first request (e.g. sanity check)."""
        server = FakeProgressTrackingServer(port=8008)
        server.start()

        try:
            params = ProgressTrackingInterceptor.Params(
                progress_tracking_url="http://localhost:8008",
                progress_tracking_interval=1,
                skip_initial_requests=1,
            )
            interceptor = ProgressTrackingInterceptor(params)

            mock_response = AdapterResponse(
                r=requests.Response(),
                rctx=AdapterRequestContext(),
            )
            context = AdapterGlobalContext(output_dir="/tmp", url="http://test")

            # First request (sanity check) - should be skipped
            interceptor.intercept_response(mock_response, context)
            assert interceptor._samples_processed == 0
            updates = server.get_updates()
            assert len(updates) == 0, "sanity check should not trigger progress update"

            # Actual evaluation samples (10 samples)
            for i in range(10):
                interceptor.intercept_response(mock_response, context)

            assert interceptor._samples_processed == 10
            updates = server.get_updates()
            assert len(updates) == 10
            assert updates[-1]["samples_processed"] == 10

        finally:
            server.stop()

    def test_skip_initial_requests_with_interval(self):
        """Test skip_initial_requests works correctly with progress_tracking_interval."""
        server = FakeProgressTrackingServer(port=8009)
        server.start()

        try:
            params = ProgressTrackingInterceptor.Params(
                progress_tracking_url="http://localhost:8009",
                progress_tracking_interval=2,
                skip_initial_requests=1,
            )
            interceptor = ProgressTrackingInterceptor(params)

            mock_response = AdapterResponse(
                r=requests.Response(),
                rctx=AdapterRequestContext(),
            )
            context = AdapterGlobalContext(output_dir="/tmp", url="http://test")

            # First request (sanity check) - skipped
            interceptor.intercept_response(mock_response, context)

            # Process 5 actual samples
            for i in range(5):
                interceptor.intercept_response(mock_response, context)

            assert interceptor._samples_processed == 5
            updates = server.get_updates()
            # With interval=2, updates at samples 2 and 4
            assert len(updates) == 2
            assert updates[0]["samples_processed"] == 2
            assert updates[1]["samples_processed"] == 4

        finally:
            server.stop()

    def test_skip_initial_requests_post_eval_hook(self):
        """Test that post_eval_hook reports correct count when skip_initial_requests is used."""
        server = FakeProgressTrackingServer(port=8010)
        server.start()

        try:
            params = ProgressTrackingInterceptor.Params(
                progress_tracking_url="http://localhost:8010",
                progress_tracking_interval=3,
                skip_initial_requests=1,
            )
            interceptor = ProgressTrackingInterceptor(params)

            mock_response = AdapterResponse(
                r=requests.Response(),
                rctx=AdapterRequestContext(),
            )
            context = AdapterGlobalContext(output_dir="/tmp", url="http://test")

            # Sanity check (skipped)
            interceptor.intercept_response(mock_response, context)

            # 10 actual samples (interval=3 sends at 3, 6, 9; post_eval sends final 10)
            for i in range(10):
                interceptor.intercept_response(mock_response, context)

            server.clear_updates()
            interceptor.post_eval_hook(context)

            updates = server.get_updates()
            assert len(updates) == 1
            assert updates[0]["samples_processed"] == 10

        finally:
            server.stop()

    def test_skip_initial_requests_default_is_zero(self):
        """Test that skip_initial_requests defaults to 0 (count all requests)."""
        params = ProgressTrackingInterceptor.Params()
        assert params.skip_initial_requests == 0

        interceptor = ProgressTrackingInterceptor(params)
        assert interceptor.skip_initial_requests == 0

    def test_skip_initial_requests_validation(self):
        """Test that skip_initial_requests rejects negative values."""
        with pytest.raises(ValidationError):
            ProgressTrackingInterceptor.Params(
                skip_initial_requests=-1,
            )

    def test_skip_multiple_initial_requests(self):
        """Test skipping more than one initial request."""
        server = FakeProgressTrackingServer(port=8011)
        server.start()

        try:
            params = ProgressTrackingInterceptor.Params(
                progress_tracking_url="http://localhost:8011",
                progress_tracking_interval=1,
                skip_initial_requests=3,
            )
            interceptor = ProgressTrackingInterceptor(params)

            mock_response = AdapterResponse(
                r=requests.Response(),
                rctx=AdapterRequestContext(),
            )
            context = AdapterGlobalContext(output_dir="/tmp", url="http://test")

            # First 3 requests should be skipped
            for i in range(3):
                interceptor.intercept_response(mock_response, context)
            assert interceptor._samples_processed == 0

            # Next 5 requests should be counted
            for i in range(5):
                interceptor.intercept_response(mock_response, context)
            assert interceptor._samples_processed == 5

            updates = server.get_updates()
            assert len(updates) == 5
            assert updates[0]["samples_processed"] == 1
            assert updates[-1]["samples_processed"] == 5

        finally:
            server.stop()


if __name__ == "__main__":
    # Simple test runner for manual testing
    test = TestProgressTrackingInterceptor()
    test.test_intercept_response_sends_progress()
    print("All tests passed!")
