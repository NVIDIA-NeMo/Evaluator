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

import threading
import time
from typing import List
from unittest.mock import patch

import requests
from flask import Flask, request
from nvidia_eval_commons.adapters.interceptors.progress_tracking_interceptor import (
    ProgressTrackingInterceptor,
)
from nvidia_eval_commons.adapters.types import (
    AdapterGlobalContext,
    AdapterRequestContext,
    AdapterResponse,
)


class FakeProgressTrackingServer:
    """Test server to receive progress tracking webhooks."""

    def __init__(self, port: int = 8000):
        self.port = port
        self.app = Flask(__name__)
        self.received_updates: List[dict] = []
        self.lock = threading.Lock()

        @self.app.route("/", methods=["POST"])
        def progress_webhook():
            """Receive progress updates."""
            data = request.get_json()
            with self.lock:
                self.received_updates.append(data)
            return {"status": "ok"}

    def start(self):
        """Start the server in a background thread."""
        self.thread = threading.Thread(
            target=self.app.run, kwargs={"host": "0.0.0.0", "port": self.port}
        )
        self.thread.daemon = True
        self.thread.start()
        # Give the server time to start
        time.sleep(0.1)

    def stop(self):
        """Stop the server."""
        # Flask doesn't have a clean shutdown, so we'll just let it run as daemon
        pass

    def get_updates(self) -> List[dict]:
        """Get all received updates."""
        with self.lock:
            return self.received_updates.copy()

    def clear_updates(self):
        """Clear received updates."""
        with self.lock:
            self.received_updates.clear()


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

    @patch("requests.post")
    def test_network_error_handling(self, mock_post):
        """Test that network errors are handled gracefully."""
        # Mock requests.post to raise an exception
        mock_post.side_effect = requests.exceptions.RequestException(
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
        mock_post.assert_called_once()

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


if __name__ == "__main__":
    # Simple test runner for manual testing
    test = TestProgressTrackingInterceptor()
    test.test_intercept_response_sends_progress()
    print("All tests passed!")
