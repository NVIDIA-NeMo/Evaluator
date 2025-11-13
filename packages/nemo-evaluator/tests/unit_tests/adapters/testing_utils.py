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

"""Utilities for adapter testing."""

import multiprocessing
import threading
import time
from typing import Any, List, Union

import pytest
from flask import Flask, jsonify, request

from nemo_evaluator.adapters.registry import InterceptorRegistry
from nemo_evaluator.adapters.server import AdapterServer, wait_for_server
from nemo_evaluator.adapters.types import (
    RequestInterceptor,
    RequestToResponseInterceptor,
    ResponseInterceptor,
)

DEFAULT_FAKE_RESPONSE = {
    "object": "chat.completion",
    "choices": [
        {
            "message": {
                "role": "assistant",
                "content": "This is a fake LLM response</think>This survives reasoning",
            }
        }
    ],
}


def create_and_run_adapter_server(url, out_dir, config):
    """Create and run an AdapterServer in a separate process."""
    adapter = AdapterServer(
        api_url=url,
        output_dir=out_dir,
        adapter_config=config,
    )
    adapter.run()


def create_and_run_fake_endpoint(port: int = 3300):
    """Create and run a fake OpenAI API endpoint on the specified port."""
    app = Flask(__name__)

    @app.route("/v1/chat/completions", methods=["POST"])
    def chat_completion():
        data = request.json
        if "fake_response" in data:
            response = data["fake_response"]
        else:
            response = DEFAULT_FAKE_RESPONSE
        return jsonify(response)

    app.run(host="localhost", port=port)


def create_fake_endpoint_process_on_port(port: int = 3300):
    """Create a process running a fake OpenAI endpoint on the specified port.

    Args:
        port: The port to run the fake endpoint on (default: 3300)

    Returns:
        The multiprocessing.Process object running the endpoint.
    """

    p = multiprocessing.Process(target=create_and_run_fake_endpoint, args=(port,))
    p.daemon = True  # Ensure process is terminated when main process exits
    p.start()

    try:
        # Wait for the server to be ready
        if not wait_for_server("localhost", port):
            p.terminate()
            p.join(timeout=5)  # Give it some time to terminate
            pytest.fail(
                f"Fake OpenAI endpoint did not start on port {port} within the timeout period"
            )
    except Exception as e:
        p.terminate()
        p.join(timeout=5)
        pytest.fail(f"Failed to start fake endpoint on port {port}: {str(e)}")

    return p


def create_fake_endpoint_process():
    """Create a process running a fake OpenAI endpoint on the default port 3300.

    Returns:
        The multiprocessing.Process object running the endpoint.
    """
    return create_fake_endpoint_process_on_port(3300)


def reset_registry() -> None:
    """Reset the interceptor registry for testing."""
    registry = InterceptorRegistry.get_instance()
    registry.reset()


def get_request_interceptors(
    configs: dict[str, dict[str, Any]],
) -> list[Union[RequestInterceptor, RequestToResponseInterceptor]]:
    """Get all request interceptors with their configurations.

    This is a testing utility that provides access to request interceptors
    from the registry. For production use, use the registry directly.

    Args:
        configs: Dictionary mapping interceptor names to their configurations

    Returns:
        List of request interceptors (RequestInterceptor or RequestToResponseInterceptor)
    """
    registry = InterceptorRegistry.get_instance()
    interceptors = []
    for name, config in configs.items():
        if name in registry._metadata:
            metadata = registry._metadata[name]
            if metadata.supports_request_interception():
                interceptor = registry._get_or_create_instance(name, config)
                if isinstance(
                    interceptor, (RequestInterceptor, RequestToResponseInterceptor)
                ):
                    interceptors.append(interceptor)
    return interceptors


def get_response_interceptors(
    configs: dict[str, dict[str, Any]],
) -> list[ResponseInterceptor]:
    """Get all response interceptors with their configurations.

    This is a testing utility that provides access to response interceptors
    from the registry. For production use, use the registry directly.

    Args:
        configs: Dictionary mapping interceptor names to their configurations

    Returns:
        List of response interceptors (ResponseInterceptor)
    """
    registry = InterceptorRegistry.get_instance()
    interceptors = []
    for name, config in configs.items():
        if name in registry._metadata:
            metadata = registry._metadata[name]
            if metadata.supports_response_interception():
                interceptor = registry._get_or_create_instance(name, config)
                if isinstance(interceptor, ResponseInterceptor):
                    interceptors.append(interceptor)
    return interceptors


def get_request_to_response_interceptors(
    configs: dict[str, dict[str, Any]],
) -> list[RequestToResponseInterceptor]:
    """Get all request-to-response interceptors with their configurations.

    This is a testing utility that provides access to request-to-response interceptors
    from the registry. For production use, use the registry directly.

    Args:
        configs: Dictionary mapping interceptor names to their configurations

    Returns:
        List of request-to-response interceptors (RequestToResponseInterceptor)
    """
    registry = InterceptorRegistry.get_instance()
    interceptors = []
    for name, config in configs.items():
        if name in registry._metadata:
            metadata = registry._metadata[name]
            if metadata.supports_request_to_response_interception():
                interceptor = registry._get_or_create_instance(name, config)
                if isinstance(interceptor, RequestToResponseInterceptor):
                    interceptors.append(interceptor)
    return interceptors


class FakeProgressTrackingServer:
    """Test server to receive progress tracking webhooks."""

    def __init__(self, port: int = 8000, request_method="PATCH"):
        self.port = port
        self.app = Flask(__name__)
        self.received_updates: List[dict] = []
        self.lock = threading.Lock()

        @self.app.route("/", methods=[request_method])
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
        time.sleep(0.5)

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
