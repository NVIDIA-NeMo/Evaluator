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

"""Pytest configuration and fixtures for e2e testing."""

import json
import time
import uuid
from typing import Any, Dict, List

import pytest
import requests
from flask import Flask, jsonify, request
from nemo_evaluator.adapters.types import AdapterGlobalContext, AdapterRequestContext
from nemo_evaluator.logging.utils import logger


class FakeEndpoint:
    """
    Fake endpoint that mimics the real endpoint behavior from eval-factory logs.
    Answers are based on the questions from nvidia-simple-evals mmlu_pro.
    """

    def __init__(self, port: int = 8000):
        self.port = port
        self.app = Flask(__name__)
        self.server = None
        self._setup_routes()

    def _setup_routes(self):
        """Set up the Flask routes."""

        @self.app.route("/v1/chat/completions", methods=["POST"])
        def chat_completions():
            """Handle chat completion requests."""
            try:
                # Log the incoming request
                logger.info(f"Incoming request: {request.method} {request.url}")
                logger.debug(f"Headers: {dict(request.headers)}")
                logger.debug(f"Body: {request.get_json()}")

                # Parse request
                data = request.get_json()
                if not data:
                    return jsonify({"error": "Invalid JSON"}), 400

                # Extract required fields
                messages = data.get("messages", [])
                model = data.get("model", "Qwen/Qwen3-8B")

                # Validate required fields
                if not messages:
                    return jsonify({"error": "Messages are required"}), 400

                # Generate fake response
                response = self._get_fake_response(messages)

                # Log the response
                logger.info(f"Generated response for model: {model}")
                logger.debug(f"Response: {json.dumps(response, indent=2)}")

                # Add endpoint-specific headers (matching real endpoint from logs)
                headers = {
                    "Date": time.strftime("%a, %d %b %Y %H:%M:%S GMT", time.gmtime()),
                    "Content-Type": "application/json",
                    "Connection": "keep-alive",
                    "Access-Control-Expose-Headers": "reqid",
                    "Reqid": str(uuid.uuid4()),
                    "Status": "fulfilled",
                    "Server": "uvicorn",
                    "Vary": "Origin",
                }

                return jsonify(response), 200, headers

            except Exception as e:
                logger.error(f"Error processing request: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route("/health", methods=["GET"])
        def health_check():
            """Health check endpoint."""
            return jsonify({"status": "healthy", "service": "fake-endpoint"})

        @self.app.route("/", methods=["GET"])
        def root():
            """Root endpoint with service information."""
            return jsonify(
                {
                    "service": "Fake Endpoint",
                    "version": "1.0.0",
                    "description": "Fake endpoint that mimics real endpoint behavior",
                    "endpoints": {
                        "/v1/chat/completions": "Chat completions endpoint",
                        "/health": "Health check",
                        "/": "Service information",
                    },
                }
            )

    def _get_fake_response(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Generate a fake response based on the input messages."""
        if not messages:
            return {"error": "No messages provided"}

        # Get the last user message
        last_message = messages[-1]["content"]

        # Determine which type of question this is based on keywords
        if (
            "advertising regulatory" in last_message.lower()
            or "adverts must not" in last_message.lower()
        ):
            response_data = self._get_advertising_response()
        elif "sizing" in last_message.lower() and "redundancy" in last_message.lower():
            response_data = self._get_downsizing_response()
        elif (
            "managers" in last_message.lower()
            and "shareholders" in last_message.lower()
        ):
            response_data = self._get_managers_response()
        else:
            # Default response for unknown questions
            response_data = {
                "content": "I understand your question. Let me provide a thoughtful response based on the information available.\n\nAnswer: A",
                "tokens": 25,
            }

        # Create response in the format expected by the client
        response = {
            "id": f"chat-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "meta/llama-3.1-8b-instruct",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_data["content"],
                    },
                    "logprobs": None,
                    "finish_reason": "stop",
                    "stop_reason": None,
                }
            ],
            "usage": {
                "prompt_tokens": response_data.get(
                    "prompt_tokens", len(str(messages)) // 4
                ),
                "total_tokens": response_data.get(
                    "total_tokens", len(str(messages)) // 4 + response_data["tokens"]
                ),
                "completion_tokens": response_data["tokens"],
            },
            "prompt_logprobs": None,
        }

        return response

    def _get_advertising_response(self) -> Dict[str, Any]:
        """Get response for advertising regulatory questions."""
        return {
            "content": """The correct answer is I) Unsafe practices, Distress, Fear, Serious.

Typical advertising regulatory bodies suggest that adverts must not:

* Encourage unsafe practices
* Cause unnecessary distress
* Cause fear
* Cause serious offence

This is a general guideline to ensure that advertisements are not misleading or harmful to consumers.

Answer: I""",
            "tokens": 1913,
            "prompt_tokens": 206,
            "total_tokens": 274,
        }

    def _get_downsizing_response(self) -> Dict[str, Any]:
        """Get response for downsizing issues questions."""
        return {
            "content": """D) Down, Privacy, Autonomy, Benefit

This question requires the test-taker to understand the concept of downsizing and its associated issues. Downsizing is a process of reducing the workforce, and it can have significant impacts on employees. The correct answer highlights two main issues associated with downsizing: privacy and autonomy. Privacy is a key issue because employees have a right to know if they are being made redundant due to the information policy of the corporation. Autonomy is a second issue, particularly the benefit package that employees receive when laid off. The other options are incorrect because they do not accurately reflect the issues associated with downsizing.

Answer: D""",
            "tokens": 131,
            "prompt_tokens": 242,
            "total_tokens": 373,
        }

    def _get_managers_response(self) -> Dict[str, Any]:
        """Get response for managers and shareholders questions."""
        return {
            "content": """The correct answer is F) Shareholders, Care and Skill, Diligence.

Managers have a fiduciary duty to act in the best interest of the company's shareholders. This includes a duty of care (to act with care and skill) and a duty of diligence (to act with diligence and prudence).

Answer: F""",
            "tokens": 1344,
            "prompt_tokens": 219,
            "total_tokens": 289,
        }

    def start(self):
        """Start the fake endpoint server."""
        import threading

        from werkzeug.serving import make_server

        self.server = make_server("localhost", self.port, self.app)
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.daemon = True
        self.thread.start()

        # Wait for server to start
        time.sleep(0.5)

        # Test health endpoint
        try:
            response = requests.get(f"http://localhost:{self.port}/health", timeout=5)
            if response.status_code == 200:
                logger.info(f"Fake endpoint started on http://localhost:{self.port}")
            else:
                raise RuntimeError(f"Health check failed: {response.status_code}")
        except Exception as e:
            raise RuntimeError(f"Failed to start fake endpoint: {e}")

    def stop(self):
        """Stop the fake endpoint server."""
        if self.server:
            self.server.shutdown()
            self.thread.join(timeout=5)
            logger.info("Fake endpoint stopped")

    def get_url(self) -> str:
        """Get the endpoint URL."""
        return f"http://localhost:{self.port}/v1/chat/completions"


@pytest.fixture(scope="session")
def fake_endpoint():
    """Fixture that provides a fake endpoint for e2e testing."""
    endpoint = FakeEndpoint(port=8000)
    endpoint.start()
    yield endpoint
    endpoint.stop()


@pytest.fixture
def fake_url(fake_endpoint):
    """Fixture that provides the URL for the fake endpoint."""
    return fake_endpoint.get_url()


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
