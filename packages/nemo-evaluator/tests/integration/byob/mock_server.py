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

"""Mock OpenAI-compatible model server for BYOB integration tests.

This module provides a deterministic mock server that responds to chat/completions
and completions endpoints with MD5-hash-based yes/no responses.
"""

import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer


class MockModelHandler(BaseHTTPRequestHandler):
    """Deterministic OpenAI-compatible model server for testing.

    Response logic: MD5(prompt) -> "Yes" if hash[:8] is even, else "No"
    This ensures the same prompt always produces the same response.
    """

    def do_POST(self):
        """Handle POST requests to /chat/completions or /completions."""
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        request = json.loads(body)

        # Extract prompt based on endpoint type
        if self.path == "/chat/completions":
            prompt = request["messages"][0]["content"]
        elif self.path == "/completions":
            prompt = request["prompt"]
        else:
            self.send_response(404)
            self.end_headers()
            return

        # Deterministic response based on MD5 of prompt
        md5_hash = hashlib.md5(prompt.encode()).hexdigest()
        response_text = "Yes" if int(md5_hash[:8], 16) % 2 == 0 else "No"

        # Build response structure based on endpoint type
        if self.path == "/chat/completions":
            response_body = {"choices": [{"message": {"content": response_text}}]}
        else:
            response_body = {"choices": [{"text": response_text}]}

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(response_body).encode())

    def log_message(self, format, *args):
        """Suppress HTTP server log output during tests."""
        pass


class MockServer:
    """Context manager for the mock model server.

    Uses port 0 (OS-assigned) to prevent port conflicts in CI.
    Runs as daemon thread with automatic cleanup.

    Example:
        with MockServer() as server:
            response = requests.post(f"{server.url}/chat/completions", ...)
    """

    def __init__(self):
        self.server = HTTPServer(("localhost", 0), MockModelHandler)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.server.shutdown()
        self.thread.join(timeout=5)

    @property
    def url(self):
        """Return base URL for the mock server."""
        return f"http://localhost:{self.port}"
