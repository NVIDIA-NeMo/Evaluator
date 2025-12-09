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

"""Main server which serves on a local port and holds chain of interceptors"""

import multiprocessing
import os
import signal
import socket
import sys
import time
from typing import List

import flask
import requests
import werkzeug.serving

from nemo_evaluator.adapters.adapter_config import AdapterConfig
from nemo_evaluator.adapters.interceptors.logging_interceptor import _get_safe_headers
from nemo_evaluator.adapters.pipeline import AdapterPipeline
from nemo_evaluator.adapters.types import (
    AdapterGlobalContext,
    AdapterRequest,
    AdapterRequestContext,
    FatalErrorException,
)
from nemo_evaluator.api.api_dataclasses import Evaluation
from nemo_evaluator.logging import get_logger

logger = get_logger(__name__)


def _setup_file_logging() -> None:
    """Set up centralized logging using NV_EVAL_LOG_DIR environment variable if set."""
    from nemo_evaluator.logging import configure_logging

    # configure_logging will automatically use NEMO_EVALUATOR_LOG_DIR if set
    configure_logging()

    logger.info(
        "File logging setup completed (uses NEMO_EVALUATOR_LOG_DIR environment variable if set)"
    )


def is_port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    """Check if the given port is open on the host.

    Args:
        host: The host to check
        port: The port to check
        timeout: Socket timeout in seconds

    Returns:
        bool: True if port is open, False otherwise
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    result = sock.connect_ex((host, port))
    sock.close()
    return result == 0


def wait_for_server(
    host: str, port: int, max_wait: float = 300, interval: float = 0.2
) -> bool:
    """Wait for server to be ready with timeout.

    Args:
        host: The host to check
        port: The port to check
        max_wait: Maximum time to wait in seconds (default: 10)
        interval: Time between checks in seconds (default: 0.2)

    Returns:
        bool: True if server is ready, False if timeout exceeded
    """
    start_time = time.time()

    while time.time() - start_time < max_wait:
        try:
            if is_port_open(host, port):
                return True
        except Exception:
            pass
        time.sleep(interval)

    return False


def _run_adapter_server(
    api_url: str,
    output_dir: str,
    adapter_config: AdapterConfig,
    port: int,
    model_name: str | None = None,
) -> None:
    """Internal function to run the adapter server."""
    # Set up centralized logging using NEMO_EVALUATOR_LOG_DIR environment variable if set
    _setup_file_logging()
    adapter = AdapterServer(
        api_url=api_url,
        output_dir=output_dir,
        adapter_config=adapter_config,
        port=port,
        model_name=model_name,
    )

    def signal_handler(signum, frame):
        """Handle termination signals by running post-eval hooks before exit."""
        if signum == signal.SIGINT:
            # Skip post-eval hooks for keyboard interrupt (Ctrl+C) for immediate termination
            logger.info(
                "Received SIGINT, shutting down immediately without post-eval hooks"
            )
            sys.exit(0)

        logger.info(
            f"Received signal {signum}, running post-eval hooks before shutdown"
        )
        try:
            adapter.run_post_eval_hooks()
            logger.info("Post-eval hooks completed successfully")
        except Exception as e:
            logger.error(f"Failed to run post-eval hooks during shutdown: {e}")
        finally:
            logger.info("Adapter server shutting down")
            sys.exit(0)

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    if os.environ.get("NEMO_EVALUATOR_LOG_DIR") is not None:
        logger.info("Starting adapter server with centralized logging enabled")
    else:
        logger.info("Starting adapter server with default logging")
    adapter.run()


class AdapterServer:
    """Adapter server with registry-based interceptor support"""

    DEFAULT_ADAPTER_HOST: str = "localhost"
    DEFAULT_ADAPTER_PORT: int = 3825

    def __init__(
        self,
        api_url: str,
        output_dir: str,
        adapter_config: AdapterConfig,
        port: int = DEFAULT_ADAPTER_PORT,
        model_name: str | None = None,
    ):
        """
        Initialize the adapter server.

        Args:
            api_url: The upstream API URL to forward requests to
            output_dir: Directory for output files
            adapter_config: Adapter configuration including interceptors and discovery
            model_name: Optional model name for logging context
        """
        self.app = flask.Flask(__name__)
        self.app.route("/", defaults={"path": ""}, methods=["POST"])(self._handler)
        self.app.route("/<path:path>", methods=["POST"])(self._handler)

        # Add route for running post-eval hooks
        self.app.route("/adapterserver/run-post-hook", methods=["POST"])(
            self._run_post_eval_hooks_handler
        )

        self.adapter_host: str = os.environ.get(
            "ADAPTER_HOST", self.DEFAULT_ADAPTER_HOST
        )
        self.adapter_port = port

        self.api_url = api_url
        self.output_dir = output_dir
        self.adapter_config = adapter_config
        self.model_name = model_name

        # Initialize the shared adapter pipeline
        self.pipeline = AdapterPipeline(adapter_config, output_dir, model_name)

        logger.info(
            "Using interceptors",
            interceptors=[ic.name for ic in adapter_config.interceptors if ic.enabled],
        )
        logger.info(
            "Using post-eval hooks",
            hooks=[
                hook.name for hook in adapter_config.post_eval_hooks if hook.enabled
            ],
        )

    @property
    def interceptor_chain(self):
        """Expose the interceptor chain from the pipeline for backward compatibility."""
        return self.pipeline.interceptor_chain

    @property
    def post_eval_hooks(self):
        """Expose the post-eval hooks from the pipeline for backward compatibility."""
        return self.pipeline.post_eval_hooks

    def run(self) -> None:
        """Start the Flask server."""
        # give way to the server

        werkzeug.serving.run_simple(
            hostname=self.adapter_host,
            port=self.adapter_port,
            application=self.app,
            threaded=True,
        )

    # The headers we don't want to let out
    _EXCLUDED_HEADERS = [
        "content-encoding",
        "content-length",
        "transfer-encoding",
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "upgrade",
    ]

    @classmethod
    def _process_response_headers(
        cls, response: requests.Response
    ) -> List[tuple[str, str]]:
        """Process response headers to remove sensitive ones."""
        headers = []
        for key, value in response.headers.items():
            if key.lower() not in cls._EXCLUDED_HEADERS:
                headers.append((key, value))
        return headers

    def _handler(self, path: str) -> flask.Response:
        """Main request handler that processes requests through the interceptor chain."""
        try:
            # Generate unique request ID for this request and bind it to logging context
            from nemo_evaluator.logging import (
                bind_model_name,
                bind_request_id,
                get_logger,
            )

            # Bind the request ID to the current context  so all loggers can access it
            request_id = bind_request_id()  # generates a new UUID

            # Bind the model name to the logging context if available
            if self.model_name:
                bind_model_name(self.model_name)

            # Get a logger for this request - context variables are automatically included
            request_logger = get_logger()

            # Log request start (request_id and model_name are automatically included from context)
            request_logger.info(
                "Request started",
                path=path,
                method=flask.request.method,
                url=self.api_url,
            )

            # Create global context
            global_context = AdapterGlobalContext(
                output_dir=self.output_dir,
                url=self.api_url,
                model_name=self.model_name,
            )

            # Create adapter request
            adapter_request = AdapterRequest(
                r=flask.request,
                rctx=AdapterRequestContext(request_id=request_id),
            )

            # Process through interceptor chain using shared pipeline
            current_request, adapter_response = self.pipeline.process_request(
                adapter_request, global_context
            )

            if adapter_response is None:
                raise RuntimeError("No adapter interceptor returned response")

            # Process through response interceptors using shared pipeline
            current_response = self.pipeline.process_response(
                adapter_response, global_context
            )

            # Log request completion (request_id is automatically included from context)
            request_logger.info(
                "Request completed",
                status_code=current_response.r.status_code,
                path=path,
            )

            # Return the final response
            headers = self._process_response_headers(current_response.r)
            return flask.Response(
                current_response.r.content,
                status=current_response.r.status_code,
                headers=headers,
            )

        except FatalErrorException as e:
            # Log failed request if enabled
            self._log_failed_request(
                500,
                f"Fatal error: {str(e)}",
                current_request if "current_request" in locals() else None,
            )

            # Send SIGTERM to parent process - the signal handler will run post-eval hooks
            logger.info("Sending SIGTERM to parent process")
            try:
                os.kill(os.getppid(), signal.SIGTERM)
            except (OSError, ProcessLookupError):
                # Fallback to SIGKILL if SIGTERM fails
                try:
                    os.kill(os.getppid(), signal.SIGKILL)
                except (OSError, ProcessLookupError):
                    sys.exit(1)

            # Return error response to Flask before exiting
            return flask.Response("Fatal error occurred", status=500)

        except Exception as e:
            # Log failed request if enabled
            self._log_failed_request(
                500,
                f"Internal server error: {str(e)}",
                current_request if "current_request" in locals() else None,
            )

            request_logger.error(f"Handler error: {e}")
            return flask.Response(
                f"Internal server error: {str(e)}", status=500, mimetype="text/plain"
            )

    def _log_failed_request(
        self, status_code: int, error_message: str, current_request=None
    ) -> None:
        """Log failed request if logging is enabled."""
        if (
            hasattr(self.adapter_config, "log_failed_requests")
            and self.adapter_config.log_failed_requests
        ):
            log_data = {
                "error": {
                    "request": {
                        "url": self.api_url,
                        "body": (
                            current_request.r.get_json() if current_request else None
                        ),
                        "headers": (
                            _get_safe_headers(current_request.r.headers)
                            if current_request
                            else {}
                        ),
                    },
                    "response": {
                        "status_code": status_code,
                        "headers": {},
                        "body": error_message,
                    },
                }
            }
            request_logger = get_logger()
            request_logger.error("failed_request_response_pair", data=log_data)

    def _run_post_eval_hooks_handler(self) -> flask.Response:
        """Handler for the post-eval hooks endpoint."""
        try:
            self.run_post_eval_hooks()
            return flask.jsonify(
                {
                    "status": "success",
                    "message": "Post-eval hooks executed successfully",
                }
            )
        except Exception as e:
            logger.error(f"Failed to run post-eval hooks: {e}")
            return flask.jsonify({"status": "error", "message": str(e)}), 500

    def run_post_eval_hooks(self) -> None:
        """Run all configured post-evaluation hooks."""
        self.pipeline.run_post_eval_hooks(url=self.api_url)

    def generate_report(self) -> None:
        """Generate HTML report of cached requests and responses."""
        # This method would need to be updated based on the new configuration structure
        # For now, we'll keep it as a placeholder
        pass


class AdapterServerProcess:
    def __init__(self, evaluation: Evaluation):
        self.evaluation = evaluation
        self.original_url = self.evaluation.target.api_endpoint.url
        self.server: None | AdapterServer = None
        self.process: None | multiprocessing.Process = None
        self.port = None

    def _find_and_reserve_free_port(
        self,
        start_port=AdapterServer.DEFAULT_ADAPTER_PORT,
        max_port=65535,
        adapter_host=AdapterServer.DEFAULT_ADAPTER_HOST,
    ) -> int:
        # If specific port has been requested, try only that one port
        adapter_server_port_env = int(os.environ.get("ADAPTER_PORT", 0))
        if adapter_server_port_env:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                s.bind((adapter_host, adapter_server_port_env))
                s.close()
                return adapter_server_port_env
            except OSError:
                s.close()
                raise OSError(
                    f"Adapter server was requested to start explicitly on {adapter_server_port_env} through 'ADAPTER_PORT' env-var, but the port seems to be taken. Exiting. "
                )
        for port in range(start_port, max_port + 1):
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                s.bind((adapter_host, port))
                # The port is now reserved by the OS while socket is open
                s.close()
                return port
            except OSError:
                s.close()
                continue
        raise OSError("No free port found in range")

    def __enter__(self):
        adapter_config = self.evaluation.target.api_endpoint.adapter_config
        if not adapter_config:
            return
        enabled_interceptors = [ic for ic in adapter_config.interceptors if ic.enabled]
        enabled_post_eval_hooks = [
            hook for hook in adapter_config.post_eval_hooks if hook.enabled
        ]
        if not enabled_interceptors and not enabled_post_eval_hooks:
            return

        # Get host from environment variable or use default
        adapter_host = os.environ.get(
            "ADAPTER_HOST", AdapterServer.DEFAULT_ADAPTER_HOST
        )

        output_dir = self.evaluation.config.output_dir
        model_name = (
            self.evaluation.target.api_endpoint.model_id
            if self.evaluation.target.api_endpoint
            else None
        )
        self.port = self._find_and_reserve_free_port(adapter_host=adapter_host)
        self.evaluation.target.api_endpoint.url = f"http://{adapter_host}:{self.port}"
        self.process = multiprocessing.get_context("spawn").Process(
            target=_run_adapter_server,
            daemon=True,
            args=(self.original_url, output_dir, adapter_config, self.port, model_name),
        )
        self.process.start()

        if wait_for_server(adapter_host, self.port):
            logger.info(f"Adapter server started on {adapter_host}:{self.port}")
            return self
        logger.error(f"Adapter server failed to start on {adapter_host}:{self.port}")
        self.process.terminate()
        self.process.join(timeout=5)
        raise RuntimeError(
            f"Adapter server failed to start on {adapter_host}:{self.port}"
        )

    def __exit__(self, type, value, traceback):
        if not self.process:
            return False
        self.evaluation.target.api_endpoint.url = self.original_url
        try:
            # Get host from environment variable or use default
            adapter_host = os.environ.get(
                "ADAPTER_HOST", AdapterServer.DEFAULT_ADAPTER_HOST
            )

            # Only run post-eval hooks if server is still responding (not shut down by signal handler)
            if is_port_open(adapter_host, self.port, timeout=1.0):
                post_hook_url = (
                    f"http://{adapter_host}:{self.port}/adapterserver/run-post-hook"
                )
                response = requests.post(post_hook_url, timeout=30)
                if response.status_code == 200:
                    logger.info("Successfully ran post-evaluation hooks")
                else:
                    logger.error(
                        f"Failed to run post-evaluation hooks: {response.status_code} - {response.text}"
                    )
            else:
                logger.info(
                    "Server not responding, post-eval hooks already run by signal handler"
                )
        except Exception as e:
            logger.error(f"Failed to run post-evaluation hooks: {e}")
        self.process.terminate()
        self.process.join()
