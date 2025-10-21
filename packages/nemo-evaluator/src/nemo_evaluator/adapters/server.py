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
from nemo_evaluator.adapters.registry import InterceptorRegistry
from nemo_evaluator.adapters.types import (
    AdapterGlobalContext,
    AdapterRequest,
    AdapterRequestContext,
    AdapterResponse,
    FatalErrorException,
    PostEvalHook,
    RequestInterceptor,
    RequestToResponseInterceptor,
    ResponseInterceptor,
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
    api_url: str, output_dir: str, adapter_config: AdapterConfig, port: int
) -> None:
    """Internal function to run the adapter server."""
    # Set up centralized logging using NEMO_EVALUATOR_LOG_DIR environment variable if set
    _setup_file_logging()
    adapter = AdapterServer(
        api_url=api_url, output_dir=output_dir, adapter_config=adapter_config, port=port
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
    ):
        """
        Initialize the adapter server.

        Args:
            api_url: The upstream API URL to forward requests to
            output_dir: Directory for output files
            adapter_config: Adapter configuration including interceptors and discovery
        """
        self.interceptor_chain: List[
            RequestInterceptor | RequestToResponseInterceptor | ResponseInterceptor
        ] = []
        self.post_eval_hooks: List[PostEvalHook] = []
        self._post_eval_hooks_executed: bool = False

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

        # Initialize registry and discover components
        self.registry = InterceptorRegistry.get_instance()
        self.registry.discover_components(
            modules=adapter_config.discovery.modules, dirs=adapter_config.discovery.dirs
        )

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

        # Validate and build chains
        self._validate_and_build_chains()

    def _validate_and_build_chains(self) -> None:
        """Validate configuration and build interceptor chains"""
        try:
            # Check if adapter chain is properly defined
            self._validate_adapter_chain_definition()

            # Validate interceptor order
            self._validate_interceptor_order()

            # Build the chains
            self._build_interceptor_chains()
            self._build_post_eval_hooks()

        except Exception as e:
            logger.error(f"Failed to build interceptor chains: {e}")
            raise

    def _validate_adapter_chain_definition(self) -> None:
        """Validate that the adapter chain is properly defined with at least one enabled interceptor or post-eval hook."""
        enabled_interceptors = [
            ic for ic in self.adapter_config.interceptors if ic.enabled
        ]
        enabled_post_eval_hooks = [
            hook for hook in self.adapter_config.post_eval_hooks if hook.enabled
        ]

        if not enabled_interceptors and not enabled_post_eval_hooks:
            warning_msg = (
                "Adapter server cannot start: No enabled interceptors or "
                "post-eval hooks found. The server requires at least one enabled "
                "interceptor or post-eval hook to function properly. "
                f"Configured interceptors: "
                f"{[ic.name for ic in self.adapter_config.interceptors]}, "
                f"Configured post-eval hooks: "
                f"{[hook.name for hook in self.adapter_config.post_eval_hooks]}"
            )
            logger.warning(warning_msg)
            raise RuntimeError(warning_msg)

    def _validate_interceptor_order(self) -> None:
        """Validate that the configured interceptor list follows the correct stage order.

        The order must be: Request -> RequestToResponse -> Response
        """
        # Define stage hierarchy and allowed transitions
        STAGE_ORDER = ["request", "request_to_response", "response"]
        current_stage_idx = 0

        for interceptor_config in self.adapter_config.interceptors:
            if not interceptor_config.enabled:
                continue

            metadata = self.registry.get_metadata(interceptor_config.name)
            if metadata is None:
                raise ValueError(f"Unknown interceptor: {interceptor_config.name}")

            # Determine the stage of this interceptor
            if metadata.supports_request_to_response_interception():
                interceptor_stage = "request_to_response"
            elif metadata.supports_request_interception():
                interceptor_stage = "request"
            elif metadata.supports_response_interception():
                interceptor_stage = "response"
            else:
                raise ValueError(
                    f"Interceptor {interceptor_config.name} doesn't implement any known interface"
                )

            # Find the stage index
            try:
                stage_idx = STAGE_ORDER.index(interceptor_stage)
            except ValueError:
                raise ValueError(f"Unknown stage: {interceptor_stage}")

            # Validate progression: can only move forward or stay at same stage
            if stage_idx < current_stage_idx:
                raise ValueError(
                    f"Invalid stage order: interceptor {interceptor_config.name} (stage: {interceptor_stage}) "
                    f"appears after {STAGE_ORDER[current_stage_idx]} stage. "
                    f"Expected order: Request -> RequestToResponse -> Response"
                )

            # Update current stage if we've moved forward
            current_stage_idx = max(current_stage_idx, stage_idx)

    def _build_interceptor_chains(self) -> None:
        """Build interceptor chains from validated configuration"""
        # Build the chain in the configured order
        self.interceptor_chain = []
        for interceptor_config in self.adapter_config.interceptors:
            if interceptor_config.enabled:
                interceptor = self.registry._get_or_create_instance(
                    interceptor_config.name,
                    interceptor_config.config,
                )

                self.interceptor_chain.append(interceptor)

        # Log the chain for debugging
        logger.info(
            "Built interceptor chain",
            interceptors=[type(i).__name__ for i in self.interceptor_chain],
        )

    def _build_post_eval_hooks(self) -> None:
        """Build post-evaluation hooks from validated configuration"""
        # Build the hooks in the configured order
        self.post_eval_hooks = []

        # Add configured post-eval hooks
        for hook_config in self.adapter_config.post_eval_hooks:
            if hook_config.enabled:
                hook = self.registry._get_or_create_instance(
                    hook_config.name, hook_config.config
                )
                self.post_eval_hooks.append(hook)

        # Also add interceptors that implement PostEvalHook
        for interceptor in self.interceptor_chain:
            if hasattr(interceptor, "post_eval_hook") and callable(
                getattr(interceptor, "post_eval_hook")
            ):
                self.post_eval_hooks.append(interceptor)

        # Log the hooks for debugging
        logger.info(
            "Built post-eval hooks",
            hooks=[type(h).__name__ for h in self.post_eval_hooks],
        )

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
            from nemo_evaluator.logging import bind_request_id, get_logger

            # Bind the request ID to the current context  so all loggers can access it
            request_id = bind_request_id()  # generates a new UUID

            # Get a logger for this request - context variables are automatically included
            request_logger = get_logger()

            # Log request start (request_id is automatically included from context)
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
            )

            # Create adapter request
            adapter_request = AdapterRequest(
                r=flask.request,
                rctx=AdapterRequestContext(request_id=request_id),
            )

            # Process through interceptor chain
            current_request = adapter_request
            adapter_response = None

            for interceptor in self.interceptor_chain:
                try:
                    if isinstance(
                        interceptor, (RequestInterceptor, RequestToResponseInterceptor)
                    ):
                        result = interceptor.intercept_request(
                            current_request, global_context
                        )

                        # If interceptor returns a response, we're done with request processing
                        if isinstance(result, AdapterResponse):
                            adapter_response = result
                            break
                        else:
                            current_request = result
                    else:
                        # This is a ResponseInterceptor, but we're still in request phase
                        # Skip it for now, it will be processed in response phase
                        continue

                except Exception as e:
                    request_logger.error(
                        f"Request interceptor {type(interceptor).__name__} failed: {e}"
                    )
                    # Continue with next interceptor
                    continue

            if adapter_response is None:
                raise RuntimeError("No adapter interceptor returned response")

            # Process through response interceptors (in reverse order for response phase)
            current_response = adapter_response
            for interceptor in reversed(self.interceptor_chain):
                try:
                    if isinstance(interceptor, ResponseInterceptor):
                        current_response = interceptor.intercept_response(
                            current_response, global_context
                        )
                except FatalErrorException:
                    # Re-raise FatalErrorException to be caught by the main handler
                    raise
                except Exception as e:
                    request_logger.error(
                        f"Response interceptor {type(interceptor).__name__} failed: {e}"
                    )
                    # Continue with next interceptor
                    continue

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
        if self._post_eval_hooks_executed:
            logger.warning("Post-eval hooks have already been executed, skipping")
            return

        global_context = AdapterGlobalContext(
            output_dir=self.output_dir,
            url=self.api_url,
        )

        for hook in self.post_eval_hooks:
            try:
                hook.post_eval_hook(global_context)
                logger.info(f"Successfully ran post-eval hook: {type(hook).__name__}")
            except Exception as e:
                logger.error(f"Post-eval hook {type(hook).__name__} failed: {e}")
                # Continue with other hooks
                continue

        self._post_eval_hooks_executed = True
        logger.info("Post-eval hooks execution completed")

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
        self.port = self._find_and_reserve_free_port(adapter_host=adapter_host)
        self.evaluation.target.api_endpoint.url = f"http://{adapter_host}:{self.port}"
        self.process = multiprocessing.get_context("spawn").Process(
            target=_run_adapter_server,
            daemon=True,
            args=(self.original_url, output_dir, adapter_config, self.port),
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
