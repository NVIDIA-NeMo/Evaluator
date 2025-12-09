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

"""Shared adapter pipeline logic used by both server and client modes."""

from typing import List

from nemo_evaluator.adapters.adapter_config import AdapterConfig
from nemo_evaluator.adapters.registry import InterceptorRegistry
from nemo_evaluator.adapters.types import (
    AdapterGlobalContext,
    AdapterRequest,
    AdapterResponse,
    FatalErrorException,
    PostEvalHook,
    RequestInterceptor,
    RequestToResponseInterceptor,
    ResponseInterceptor,
)
from nemo_evaluator.logging import get_logger

logger = get_logger(__name__)


class AdapterPipeline:
    """Shared adapter pipeline that processes requests/responses through interceptors.

    This class encapsulates the core adapter logic that is used by both:
    - Server mode (AdapterServer with Flask)
    - Client mode (AdapterTransport with httpx)
    """

    def __init__(
        self,
        adapter_config: AdapterConfig,
        output_dir: str,
        model_name: str | None = None,
    ):
        """Initialize the adapter pipeline.

        Args:
            adapter_config: Adapter configuration with interceptors and hooks
            output_dir: Directory for output files
            model_name: Optional model name for logging context
        """
        self.adapter_config = adapter_config
        self.output_dir = output_dir
        self.model_name = model_name

        # Initialize interceptor chain and hooks
        self.interceptor_chain: List[
            RequestInterceptor | RequestToResponseInterceptor | ResponseInterceptor
        ] = []
        self.post_eval_hooks: List[PostEvalHook] = []
        self._post_eval_hooks_executed: bool = False

        # Initialize registry and discover components
        self.registry = InterceptorRegistry.get_instance()
        self.registry.discover_components(
            modules=adapter_config.discovery.modules,
            dirs=adapter_config.discovery.dirs,
        )

        # Validate and build chains
        self._validate_and_build_chains()

    def _validate_and_build_chains(self) -> None:
        """Validate configuration and build interceptor chains."""
        try:
            # Check if adapter chain is properly defined
            self._validate_adapter_chain_definition()

            # Validate interceptor order
            self._validate_interceptor_order()

            # Build the chains
            self._build_interceptor_chain()
            self._build_post_eval_hooks()

        except Exception as e:
            logger.error(f"Failed to build interceptor chains: {e}")
            raise

    def _validate_adapter_chain_definition(self) -> None:
        """Validate that the adapter chain is properly defined with at least one enabled component."""
        enabled_interceptors = [
            ic for ic in self.adapter_config.interceptors if ic.enabled
        ]
        enabled_post_eval_hooks = [
            hook for hook in self.adapter_config.post_eval_hooks if hook.enabled
        ]

        if not enabled_interceptors and not enabled_post_eval_hooks:
            warning_msg = (
                "Adapter pipeline cannot start: No enabled interceptors or "
                "post-eval hooks found. The pipeline requires at least one enabled "
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

    def _build_interceptor_chain(self) -> None:
        """Build interceptor chain from validated configuration."""
        self.interceptor_chain = []
        for interceptor_config in self.adapter_config.interceptors:
            if interceptor_config.enabled:
                interceptor = self.registry._get_or_create_instance(
                    interceptor_config.name,
                    interceptor_config.config,
                )
                self.interceptor_chain.append(interceptor)

        logger.info(
            "Built interceptor chain",
            interceptors=[type(i).__name__ for i in self.interceptor_chain],
        )

    def _build_post_eval_hooks(self) -> None:
        """Build post-evaluation hooks from validated configuration."""
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

        logger.info(
            "Built post-eval hooks",
            hooks=[type(h).__name__ for h in self.post_eval_hooks],
        )

    def process_request(
        self, adapter_request: AdapterRequest, global_context: AdapterGlobalContext
    ) -> tuple[AdapterRequest, AdapterResponse | None]:
        """Process request through the interceptor chain.

        Args:
            adapter_request: The request to process
            global_context: Global context for the request

        Returns:
            Tuple of (modified_request, optional_response)
            - If an interceptor returns a response, it's returned as the second element
            - Otherwise, the second element is None and the first is the modified request
        """
        current_request = adapter_request
        request_logger = get_logger()

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
                        return current_request, result
                    else:
                        current_request = result
                else:
                    # This is a ResponseInterceptor, skip in request phase
                    continue

            except FatalErrorException:
                # Re-raise fatal errors
                raise
            except Exception as e:
                request_logger.error(
                    f"Request interceptor {type(interceptor).__name__} failed: {e}"
                )
                # Continue with next interceptor
                continue

        return current_request, None

    def process_response(
        self, adapter_response: AdapterResponse, global_context: AdapterGlobalContext
    ) -> AdapterResponse:
        """Process response through the interceptor chain (in reverse order).

        Args:
            adapter_response: The response to process
            global_context: Global context for the response

        Returns:
            Modified response after processing through all response interceptors
        """
        current_response = adapter_response
        request_logger = get_logger()

        for interceptor in reversed(self.interceptor_chain):
            try:
                if isinstance(interceptor, ResponseInterceptor):
                    current_response = interceptor.intercept_response(
                        current_response, global_context
                    )
            except FatalErrorException:
                # Re-raise fatal errors
                raise
            except Exception as e:
                request_logger.error(
                    f"Response interceptor {type(interceptor).__name__} failed: {e}"
                )
                # Continue with next interceptor
                continue

        return current_response

    def run_post_eval_hooks(self, url: str = "") -> None:
        """Run all configured post-evaluation hooks.

        Args:
            url: Optional URL for global context (not always relevant)
        """
        if self._post_eval_hooks_executed:
            logger.warning("Post-eval hooks have already been executed, skipping")
            return

        global_context = AdapterGlobalContext(
            output_dir=self.output_dir,
            url=url,
            model_name=self.model_name,
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
