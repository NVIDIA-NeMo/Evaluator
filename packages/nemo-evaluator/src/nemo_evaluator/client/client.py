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

"""NeMo Evaluator client with integrated adapter support for client-mode evaluation."""

from typing import Any

import httpx
from openai import OpenAI

from nemo_evaluator.adapters.adapter_config import AdapterConfig
from nemo_evaluator.client.adapter_transport import create_adapter_http_client
from nemo_evaluator.logging import get_logger

logger = get_logger(__name__)


class NeMoEvaluatorClient(OpenAI):
    """OpenAI-compatible client with integrated adapter pipeline support.

    This client enables "client mode" for adapters, where the adapter interceptor
    pipeline runs in-process through the HTTP client, instead of requiring a
    separate proxy server.

    Args:
        adapter_config: Configuration for adapter interceptors and hooks
        endpoint_url: The endpoint URL - can be base URL (e.g., http://host/v1)
                     or full endpoint (e.g., http://custom:2137/submit)
        output_dir: Directory for adapter output files (logs, cache, etc.)
        http_client: Optional custom httpx client. If provided, it will be wrapped
                    with adapter transport. If not provided, a new client with
                    adapter transport is created.
        **kwargs: Additional arguments passed to OpenAI client (api_key, etc.)
    """

    def __init__(
        self,
        *,
        adapter_config: AdapterConfig,
        endpoint_url: str,
        output_dir: str = "./nemo_eval_output",
        http_client: httpx.Client | None = None,
        **kwargs: Any,
    ):
        """Initialize NeMoEvaluatorClient with adapter support.

        Args:
            adapter_config: Adapter configuration with interceptors
            endpoint_url: The endpoint URL (can be base URL or full endpoint)
            output_dir: Directory for output files
            http_client: Optional httpx client to wrap. If None, creates a new one.
            **kwargs: Additional OpenAI client arguments (api_key, etc.)
        """
        self.adapter_config = adapter_config
        self.endpoint_url = endpoint_url
        self.output_dir = output_dir

        # Extract base transport if http_client is provided
        if http_client is not None:
            base_transport = http_client._transport
            http_client.close()  # Close the provided client since we'll create a new one
        else:
            base_transport = None

        # Create adapter HTTP client using shared factory
        adapter_http_client, self.adapter_transport = create_adapter_http_client(
            adapter_config=adapter_config,
            output_dir=output_dir,
            endpoint_url=endpoint_url,
            base_transport=base_transport,
        )

        # Initialize OpenAI client with our custom http_client
        # Pass endpoint_url as base_url to OpenAI
        super().__init__(
            http_client=adapter_http_client, base_url=endpoint_url, **kwargs
        )

        logger.info("NeMoEvaluatorClient initialized")

    def close(self) -> None:
        """Close the client and run post-evaluation hooks.

        This method should be called when you're done with evaluations to ensure
        post-eval hooks are executed and resources are cleaned up properly.
        """
        logger.info("Closing NeMoEvaluatorClient and running post-eval hooks")
        try:
            self.adapter_transport.run_post_eval_hooks()
        finally:
            super().close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures post-eval hooks run."""
        self.close()
        return False
