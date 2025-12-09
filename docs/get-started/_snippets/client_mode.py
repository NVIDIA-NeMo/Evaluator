# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Example: Using NeMo Evaluator Client Mode with adapters."""

import asyncio

from nemo_evaluator.adapters.adapter_config import AdapterConfig, InterceptorConfig
from nemo_evaluator.api.api_dataclasses import EndpointModelConfig
from nemo_evaluator.client import NeMoEvaluatorClient


async def main():
    # Configure model and adapters
    config = EndpointModelConfig(
        model_id="my-model",
        url="https://api.example.com/v1/chat/completions",
        api_key_name="API_KEY",
        is_base_url=False,  # URL is complete endpoint, not base
        adapter_config=AdapterConfig(
            mode="client",  # Use client mode (no server spawned)
            interceptors=[
                # Add system message to all requests
                InterceptorConfig(
                    name="system_message",
                    config={"system_message": "You are a helpful AI assistant."},
                ),
                # Log requests
                InterceptorConfig(name="request_logging"),
                # Cache responses for efficiency
                InterceptorConfig(
                    name="caching",
                    config={
                        "cache_dir": "./cache",
                        "reuse_cached_responses": True,
                        "save_requests": True,
                        "save_responses": True,
                    },
                ),
                # Extract reasoning tokens
                InterceptorConfig(name="reasoning"),
                # Make HTTP call (required)
                InterceptorConfig(name="endpoint"),
            ],
            post_eval_hooks=[
                {"name": "post_eval_report", "config": {"report_types": ["html"]}}
            ],
        ),
        temperature=0.7,
        max_new_tokens=100,
    )

    # Create client with adapters integrated (runs in-process, no proxy server)
    async with NeMoEvaluatorClient(
        endpoint_model_config=config, output_dir="./eval_output"
    ) as client:
        # Use exactly like OpenAI client - adapters run automatically
        response = await client.chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": "Explain quantum computing in one sentence.",
                }
            ],
        )

        print(response)

        # Second request with same params hits cache
        response2 = await client.chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": "Explain quantum computing in one sentence.",
                }
            ],
        )

        print(f"Cached: {response == response2}")

    # Post-eval hooks run automatically on context exit


if __name__ == "__main__":
    asyncio.run(main())
