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
#!/usr/bin/env python3
"""Example: Using NeMo Evaluator Client Mode with adapters."""

from nemo_evaluator.adapters.adapter_config import AdapterConfig, InterceptorConfig
from nemo_evaluator.client import NeMoEvaluatorClient

# Configure adapters - same config works for both server and client mode
adapter_config = AdapterConfig(
    interceptors=[
        # Add system message to all requests
        InterceptorConfig(
            name="system_message",
            enabled=True,
            config={"system_message": "You are a helpful AI assistant."},
        ),
        # Log requests and responses
        InterceptorConfig(name="request_logging", enabled=True),
        # Cache responses for efficiency
        InterceptorConfig(
            name="caching",
            enabled=True,
            config={"cache_dir": "./cache", "reuse_cached_responses": True},
        ),
        # Extract reasoning tokens
        InterceptorConfig(name="reasoning", enabled=True),
        # Make the actual HTTP call (required)
        InterceptorConfig(name="endpoint", enabled=True),
    ]
)

# Create client with adapters integrated (runs in-process, no proxy server)
with NeMoEvaluatorClient(
    endpoint_url="http://localhost:8000/v1",
    api_key="your-api-key",
    adapter_config=adapter_config,
    output_dir="./eval_output",
) as client:
    # Use exactly like OpenAI client - adapters run automatically
    response = client.chat.completions.create(
        model="my-model",
        messages=[
            {"role": "user", "content": "Explain quantum computing in one sentence."}
        ],
        temperature=0.7,
        max_tokens=100,
    )

    print(response.choices[0].message.content)

    # Second request with same params hits cache
    response2 = client.chat.completions.create(
        model="my-model",
        messages=[
            {"role": "user", "content": "Explain quantum computing in one sentence."}
        ],
        temperature=0.7,
        max_tokens=100,
    )

# Post-eval hooks run automatically on context exit
