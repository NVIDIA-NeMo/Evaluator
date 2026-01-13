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
"""
Basic NeMo Evaluator Core API quickstart example.
"""

# Prerequisites: Set your API key
# export NGC_API_KEY="nvapi-..."

import os

# [snippet-start]
from nemo_evaluator.api.api_dataclasses import (
    ApiEndpoint,
    ConfigParams,
    EndpointType,
    EvaluationConfig,
    EvaluationTarget,
)
from nemo_evaluator.core.evaluate import evaluate

# Configure evaluation
eval_config = EvaluationConfig(
    type="mmlu_pro",
    output_dir="./results",
    # Remove limit_samples for full dataset
    params=ConfigParams(limit_samples=10),
)

# Configure target endpoint
target_config = EvaluationTarget(
    api_endpoint=ApiEndpoint(
        url="https://integrate.api.nvidia.com/v1/chat/completions",
        model_id="meta/llama-3.2-3b-instruct",
        api_key="NGC_API_KEY",
        type=EndpointType.CHAT,
    )
)

# Run evaluation
result = evaluate(eval_cfg=eval_config, target_cfg=target_config)
print(f"Evaluation completed: {result}")
# [snippet-end]

if __name__ == "__main__":
    # Note: This requires a valid API key to actually run
    api_key = os.getenv("NGC_API_KEY")
    if not api_key:
        print("Set NGC_API_KEY environment variable to run this example")
        print("export NGC_API_KEY='your-key-here'")
