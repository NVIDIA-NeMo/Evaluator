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
Complete working example with proper error handling.
"""

# [snippet-start]
# Prerequisites: Set your API key before running the snippet
# export NGC_API_KEY="nvapi-..."

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
    params=ConfigParams(
        limit_samples=3,
        temperature=0.0,
        max_new_tokens=1024,
        parallelism=1,
        max_retries=5,
    ),
)

# Configure target
target_config = EvaluationTarget(
    api_endpoint=ApiEndpoint(
        model_id="meta/llama-3.2-3b-instruct",
        url="https://integrate.api.nvidia.com/v1/chat/completions",
        type=EndpointType.CHAT,
        api_key="NGC_API_KEY",
    )
)

if __name__ == "__main__":
    # Run evaluation
    result = evaluate(eval_cfg=eval_config, target_cfg=target_config)
    print(f"Evaluation completed. Results saved to: {eval_config.output_dir}")

# [snippet-end]
