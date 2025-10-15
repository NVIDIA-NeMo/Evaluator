#!/usr/bin/env python3
"""
Complete working example with proper error handling.
"""

# [snippet-start]
# Prerequisites: Set your API key
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
        model_id="meta/llama-3.1-8b-instruct",
        url="https://integrate.api.nvidia.com/v1/chat/completions",
        type=EndpointType.CHAT,
        api_key="NGC_API_KEY",
    )
)

# Run evaluation
try:
    result = evaluate(eval_cfg=eval_config, target_cfg=target_config)
    print(f"Evaluation completed. Results saved to: {eval_config.output_dir}")
except Exception as e:
    print(f"Evaluation failed: {e}")
# [snippet-end]

if __name__ == "__main__":
    print(
        "Replace 'nvapi-your-key-here' with your actual NGC API key to run this example"
    )
