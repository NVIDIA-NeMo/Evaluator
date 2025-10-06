#!/usr/bin/env python3
"""
Basic NeMo Evaluator Core API quickstart example.
"""
import os

# [snippet-start]
from nemo_evaluator.core.evaluate import evaluate
from nemo_evaluator.api.api_dataclasses import (
    EvaluationConfig, 
    EvaluationTarget, 
    ApiEndpoint, 
    EndpointType,
    ConfigParams
)

# Configure evaluation
eval_config = EvaluationConfig(
    type="mmlu_pro",
    output_dir="./results",
    params=ConfigParams(
        limit_samples=10,
        temperature=0.0,
        max_new_tokens=1024,
        parallelism=1
    )
)

# Configure target endpoint
target_config = EvaluationTarget(
    api_endpoint=ApiEndpoint(
        url="https://integrate.api.nvidia.com/v1/chat/completions",
        model_id="meta/llama-3.1-8b-instruct",
        api_key="your_api_key_here",
        type=EndpointType.CHAT
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

