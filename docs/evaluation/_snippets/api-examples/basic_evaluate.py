#!/usr/bin/env python3
"""
Basic evaluation example: Evaluate a model on a single academic benchmark.
"""
import os

# [snippet-start]
from nemo_evaluator.core.evaluate import evaluate
from nemo_evaluator.api.api_dataclasses import (
    EvaluationConfig, EvaluationTarget, ApiEndpoint, ConfigParams, EndpointType
)

# Configure evaluation
eval_config = EvaluationConfig(
    type="mmlu_pro",
    output_dir="./results",
    params=ConfigParams(
        limit_samples=100,      # Remove for full dataset
        temperature=0.01,       # Near-deterministic for reproducibility
        max_new_tokens=512,
        top_p=0.95,
        parallelism=4
    )
)

# Configure target endpoint
target_config = EvaluationTarget(
    api_endpoint=ApiEndpoint(
        url="https://integrate.api.nvidia.com/v1/chat/completions",
        model_id="meta/llama-3.1-8b-instruct",
        type=EndpointType.CHAT,
        api_key="YOUR_API_KEY"  # Environment variable name
    )
)

# Run evaluation
result = evaluate(eval_cfg=eval_config, target_cfg=target_config)
print(f"Evaluation completed: {result}")
# [snippet-end]


if __name__ == "__main__":
    # Override with environment variables if provided
    api_key_name = os.getenv("API_KEY_NAME", "YOUR_API_KEY")
    
    if not os.getenv(api_key_name):
        print(f"Warning: Environment variable {api_key_name} not set")
        print("Set it before running: export YOUR_API_KEY='your-key-here'")
    
    # Run the evaluation (code above will execute)
    pass

