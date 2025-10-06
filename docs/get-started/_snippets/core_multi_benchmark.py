#!/usr/bin/env python3
"""
Multi-benchmark evaluation example.
"""

# [snippet-start]
from nemo_evaluator.core.evaluate import evaluate
from nemo_evaluator.api.api_dataclasses import (
    EvaluationConfig, EvaluationTarget, ApiEndpoint, EndpointType, ConfigParams
)

# Configure target once
target_config = EvaluationTarget(
    api_endpoint=ApiEndpoint(
        url="https://integrate.api.nvidia.com/v1/chat/completions",
        model_id="meta/llama-3.1-8b-instruct",
        api_key="your_api_key_here",
        type=EndpointType.CHAT
    )
)

# Run multiple benchmarks
benchmarks = ["gsm8k", "hellaswag", "arc_easy"]
results = {}

for benchmark in benchmarks:
    config = EvaluationConfig(
        type=benchmark,
        output_dir=f"./results/{benchmark}",
        params=ConfigParams(limit_samples=10)
    )
    
    result = evaluate(eval_cfg=config, target_cfg=target_config)
    results[benchmark] = result
# [snippet-end]

if __name__ == "__main__":
    print("Multi-benchmark evaluation example")
    print("Replace 'your_api_key_here' with your actual API key to run")

