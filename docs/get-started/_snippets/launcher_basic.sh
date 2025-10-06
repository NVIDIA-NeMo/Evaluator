#!/bin/bash
# Basic NeMo Evaluator Launcher quickstart example

# Prerequisites: Set your API key
export NGC_API_KEY="${NGC_API_KEY:-your-api-key-here}"

# [snippet-start]
# Run evaluation against a hosted endpoint
nv-eval run \
    --config-dir examples \
    --config-name local_llama_3_1_8b_instruct \
    -o target.api_endpoint.url=https://integrate.api.nvidia.com/v1/chat/completions \
    -o target.api_endpoint.api_key_name=NGC_API_KEY \
    -o execution.output_dir=./results
# [snippet-end]

echo "Evaluation started. Use 'nv-eval status <invocation_id>' to check progress."

