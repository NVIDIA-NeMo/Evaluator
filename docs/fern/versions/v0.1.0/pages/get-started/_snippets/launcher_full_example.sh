#!/bin/bash
# Complete working example using NVIDIA Build
# Assuming "Prerequisites" are fulfilled

# Set up your API key
export NGC_API_KEY="${NGC_API_KEY:-nvapi-your-key-here}"

# [snippet-start]
# Run a quick test evaluation with limited samples
nemo-evaluator-launcher run \
    --config packages/nemo-evaluator-launcher/examples/local_basic.yaml \
    -o target.api_endpoint.url=https://integrate.api.nvidia.com/v1/chat/completions \
    -o target.api_endpoint.model_id=meta/llama-3.2-3b-instruct \
    -o target.api_endpoint.api_key_name=NGC_API_KEY \
    -o execution.output_dir=./results
# [snippet-end]

# Note: Replace <invocation_id> with actual ID from output
echo ""
echo "Evaluation started! Next steps:"
echo "1. Monitor progress: nemo-evaluator-launcher status <invocation_id>"
echo "2. View results: ls -la ./results/<invocation_id>/"

