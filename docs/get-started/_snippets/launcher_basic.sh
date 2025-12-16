#!/bin/bash
# Basic NeMo Evaluator Launcher quickstart example

# Prerequisites: Set your API key
export NGC_API_KEY="${NGC_API_KEY:-your-api-key-here}"

# Run evaluation against a hosted endpoint
# [snippet-start]
nemo-evaluator-launcher run \
    --config packages/nemo-evaluator-launcher/examples/local_basic.yaml \
    -o target.api_endpoint.url=https://integrate.api.nvidia.com/v1/chat/completions \
    -o target.api_endpoint.api_key_name=NGC_API_KEY \
    -o execution.output_dir=./results
# [snippet-end]

echo "Evaluation started. Use 'nemo-evaluator-launcher status <invocation_id>' to check progress."

