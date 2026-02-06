#!/bin/bash
# Complete working example using NVIDIA Build
# Assuming "Prerequisites" are fulfilled

# Set up your API key
export NGC_API_KEY="${NGC_API_KEY:-nvapi-your-key-here}"

# [snippet-start]
# Quick evaluation using direct CLI flags (no config file needed)
nemo-evaluator-launcher run \
    --model meta/llama-3.2-3b-instruct \
    --task gpqa_diamond \
    --output-dir ./results

# Or use a config file for more complex setups
# nemo-evaluator-launcher run \
#     --config packages/nemo-evaluator-launcher/examples/local_basic.yaml \
#     --output-dir ./results
# [snippet-end]

# Note: Replace <invocation_id> with actual ID from output
echo ""
echo "Evaluation started! Next steps:"
echo "1. Monitor progress: nemo-evaluator-launcher status <invocation_id>"
echo "2. View results: ls -la ./results/<invocation_id>/"

