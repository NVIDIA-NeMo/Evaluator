#!/bin/bash
# Basic NeMo Evaluator Launcher quickstart example

# Prerequisites: Set your API key
export NGC_API_KEY="${NGC_API_KEY:-your-api-key-here}"

# Run evaluation against a hosted endpoint
# [snippet-start]
# Quick evaluation using direct CLI flags (no config file needed)
nemo-evaluator-launcher run \
    --model meta/llama-3.2-3b-instruct \
    --task gpqa_diamond \
    --output-dir ./results
# [snippet-end]

echo "Evaluation started. Use 'nemo-evaluator-launcher status <invocation_id>' to check progress."

