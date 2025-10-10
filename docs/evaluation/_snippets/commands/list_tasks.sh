#!/bin/bash
# Task discovery commands for NeMo Evaluator

# [snippet-start]
# List all available benchmarks
nemo-evaluator-launcher ls tasks

# Output as JSON for programmatic filtering
nemo-evaluator-launcher ls tasks --json

# Filter for specific task types (example: academic benchmarks)
nemo-evaluator-launcher ls tasks | grep -E "(mmlu|gsm8k|arc)"
# [snippet-end]

