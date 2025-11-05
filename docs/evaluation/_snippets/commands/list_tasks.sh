#!/bin/bash
# Task discovery commands for NeMo Evaluator Launcher

# [snippet-start]
# List all available benchmarks
nemo-evaluator-launcher ls

# Output as JSON for programmatic filtering
nemo-evaluator-launcher ls --json

# Filter for specific task types (example: mmlu, gsm8k, and arc_challenge)
nemo-evaluator-launcher ls | grep -E "(mmlu|gsm8k|arc)"
# [snippet-end]

