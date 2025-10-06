#!/bin/bash
# Task discovery commands for NeMo Evaluator

# [snippet-start]
# List all available benchmarks
nv-eval ls tasks

# Output as JSON for programmatic filtering
nv-eval ls tasks --json

# Filter for specific task types (example: academic benchmarks)
nv-eval ls tasks | grep -E "(mmlu|gsm8k|arc)"
# [snippet-end]

