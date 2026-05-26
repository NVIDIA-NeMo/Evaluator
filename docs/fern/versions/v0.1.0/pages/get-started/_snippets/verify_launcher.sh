#!/bin/bash
# Verify NeMo Evaluator Launcher installation

# [snippet-start]
# Verify installation
nemo-evaluator-launcher --version

# Test basic functionality - list available tasks
nemo-evaluator-launcher ls tasks | head -10
# [snippet-end]

echo "✓ Launcher installed successfully"

