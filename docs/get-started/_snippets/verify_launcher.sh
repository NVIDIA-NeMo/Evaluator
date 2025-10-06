#!/bin/bash
# Verify NeMo Evaluator Launcher installation

# [snippet-start]
# Verify installation
nv-eval --version

# Test basic functionality - list available tasks
nv-eval ls tasks | head -10
# [snippet-end]

echo "✓ Launcher installed successfully"

