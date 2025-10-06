#!/bin/bash
# Install NeMo Evaluator Launcher with all exporters

# [snippet-start]
# Create and activate virtual environment
python3 -m venv nemo-eval-env
source nemo-eval-env/bin/activate

# Install launcher with all exporters (recommended)
pip install nemo-evaluator-launcher[all]
# [snippet-end]

# Verify installation
if command -v nv-eval &> /dev/null; then
    echo "✓ NeMo Evaluator Launcher installed successfully"
    nv-eval --version
else
    echo "✗ Installation failed"
    exit 1
fi

