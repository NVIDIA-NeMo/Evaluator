#!/bin/bash
# Install NeMo Evaluator Core library with dependencies

# [snippet-start]
# Create and activate virtual environment
python3 -m venv nemo-eval-env
source nemo-eval-env/bin/activate

# Install core library with dependencies
pip install nemo-evaluator

# Install evaluation frameworks
pip install nvidia-simple-evals nvidia-lm-eval
# [snippet-end]

# Verify installation
nemo-evaluator ls | head && echo '✓ CLI available' || exit 1
python3 -c "from nemo_evaluator.api import evaluate; print('✓ Python API available')" || exit 1
echo "✓ NeMo Evaluator Core installed successfully"

