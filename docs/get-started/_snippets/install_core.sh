#!/bin/bash
# Install NeMo Evaluator Core library with dependencies

# [snippet-start]
# Create and activate virtual environment
python3 -m venv nemo-eval-env
source nemo-eval-env/bin/activate

# Install core library with dependencies
pip install torch==2.7.0 setuptools pybind11 wheel_stub  # Required for TE
pip install --no-build-isolation nemo-evaluator

# Install evaluation frameworks
pip install nvidia-simple-evals nvidia-lm-eval
# [snippet-end]

# Verify installation
echo "Verifying installation..."
python3 -c "from nemo_evaluator.core.evaluate import evaluate; print('✓ Core library installed')" || exit 1
python3 -c "from nemo_evaluator.adapters.adapter_config import AdapterConfig; print('✓ Adapter system available')" || exit 1
echo "✓ NeMo Evaluator Core installed successfully"

