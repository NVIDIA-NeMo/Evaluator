#!/bin/bash
# Verify NeMo Evaluator Core installation

# [snippet-start]
# Verify installation
nemo-evaluator ls | head && echo '✓ CLI available' || exit 1
python3 -c "from nemo_evaluator.api import evaluate; print('✓ Python API available')" || exit 1
echo "✓ NeMo Evaluator Core installed successfully"
# [snippet-end]


