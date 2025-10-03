#!/usr/bin/env python3
"""
Minimal configuration for academic benchmark evaluation.
"""
from nemo_evaluator.api.api_dataclasses import ConfigParams

# [snippet-start]
# Minimal configuration for academic benchmark evaluation
params = ConfigParams(
    temperature=0.01,       # Near-deterministic (0.0 not supported by all endpoints)
    top_p=1.0,             # No nucleus sampling
    max_new_tokens=256,    # Sufficient for most academic tasks
    limit_samples=100,     # Remove for full dataset
    parallelism=4          # Adjust based on endpoint capacity
)
# [snippet-end]

