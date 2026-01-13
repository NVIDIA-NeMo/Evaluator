#!/bin/bash
"""
Basic NeMo Evaluator Core CLI quickstart example.
"""

# [snippet-start]

nemo-evaluator run_eval \
    --eval_type mmlu_pro \
    --model_id meta/llama-3.2-3b-instruct \
    --model_url https://integrate.api.nvidia.com/v1/chat/completions \
    --model_type chat \
    --api_key_name NGC_API_KEY \  # pass variable name not the key itself
    --output_dir ./results \
    --overrides 'config.params.limit_samples=3,config.params.temperature=0.0,config.params.max_new_tokens=1024,config.params.parallelism=1,config.params.max_retries=5'

# [snippet-end]

