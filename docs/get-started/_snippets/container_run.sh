#!/bin/bash
# Run evaluation using NGC containers directly

# Set container version (or use environment variable)
DOCKER_TAG="${DOCKER_TAG:-25.09.1}"
export MY_API_KEY="${MY_API_KEY:-your-api-key}"

# [snippet-start]
# Run evaluation directly in container
docker run --rm --gpus all \
    -v $(pwd)/results:/workspace/results \
    -e MY_API_KEY="${MY_API_KEY}" \
    nvcr.io/nvidia/eval-factory/simple-evals:${DOCKER_TAG} \
    nemo-evaluator run_eval \
        --eval_type mmlu_pro \
        --model_url https://integrate.api.nvidia.com/v1/chat/completions \
        --model_id meta/llama-3.1-8b-instruct \
        --api_key_name MY_API_KEY \
        --output_dir /workspace/results
# [snippet-end]

echo "✓ Evaluation complete. Check ./results/ for output."

