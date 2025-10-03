(gs-quickstart-container)=
# Container Direct

**Best for**: Users who prefer container-based workflows

The Container Direct approach gives you full control over the container environment with volume mounting, environment variable management, and integration into Docker-based CI/CD pipelines.

## Prerequisites

- Docker with GPU support
- OpenAI-compatible endpoint

## Quick Start

```bash
# 1. Pull evaluation container
docker pull nvcr.io/nvidia/eval-factory/simple-evals:{{ docker_compose_latest }}

# 2. Run container interactively
docker run --rm -it --gpus all nvcr.io/nvidia/eval-factory/simple-evals:{{ docker_compose_latest }} bash

# 3. Inside container - set up environment
export MY_API_KEY=nvapi-your-key-here
export HF_TOKEN=hf_your-token-here  # If using Hugging Face models

# 4. Run evaluation
eval-factory run_eval \
    --eval_type mmlu_pro \
    --model_id meta/llama-3.1-8b-instruct \
    --model_url https://integrate.api.nvidia.com/v1/chat/completions \
    --model_type chat \
    --api_key_name MY_API_KEY \
    --output_dir /tmp/results \
    --overrides 'config.params.limit_samples=10'
```

## Complete Container Workflow

Here's a complete example with volume mounting and advanced configuration:

```bash
# 1. Create local directories for persistent storage
mkdir -p ./results ./cache ./logs

# 2. Run container with volume mounts
docker run --rm -it --gpus all \
    -v $(pwd)/results:/workspace/results \
    -v $(pwd)/cache:/workspace/cache \
    -v $(pwd)/logs:/workspace/logs \
    -e MY_API_KEY=nvapi-your-key-here \
    -e HF_TOKEN=hf_your-token-here \
    nvcr.io/nvidia/eval-factory/simple-evals:{{ docker_compose_latest }} bash

# 3. Inside container - run evaluation
eval-factory run_eval \
    --eval_type mmlu_pro \
    --model_id meta/llama-3.1-8b-instruct \
    --model_url https://integrate.api.nvidia.com/v1/chat/completions \
    --model_type chat \
    --api_key_name MY_API_KEY \
    --output_dir /workspace/results \
    --overrides 'config.params.limit_samples=3'

# 4. Exit container and check results
exit
ls -la ./results/
```

## One-Liner Container Execution

For automated workflows, you can run everything in a single command:

```bash
# Run evaluation directly in container
docker run --rm --gpus all \
    -v $(pwd)/results:/workspace/results \
    -e MY_API_KEY="${MY_API_KEY}" \
    nvcr.io/nvidia/eval-factory/simple-evals:{{ docker_compose_latest }} \
    eval-factory run_eval \
        --eval_type mmlu_pro \
        --model_url https://integrate.api.nvidia.com/v1/chat/completions \
        --model_id meta/llama-3.1-8b-instruct \
        --api_key_name MY_API_KEY \
        --output_dir /workspace/results
```

## Key Features

### Full Container Control

- Direct access to container environment
- Custom volume mounting strategies
- Environment variable management
- GPU resource allocation

### CI/CD Integration

- Single-command execution for automation
- Docker Compose compatibility
- Kubernetes deployment ready
- Pipeline integration capabilities

### Persistent Storage

- Volume mounting for results persistence
- Cache directory management
- Log file preservation
- Custom configuration mounting

### Environment Isolation

- Clean, reproducible environments
- Dependency management handled
- Version pinning through container tags
- No local Python environment conflicts

## Advanced Container Patterns

### Docker Compose Integration

```yaml
# docker-compose.yml
version: '3.8'
services:
  nemo-eval:
    image: nvcr.io/nvidia/eval-factory/simple-evals:{{ docker_compose_latest }}
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    volumes:
      - ./results:/workspace/results
      - ./cache:/workspace/cache
      - ./configs:/workspace/configs
    environment:
      - MY_API_KEY=${NGC_API_KEY}
    command: |
      eval-factory run_eval 
        --eval_type mmlu_pro 
        --model_id meta/llama-3.1-8b-instruct 
        --model_url https://integrate.api.nvidia.com/v1/chat/completions 
        --model_type chat 
        --api_key_name MY_API_KEY 
        --output_dir /workspace/results
```

### Batch Processing Script

```bash
#!/bin/bash
# batch_eval.sh

BENCHMARKS=("mmlu_pro" "gpqa_diamond" "humaneval")
API_KEY=${NGC_API_KEY}

for benchmark in "${BENCHMARKS[@]}"; do
    echo "Running evaluation for $benchmark..."
    
    docker run --rm --gpus all \
        -v $(pwd)/results:/workspace/results \
        -e MY_API_KEY=$API_KEY \
        -e HF_TOKEN=$HF_TOKEN \
        nvcr.io/nvidia/eval-factory/simple-evals:{{ docker_compose_latest }} \
        eval-factory run_eval \
            --eval_type $benchmark \
            --model_id meta/llama-3.1-8b-instruct \
            --model_url https://integrate.api.nvidia.com/v1/chat/completions \
            --model_type chat \
            --api_key_name MY_API_KEY \
            --output_dir /workspace/results/$benchmark \
            --overrides 'config.params.limit_samples=10'
            
    echo "Completed $benchmark evaluation"
done

echo "All evaluations completed. Results in ./results/"
```

## Next Steps

- Integrate into your CI/CD pipelines
- Explore Docker Compose for multi-service setups
- Consider Kubernetes deployment for scale
- Try {ref}`gs-quickstart-launcher` for simplified workflows
- See {ref}`gs-quickstart-core` for programmatic API and advanced adapter features
