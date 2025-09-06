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
docker pull nvcr.io/nvidia/eval-factory/simple-evals:25.07.3

# 2. Run container interactively
docker run --rm -it --gpus all nvcr.io/nvidia/eval-factory/simple-evals:25.07.3 bash

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
    nvcr.io/nvidia/eval-factory/simple-evals:25.07.3 bash

# 3. Inside container - run evaluation with advanced features
eval-factory run_eval \
    --eval_type mmlu_pro \
    --model_id meta/llama-3.1-8b-instruct \
    --model_url https://integrate.api.nvidia.com/v1/chat/completions \
    --model_type chat \
    --api_key_name MY_API_KEY \
    --output_dir /workspace/results \
    --overrides 'config.params.limit_samples=3,target.api_endpoint.adapter_config.use_request_logging=True,target.api_endpoint.adapter_config.use_caching=True,target.api_endpoint.adapter_config.caching_dir=/workspace/cache'

# 4. Exit container and check results
exit
ls -la ./results/
```

## One-Liner Container Execution

For automated workflows, you can run everything in a single command:

```bash
docker run --rm --gpus all \
    -v $(pwd)/results:/workspace/results \
    -e MY_API_KEY=nvapi-your-key-here \
    nvcr.io/nvidia/eval-factory/simple-evals:25.07.3 \
    eval-factory run_eval \
        --eval_type mmlu_pro \
        --model_id meta/llama-3.1-8b-instruct \
        --model_url https://integrate.api.nvidia.com/v1/chat/completions \
        --model_type chat \
        --api_key_name MY_API_KEY \
        --output_dir /workspace/results \
        --overrides 'config.params.limit_samples=3'
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
    image: nvcr.io/nvidia/eval-factory/simple-evals:25.07.3
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

BENCHMARKS=("mmlu_pro" "gsm8k" "hellaswag" "arc_easy")
API_KEY=${NGC_API_KEY}

for benchmark in "${BENCHMARKS[@]}"; do
    echo "Running evaluation for $benchmark..."
    
    docker run --rm --gpus all \
        -v $(pwd)/results:/workspace/results \
        -e MY_API_KEY=$API_KEY \
        nvcr.io/nvidia/eval-factory/simple-evals:25.07.3 \
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

### Custom Configuration Mounting

```bash
# Mount custom configuration files
docker run --rm --gpus all \
    -v $(pwd)/results:/workspace/results \
    -v $(pwd)/my-configs:/workspace/my-configs \
    -e MY_API_KEY=nvapi-your-key-here \
    nvcr.io/nvidia/eval-factory/simple-evals:25.07.3 \
    eval-factory run_eval \
        --config-dir /workspace/my-configs \
        --config-name custom_evaluation \
        --output-dir /workspace/results
```

## Next Steps

- Integrate into your CI/CD pipelines
- Explore Docker Compose for multi-service setups
- Consider Kubernetes deployment for scale
- Try [NeMo Evaluator Launcher](launcher.md) for simplified workflows
- Explore [Complete Stack Integration](full-stack.md) for advanced use cases
