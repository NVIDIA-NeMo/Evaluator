(launcher-orchestrated-local)=

# Local Execution

Run evaluations on your local machine using Docker containers. The local executor connects to existing model endpoints and orchestrates evaluation tasks locally.

:::{important}
The local executor does **not** deploy models. You must have an existing model endpoint running before starting evaluation. For launcher-orchestrated model deployment, use {ref}`launcher-orchestrated-slurm` or {ref}`launcher-orchestrated-lepton`.
:::

## Overview

Local execution:

- Runs evaluation containers locally using Docker
- Connects to existing model endpoints (local or remote)
- Suitable for development, testing, and small-scale evaluations
- Supports parallel or sequential task execution

## Quick Start

```bash
# Run evaluation against existing endpoint
nemo-evaluator-launcher run \
    --config-dir examples \
    --config-name local_llama_3_1_8b_instruct
```

## Configuration

### Basic Configuration

```yaml
# examples/local_llama_3_1_8b_instruct.yaml
defaults:
  - execution: local
  - deployment: none
  - _self_

execution:
  output_dir: llama_3_1_8b_instruct_results
  # mode: sequential  # Optional: run tasks sequentially instead of parallel

target:
  api_endpoint:
    model_id: meta/llama-3.1-8b-instruct
    url: https://integrate.api.nvidia.com/v1/chat/completions
    api_key_name: API_KEY

evaluation:
  tasks:
    - name: ifeval
    - name: gpqa_diamond
```

**Required fields:**

- `execution.output_dir`: Directory for results
- `target.api_endpoint.url`: Model endpoint URL
- `evaluation.tasks`: List of evaluation tasks

### Execution Modes

```yaml
execution:
  output_dir: ./results
  mode: parallel  # Default: run tasks in parallel
  # mode: sequential  # Run tasks one at a time
```

### Multi-Task Evaluation

```yaml
evaluation:
  tasks:
    - name: mmlu_pro
      overrides:
        config.params.limit_samples: 200
    - name: gsm8k
      overrides:
        config.params.limit_samples: 100
    - name: humaneval
      overrides:
        config.params.limit_samples: 50
```

### Task-Specific Configuration

```yaml
evaluation:
  tasks:
    - name: gpqa_diamond
      overrides:
        config.params.temperature: 0.6
        config.params.top_p: 0.95
        config.params.max_new_tokens: 8192
        config.params.parallelism: 4
      env_vars:
        HF_TOKEN: HF_TOKEN_FOR_GPQA_DIAMOND
```

### With Adapter Interceptors

```yaml
target:
  api_endpoint:
    url: http://localhost:8080/v1/chat/completions
    model_id: my-model
    adapter_config:
      interceptors:
        - name: reasoning
          config:
            start_reasoning_token: "<think>"
            end_reasoning_token: "</think>"
        - name: caching
          config:
            cache_dir: ./evaluation_cache
        - name: request_logging
          config:
            max_requests: 50
```

## Command-Line Usage

### Basic Commands

```bash
# Run evaluation
nemo-evaluator-launcher run \
    --config-dir examples \
    --config-name local_llama_3_1_8b_instruct

# Dry run to preview configuration
nemo-evaluator-launcher run \
    --config-dir examples \
    --config-name local_llama_3_1_8b_instruct \
    --dry-run

# Override endpoint URL
nemo-evaluator-launcher run \
    --config-dir examples \
    --config-name local_llama_3_1_8b_instruct \
    -o target.api_endpoint.url=http://localhost:8080/v1/chat/completions
```

### Job Management

```bash
# Check job status
nemo-evaluator-launcher status <job_id>

# Check entire invocation
nemo-evaluator-launcher status <invocation_id>

# Kill running job
nemo-evaluator-launcher kill <job_id>

# List available tasks
nemo-evaluator-launcher ls tasks

# List recent runs
nemo-evaluator-launcher ls runs
```

## Requirements

### System Requirements

- **Docker**: Docker Engine installed and running
- **Storage**: Adequate space for evaluation containers and results
- **Network**: Internet access to pull Docker images

### Model Endpoint

You must have a model endpoint running and accessible before starting evaluation. Options include:

- {ref}`manual-deployment` using vLLM, TensorRT-LLM, or other frameworks
- {ref}`hosted-services` like NVIDIA API Catalog or OpenAI
- Custom deployment solutions

## Troubleshooting

### Docker Issues

**Docker not running:**

```bash
# Check Docker status
docker ps

# Start Docker daemon (varies by platform)
sudo systemctl start docker  # Linux
# Or open Docker Desktop on macOS/Windows
```

**Permission denied:**

```bash
# Add user to docker group (Linux)
sudo usermod -aG docker $USER
# Log out and back in for changes to take effect
```

### Endpoint Connectivity

**Cannot connect to endpoint:**

```bash
# Test endpoint availability
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "test", "messages": [{"role": "user", "content": "Hi"}]}'

# Check if endpoint is accessible from Docker container
docker run --rm curlimages/curl:latest curl http://host.docker.internal:8080/health
```

**API authentication errors:**

- Verify `api_key_name` matches your environment variable
- Check that the environment variable has a value: `echo $API_KEY`
- Check API key has proper permissions

### Evaluation Issues

**Job hangs or shows no progress:**

```bash
# Track logs in real-time
tail -f <output_dir>/<task_name>/logs/stdout.log

# Check Docker container status
docker ps -a

# Kill and restart if needed
nemo-evaluator-launcher kill <job_id>
```

**Tasks fail with errors:**

- Check logs in `<output_dir>/<task_name>/logs/stdout.log`
- Verify model endpoint supports required request format
- Ensure adequate disk space for results

### Configuration Validation

```bash
# Validate configuration before running
nemo-evaluator-launcher run \
    --config-dir examples \
    --config-name local_llama_3_1_8b_instruct \
    --dry-run
```

## Best Practices

### Before Running

1. Verify endpoint is running and accessible
2. Test configuration with `--dry-run`
3. Use `limit_samples` for initial testing
4. Ensure adequate disk space for results

### During Execution

- Track logs for errors: `tail -f <output_dir>/*/logs/stdout.log`
- Check job status: `nemo-evaluator-launcher status <invocation_id>`
- Use sequential mode (`mode: sequential`) if running several tasks with limited resources

### Configuration Tips

- Use environment variables for API keys: `api_key_name: API_KEY`
- Store configuration files in version control
- Use task-specific overrides for custom parameters

## Next Steps

- **Deploy your own model**: See {ref}`manual-deployment` for local model serving
- **Scale to HPC**: Use {ref}`launcher-orchestrated-slurm` for cluster deployments
- **Cloud execution**: Try {ref}`launcher-orchestrated-lepton` for cloud-based evaluation
- **Configure adapters**: Add interceptors with {ref}`adapters`
