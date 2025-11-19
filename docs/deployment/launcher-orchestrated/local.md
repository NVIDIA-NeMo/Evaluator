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
    --config-dir packages/nemo-evaluator-launcher/examples \
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
    api_key_name: NGC_API_KEY

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

### With Adapter Configuration

Configure adapters using evaluation overrides:

```yaml
target:
  api_endpoint:
    url: http://localhost:8080/v1/chat/completions
    model_id: my-model

evaluation:
  overrides:
    target.api_endpoint.adapter_config.use_reasoning: true
    target.api_endpoint.adapter_config.use_system_prompt: true
    target.api_endpoint.adapter_config.custom_system_prompt: "Think step by step."
```

For detailed adapter configuration options, refer to {ref}`adapters`.

### Tasks Requiring Dataset Mounting

Some tasks require access to local datasets. For these tasks, specify the dataset location:

```yaml
evaluation:
  tasks:
    - name: mteb.techqa
      dataset_dir: /path/to/your/techqa/dataset
```

The system will automatically:
- Mount the dataset directory into the evaluation container at `/datasets` (or a custom path if specified)
- Set the `NEMO_EVALUATOR_DATASET_DIR` environment variable
- Validate that all required environment variables are configured

**Custom mount path example:**

```yaml
evaluation:
  tasks:
    - name: mteb.techqa
      dataset_dir: /mnt/data/techqa
      dataset_mount_path: /custom/path  # Optional: customize container mount point
```


### Advanced settings

If you are deploying the model locally with Docker, you can use a dedicated docker network.
This will provide a secure connetion between deployment and evaluation docker containers.

```shell
docker network create my-custom-network

docker run --gpus all --network my-custom-network --name my-phi-container vllm/vllm-openai:latest \
    --model microsoft/Phi-4-mini-instruct
```

Then use the same network in the evaluator config:

```yaml
defaults:
  - execution: local
  - deployment: none
  - _self_

execution:
  output_dir: my_phi_test
  extra_docker_args: "--network my-custom-network"

target:
  api_endpoint:
    model_id: microsoft/Phi-4-mini-instruct
    url: http://my-phi-container:8000/v1/chat/completions
    api_key_name: null

evaluation:
  tasks:
    - name: simple_evals.mmlu_pro
      overrides:
        config.params.limit_samples: 10 # TEST ONLY: Limits to 10 samples for quick testing
        config.params.parallelism: 1
```

Alternatively you can expose ports and use the host network:

```shell
docker run --gpus all -p 8000:8000 vllm/vllm-openai:latest \
    --model microsoft/Phi-4-mini-instruct
```

```yaml
execution:
  extra_docker_args: "--network host"
```

## Command-Line Usage

### Basic Commands

```bash
# Run evaluation
nemo-evaluator-launcher run \
    --config-dir packages/nemo-evaluator-launcher/examples \
    --config-name local_llama_3_1_8b_instruct

# Dry run to preview configuration
nemo-evaluator-launcher run \
    --config-dir packages/nemo-evaluator-launcher/examples \
    --config-name local_llama_3_1_8b_instruct \
    --dry-run

# Override endpoint URL
nemo-evaluator-launcher run \
    --config-dir packages/nemo-evaluator-launcher/examples \
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

- {ref}`bring-your-own-endpoint-manual` using vLLM, TensorRT-LLM, or other frameworks
- {ref}`bring-your-own-endpoint-hosted` like NVIDIA API Catalog or OpenAI
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
```

**API authentication errors:**

- Verify `api_key_name` matches your environment variable
- Check that the environment variable has a value: `echo $NGC_API_KEY`
- Check API key has proper permissions

### Evaluation Issues

**Job hangs or shows no progress:**

Check logs in the output directory:

```bash
# Track logs in real-time
tail -f <output_dir>/<task_name>/logs/stdout.log

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
    --config-dir packages/nemo-evaluator-launcher/examples \
    --config-name local_llama_3_1_8b_instruct \
    --dry-run
```

## Next Steps

- **Deploy your own model**: See {ref}`bring-your-own-endpoint-manual` for local model serving
- **Scale to HPC**: Use {ref}`launcher-orchestrated-slurm` for cluster deployments
- **Cloud execution**: Try {ref}`launcher-orchestrated-lepton` for cloud-based evaluation
- **Configure adapters**: Add interceptors with {ref}`adapters`
