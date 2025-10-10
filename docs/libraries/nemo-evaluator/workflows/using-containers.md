(container-workflows)=

# Container Workflows

This document explains how to use evaluation containers within NeMo Evaluator workflows, focusing on command execution and configuration.

## Overview

Evaluation containers provide consistent, reproducible environments for running AI model evaluations. For a comprehensive list of all available containers, see {ref}`nemo-evaluator-containers`.

## Basic Container Usage

### Running an Evaluation

```bash
docker run --rm -it nvcr.io/nvidia/eval-factory/simple-evals:{{ docker_compose_latest }} bash

export HF_TOKEN=hf_xxx
export MY_API_KEY=nvapi-xxx

nemo-evaluator run_eval \
    --eval_type mmlu_pro \
    --model_id meta/llama-3.1-8b-instruct \
    --model_url https://integrate.api.nvidia.com/v1/chat/completions \
    --model_type chat \
    --api_key_name MY_API_KEY \
    --output_dir /workspace/results \
    --overrides 'config.params.limit_samples=10'
```

## Interceptor Configuration

The adapter system uses interceptors to modify requests and responses. Configure interceptors using the `--overrides` parameter.

### Enable Request Logging

```bash
nemo-evaluator run_eval \
    --eval_type mmlu_pro \
    --model_id meta/llama-3.1-8b-instruct \
    --model_url https://integrate.api.nvidia.com/v1/chat/completions \
    --model_type chat \
    --api_key_name MY_API_KEY \
    --output_dir ./results \
    --overrides 'target.api_endpoint.adapter_config.interceptors=[{"name":"request_logging","config":{"max_requests":100}}]'
```

### Enable Caching

```bash
nemo-evaluator run_eval \
    --eval_type mmlu_pro \
    --model_id meta/llama-3.1-8b-instruct \
    --model_url https://integrate.api.nvidia.com/v1/chat/completions \
    --model_type chat \
    --api_key_name MY_API_KEY \
    --output_dir ./results \
    --overrides 'target.api_endpoint.adapter_config.interceptors=[{"name":"caching","config":{"cache_dir":"./cache","reuse_cached_responses":true}}]'
```

### Multiple Interceptors

Combine multiple interceptors in a single command:

```bash
nemo-evaluator run_eval \
    --eval_type mmlu_pro \
    --model_id meta/llama-3.1-8b-instruct \
    --model_url https://integrate.api.nvidia.com/v1/chat/completions \
    --model_type chat \
    --api_key_name MY_API_KEY \
    --output_dir ./results \
    --overrides 'target.api_endpoint.adapter_config.interceptors=[{"name":"request_logging"},{"name":"caching","config":{"cache_dir":"./cache"}},{"name":"reasoning","config":{"start_reasoning_token":"<think>","end_reasoning_token":"</think>"}}]'
```

For detailed interceptor configuration, see {ref}`nemo-evaluator-interceptors`.

## Legacy Configuration Support

Legacy parameter names are still supported for backward compatibility:

```bash
--overrides 'target.api_endpoint.adapter_config.use_request_logging=true,target.api_endpoint.adapter_config.use_caching=true'
```

:::{note}
Legacy parameters will be automatically converted to the modern interceptor-based configuration. For new projects, use the interceptor syntax shown above.
:::

## Troubleshooting

### Port Conflicts

If you encounter adapter server port conflicts:

```bash
export ADAPTER_PORT=3828
export ADAPTER_HOST=localhost
```

### API Key Issues

Verify your API key environment variable:

```bash
echo $MY_API_KEY
```

## Environment Variables

### Adapter Server Configuration

```bash
export ADAPTER_PORT=3828  # Default: 3825
export ADAPTER_HOST=localhost
```

### API Key Management

```bash
export MY_API_KEY=your_api_key_here
export HF_TOKEN=your_hf_token_here
```
