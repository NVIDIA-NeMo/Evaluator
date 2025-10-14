(cli-workflows)=

# CLI Workflows

This document explains how to use evaluation containers within NeMo Evaluator workflows, focusing on command execution and configuration.

## Overview

Evaluation containers provide consistent, reproducible environments for running AI model evaluations. For a comprehensive list of all available containers, refer to {ref}`nemo-evaluator-containers`.

## Basic CLI

### Using YAML Configuration

Define your config:

```yaml
config:
  type: mmlu_pro
  output_dir: /workspace/results
  params:
    limit_samples: 10
target:
  api_endpoint:
    url: https://integrate.api.nvidia.com/v1/chat/completions
    model_id: meta/llama-3.1-8b-instruct
    type: chat
    api_key: NGC_API_KEY
```

Run evaluation:

```bash
export HF_TOKEN=hf_xxx
export MY_API_KEY=nvapi-xxx

nemo-evaluator run_eval \
  --run_config /workspace/my_config.yml
```

### Using CLI overrides

Provide all arguments through CLI:

```bash
export HF_TOKEN=hf_xxx
export MY_API_KEY=nvapi-xxx

nemo-evaluator run_eval \
    --eval_type mmlu_pro \
    --model_id meta/llama-3.1-8b-instruct \
    --model_url https://integrate.api.nvidia.com/v1/chat/completions \
    --model_type chat \
    --api_key_name NGC_API_KEY \
    --output_dir /workspace/results \
    --overrides 'config.params.limit_samples=10'
```

## Interceptor Configuration

The adapter system uses interceptors to modify requests and responses. Configure interceptors using the `--overrides` parameter.

For detailed interceptor configuration, refer to {ref}`nemo-evaluator-interceptors`.

:::{note}
Always remember to include `endpoint` Interceptor at the and of your custom Interceptors chain. 
:::


### Enable Request Logging

```yaml
config:
  type: mmlu_pro
  output_dir: /workspace/results
  params:
    limit_samples: 10
target:
  api_endpoint:
    url: https://integrate.api.nvidia.com/v1/chat/completions
    model_id: meta/llama-3.1-8b-instruct
    type: chat
    api_key: NGC_API_KEY
    adapter_config:
      interceptors:
        - name: "request_logging"
            enabled: true
            config:
              max_requests: 1000
        - name: "endpoint"
          enabled: true
          config: {}
```

```bash
export HF_TOKEN=hf_xxx
export MY_API_KEY=nvapi-xxx

nemo-evaluator run_eval \
  --run_config /workspace/my_config.yml
```


### Enable Caching

```yaml
config:
  type: mmlu_pro
  output_dir: /workspace/results
  params:
    limit_samples: 10
target:
  api_endpoint:
    url: https://integrate.api.nvidia.com/v1/chat/completions
    model_id: meta/llama-3.1-8b-instruct
    type: chat
    api_key: NGC_API_KEY
    adapter_config:
      interceptors:
        - name: "caching"
          enabled: true
          config:
            cache_dir: "./evaluation_cache"
            reuse_cached_responses: true
            save_requests: true
            save_responses: true
            max_saved_requests: 1000
            max_saved_responses: 1000
        - name: "endpoint"
          enabled: true
          config: {}
```

```bash
export HF_TOKEN=hf_xxx
export MY_API_KEY=nvapi-xxx

nemo-evaluator run_eval \
  --run_config /workspace/my_config.yml
```

### Multiple Interceptors

```yaml
config:
  type: mmlu_pro
  output_dir: /workspace/results
  params:
    limit_samples: 10
target:
  api_endpoint:
    url: https://integrate.api.nvidia.com/v1/chat/completions
    model_id: meta/llama-3.1-8b-instruct
    type: chat
    api_key: NGC_API_KEY
    adapter_config:
      interceptors:
        - name: "caching"
          enabled: true
          config:
            cache_dir: "./evaluation_cache"
            reuse_cached_responses: true
            save_requests: true
            save_responses: true
            max_saved_requests: 1000
            max_saved_responses: 1000
        - name: "request_logging"
            enabled: true
            config:
              max_requests: 1000
        - name: "reasoning"
          config:
            start_reasoning_token: "<think>"
            end_reasoning_token: "</think>"
            add_reasoning: true
            enable_reasoning_tracking: true
        - name: "endpoint"
          enabled: true
          config: {}
```

```bash
export HF_TOKEN=hf_xxx
export MY_API_KEY=nvapi-xxx

nemo-evaluator run_eval \
  --run_config /workspace/my_config.yml
```

### Legacy Configuration Support

Provide Interceptor configuration with `--overrides` flag:

```bash
nemo-evaluator run_eval \
    --eval_type mmlu_pro \
    --model_id meta/llama-3.1-8b-instruct \
    --model_url https://integrate.api.nvidia.com/v1/chat/completions \
    --model_type chat \
    --api_key_name MY_API_KEY \
    --output_dir ./results \
    --overrides 'target.api_endpoint.adapter_config.use_request_logging=True,target.api_endpoint.adapter_config.max_saved_requests=1000,target.api_endpoint.adapter_config.use_caching=True,target.api_endpoint.adapter_config.caching_dir=./cache,target.api_endpoint.adapter_config.reuse_cached_responses=True'
```

:::{note}
Legacy parameters will be automatically converted to the modern interceptor-based configuration. For new projects, use the YAML interceptor configutation shown above.
:::

## Troubleshooting

### Port Conflicts

If you encounter adapter server port conflicts:

```bash
export ADAPTER_PORT=3828
export ADAPTER_HOST=localhost
```

:::{note}
You can manually set the port, or rely on NeMo Evaluator's dynamic port binding feature.
:::

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
