---
orphan: true
---

(bring-your-own-endpoint-manual)=

# Manual Deployment

Deploy models yourself using popular serving frameworks, then point NeMo Evaluator to your endpoints. This approach gives you full control over deployment infrastructure and serving configuration.

## Overview

Manual deployment involves:

- Setting up model serving using frameworks like vLLM, TensorRT-LLM, or custom solutions
- Configuring OpenAI-compatible endpoints
- Managing infrastructure, scaling, and monitoring yourself
- Using either the launcher or core library to run evaluations against your endpoints

:::{note}
This guide focuses on NeMo Evaluator configuration. For specific serving framework installation and deployment instructions, refer to their official documentation:

- [vLLM Documentation](https://docs.vllm.ai/)
- [TensorRT-LLM Documentation](https://nvidia.github.io/TensorRT-LLM/)
- [Hugging Face TGI Documentation](https://huggingface.co/docs/text-generation-inference/)
:::

## Using Manual Deployments with NeMo Evaluator

Before connecting to your manual deployment, verify it's properly configured using our {ref}`deployment-testing-compatibility` guide.

### With Launcher

Once your manual deployment is running, use the launcher to evaluate:

```bash
# Basic evaluation against manual deployment
nemo-evaluator-launcher run \
    --config-dir packages/nemo-evaluator-launcher/examples \
    --config-name local_llama_3_1_8b_instruct \
    -o target.api_endpoint.url=http://localhost:8080/v1/completions \
    -o target.api_endpoint.model_id=your-model-name
```

#### Configuration File Approach

```yaml
# config/manual_deployment.yaml
defaults:
  - execution: local
  - deployment: none  # No deployment by launcher
  - _self_

target:
  api_endpoint:
    url: http://localhost:8080/v1/completions
    model_id: llama-3.1-8b
    # Optional authentication (name of environment variable holding API key)
    api_key_name: API_KEY

execution:
  output_dir: ./results

evaluation:
  tasks:
    - name: mmlu_pro
      overrides:
        config.params.limit_samples: 100
    - name: gsm8k
      overrides:
        config.params.limit_samples: 50
```

### With Core Library

Direct API usage for manual deployments:

```python
from nemo_evaluator import (
    ApiEndpoint,
    ConfigParams,
    EndpointType,
    EvaluationConfig,
    EvaluationTarget,
    evaluate
)

# Configure your manual deployment endpoint
api_endpoint = ApiEndpoint(
    url="http://localhost:8080/v1/completions",
    type=EndpointType.COMPLETIONS,
    model_id="llama-3.1-8b",
    api_key="API_KEY"  # Name of environment variable holding API key
)

target = EvaluationTarget(api_endpoint=api_endpoint)

# Configure evaluation
config = EvaluationConfig(
    type="mmlu_pro",
    output_dir="./results",
    params=ConfigParams(
        limit_samples=100,
        parallelism=4
    )
)

# Run evaluation
results = evaluate(eval_cfg=config, target_cfg=target)
print(f"Results: {results}")
```

#### With Adapter Configuration

```python
from nemo_evaluator import (
    ApiEndpoint,
    ConfigParams,
    EndpointType,
    EvaluationConfig,
    EvaluationTarget,
    evaluate
)
from nemo_evaluator.adapters.adapter_config import AdapterConfig, InterceptorConfig

# Configure adapter with interceptors
adapter_config = AdapterConfig(
    interceptors=[
        InterceptorConfig(
            name="caching",
            config={
                "cache_dir": "./cache",
                "reuse_cached_responses": True
            }
        ),
        InterceptorConfig(
            name="request_logging",
            config={"max_requests": 10}
        ),
        InterceptorConfig(
            name="response_logging",
            config={"max_responses": 10}
        )
    ]
)

# Configure endpoint with adapter
api_endpoint = ApiEndpoint(
    url="http://localhost:8080/v1/completions",
    type=EndpointType.COMPLETIONS,
    model_id="llama-3.1-8b",
    api_key="API_KEY",
    adapter_config=adapter_config
)

target = EvaluationTarget(api_endpoint=api_endpoint)

# Configure evaluation
config = EvaluationConfig(
    type="mmlu_pro",
    output_dir="./results",
    params=ConfigParams(
        limit_samples=100,
        parallelism=4
    )
)

# Run evaluation
results = evaluate(eval_cfg=config, target_cfg=target)
print(f"Results: {results}")
```

## Prerequisites

Before using a manually deployed endpoint with NeMo Evaluator, ensure:

- Your model endpoint is running and accessible
- The endpoint supports OpenAI-compatible API format
- You have any required API keys or authentication credentials
- Your endpoint supports the required generation parameters (see below)

### Endpoint Requirements

Your endpoint must support the following generation parameters for compatibility with NeMo Evaluator:

- `temperature`: Controls randomness in generation (0.0 to 1.0)
- `top_p`: Nucleus sampling threshold (0.0 to 1.0)
- `max_tokens`: Maximum tokens to generate

## Testing Your Endpoint

Before running evaluations, verify your endpoint is working as expected.

::::{dropdown} Test Completions Endpoint
:icon: code-square

```bash
# Basic test (no authentication)
curl -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-model-name",
    "prompt": "What is machine learning?",
    "temperature": 0.6,
    "top_p": 0.95,
    "max_tokens": 256,
    "stream": false
  }'

# With authentication
curl -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "your-model-name",
    "prompt": "What is machine learning?",
    "temperature": 0.6,
    "top_p": 0.95,
    "max_tokens": 256,
    "stream": false
  }'
```

::::

::::{dropdown} Test Chat Completions Endpoint
:icon: code-square

```bash
# Basic test (no authentication)
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-model-name",
    "messages": [
      {
        "role": "user",
        "content": "What is machine learning?"
      }
    ],
    "temperature": 0.6,
    "top_p": 0.95,
    "max_tokens": 256,
    "stream": false
  }'

# With authentication
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "your-model-name",
    "messages": [
      {
        "role": "user",
        "content": "What is machine learning?"
      }
    ],
    "temperature": 0.6,
    "top_p": 0.95,
    "max_tokens": 256,
    "stream": false
  }'
```

::::

:::{note}
Each evaluation task requires a specific endpoint type. Verify your endpoint supports the correct type for your chosen tasks. Use `nemo-evaluator-launcher ls tasks` to see which endpoint type each task requires.
:::

## OpenAI API Compatibility

Your endpoint must implement the OpenAI API format:

::::{dropdown} Completions Endpoint Format
:icon: code-square

**Request**: `POST /v1/completions`

```json
{
  "model": "model-name",
  "prompt": "string",
  "max_tokens": 100,
  "temperature": 0.7,
  "top_p": 0.9
}
```

**Response**:

```json
{
  "id": "cmpl-xxx",
  "object": "text_completion",
  "created": 1234567890,
  "model": "model-name",
  "choices": [{
    "text": "generated text",
    "index": 0,
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 20,
    "total_tokens": 30
  }
}
```

::::

::::{dropdown} Chat Completions Endpoint Format
:icon: code-square

**Request**: `POST /v1/chat/completions`

```json
{
  "model": "model-name",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  "max_tokens": 100,
  "temperature": 0.7
}
```

**Response**:

```json
{
  "id": "chatcmpl-xxx",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "model-name",
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "Hello! How can I help you?"
    },
    "index": 0,
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 15,
    "completion_tokens": 10,
    "total_tokens": 25
  }
}
```

::::

## Troubleshooting

### Connection Issues

If you encounter connection errors:

1. Verify the endpoint is running and accessible. Check the health endpoint (path varies by framework):

   ```bash
   # For vLLM, SGLang, NIM
   curl http://localhost:8080/health
   
   # For NeMo/Triton deployments
   curl http://localhost:8080/v1/triton_health
   ```

2. Check that the URL in your configuration matches your deployment:
   - Include the full path (e.g., `/v1/completions` or `/v1/chat/completions`)
   - Verify the port number matches your server configuration
   - Ensure no firewall rules are blocking connections

3. Test with a simple curl command before running full evaluations

### Authentication Errors

If you see authentication failures:

1. Verify the environment variable has a value:

   ```bash
   echo $API_KEY
   ```

2. Ensure the `api_key_name` in your YAML configuration matches the environment variable name

3. Check that your endpoint requires the same authentication method

### Timeout Errors

If requests are timing out:

1. Increase the timeout in your configuration:

   ```yaml
   evaluation:
     overrides:
       config.params.request_timeout: 300  # 5 minutes
   ```

2. Reduce parallelism to avoid overwhelming your endpoint:

   ```yaml
   evaluation:
     overrides:
       config.params.parallelism: 1
   ```

3. Check your endpoint's logs for performance issues

## Next Steps

- **Hosted services**: Compare with [hosted services](hosted-services.md) for managed solutions
- **Launcher-orchestrated deployment**: [Deploy](../launcher-orchestrated/index.md) models for evaluation with `nemo-evaluator-launcher` 
