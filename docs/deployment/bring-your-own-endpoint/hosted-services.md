<!-- vale off -->
(bring-your-own-endpoint-hosted)=

# Hosted Services

Use existing hosted model APIs from cloud providers without managing your own infrastructure. This approach offers the fastest path to evaluation with minimal setup requirements.
<!-- vale on -->

## Overview

Hosted services provide:

- Pre-deployed models accessible via API
- No infrastructure management required
- Pay-per-use pricing models
- Instant availability and global access
- Professional SLA and support

## NVIDIA Build

<!-- vale off -->
NVIDIA's catalog of ready-to-use AI models with OpenAI-compatible APIs.
<!-- vale on -->

### NVIDIA Build Setup and Authentication

```bash
# Get your NGC API key from https://build.nvidia.com
export NGC_API_KEY="your-ngc-api-key"

# Test authentication
curl -H "Authorization: Bearer $NGC_API_KEY" \
     "https://integrate.api.nvidia.com/v1/models"
```

Refer to the [NVIDIA Build catalog](https://build.nvidia.com) for available models.

### NVIDIA Build Configuration

#### Basic NVIDIA Build Evaluation

```yaml
# config/nvidia_build_basic.yaml
defaults:
  - execution: local
  - deployment: none  # No deployment needed
  - _self_

target:
  api_endpoint:
    url: https://integrate.api.nvidia.com/v1/chat/completions
    model_id: meta/llama-3.1-8b-instruct
    api_key_name: NGC_API_KEY  # Name of environment variable

execution:
  output_dir: ./results

evaluation:
  overrides:
    config.params.limit_samples: 100
  tasks:
    - name: mmlu_pro
    - name: gsm8k
```

#### Multi-Model Comparison

For multi-model comparison, run separate evaluations for each model and compare results:

```bash
# Evaluate first model
nv-eval run \
    --config-dir examples \
    --config-name local_llama_3_1_8b_instruct \
    -o target.api_endpoint.model_id=meta/llama-3.1-8b-instruct \
    -o execution.output_dir=./results/llama-3.1-8b

# Evaluate second model
nv-eval run \
    --config-dir examples \
    --config-name local_llama_3_1_8b_instruct \
    -o target.api_endpoint.model_id=meta/llama-3.1-70b-instruct \
    -o execution.output_dir=./results/llama-3.1-70b
```

#### With Custom Adapters

Configure adapters using the interceptor structure in Python. For detailed YAML configuration, see {ref}`adapters-configuration`.

```python
from nemo_evaluator import evaluate, ApiEndpoint, EvaluationTarget, EvaluationConfig
from nemo_evaluator.adapters.adapter_config import AdapterConfig, InterceptorConfig

adapter_config = AdapterConfig(
    interceptors=[
        InterceptorConfig(
            name="system_message",
            config={"system_message": "You are a helpful assistant that provides accurate and concise answers."}
        ),
        InterceptorConfig(
            name="caching",
            config={"cache_dir": "./nvidia_build_cache", "reuse_cached_responses": True}
        ),
        InterceptorConfig(
            name="request_logging",
            config={"max_requests": 20}
        )
    ]
)

api_endpoint = ApiEndpoint(
    url="https://integrate.api.nvidia.com/v1/chat/completions",
    model_id="meta/llama-3.1-8b-instruct",
    api_key="NGC_API_KEY",  # Name of environment variable
    adapter_config=adapter_config
)
target = EvaluationTarget(api_endpoint=api_endpoint)
config = EvaluationConfig(type="mmlu_pro", output_dir="./results")
results = evaluate(target_cfg=target, eval_cfg=config)
```

### NVIDIA Build CLI Usage

Use `nv-eval` (recommended) or `nemo-evaluator-launcher`:

```bash
# Basic evaluation
nv-eval run \
    --config-dir examples \
    --config-name local_llama_3_1_8b_instruct \
    -o target.api_endpoint.url=https://integrate.api.nvidia.com/v1/chat/completions \
    -o target.api_endpoint.model_id=meta/llama-3.1-8b-instruct \
    -o target.api_endpoint.api_key=${NGC_API_KEY}

# Large model evaluation with limited samples
nv-eval run \
    --config-dir examples \
    --config-name local_llama_3_1_8b_instruct \
    -o target.api_endpoint.model_id=meta/llama-3.1-405b-instruct \
    -o config.params.limit_samples=50
```

## OpenAI API

Direct integration with OpenAI's GPT models for comparison and benchmarking.

### OpenAI Setup and Authentication

```bash
# Get API key from https://platform.openai.com/api-keys
export OPENAI_API_KEY="your-openai-api-key"

# Test authentication
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
     "https://api.openai.com/v1/models"
```

Refer to the [OpenAI model documentation](https://platform.openai.com/docs/models) for available models.

### OpenAI Configuration

#### Basic OpenAI Evaluation

```yaml
# config/openai_basic.yaml
defaults:
  - execution: local
  - deployment: none
  - _self_

target:
  api_endpoint:
    url: https://api.openai.com/v1/chat/completions
    model_id: gpt-4
    api_key_name: OPENAI_API_KEY  # Name of environment variable

execution:
  output_dir: ./results

evaluation:
  overrides:
    config.params.limit_samples: 100
  tasks:
    - name: mmlu_pro
    - name: gsm8k
```

#### Cost-Optimized Configuration

```yaml
# config/openai_cost_optimized.yaml
defaults:
  - execution: local
  - deployment: none
  - _self_

target:
  api_endpoint:
    url: https://api.openai.com/v1/chat/completions
    model_id: gpt-3.5-turbo  # Less expensive model
    api_key_name: OPENAI_API_KEY

execution:
  output_dir: ./results

evaluation:
  overrides:
    config.params.limit_samples: 50  # Smaller sample size
    config.params.parallelism: 2  # Lower parallelism to respect rate limits
  tasks:
    - name: mmlu_pro
```

### OpenAI CLI Usage

Use `nv-eval` (recommended) or `nemo-evaluator-launcher`:

```bash
# GPT-4 evaluation
nv-eval run \
    --config-dir examples \
    --config-name local_llama_3_1_8b_instruct \
    -o target.api_endpoint.url=https://api.openai.com/v1/chat/completions \
    -o target.api_endpoint.model_id=gpt-4 \
    -o target.api_endpoint.api_key=${OPENAI_API_KEY}

# Cost-effective GPT-3.5 evaluation
nv-eval run \
    --config-dir examples \
    --config-name local_llama_3_1_8b_instruct \
    -o target.api_endpoint.model_id=gpt-3.5-turbo \
    -o config.params.limit_samples=50 \
    -o config.params.parallelism=2
```

## Troubleshooting

### Authentication Errors

Verify that your API key has the correct value:

```bash
# Verify NVIDIA Build API key
curl -H "Authorization: Bearer $NGC_API_KEY" \
     "https://integrate.api.nvidia.com/v1/models"

# Verify OpenAI API key
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
     "https://api.openai.com/v1/models"
```

### Rate Limiting

If you encounter rate limit errors (HTTP 429), reduce the `parallelism` parameter in your configuration:

```yaml
evaluation:
  overrides:
    config.params.parallelism: 2  # Lower parallelism to respect rate limits
```

## Next Steps

- **Add adapters**: Explore [adapter configurations](../adapters/configuration.md) for custom processing
- **Self-host models**: Consider [manual deployment](manual-deployment.md) for full control
