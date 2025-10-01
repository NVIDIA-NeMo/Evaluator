(bring-your-own-endpoint-hosted)=

# Hosted Services

Use existing hosted model APIs from cloud providers without managing your own infrastructure. This approach offers the fastest path to evaluation with minimal setup requirements.

## Overview

Hosted services provide:
- Pre-deployed models accessible via API
- No infrastructure management required
- Pay-per-use pricing models
- Instant availability and global access
- Professional SLA and support

## NVIDIA Build

NVIDIA's catalog of ready-to-use AI models with OpenAI-compatible APIs.

### Setup and Authentication

```bash
# Get your NGC API key from https://build.nvidia.com
export NGC_API_KEY="your-ngc-api-key"

# Test authentication
curl -H "Authorization: Bearer $NGC_API_KEY" \
     "https://integrate.api.nvidia.com/v1/models"
```

### Available Models

Popular models available on NVIDIA Build:

```yaml
# Common models and their identifiers
models:
  # Llama family
  - id: meta/llama-3.1-8b-instruct
    name: "Llama 3.1 8B Instruct"
    context_length: 128000
    
  - id: meta/llama-3.1-70b-instruct  
    name: "Llama 3.1 70B Instruct"
    context_length: 128000
    
  - id: meta/llama-3.1-405b-instruct
    name: "Llama 3.1 405B Instruct"
    context_length: 128000
    
  # Nemotron family
  - id: nvidia/nemotron-4-340b-instruct
    name: "Nemotron 4 340B Instruct"
    context_length: 4096
    
  # Code generation
  - id: meta/codellama-70b
    name: "Code Llama 70B"
    context_length: 16384
```

### Configuration Examples

#### Basic NVIDIA Build Evaluation

```yaml
# config/nvidia_build_basic.yaml
deployment:
  type: none  # No deployment needed

target:
  api_endpoint:
    url: https://integrate.api.nvidia.com/v1/chat/completions
    model_id: meta/llama-3.1-8b-instruct
    api_key: ${NGC_API_KEY}

execution:
  backend: local

evaluation:
  tasks:
    - name: mmlu_pro
      params:
        limit_samples: 100
    - name: gsm8k
      params:
        limit_samples: 50
```

#### Multi-Model Comparison

```yaml
# config/nvidia_build_comparison.yaml
models:
  - name: llama-3.1-8b
    target:
      api_endpoint:
        url: https://integrate.api.nvidia.com/v1/chat/completions
        model_id: meta/llama-3.1-8b-instruct
        api_key: ${NGC_API_KEY}
        
  - name: llama-3.1-70b
    target:
      api_endpoint:
        url: https://integrate.api.nvidia.com/v1/chat/completions
        model_id: meta/llama-3.1-70b-instruct
        api_key: ${NGC_API_KEY}

evaluation:
  tasks:
    - name: mmlu_pro
    - name: gsm8k
    - name: humaneval
```

#### With Custom Adapters

```yaml
# config/nvidia_build_with_adapters.yaml
target:
  api_endpoint:
    url: https://integrate.api.nvidia.com/v1/chat/completions
    model_id: meta/llama-3.1-8b-instruct
    api_key: ${NGC_API_KEY}
    
    adapter_config:
      # Custom system prompting
      use_system_prompt: true
      custom_system_prompt: "You are a helpful assistant that provides accurate and concise answers."
      
      # Caching for cost efficiency
      use_caching: true
      caching_dir: ./nvidia_build_cache
      
      # Request logging
      use_request_logging: true
      max_logged_requests: 20
```

### Command-Line Usage

```bash
# Basic evaluation
nemo-evaluator-launcher run \
    --config-dir examples \
    --config-name local_llama_3_1_8b_instruct \
    -o target.api_endpoint.url=https://integrate.api.nvidia.com/v1/chat/completions \
    -o target.api_endpoint.model_id=meta/llama-3.1-8b-instruct \
    -o target.api_endpoint.api_key=${NGC_API_KEY}

# Large model evaluation
nemo-evaluator-launcher run \
    --config-dir examples \
    --config-name local_llama_3_1_8b_instruct \
    -o target.api_endpoint.model_id=meta/llama-3.1-405b-instruct \
    -o 'evaluation.tasks=["mmlu_pro"]' \
    -o evaluation.params.limit_samples=50
```

## OpenAI API

Direct integration with OpenAI's GPT models for comparison and benchmarking.

### Setup and Authentication

```bash
# Get API key from https://platform.openai.com/api-keys
export OPENAI_API_KEY="your-openai-api-key"

# Test authentication
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
     "https://api.openai.com/v1/models"
```

### Available Models

```yaml
# OpenAI model identifiers
models:
  # GPT-4 family
  - id: gpt-4
    name: "GPT-4"
    context_length: 8192
    
  - id: gpt-4-turbo
    name: "GPT-4 Turbo"
    context_length: 128000
    
  - id: gpt-4o
    name: "GPT-4o"
    context_length: 128000
    
  # GPT-3.5 family  
  - id: gpt-3.5-turbo
    name: "GPT-3.5 Turbo"
    context_length: 4096
```

### Configuration Examples

#### Basic OpenAI Evaluation

```yaml
# config/openai_basic.yaml
deployment:
  type: none

target:
  api_endpoint:
    url: https://api.openai.com/v1/chat/completions
    model_id: gpt-4
    api_key: ${OPENAI_API_KEY}

evaluation:
  tasks:
    - name: mmlu_pro
      params:
        limit_samples: 100
    - name: gsm8k
      params:
        limit_samples: 50
```

#### Cost-Optimized Configuration

```yaml
# config/openai_cost_optimized.yaml
target:
  api_endpoint:
    url: https://api.openai.com/v1/chat/completions
    model_id: gpt-3.5-turbo  # Less expensive model
    api_key: ${OPENAI_API_KEY}
    
    adapter_config:
      # Aggressive caching to reduce API calls
      use_caching: true
      caching_dir: ./openai_cache

evaluation:
  tasks:
    - name: mmlu_pro
      params:
        limit_samples: 50  # Smaller sample size
        parallelism: 2  # Lower parallelism to respect rate limits
```

### Command-Line Usage

```bash
# GPT-4 evaluation
nemo-evaluator-launcher run \
    --config-dir examples \
    --config-name local_llama_3_1_8b_instruct \
    -o target.api_endpoint.url=https://api.openai.com/v1/chat/completions \
    -o target.api_endpoint.model_id=gpt-4 \
    -o target.api_endpoint.api_key=${OPENAI_API_KEY}

# Cost-effective GPT-3.5 evaluation
nemo-evaluator-launcher run \
    --config-dir examples \
    --config-name local_llama_3_1_8b_instruct \
    -o target.api_endpoint.model_id=gpt-3.5-turbo \
    -o +config.params.limit_samples=50 \
    -o +config.params.parallelism=2
```

## Anthropic Claude

Access Claude models through Anthropic's API.

### Setup and Authentication

```bash
# Get API key from https://console.anthropic.com/
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

### Configuration

```yaml
# config/anthropic_claude.yaml
target:
  api_endpoint:
    url: https://api.anthropic.com/v1/messages
    model_id: claude-3-sonnet-20240229
    api_key: ${ANTHROPIC_API_KEY}
    
    # Anthropic-specific headers
    custom_headers:
      anthropic-version: "2023-06-01"

evaluation:
  tasks:
    - name: mmlu_pro
    - name: gsm8k
```

## Google Gemini

Use Google's Gemini models through their API.

### Setup and Authentication

```bash
# Get API key from Google AI Studio
export GOOGLE_API_KEY="your-google-api-key"
```

### Configuration

```yaml
# config/google_gemini.yaml
target:
  api_endpoint:
    url: https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent
    model_id: gemini-pro
    api_key: ${GOOGLE_API_KEY}

evaluation:
  tasks:
    - name: mmlu_pro
    - name: gsm8k
```

## Troubleshooting

### Common Issues

**Authentication Errors:**

```bash
# Verify API key
curl -H "Authorization: Bearer $NGC_API_KEY" \
     "https://integrate.api.nvidia.com/v1/models"

# Check key format
echo $NGC_API_KEY | wc -c  # Should be appropriate length
```

**Rate Limiting:**

```python
# Handle rate limits gracefully
async def handle_rate_limit(response):
    if response.status == 429:
        retry_after = int(response.headers.get('Retry-After', 60))
        print(f"Rate limited. Waiting {retry_after} seconds...")
        await asyncio.sleep(retry_after)
        return True
    return False
```

**Model Availability:**

```bash
# Check available models
curl -H "Authorization: Bearer $API_KEY" \
w     "$BASE_URL/models" | jq '.data[].id'
```

## Best Practices

### Cost Optimization

- **Use caching**: Enable response caching to reduce repeated API calls
- **Choose appropriate models**: Balance cost versus performance needs
- **Limit evaluation scope**: Use `limit_samples` to test with smaller datasets first
- **Monitor API usage**: Track token consumption and request patterns

### Performance

- **Adjust parallelism**: Configure `params.parallelism` based on provider rate limits
- **Use caching**: Enable the caching adapter to avoid redundant API calls
- **Limit sample sizes**: Test with smaller datasets before full evaluations
- **Track response times**: Monitor latency to identify performance issues

### Security

- **Secure API keys**: Use environment variables (e.g., `${NGC_API_KEY}`) for API credentials
- **Audit logging**: Enable `use_request_logging` and `use_response_logging` for compliance
- **Check responses**: Verify response format and status codes

### Reliability

- **Configure retries**: Set `max_retries` in evaluation configuration for transient failures
- **Set appropriate timeouts**: Configure `request_timeout` for long-running requests
- **Track API health**: Monitor error rates and response codes
- **Test before production**: Use `limit_samples` for initial validation

## Next Steps

- **Compare providers**: Evaluate different hosted services for your use cases
- **Add adapters**: Explore [adapter configurations](../adapters/configuration.md) for advanced features
- **Self-host models**: Consider [manual deployment](manual-deployment.md) for full control
