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
  
  compare_models: true
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
      
      # Rate limiting
      max_requests_per_minute: 50
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
    -o evaluation.tasks='["mmlu_pro"]' \
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
      cache_key_fields: ["messages", "temperature", "max_tokens"]
      
      # Rate limiting to control costs
      max_requests_per_minute: 20
      max_tokens_per_minute: 10000

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
    -o evaluation.parallelism=2 \
    -o evaluation.params.limit_samples=50
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

## Cost Management

### Cost Tracking Configuration

```yaml
# config/cost_aware_hosted.yaml
target:
  api_endpoint:
    url: https://integrate.api.nvidia.com/v1/chat/completions
    model_id: meta/llama-3.1-8b-instruct
    api_key: ${NGC_API_KEY}
    
    adapter_config:
      # Cost tracking
      track_costs: true
      cost_per_1k_tokens: 0.0015  # Update based on provider pricing
      
      # Budget controls
      max_cost_per_evaluation: 50.0  # USD
      cost_alert_threshold: 40.0     # Alert at 80% of budget
      
      # Aggressive caching to reduce costs
      use_caching: true
      cache_ttl: 86400  # 24 hours

evaluation:
  # Cost-conscious evaluation settings
  tasks:
    - name: mmlu_pro
      params:
        limit_samples: 200  # Balanced sample size
        
  parallelism: 4  # Moderate parallelism
```

### Cost Monitoring

```python
# cost_monitor.py - Track evaluation costs
import json
from datetime import datetime

class CostMonitor:
    def __init__(self, cost_per_1k_tokens=0.0015):
        self.cost_per_1k_tokens = cost_per_1k_tokens
        self.total_tokens = 0
        self.total_cost = 0.0
        self.requests = []
    
    def track_request(self, prompt_tokens, completion_tokens):
        total_tokens = prompt_tokens + completion_tokens
        cost = (total_tokens / 1000) * self.cost_per_1k_tokens
        
        self.total_tokens += total_tokens
        self.total_cost += cost
        
        self.requests.append({
            "timestamp": datetime.now().isoformat(),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost": cost
        })
    
    def get_summary(self):
        return {
            "total_requests": len(self.requests),
            "total_tokens": self.total_tokens,
            "total_cost": round(self.total_cost, 4),
            "average_tokens_per_request": self.total_tokens / len(self.requests) if self.requests else 0,
            "average_cost_per_request": self.total_cost / len(self.requests) if self.requests else 0
        }
    
    def save_report(self, filename):
        with open(filename, 'w') as f:
            json.dump({
                "summary": self.get_summary(),
                "requests": self.requests
            }, f, indent=2)

# Usage in evaluation
monitor = CostMonitor(cost_per_1k_tokens=0.0015)
# Track each request...
monitor.save_report("cost_report.json")
```

## Rate Limiting and Quotas

### Provider-Specific Limits

```yaml
# Rate limiting configurations for different providers

# NVIDIA Build
nvidia_build_limits:
  requests_per_minute: 100
  tokens_per_minute: 50000
  concurrent_requests: 10

# OpenAI
openai_limits:
  # Varies by tier and model
  gpt_4:
    requests_per_minute: 10000
    tokens_per_minute: 300000
  gpt_3_5_turbo:
    requests_per_minute: 10000  
    tokens_per_minute: 1000000

# Anthropic
anthropic_limits:
  requests_per_minute: 1000
  tokens_per_minute: 200000
```

### Adaptive Rate Limiting

```python
# adaptive_rate_limiter.py
import asyncio
import time
from collections import deque

class AdaptiveRateLimiter:
    def __init__(self, max_requests_per_minute=100, max_tokens_per_minute=50000):
        self.max_requests_per_minute = max_requests_per_minute
        self.max_tokens_per_minute = max_tokens_per_minute
        
        self.request_times = deque()
        self.token_counts = deque()
        
    async def acquire(self, estimated_tokens=100):
        now = time.time()
        
        # Remove old entries
        cutoff = now - 60  # 1 minute ago
        while self.request_times and self.request_times[0] < cutoff:
            self.request_times.popleft()
        while self.token_counts and self.token_counts[0][0] < cutoff:
            self.token_counts.popleft()
        
        # Check limits
        if len(self.request_times) >= self.max_requests_per_minute:
            sleep_time = 60 - (now - self.request_times[0])
            await asyncio.sleep(sleep_time)
        
        current_tokens = sum(count for _, count in self.token_counts)
        if current_tokens + estimated_tokens > self.max_tokens_per_minute:
            sleep_time = 60 - (now - self.token_counts[0][0])
            await asyncio.sleep(sleep_time)
        
        # Record this request
        self.request_times.append(now)
        self.token_counts.append((now, estimated_tokens))
```

## Error Handling and Retries

### Robust Error Handling

```python
# robust_client.py
import asyncio
import aiohttp
import backoff
from typing import Dict, Any

class RobustHostedClient:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=300),
            headers={"Authorization": f"Bearer {self.api_key}"}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.session.close()
    
    @backoff.on_exception(
        backoff.expo,
        (aiohttp.ClientError, asyncio.TimeoutError),
        max_tries=5,
        max_time=300
    )
    async def make_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        async with self.session.post(
            f"{self.base_url}/chat/completions",
            json=payload
        ) as response:
            if response.status == 429:  # Rate limited
                retry_after = int(response.headers.get('Retry-After', 60))
                await asyncio.sleep(retry_after)
                raise aiohttp.ClientError("Rate limited")
            
            if response.status >= 400:
                error_text = await response.text()
                raise aiohttp.ClientError(f"HTTP {response.status}: {error_text}")
            
            return await response.json()

# Usage
async def evaluate_with_retries():
    async with RobustHostedClient(
        "https://integrate.api.nvidia.com/v1",
        os.getenv("NGC_API_KEY")
    ) as client:
        result = await client.make_request({
            "model": "meta/llama-3.1-8b-instruct",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100
        })
        return result
```

## Security and Compliance

### API Key Management

```yaml
# config/secure_hosted.yaml
target:
  api_endpoint:
    url: https://integrate.api.nvidia.com/v1/chat/completions
    model_id: meta/llama-3.1-8b-instruct
    
    # Use environment variables for secrets
    api_key: ${NGC_API_KEY}
    
    # Additional security headers
    custom_headers:
      X-Request-ID: "${REQUEST_ID}"
      X-Client-Version: "nemo-evaluator-1.0"
    
    adapter_config:
      # Audit logging
      use_request_logging: true
      use_response_logging: true
      log_level: "INFO"
      
      # Request sanitization
      sanitize_requests: true
      remove_sensitive_data: true
```

### Compliance Configuration

```python
# compliance_config.py
class ComplianceConfig:
    def __init__(self):
        self.data_residency = "US"  # Data location requirements
        self.encryption_in_transit = True
        self.audit_logging = True
        self.data_retention_days = 30
        
    def get_compliant_headers(self):
        return {
            "X-Data-Residency": self.data_residency,
            "X-Audit-Required": str(self.audit_logging).lower(),
            "X-Retention-Policy": f"{self.data_retention_days}d"
        }
```

## Performance Optimization

### Connection Pooling

```python
# optimized_client.py
import aiohttp
import asyncio
from aiohttp import TCPConnector

class OptimizedHostedClient:
    def __init__(self, base_url: str, api_key: str, max_connections=100):
        self.base_url = base_url
        self.api_key = api_key
        
        # Optimized connector
        connector = TCPConnector(
            limit=max_connections,
            limit_per_host=max_connections,
            keepalive_timeout=30,
            enable_cleanup_closed=True
        )
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=300),
            headers={"Authorization": f"Bearer {self.api_key}"}
        )
    
    async def batch_requests(self, payloads):
        """Process multiple requests concurrently"""
        semaphore = asyncio.Semaphore(20)  # Limit concurrent requests
        
        async def process_request(payload):
            async with semaphore:
                return await self.make_request(payload)
        
        tasks = [process_request(payload) for payload in payloads]
        return await asyncio.gather(*tasks, return_exceptions=True)
```

### Caching Strategies

```python
# smart_cache.py
import hashlib
import json
import time
from typing import Optional, Dict, Any

class SmartCache:
    def __init__(self, ttl=3600, max_size=10000):
        self.ttl = ttl
        self.max_size = max_size
        self.cache = {}
        self.access_times = {}
    
    def _get_key(self, request: Dict[str, Any]) -> str:
        # Create cache key from request parameters
        cache_data = {
            "model": request.get("model"),
            "messages": request.get("messages", []),
            "temperature": request.get("temperature", 0.7),
            "max_tokens": request.get("max_tokens", 100)
        }
        return hashlib.md5(json.dumps(cache_data, sort_keys=True).encode()).hexdigest()
    
    def get(self, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        key = self._get_key(request)
        
        if key in self.cache:
            # Check TTL
            if time.time() - self.cache[key]["timestamp"] < self.ttl:
                self.access_times[key] = time.time()
                return self.cache[key]["response"]
            else:
                # Expired
                del self.cache[key]
                del self.access_times[key]
        
        return None
    
    def set(self, request: Dict[str, Any], response: Dict[str, Any]):
        key = self._get_key(request)
        
        # Evict oldest if at capacity
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
            del self.cache[oldest_key]
            del self.access_times[oldest_key]
        
        self.cache[key] = {
            "response": response,
            "timestamp": time.time()
        }
        self.access_times[key] = time.time()
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
     "$BASE_URL/models" | jq '.data[].id'
```

**Cost Overruns:**
```python
# Implement cost circuit breaker
class CostCircuitBreaker:
    def __init__(self, max_cost=100.0):
        self.max_cost = max_cost
        self.current_cost = 0.0
        
    def check_budget(self, estimated_cost):
        if self.current_cost + estimated_cost > self.max_cost:
            raise Exception(f"Budget exceeded: ${self.current_cost:.2f} + ${estimated_cost:.2f} > ${self.max_cost:.2f}")
        
    def add_cost(self, actual_cost):
        self.current_cost += actual_cost
```

## Best Practices

### Cost Optimization
- **Use caching aggressively**: Cache responses for repeated evaluations
- **Choose appropriate models**: Balance cost vs performance needs
- **Implement budget controls**: Set spending limits and alerts
- **Monitor usage patterns**: Track costs and optimize based on usage

### Performance
- **Batch requests**: Group multiple requests when possible
- **Use connection pooling**: Reuse connections for better performance
- **Implement smart retries**: Exponential backoff with jitter
- **Monitor rate limits**: Stay within provider limits

### Security
- **Secure API keys**: Use environment variables and secret management
- **Audit requests**: Log all API interactions for compliance
- **Implement timeouts**: Prevent hanging requests
- **Validate responses**: Check response format and content

### Reliability
- **Handle failures gracefully**: Implement proper error handling
- **Use circuit breakers**: Prevent cascading failures
- **Monitor service health**: Track API availability and latency
- **Have fallback plans**: Multiple providers or local deployment

## Next Steps

- **Compare providers**: Evaluate different hosted services for your use cases
- **Optimize costs**: Implement advanced cost management strategies
- **Scale operations**: Move to [enterprise infrastructure](enterprise-infrastructure.md) for production
- **Hybrid approach**: Combine hosted services with [manual deployment](manual-deployment.md)
