# Endpoint

The endpoint interceptor is the final interceptor in the pipeline that routes requests to the actual model API and handles the HTTP communication.

## Overview

The `EndpointInterceptor` is responsible for making the actual API calls to model endpoints. It handles different API formats, implements retry logic, manages timeouts, and provides error handling for HTTP communication.

## Configuration

### AdapterConfig Parameters

```python
from nemo_evaluator.adapters.adapter_config import AdapterConfig

adapter_config = AdapterConfig(
    # Endpoint configuration
    api_url="http://localhost:8080/v1/completions",
    
    # Retry configuration
    max_retries=3,
    retry_delay=1.0,
    retry_on_timeout=True,
    
    # Timeout settings
    timeout=30.0,
    
    # Rate limiting
    rate_limit_rpm=60
)
```

### CLI Configuration
```bash
--overrides 'target.api_endpoint.url=http://localhost:8080/v1/completions,target.api_endpoint.adapter_config.max_retries=3,target.api_endpoint.adapter_config.timeout=30.0'
```

### YAML Configuration
```yaml
target:
  api_endpoint:
    url: http://localhost:8080/v1/completions
    model_id: megatron_model
    adapter_config:
      max_retries: 3
      retry_delay: 1.0
      retry_on_timeout: true
      timeout: 30.0
      rate_limit_rpm: 60
```

## Configuration Options

| Parameter | Description | Default | Type |
|-----------|-------------|---------|------|
| `api_url` | Model API endpoint URL | Required | str |
| `max_retries` | Maximum number of retry attempts | `3` | int |
| `retry_delay` | Delay between retries (seconds) | `1.0` | float |
| `retry_on_timeout` | Retry on timeout errors | `True` | bool |
| `timeout` | Request timeout (seconds) | `30.0` | float |
| `rate_limit_rpm` | Rate limit (requests per minute) | `None` | int |

## Supported API Formats

### OpenAI Completions API
```python
# Request format
{
    "model": "megatron_model",
    "prompt": "What is 2+2?",
    "max_tokens": 100,
    "temperature": 0.0
}

# Response format  
{
    "choices": [
        {
            "text": "2+2 equals 4.",
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 8,
        "total_tokens": 18
    }
}
```

### OpenAI Chat API
```python
# Request format
{
    "model": "megatron_model", 
    "messages": [
        {"role": "user", "content": "What is 2+2?"}
    ],
    "max_tokens": 100,
    "temperature": 0.0
}

# Response format
{
    "choices": [
        {
            "message": {
                "role": "assistant",
                "content": "2+2 equals 4."
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 15,
        "completion_tokens": 8, 
        "total_tokens": 23
    }
}
```

## Error Handling

### Retry Logic
```python
# Automatic retry for transient errors
retry_conditions = [
    "ConnectionError",
    "TimeoutError", 
    "HTTPStatus.500",  # Internal Server Error
    "HTTPStatus.502",  # Bad Gateway
    "HTTPStatus.503",  # Service Unavailable
    "HTTPStatus.504"   # Gateway Timeout
]
```

### Error Response Format
```json
{
    "error": {
        "type": "timeout_error",
        "message": "Request timed out after 30.0 seconds",
        "details": {
            "endpoint": "http://localhost:8080/v1/completions",
            "attempt": 3,
            "total_attempts": 3
        }
    },
    "success": false,
    "status_code": null
}
```

## Rate Limiting

### Request Throttling
```python
# Configure rate limiting
adapter_config = AdapterConfig(
    api_url="http://localhost:8080/v1/completions",
    rate_limit_rpm=120,  # 120 requests per minute
    rate_limit_burst=10  # Allow bursts of up to 10 requests
)
```

### Rate Limit Headers
The interceptor respects standard rate limiting headers:
- `X-RateLimit-Limit`
- `X-RateLimit-Remaining` 
- `X-RateLimit-Reset`
- `Retry-After`

## Authentication

### API Key Authentication
```python
adapter_config = AdapterConfig(
    api_url="https://api.openai.com/v1/chat/completions",
    api_key="your-api-key-here"
)
```

### Custom Headers
```python
adapter_config = AdapterConfig(
    api_url="http://custom-endpoint.com/v1/completions",
    custom_headers={
        "Authorization": "Bearer your-token",
        "X-Custom-Header": "custom-value"
    }
)
```

## Load Balancing

### Multiple Endpoints
```python
adapter_config = AdapterConfig(
    api_urls=[
        "http://endpoint1.com/v1/completions",
        "http://endpoint2.com/v1/completions", 
        "http://endpoint3.com/v1/completions"
    ],
    load_balance_strategy="round_robin"  # or "random", "least_loaded"
)
```

## Monitoring and Metrics

### Request Metrics
The endpoint interceptor tracks:
- Request latency
- Success/failure rates
- Retry attempts
- Rate limit hits
- Error types and frequencies

### Health Checks
```python
# Endpoint health monitoring
adapter_config = AdapterConfig(
    api_url="http://localhost:8080/v1/completions",
    enable_health_checks=True,
    health_check_interval=60,  # seconds
    health_check_timeout=5.0
)
```

## Use Cases

- **Model API Integration**: Connect to any OpenAI-compatible model API
- **Fault Tolerance**: Handle transient network and server errors gracefully
- **Performance Optimization**: Implement retry logic and rate limiting
- **Load Distribution**: Balance requests across multiple model instances
- **Monitoring**: Track API performance and reliability metrics

## Best Practices

### Timeout Configuration
```python
# Configure appropriate timeouts based on model size
adapter_config = AdapterConfig(
    timeout=60.0,     # Longer timeout for large models
    max_retries=5,    # More retries for critical evaluations
    retry_delay=2.0   # Longer delay between retries
)
```

### Error Handling
```python
# Comprehensive error handling
adapter_config = AdapterConfig(
    api_url="http://localhost:8080/v1/completions",
    max_retries=3,
    retry_on_timeout=True,
    continue_on_error=True,      # Continue evaluation on individual failures
    error_threshold=0.1,         # Stop if >10% of requests fail
    log_failed_requests=True     # Log errors for debugging
)
```

### Production Configuration
```python
# Production-ready endpoint configuration
adapter_config = AdapterConfig(
    api_url="https://production-api.com/v1/completions",
    api_key=os.environ["API_KEY"],
    max_retries=5,
    retry_delay=2.0,
    timeout=45.0,
    rate_limit_rpm=1000,
    enable_health_checks=True,
    custom_headers={
        "User-Agent": "NeMo-Evaluator/1.0",
        "X-Evaluation-ID": "eval-12345"
    }
)
```
