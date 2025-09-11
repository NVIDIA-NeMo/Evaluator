(interceptor-logging)=

# Log Requests and Responses

The logging interceptor captures detailed request and response data for debugging, analysis, and audit trails.

## Overview

The `LoggingInterceptor` provides comprehensive logging capabilities for evaluation requests and responses. It can log successful requests, failed requests, and detailed response data with configurable limits and storage options.

## Configuration

### AdapterConfig Parameters

```python
from nemo_evaluator.adapters.adapter_config import AdapterConfig

adapter_config = AdapterConfig(
    # Enable logging
    use_request_logging=True,
    use_response_logging=True,
    
    # Logging limits
    max_logged_requests=1000,
    max_logged_responses=1000,
    
    # Error logging
    log_failed_requests=True,
    
    # Storage options
    save_responses=True,
    logging_dir="./evaluation_logs"
)
```

### CLI Configuration
```bash
--overrides 'target.api_endpoint.adapter_config.use_request_logging=True,target.api_endpoint.adapter_config.use_response_logging=True,target.api_endpoint.adapter_config.max_logged_requests=1000,target.api_endpoint.adapter_config.log_failed_requests=True'
```

### YAML Configuration
```yaml
target:
  api_endpoint:
    adapter_config:
      use_request_logging: true
      use_response_logging: true
      max_logged_requests: 1000
      max_logged_responses: 1000
      log_failed_requests: true
      save_responses: true
      logging_dir: "./evaluation_logs"
```

## Configuration Options

| Parameter | Description | Default | Type |
|-----------|-------------|---------|------|
| `use_request_logging` | Enable request logging | `False` | bool |
| `use_response_logging` | Enable response logging | `False` | bool |
| `max_logged_requests` | Maximum number of requests to log | `100` | int |
| `max_logged_responses` | Maximum number of responses to log | `100` | int |
| `log_failed_requests` | Log requests that result in errors | `True` | bool |
| `save_responses` | Save response content to files | `False` | bool |
| `logging_dir` | Directory to save log files | `"./logs"` | str |

## Log Format

### Request Log Entry
```json
{
    "timestamp": "2025-01-08T10:30:00Z",
    "request_id": "req_12345",
    "benchmark": "gsm8k",
    "sample_id": "sample_001",
    "request": {
        "messages": [
            {"role": "user", "content": "What is 2+2?"}
        ],
        "temperature": 0.0,
        "max_new_tokens": 512
    },
    "endpoint_url": "http://localhost:8080/v1/completions",
    "model_id": "megatron_model"
}
```

### Response Log Entry
```json
{
    "timestamp": "2025-01-08T10:30:01Z",
    "request_id": "req_12345",
    "response": {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "2+2 equals 4."
                }
            }
        ],
        "usage": {
            "prompt_tokens": 45,
            "completion_tokens": 8,
            "total_tokens": 53
        }
    },
    "latency_ms": 1250,
    "status_code": 200,
    "success": true
}
```

## Use Cases

- **Debugging**: Analyze failed requests and unexpected responses
- **Performance Analysis**: Track response times and token usage
- **Audit Trails**: Maintain records of all evaluation interactions
- **Dataset Creation**: Build datasets from evaluation runs
- **Compliance**: Meet regulatory requirements for AI system logging
