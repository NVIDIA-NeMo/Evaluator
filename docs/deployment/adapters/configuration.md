(adapters-configuration)=

# Configuration

The adapter system is configured using the `AdapterConfig` class from `nemo_evaluator.adapters.adapter_config`. This class provides comprehensive configuration options for all interceptors in the pipeline.

## Core Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `api_url` | `str` | Required | URL of the model API endpoint |

## Interceptor Configuration

### System Message Interceptor
| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `use_system_prompt` | `bool` | `False` | Enable system message injection |
| `custom_system_prompt` | `Optional[str]` | `None` | Custom system message content |

### Reasoning Interceptor
| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `use_reasoning` | `bool` | `False` | Enable reasoning processing |
| `start_reasoning_token` | `str` | `"<think>"` | Token marking start of reasoning |
| `end_reasoning_token` | `str` | `"</think>"` | Token marking end of reasoning |
| `extract_reasoning` | `bool` | `True` | Extract reasoning into separate field |
| `reasoning_field` | `str` | `"reasoning"` | Field name for extracted reasoning |

### Logging Interceptors
| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `use_request_logging` | `bool` | `False` | Enable request logging |
| `use_response_logging` | `bool` | `False` | Enable response logging |
| `max_logged_requests` | `int` | `100` | Maximum requests to log |
| `max_logged_responses` | `int` | `100` | Maximum responses to log |
| `log_failed_requests` | `bool` | `True` | Log failed requests |
| `save_responses` | `bool` | `False` | Save response content to files |
| `logging_dir` | `str` | `"./logs"` | Directory for log files |

### Caching Interceptor
| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `use_caching` | `bool` | `False` | Enable response caching |
| `caching_dir` | `str` | `"./cache"` | Cache storage directory |
| `reuse_cached_responses` | `bool` | `True` | Use cached responses when available |
| `cache_key_fields` | `list` | `["messages"]` | Fields for cache key generation |

### Progress Tracking Interceptor
| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `use_progress_tracking` | `bool` | `False` | Enable progress monitoring |
| `progress_tracking_url` | `Optional[str]` | `None` | URL for progress updates |
| `progress_update_interval` | `int` | `10` | Update interval in requests |

### Endpoint Interceptor
| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `max_retries` | `int` | `3` | Maximum retry attempts |
| `retry_delay` | `float` | `1.0` | Delay between retries (seconds) |
| `timeout` | `float` | `30.0` | Request timeout (seconds) |
| `retry_on_timeout` | `bool` | `True` | Retry on timeout errors |

## Configuration Examples

### Basic Configuration
```python
from nemo_evaluator.adapters.adapter_config import AdapterConfig

adapter_config = AdapterConfig(
    api_url="http://localhost:8080/v1/completions/",
    use_request_logging=True,
    use_caching=True
)
```

### Advanced Configuration
```python
adapter_config = AdapterConfig(
    # Endpoint configuration
    api_url="http://localhost:8080/v1/completions/",
    max_retries=5,
    timeout=45.0,
    
    # System prompting
    use_system_prompt=True,
    custom_system_prompt="You are an expert AI assistant.",
    
    # Reasoning processing
    use_reasoning=True,
    start_reasoning_token="<think>",
    end_reasoning_token="</think>",
    extract_reasoning=True,
    
    # Comprehensive logging
    use_request_logging=True,
    use_response_logging=True,
    max_logged_requests=1000,
    log_failed_requests=True,
    
    # Performance optimization
    use_caching=True,
    caching_dir="./production_cache",
    reuse_cached_responses=True,
    
    # Monitoring
    use_progress_tracking=True,
    progress_tracking_url="http://monitoring:3828/progress"
)
```


