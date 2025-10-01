<!-- markdownlint-disable MD041 -->
<!-- vale off -->
(adapters-configuration)=

# Configuration

Configure the adapter system using the `AdapterConfig` class from `nemo_evaluator.adapters.adapter_config`. This class uses a registry-based interceptor architecture where you configure a list of interceptors, each with their own parameters.

## Core Configuration Structure

`AdapterConfig` accepts the following structure:

```python
from nemo_evaluator.adapters.adapter_config import AdapterConfig, InterceptorConfig

adapter_config = AdapterConfig(
    interceptors=[
        InterceptorConfig(
            name="interceptor_name",
            enabled=True,  # Optional, defaults to True
            config={
                # Interceptor-specific parameters
            }
        )
    ],
    endpoint_type="chat"  # Optional, defaults to "chat"
)
```

## Available Interceptors

### System Message Interceptor

**Name:** `system_message`

Adds a system message to requests by adding it as a system role message.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `system_message` | `str` | Required | System message to add to requests |

**Example:**

```python
InterceptorConfig(
    name="system_message",
    config={
        "system_message": "You are a helpful assistant."
    }
)
```

### Reasoning Interceptor

**Name:** `reasoning`

Processes reasoning content in responses by detecting and removing reasoning tokens, tracking reasoning statistics, and optionally extracting reasoning to separate fields.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `start_reasoning_token` | `str \| None` | `"<think>"` | Token marking start of reasoning section |
| `end_reasoning_token` | `str` | `"</think>"` | Token marking end of reasoning section |
| `add_reasoning` | `bool` | `True` | Whether to add reasoning information |
| `migrate_reasoning_content` | `bool` | `False` | Migrate reasoning_content to content field with tokens |
| `enable_reasoning_tracking` | `bool` | `True` | Enable reasoning tracking and logging |
| `include_if_not_finished` | `bool` | `True` | Include reasoning if end token not found |
| `enable_caching` | `bool` | `True` | Cache individual request reasoning statistics |
| `cache_dir` | `str` | `"/tmp/reasoning_interceptor"` | Cache directory for reasoning stats |
| `stats_file_saving_interval` | `int \| None` | `None` | Save stats to file every N responses (None = only save via post_eval_hook) |
| `logging_aggregated_stats_interval` | `int` | `100` | Log aggregated stats every N responses |

**Example:**

```python
InterceptorConfig(
    name="reasoning",
    config={
        "start_reasoning_token": "<think>",
        "end_reasoning_token": "</think>",
        "enable_reasoning_tracking": True
    }
)
```

### Request Logging Interceptor

**Name:** `request_logging`

Logs incoming requests with configurable limits and detail levels.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `log_request_body` | `bool` | `True` | Whether to log request body |
| `log_request_headers` | `bool` | `True` | Whether to log request headers |
| `max_requests` | `int \| None` | `2` | Maximum requests to log (None for unlimited) |

**Example:**

```python
InterceptorConfig(
    name="request_logging",
    config={
        "max_requests": 50,
        "log_request_body": True
    }
)
```

### Response Logging Interceptor

**Name:** `response_logging`

Logs outgoing responses with configurable limits and detail levels.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `log_response_body` | `bool` | `True` | Whether to log response body |
| `log_response_headers` | `bool` | `True` | Whether to log response headers |
| `max_responses` | `int \| None` | `None` | Maximum responses to log (None for unlimited) |

**Example:**

```python
InterceptorConfig(
    name="response_logging",
    config={
        "max_responses": 50,
        "log_response_body": True
    }
)
```

### Caching Interceptor

**Name:** `caching`

Caches requests and responses to disk with options for reusing cached responses.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cache_dir` | `str` | `"/tmp"` | Directory to store cache files |
| `reuse_cached_responses` | `bool` | `False` | Whether to reuse cached responses |
| `save_requests` | `bool` | `False` | Whether to save requests to cache |
| `save_responses` | `bool` | `True` | Whether to save responses to cache |
| `max_saved_requests` | `int \| None` | `None` | Maximum requests to save (None for unlimited) |
| `max_saved_responses` | `int \| None` | `None` | Maximum responses to save (None for unlimited) |

**Notes:**

- If `reuse_cached_responses` is `True`, `save_responses` is automatically set to `True` and `max_saved_responses` to `None`
- The system generates cache keys automatically using SHA256 hash of request data

**Example:**

```python
InterceptorConfig(
    name="caching",
    config={
        "cache_dir": "./evaluation_cache",
        "reuse_cached_responses": True
    }
)
```

### Progress Tracking Interceptor

**Name:** `progress_tracking`

Tracks evaluation progress by counting processed samples and optionally sending updates to a webhook.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `progress_tracking_url` | `str \| None` | `"http://localhost:8000"` | URL to post progress updates. Supports shell variable expansion. |
| `progress_tracking_interval` | `int` | `1` | Update every N samples |
| `request_method` | `str` | `"PATCH"` | HTTP method for progress updates |
| `output_dir` | `str \| None` | `None` | Directory to save progress file (creates a `progress` file in this directory) |

**Example:**

```python
InterceptorConfig(
    name="progress_tracking",
    config={
        "progress_tracking_url": "http://monitor:8000/progress",
        "progress_tracking_interval": 10
    }
)
```

### Endpoint Interceptor

**Name:** `endpoint`

Makes the actual HTTP request to the upstream API. This interceptor has no configurable parameters and is typically added automatically as the final interceptor in the chain.

**Example:**

```python
InterceptorConfig(name="endpoint")
```

## Configuration Examples

### Basic Configuration

```python
from nemo_evaluator.adapters.adapter_config import AdapterConfig, InterceptorConfig

adapter_config = AdapterConfig(
    interceptors=[
        InterceptorConfig(
            name="request_logging",
            config={"max_requests": 10}
        ),
        InterceptorConfig(
            name="caching",
            config={"cache_dir": "./cache"}
        )
    ]
)
```

### Advanced Configuration

```python
from nemo_evaluator.adapters.adapter_config import AdapterConfig, InterceptorConfig

adapter_config = AdapterConfig(
    interceptors=[
        # System prompting
        InterceptorConfig(
            name="system_message",
            config={
                "system_message": "You are an expert AI assistant."
            }
        ),
        # Reasoning processing
        InterceptorConfig(
            name="reasoning",
            config={
                "start_reasoning_token": "<think>",
                "end_reasoning_token": "</think>",
                "enable_reasoning_tracking": True
            }
        ),
        # Request logging
        InterceptorConfig(
            name="request_logging",
            config={
                "max_requests": 1000,
                "log_request_body": True
            }
        ),
        # Response logging
        InterceptorConfig(
            name="response_logging",
            config={
                "max_responses": 1000,
                "log_response_body": True
            }
        ),
        # Caching
        InterceptorConfig(
            name="caching",
            config={
                "cache_dir": "./production_cache",
                "reuse_cached_responses": True
            }
        ),
        # Progress tracking
        InterceptorConfig(
            name="progress_tracking",
            config={
                "progress_tracking_url": "http://monitoring:3828/progress",
                "progress_tracking_interval": 10
            }
        )
    ],
    endpoint_type="chat"
)
```

### YAML Configuration

You can also configure adapters through YAML files in your evaluation configuration:

```yaml
target:
  api_endpoint:
    url: http://localhost:8080/v1/chat/completions
    type: chat
    model_id: megatron_model
    adapter_config:
      interceptors:
        - name: system_message
          config:
            system_message: "You are a helpful assistant."
        - name: reasoning
          config:
            start_reasoning_token: "<think>"
            end_reasoning_token: "</think>"
        - name: request_logging
          config:
            max_requests: 50
        - name: response_logging
          config:
            max_responses: 50
        - name: caching
          config:
            cache_dir: ./cache
            reuse_cached_responses: true
```

## Interceptor Order

Interceptors are executed in the order they appear in the `interceptors` list:

1. **Request interceptors** process the request in list order
2. The **endpoint interceptor** makes the actual API call (automatically added if not present)
3. **Response interceptors** process the response in reverse list order

For example, with interceptors `[system_message, request_logging, caching, response_logging, reasoning]`:

- Request flow: `system_message` → `request_logging` → `caching` (check cache) → API call (if cache miss)
- Response flow: API call → `caching` (save to cache) → `response_logging` → `reasoning`

## Shorthand Syntax

You can use string names as shorthand for interceptors with default configuration:

```python
adapter_config = AdapterConfig(
    interceptors=["request_logging", "caching", "response_logging"]
)
```

This is equivalent to:

```python
adapter_config = AdapterConfig(
    interceptors=[
        InterceptorConfig(name="request_logging"),
        InterceptorConfig(name="caching"),
        InterceptorConfig(name="response_logging")
    ]
)
```
