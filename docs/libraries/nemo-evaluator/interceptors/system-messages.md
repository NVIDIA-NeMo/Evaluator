<!-- markdownlint-disable MD041 -->
(interceptor-system-messages)=

# System Messages

The system message interceptor injects custom system prompts into evaluation requests, enabling consistent prompting and role-specific behavior across evaluations.

## Overview

The `SystemMessageInterceptor` modifies incoming requests to include custom system messages. This interceptor works with chat-format requests, replacing any existing system messages with the configured message.

## Configuration

### Python Configuration

```python
from nemo_evaluator.adapters.adapter_config import AdapterConfig, InterceptorConfig

adapter_config = AdapterConfig(
    interceptors=[
        InterceptorConfig(
            name="system_message",
            config={
                "system_message": "You are a helpful AI assistant."
            }
        )
    ]
)
```

### YAML Configuration

```yaml
target:
  api_endpoint:
    adapter_config:
      interceptors:
        - name: system_message
          config:
            system_message: "You are a helpful AI assistant."
```

## Configuration Options

| Parameter | Description | Type | Required |
|-----------|-------------|------|----------|
| `system_message` | System message to add to requests | str | Yes |

## Behavior

The interceptor modifies chat-format requests by:

1. Removing any existing system messages from the messages array
2. Inserting the configured system message as the first message
3. Preserving all other request parameters

### Example Request Transformation

```python
# Original request
{
    "messages": [
        {"role": "user", "content": "What is 2+2?"}
    ]
}

# After system message interceptor
{
    "messages": [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "What is 2+2?"}
    ]
}
```

If an existing system message is present, the interceptor replaces it:

```python
# Original request with existing system message
{
    "messages": [
        {"role": "system", "content": "Old system message"},
        {"role": "user", "content": "What is 2+2?"}
    ]
}

# After system message interceptor
{
    "messages": [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "What is 2+2?"}
    ]
}
```

## Usage Example

```python
from nemo_evaluator.adapters.adapter_config import AdapterConfig, InterceptorConfig

# System message with other interceptors
adapter_config = AdapterConfig(
    interceptors=[
        InterceptorConfig(
            name="system_message",
            config={
                "system_message": "You are an expert problem solver."
            }
        ),
        InterceptorConfig(
            name="caching",
            config={
                "cache_dir": "./cache"
            }
        )
    ]
)
```
