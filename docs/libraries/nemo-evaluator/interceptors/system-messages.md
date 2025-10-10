<!-- markdownlint-disable MD041 -->
(interceptor-system-messages)=

# System Messages

## Overview

The `SystemMessageInterceptor` modifies incoming requests to include custom system messages. This interceptor works with chat-format requests, replacing any existing system messages with the configured message.

## Configuration

### CLI Configuration
```bash
--overrides 'target.api_endpoint.adapter_config.use_system_prompt=True,target.api_endpoint.adapter_config.custom_system_prompt="You are a helpful assistant."'
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
        - name: "endpoint"
          enabled: true
          config: {}
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
