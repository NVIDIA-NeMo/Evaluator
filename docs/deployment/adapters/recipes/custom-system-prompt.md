<!-- markdownlint-disable MD012 MD041 -->
(adapters-recipe-system-prompt)=

# Custom System Prompt (Chat)

Apply a standard system message to chat endpoints for consistent behavior.

```python
from nemo_evaluator import (
    ApiEndpoint, EndpointType, EvaluationConfig, EvaluationTarget, evaluate
)
from nemo_evaluator.adapters.adapter_config import AdapterConfig, InterceptorConfig

# Configure chat endpoint
chat_url = "http://0.0.0.0:8080/v1/chat/completions/"
api_endpoint = ApiEndpoint(url=chat_url, type=EndpointType.CHAT, model_id="megatron_model")

# Configure adapter with custom system prompt using interceptor
api_endpoint.adapter_config = AdapterConfig(
    interceptors=[
        InterceptorConfig(
            name="system_message",
            config={
                "system_message": "You are a precise, concise assistant. Answer questions directly and accurately.",
                "strategy": "prepend"  # Optional: "replace", "append", or "prepend" (default)
            }
        )
    ]
)

target = EvaluationTarget(api_endpoint=api_endpoint)
config = EvaluationConfig(type="mmlu_pro", output_dir="results")

results = evaluate(target_cfg=target, eval_cfg=config)
```

## How It Works

The `system_message` interceptor modifies chat-format requests based on the configured strategy:

- **`prepend` (default)**: Prepends the configured system message before any existing system message
- **`replace`**: Removes any existing system messages and replaces with the configured message
- **`append`**: Appends the configured system message after any existing system message

All strategies:
1. Insert or modify the system message as the first message with `role: "system"`
2. Preserve all other request parameters

## Strategy Examples

```python
# Replace existing system messages (ignore any existing ones)
InterceptorConfig(
    name="system_message",
    config={
        "system_message": "You are a helpful assistant.",
        "strategy": "replace"
    }
)

# Prepend to existing system messages (default behavior)
InterceptorConfig(
    name="system_message",
    config={
        "system_message": "Important: ",
        "strategy": "prepend"
    }
)

# Append to existing system messages
InterceptorConfig(
    name="system_message",
    config={
        "system_message": "\nRemember to be concise.",
        "strategy": "append"
    }
)
```

Refer to {ref}`adapters-configuration` for more configuration options.
