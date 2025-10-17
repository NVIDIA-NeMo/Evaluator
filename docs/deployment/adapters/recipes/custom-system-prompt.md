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
            config={"system_message": "You are a precise, concise assistant. Answer questions directly and accurately."}
        )
    ]
)

target = EvaluationTarget(api_endpoint=api_endpoint)
config = EvaluationConfig(type="mmlu_pro", output_dir="results")

results = evaluate(target_cfg=target, eval_cfg=config)
```

## How It Works

The `system_message` interceptor modifies chat-format requests by:

1. Removing any existing system messages from the messages array
2. Inserting the configured system message as the first message with `role: "system"`
3. Preserving all other request parameters

Refer to {ref}`adapters-configuration` for more configuration options.
