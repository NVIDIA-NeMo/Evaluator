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
            enabled=True,
            config={"system_message": "You are a precise, concise assistant. Answer questions directly and accurately."}
        )
    ]
)

target = EvaluationTarget(api_endpoint=api_endpoint)
config = EvaluationConfig(type="mmlu_pro", output_dir="results")

results = evaluate(target_cfg=target, eval_cfg=config)
```

Notes:

- Include the `system_message` interceptor in the interceptors list to enable this feature
- The system message applies to both chat and completions endpoints
- The adapter injects the system message into the request before sending to the model
- For chat endpoints, the interceptor adds the system message as a message with `role: "system"`
- For completions endpoints, the interceptor prepends the system message to the prompt
- Refer to {ref}`adapters-configuration` for available options and defaults

