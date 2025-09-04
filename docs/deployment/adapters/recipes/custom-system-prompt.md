<!-- markdownlint-disable MD012 MD041 -->
(adapters-recipe-system-prompt)=

# Custom System Prompt (Chat)

Apply a standard system message to chat endpoints for consistent behavior.

```python
from nvidia_eval_commons.core.evaluate import evaluate
from nvidia_eval_commons.api.api_dataclasses import (
    ApiEndpoint, EndpointType, EvaluationConfig, EvaluationTarget, AdapterConfig
)

chat_url = "http://0.0.0.0:8080/v1/chat/completions/"
target = EvaluationTarget(api_endpoint=ApiEndpoint(url=chat_url, type=EndpointType.CHAT))
config = EvaluationConfig(type="mmlu", output_dir="results")

adapter_config = AdapterConfig(
    api_url=target.api_endpoint.url,
    custom_system_prompt="You are a precise, concise assistant.",
)

results = evaluate(target_cfg=target, eval_cfg=config, adapter_cfg=adapter_config)
```

Notes:

- `custom_system_prompt` applies to chat endpoints.
- The adapter replaces the upstream system message when provided.
- Refer to {ref}`adapters-configuration` for available options and defaults.

