<!-- markdownlint-disable MD012 MD041 -->
(adapters-recipe-system-prompt)=

# Custom System Prompt (Chat)

Apply a standard system message to chat endpoints for consistent behavior.

```python
from nemo_evaluator.core.evaluate import evaluate
from nemo_evaluator.adapters.adapter_config import AdapterConfig
from nvidia_eval_commons.api.api_dataclasses import (
    ApiEndpoint, EndpointType, EvaluationConfig, EvaluationTarget
)

# Configure chat endpoint
chat_url = "http://0.0.0.0:8080/v1/chat/completions/"
api_endpoint = ApiEndpoint(url=chat_url, type=EndpointType.CHAT, model_id="megatron_model")
target = EvaluationTarget(api_endpoint=api_endpoint)
config = EvaluationConfig(type="mmlu_pro", output_dir="results")

# Configure adapter with custom system prompt
adapter_config = AdapterConfig(
    api_url=chat_url,
    use_system_prompt=True,
    custom_system_prompt="You are a precise, concise assistant. Answer questions directly and accurately."
)

results = evaluate(target_cfg=target, eval_cfg=config, adapter_cfg=adapter_config)
```

Notes:

- `use_system_prompt=True` must be set to enable system message injection
- `custom_system_prompt` applies to both chat and completions endpoints
- The adapter injects the system message into the request before sending to the model
- Refer to {ref}`adapters-configuration` for available options and defaults.

