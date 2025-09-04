<!-- markdownlint-disable MD012 MD041 -->
(adapters-recipe-logging)=

# Logging Caps

Limit logging volume during evaluations to control overhead.

```python
from nvidia_eval_commons.core.evaluate import evaluate
from nvidia_eval_commons.api.api_dataclasses import (
    ApiEndpoint, EvaluationConfig, EvaluationTarget, AdapterConfig
)

target = EvaluationTarget(api_endpoint={"url": completions_url, "type": "completions"})
config = EvaluationConfig(type="hellaswag", output_dir="results")

adapter_cfg = AdapterConfig(
    api_url=target.api_endpoint.url,
    max_logged_requests=5,
    max_logged_responses=5,
)

results = evaluate(target_cfg=target, eval_cfg=config, adapter_cfg=adapter_cfg)
```

Tips:

- Set either value to `0` to disable logging for that direction.
- Use small caps for quick debugging; increase when necessary.

