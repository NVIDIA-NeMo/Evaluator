<!-- markdownlint-disable MD012 MD041 -->
(adapters-recipe-response-shaping)=

# Response Shaping

Normalize provider-specific response formats for evaluators.

```python
from nvidia_eval_commons.core.evaluate import evaluate
from nvidia_eval_commons.api.api_dataclasses import (
    ApiEndpoint, EvaluationConfig, EvaluationTarget, AdapterConfig
)

target = EvaluationTarget(api_endpoint={"url": completions_url, "type": "completions"})
config = EvaluationConfig(type="lambada", output_dir="results")

adapter_cfg = AdapterConfig(
    api_url=target.api_endpoint.url,
    # Example: use a custom interceptor in your adapter stack that maps
    # upstream fields into a standard `text` field expected by evaluators.
)

results = evaluate(target_cfg=target, eval_cfg=config, adapter_cfg=adapter_cfg)
```

Guidance:

- Map upstream fields (for example, `choices[0].message.content`) to a single `text` property.
- Keep transformations minimal to avoid masking upstream issues.


