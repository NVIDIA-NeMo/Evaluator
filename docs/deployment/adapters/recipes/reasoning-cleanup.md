<!-- markdownlint-disable MD012 MD041 -->
(adapters-recipe-reasoning)=

# Reasoning Cleanup

Use the reasoning adapter to remove intermediate thoughts from model outputs before scoring.

```python
from nvidia_eval_commons.core.evaluate import evaluate
from nvidia_eval_commons.api.api_dataclasses import (
    ApiEndpoint, EvaluationConfig, EvaluationTarget, AdapterConfig
)

target = EvaluationTarget(api_endpoint={"url": completions_url, "type": "completions"})
config = EvaluationConfig(type="gsm8k", output_dir="results")

adapter_cfg = AdapterConfig(
    api_url=target.api_endpoint.url,
    use_reasoning=True,
    end_reasoning_token="</think>",
)

results = evaluate(target_cfg=target, eval_cfg=config, adapter_cfg=adapter_cfg)
```

Tips:

- Set `end_reasoning_token` to match your model’s delimiter.
- Keep logging caps enabled to inspect sample outputs during setup.


