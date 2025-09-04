<!-- markdownlint-disable MD012 MD041 -->
(adapters-recipe-reasoning)=

# Reasoning Cleanup

Use the reasoning adapter to remove intermediate thoughts from model outputs before scoring.

```python
from nvidia_eval_commons.core.evaluate import evaluate
from nvidia_eval_commons.api.api_dataclasses import (
    ApiEndpoint, EndpointType, EvaluationConfig, EvaluationTarget, AdapterConfig
)

completions_url = "http://0.0.0.0:8080/v1/completions"
target = EvaluationTarget(api_endpoint=ApiEndpoint(url=completions_url, type=EndpointType.COMPLETIONS))
config = EvaluationConfig(type="gsm8k", output_dir="results")

adapter_config = AdapterConfig(
    api_url=target.api_endpoint.url,
    use_reasoning=True,
    end_reasoning_token="</think>",
)

results = evaluate(target_cfg=target, eval_cfg=config, adapter_cfg=adapter_config)
```

Tips:

- Set `end_reasoning_token` to match your model’s delimiter.
- Refer to {ref}`adapters-configuration` for all `AdapterConfig` options and defaults.


