<!-- markdownlint-disable MD012 MD041 -->
(adapters-recipe-reasoning)=

# Reasoning Cleanup

Use the reasoning adapter to remove intermediate thoughts from model outputs before scoring.

```python
from nemo_evaluator.core.evaluate import evaluate
from nemo_evaluator.adapters.adapter_config import AdapterConfig
from nvidia_eval_commons.api.api_dataclasses import (
    ApiEndpoint, EndpointType, EvaluationConfig, EvaluationTarget
)

# Configure completions endpoint
completions_url = "http://0.0.0.0:8080/v1/completions/"
api_endpoint = ApiEndpoint(url=completions_url, type=EndpointType.COMPLETIONS, model_id="megatron_model")
target = EvaluationTarget(api_endpoint=api_endpoint)
config = EvaluationConfig(type="gsm8k", output_dir="results")

# Configure adapter with reasoning extraction
adapter_config = AdapterConfig(
    api_url=completions_url,
    use_reasoning=True,
    start_reasoning_token="<think>",
    end_reasoning_token="</think>",
    extract_reasoning=True,
    reasoning_field="reasoning"
)

results = evaluate(target_cfg=target, eval_cfg=config, adapter_cfg=adapter_config)
```

Tips:

- Set both `start_reasoning_token` and `end_reasoning_token` to match your model's delimiters
- `extract_reasoning=True` separates reasoning from final answer into different fields
- `reasoning_field` specifies the field name for extracted reasoning content
- The reasoning interceptor works with both chat and completions endpoints
- Refer to {ref}`adapters-configuration` for all `AdapterConfig` options and defaults.


