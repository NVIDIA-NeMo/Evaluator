<!-- markdownlint-disable MD012 MD041 -->
(adapters-recipe-reasoning)=

# Reasoning Cleanup

Use the reasoning adapter to remove intermediate thoughts from model outputs before scoring.

```python
from nemo_evaluator import (
    ApiEndpoint, EndpointType, EvaluationConfig, EvaluationTarget, evaluate
)
from nemo_evaluator.adapters.adapter_config import AdapterConfig, InterceptorConfig

# Configure completions endpoint
completions_url = "http://0.0.0.0:8080/v1/completions/"
api_endpoint = ApiEndpoint(url=completions_url, type=EndpointType.COMPLETIONS, model_id="megatron_model")

# Configure adapter with reasoning extraction
api_endpoint.adapter_config = AdapterConfig(
    interceptors=[
        InterceptorConfig(
            name="reasoning",
            enabled=True,
            config={
                "start_reasoning_token": "<think>",
                "end_reasoning_token": "</think>"
            }
        )
    ]
)

target = EvaluationTarget(api_endpoint=api_endpoint)
config = EvaluationConfig(type="gsm8k", output_dir="results")

results = evaluate(target_cfg=target, eval_cfg=config)
```

Tips:

- Set both `start_reasoning_token` and `end_reasoning_token` to match your model's delimiters
- The reasoning interceptor removes content between these tokens from the final response before scoring
- Reasoning statistics (word counts, token counts, completion status) are automatically tracked and logged
- The interceptor works with both chat and completions endpoints
- Other parameters include `include_if_not_finished` (default: `True`) and `enable_reasoning_tracking` (default: `True`)
- Refer to {ref}`adapters-configuration` for all interceptor options and defaults


