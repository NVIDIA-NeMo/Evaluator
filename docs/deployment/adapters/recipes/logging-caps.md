<!-- markdownlint-disable MD012 MD041 -->
(adapters-recipe-logging)=

# Logging Caps

Limit logging volume during evaluations to control overhead.

```python
from nemo_evaluator import (
    ApiEndpoint, EndpointType, EvaluationConfig, EvaluationTarget, evaluate
)
from nemo_evaluator.adapters.adapter_config import AdapterConfig, InterceptorConfig

# Configure completions endpoint
completions_url = "http://0.0.0.0:8080/v1/completions/"
api_endpoint = ApiEndpoint(url=completions_url, type=EndpointType.COMPLETIONS, model_id="megatron_model")

# Configure adapter with logging limits
api_endpoint.adapter_config = AdapterConfig(
    interceptors=[
        InterceptorConfig(
            name="request_logging",
            enabled=True,
            config={"max_requests": 5}  # Limit request logging
        ),
        InterceptorConfig(
            name="response_logging",
            enabled=True,
            config={"max_responses": 5}  # Limit response logging
        )
    ]
)

target = EvaluationTarget(api_endpoint=api_endpoint)
config = EvaluationConfig(type="hellaswag", output_dir="results")

results = evaluate(target_cfg=target, eval_cfg=config)
```

Use the following tips to control logging caps:

- Include `request_logging` and `response_logging` interceptors to enable logging
- Set `max_requests` and `max_responses` in the interceptor config to limit volume
- Omit or disable interceptors to turn off logging for that direction
- Use low limits for quick debugging, and increase when needed

Refer to {ref}`adapters-configuration` for all `AdapterConfig` options and defaults

