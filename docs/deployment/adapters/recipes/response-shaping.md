<!-- markdownlint-disable MD012 MD041 -->
(adapters-recipe-response-shaping)=

# Response Shaping

Normalize provider-specific response formats for evaluators.

```python
from nemo_evaluator import (
    ApiEndpoint, EndpointType, EvaluationConfig, EvaluationTarget, evaluate
)
from nemo_evaluator.adapters.adapter_config import AdapterConfig, InterceptorConfig

# Configure completions endpoint
completions_url = "http://0.0.0.0:8080/v1/completions/"
api_endpoint = ApiEndpoint(url=completions_url, type=EndpointType.COMPLETIONS, model_id="megatron_model")

# Configure adapter with payload modification for response shaping
api_endpoint.adapter_config = AdapterConfig(
    interceptors=[
        InterceptorConfig(
            name="payload_modifier",
            enabled=True,
            config={
                "params_to_add": {"temperature": 0.0, "max_new_tokens": 100},
                "params_to_remove": ["max_tokens"]  # Remove conflicting parameters
            }
        ),
        InterceptorConfig(
            name="request_logging",
            enabled=True,
            config={"max_requests": 10}
        ),
        InterceptorConfig(
            name="response_logging",
            enabled=True,
            config={"max_responses": 10}
        )
    ]
)

target = EvaluationTarget(api_endpoint=api_endpoint)
config = EvaluationConfig(type="lambada", output_dir="results")

results = evaluate(target_cfg=target, eval_cfg=config)
```

Guidance:

- Use the `payload_modifier` interceptor to standardize request parameters across different endpoints
- Configure `params_to_add` in the interceptor config to add or override parameters
- Configure `params_to_remove` in the interceptor config to eliminate conflicting or unsupported parameters
- Configure `params_to_rename` in the interceptor config to map parameter names between different API formats
- Use `request_logging` and `response_logging` interceptors to monitor transformations
- Keep transformations minimal to avoid masking upstream issues
- The payload modifier interceptor works with both chat and completions endpoints


