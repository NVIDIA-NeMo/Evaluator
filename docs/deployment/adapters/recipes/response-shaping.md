<!-- markdownlint-disable MD012 MD041 -->
(adapters-recipe-response-shaping)=

# Response Shaping

Normalize provider-specific response formats for evaluators.

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
config = EvaluationConfig(type="lambada", output_dir="results")

# Configure adapter with payload modification for response shaping
adapter_cfg = AdapterConfig(
    api_url=completions_url,
    
    # Use payload modifier to standardize request parameters
    params_to_add={"temperature": 0.0, "max_new_tokens": 100},
    params_to_remove=["max_tokens"],  # Remove conflicting parameters
    
    # Enable logging to monitor transformations
    use_request_logging=True,
    use_response_logging=True,
    max_logged_requests=10
)

results = evaluate(target_cfg=target, eval_cfg=config, adapter_cfg=adapter_cfg)
```

Guidance:

- Use `params_to_add` to standardize request parameters across different endpoints
- Use `params_to_remove` to eliminate conflicting or unsupported parameters
- Use `params_to_rename` to map parameter names between different API formats
- Enable logging to monitor parameter transformations and ensure they work correctly
- Keep transformations minimal to avoid masking upstream issues
- The payload modifier interceptor works with both chat and completions endpoints


