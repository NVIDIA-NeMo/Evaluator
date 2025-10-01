<!-- markdownlint-disable MD012 MD041 -->
(adapters-recipe-logging)=

# Logging Caps

Limit logging volume during evaluations to control overhead.

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
config = EvaluationConfig(type="hellaswag", output_dir="results")

# Configure adapter with logging limits
adapter_cfg = AdapterConfig(
    api_url=completions_url,
    use_request_logging=True,
    use_response_logging=True,
    max_logged_requests=5,      # Limit request logging
    max_logged_responses=5,     # Limit response logging
    log_failed_requests=True    # Always log failures for debugging
)

results = evaluate(target_cfg=target, eval_cfg=config, adapter_cfg=adapter_cfg)
```

Use the following tips to control logging caps:

- Enable logging with `use_request_logging=True` and `use_response_logging=True`
- Set `max_logged_requests` and `max_logged_responses` to limit volume
- Set either value to `0` to disable logging for that direction
- Use low limits for quick debugging, and increase when needed
- `log_failed_requests=True` ensures errors are always captured for debugging

Refer to {ref}`adapters-configuration` for all `AdapterConfig` options and defaults.

