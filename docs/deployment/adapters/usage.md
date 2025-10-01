(adapters-usage)=

# Usage

Configure the adapter system using the `AdapterConfig` class with interceptors. Pass the configuration through the `ApiEndpoint.adapter_config` parameter:

```python
from nemo_evaluator import (
    ApiEndpoint,
    EndpointType,
    EvaluationConfig,
    EvaluationTarget,
    evaluate
)
from nemo_evaluator.adapters.adapter_config import AdapterConfig, InterceptorConfig

# Configure adapter with multiple interceptors
adapter_config = AdapterConfig(
    interceptors=[
        # Reasoning interceptor
        InterceptorConfig(
            name="reasoning",
            config={
                "start_reasoning_token": "<think>",
                "end_reasoning_token": "</think>"
            }
        ),
        # System message interceptor
        InterceptorConfig(
            name="system_message",
            config={
                "system_message": "You are a helpful assistant that thinks step by step."
            }
        ),
        # Logging interceptors
        InterceptorConfig(
            name="request_logging",
            config={"max_requests": 50}
        ),
        InterceptorConfig(
            name="response_logging",
            config={"max_responses": 50}
        ),
        # Caching interceptor
        InterceptorConfig(
            name="caching",
            config={
                "cache_dir": "./evaluation_cache"
            }
        ),
        # Progress tracking
        InterceptorConfig(
            name="progress_tracking"
        )
    ]
)

# Configure evaluation target
api_endpoint = ApiEndpoint(
    url="http://localhost:8080/v1/completions/",
    type=EndpointType.COMPLETIONS,
    model_id="megatron_model",
    adapter_config=adapter_config
)
target_config = EvaluationTarget(api_endpoint=api_endpoint)

# Configure evaluation
eval_config = EvaluationConfig(
    type="mmlu_pro",
    params={"limit_samples": 10},
    output_dir="./results/mmlu",
)

# Run evaluation with adapter system
results = evaluate(
    eval_cfg=eval_config,
    target_cfg=target_config
)
```

## YAML Configuration

You can also configure adapters through YAML configuration files:

```yaml
target:
  api_endpoint:
    url: http://localhost:8080/v1/completions/
    type: completions
    model_id: megatron_model
    adapter_config:
      interceptors:
        - name: reasoning
          config:
            start_reasoning_token: "<think>"
            end_reasoning_token: "</think>"
        - name: system_message
          config:
            system_message: "You are a helpful assistant that thinks step by step."
        - name: request_logging
          config:
            max_requests: 50
        - name: response_logging
          config:
            max_responses: 50
        - name: caching
          config:
            cache_dir: ./cache
        - name: progress_tracking

config:
  type: mmlu_pro
  output_dir: ./results
  params:
    limit_samples: 10
```
