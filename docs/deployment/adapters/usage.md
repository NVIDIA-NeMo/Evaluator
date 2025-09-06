(adapters-usage)=

# Usage

To enable the adapters, pass the `AdapterConfig` class to the `evaluate` method from `nemo_evaluator.core.evaluate`. The example below configures the adapter with reasoning, caching, and logging:

```python
from nemo_evaluator.core.evaluate import evaluate
from nemo_evaluator.adapters.adapter_config import AdapterConfig
from nvidia_eval_commons.api.api_dataclasses import (
    ApiEndpoint,
    EvaluationConfig,
    EvaluationTarget
)

# Configure evaluation target
api_endpoint = ApiEndpoint(
    url="http://localhost:8080/v1/completions/",
    model_id="megatron_model"
)
target_config = EvaluationTarget(api_endpoint=api_endpoint)

# Configure evaluation
eval_config = EvaluationConfig(
    type="mmlu_pro",
    params={"limit_samples": 10},
    output_dir="./results/mmlu",
)

# Configure adapter with multiple interceptors
adapter_config = AdapterConfig(
    # Core endpoint configuration
    api_url="http://localhost:8080/v1/completions/",
    
    # Reasoning interceptor
    use_reasoning=True,
    start_reasoning_token="<think>",
    end_reasoning_token="</think>",
    extract_reasoning=True,
    
    # System message interceptor
    use_system_prompt=True,
    custom_system_prompt="You are a helpful assistant that thinks step by step.",
    
    # Logging interceptors
    use_request_logging=True,
    use_response_logging=True,
    max_logged_requests=50,
    max_logged_responses=50,
    log_failed_requests=True,
    
    # Caching interceptor
    use_caching=True,
    caching_dir="./evaluation_cache",
    reuse_cached_responses=True,
    
    # Progress tracking
    use_progress_tracking=True
)

# Run evaluation with adapter system
results = evaluate(
    target_cfg=target_config,
    eval_cfg=eval_config,
    adapter_cfg=adapter_config,
)
```

## YAML Configuration

You can also configure adapters through YAML configuration files:

```yaml
target:
  api_endpoint:
    url: http://localhost:8080/v1/completions/
    model_id: megatron_model
    adapter_config:
      # Reasoning capabilities
      use_reasoning: true
      start_reasoning_token: "<think>"
      end_reasoning_token: "</think>"
      
      # System prompting
      use_system_prompt: true
      custom_system_prompt: "You are a helpful assistant that thinks step by step."
      
      # Logging and caching
      use_request_logging: true
      use_response_logging: true
      use_caching: true
      caching_dir: ./cache
      
      # Progress monitoring
      use_progress_tracking: true

config:
  type: mmlu_pro
  output_dir: ./results
  params:
    limit_samples: 10
```


