(python-api-workflows)=

# Python API

The NeMo Evaluator Python API provides programmatic access to evaluation capabilities through the `nemo-evaluator` package, allowing you to integrate evaluations into existing ML pipelines, automate workflows, and build custom evaluation applications.

## Overview

The Python API is built on top of NeMo Evaluator and provides:

- **Programmatic Evaluation**: Run evaluations from Python code using `evaluate`
- **Configuration Management**: Dynamic configuration and parameter management
- **Adapter Integration**: Access to the full adapter system capabilities
- **Result Processing**: Programmatic access to evaluation results
- **Pipeline Integration**: Seamless integration with existing ML workflows

## Supported PyPI Packages

| Package Name | PyPI URL |
|--------------|----------|
| nvidia-bfcl | https://pypi.org/project/nvidia-bfcl/ |
| nvidia-bigcode-eval | https://pypi.org/project/nvidia-bigcode-eval/ |
| nvidia-crfm-helm | https://pypi.org/project/nvidia-crfm-helm/ |
| nvidia-eval-factory-garak | https://pypi.org/project/nvidia-eval-factory-garak/ |
| nvidia-lm-eval | https://pypi.org/project/nvidia-lm-eval/ |
| nvidia-mtbench-evaluator | https://pypi.org/project/nvidia-mtbench-evaluator/ |
| nvidia-safety-harness | https://pypi.org/project/nvidia-safety-harness/ |
| nvidia-simple-evals | https://pypi.org/project/nvidia-simple-evals/ |
| nvidia-tooltalk | https://pypi.org/project/nvidia-tooltalk/ |
| nvidia-vlmeval | https://pypi.org/project/nvidia-vlmeval/ |

## Basic Usage

### Basic Evaluation

Run a simple evaluation with minimal configuration:

```python
from nemo_evaluator.core.evaluate import evaluate
from nemo_evaluator.api.api_dataclasses import (
    EvaluationConfig, 
    EvaluationTarget, 
    ApiEndpoint, 
    EndpointType, 
    ConfigParams
)

# Configure evaluation
eval_config = EvaluationConfig(
    type="mmlu_pro",
    output_dir="./results",
    params=ConfigParams(
        limit_samples=3,
        temperature=0.0,
        max_new_tokens=1024,
        parallelism=1
    )
)

# Configure target endpoint
target_config = EvaluationTarget(
    api_endpoint=ApiEndpoint(
        model_id="meta/llama-3.1-8b-instruct",
        url="https://integrate.api.nvidia.com/v1/chat/completions",
        type=EndpointType.CHAT,
        api_key="nvapi-your-key-here"
    )
)

# Run evaluation
result = evaluate(eval_cfg=eval_config, target_cfg=target_config)
print(f"Evaluation completed: {result}")
```

### Evaluation With Adapter Interceptors

Use interceptors for advanced features such as caching, logging, and reasoning:

```python
from nemo_evaluator.core.evaluate import evaluate
from nemo_evaluator.api.api_dataclasses import (
    EvaluationConfig,
    EvaluationTarget,
    ApiEndpoint,
    EndpointType,
    ConfigParams
)
from nemo_evaluator.adapters.adapter_config import AdapterConfig, InterceptorConfig

# Configure evaluation
eval_config = EvaluationConfig(
    type="mmlu_pro",
    output_dir="./results",
    params=ConfigParams(
        limit_samples=10,
        temperature=0.0,
        max_new_tokens=1024,
        parallelism=1
    )
)

# Configure adapter with interceptors
adapter_config = AdapterConfig(
    interceptors=[
        # Add custom system message
        InterceptorConfig(
            name="system_message",
            config={
                "system_message": "You are a helpful AI assistant. Please provide accurate and detailed answers."
            }
        ),
        # Enable request logging
        InterceptorConfig(
            name="request_logging",
            config={"max_requests": 50}
        ),
        # Enable caching
        InterceptorConfig(
            name="caching",
            config={
                "cache_dir": "./evaluation_cache",
                "reuse_cached_responses": True
            }
        ),
        # Enable response logging
        InterceptorConfig(
            name="response_logging",
            config={"max_responses": 50}
        ),
        # Enable reasoning extraction
        InterceptorConfig(
            name="reasoning",
            config={
                "start_reasoning_token": "<think>",
                "end_reasoning_token": "</think>"
            }
        ),
        # Enable progress tracking
        InterceptorConfig(
            name="progress_tracking"
        ),
        InterceptorConfig(
            name="endpoint"
        ),
    ]
)

# Configure target with adapter
target_config = EvaluationTarget(
    api_endpoint=ApiEndpoint(
        model_id="meta/llama-3.1-8b-instruct",
        url="https://integrate.api.nvidia.com/v1/chat/completions",
        type=EndpointType.CHAT,
        api_key="nvapi-your-key-here",
        adapter_config=adapter_config
    )
)

# Run evaluation
result = evaluate(eval_cfg=eval_config, target_cfg=target_config)
print(f"Evaluation completed: {result}")
```

## Related Documentation

- **API Reference**: For complete API documentation, refer to the [API Reference](../api.md) page
- **Adapter Configuration**: For detailed interceptor configuration options, refer to the {ref}`adapters-usage` page
- **Interceptor Documentation**: For information about available interceptors, refer to the [Interceptors](../interceptors/index.md) page
