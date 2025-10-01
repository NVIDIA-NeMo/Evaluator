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

## Supported PyPi Wheels:

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

### Run Evaluations

Below is an example script to run evaluations through a Python script. Please ensure that the `nvidia-simple-evals` package is installed. If you haven't installed it yet, you can find the installation instructions [here](https://pypi.org/project/nvidia-simple-evals/).

```python
from nemo_evaluator.core.evaluate import evaluate
from nemo_evaluator.api.api_dataclasses import (
    EvaluationConfig, 
    EvaluationTarget, 
    ApiEndpoint, 
    EndpointType, 
    ConfigParams,
    AdapterConfig
)

# Method 1: Direct configuration in dataclasses
eval_config = EvaluationConfig(
    type="mmlu_pro",
    output_dir="./results",
    params=ConfigParams(
        limit_samples=3,  # Limit to only 3 samples for testing
        temperature=0.0,
        max_new_tokens=1024,
        parallelism=1
    )
)

# Create adapter configuration for advanced features
adapter_config = AdapterConfig(
    use_request_logging=True,      # Log all requests
    use_response_logging=True,     # Log all responses
    use_caching=True,              # Enable caching
    caching_dir="/tmp/cache",      # Cache directory
    use_reasoning=True,            # Enable reasoning capabilities
    save_responses=True,           # Save responses to disk
    log_failed_requests=True,      # Log failed requests
    use_system_prompt=True,        # Use custom system prompt
    custom_system_prompt="You are a helpful AI assistant. Please provide accurate and detailed answers."
)

target_config = EvaluationTarget(
    api_endpoint=ApiEndpoint(
        model_id="meta/llama-3.1-8b-instruct",
        url="https://integrate.api.nvidia.com/v1/chat/completions",
        type=EndpointType.CHAT,
        api_key="MY_API_KEY",
        adapter_config=adapter_config
    )
)

print("=== Method 1: Direct Configuration ===")
print(f"Evaluation config: {eval_config}")
print(f"Target config: {target_config}")

# Method 2: Using overrides (similar to CLI --overrides)
# This is useful when you want to override specific values programmatically
overrides = {
    "config": {
        "params": {
            "limit_samples": 5,  # Override to 5 samples
            "temperature": 0.1,   # Override temperature
        }
    },
    "target": {
        "api_endpoint": {
            "adapter_config": {
                "use_caching": True,
                "caching_dir": "/tmp/custom_cache",
                "use_request_logging": True
            }
        }
    }
}

# Apply overrides to the base configuration
from nemo_evaluator.core.utils import deep_update

# Create a base config dict
base_config = {
    "config": eval_config.model_dump(),
    "target": target_config.model_dump()
}

# Apply overrides
final_config = deep_update(base_config, overrides, skip_nones=True)

print("\n=== Method 2: Using Overrides ===")
print(f"Base config limit_samples: {eval_config.params.limit_samples}")
print(f"After override limit_samples: {final_config['config']['params']['limit_samples']}")

# Method 3: Environment variable overrides
# You can also set environment variables for dynamic configuration
import os
os.environ["ADAPTER_PORT"] = "3828"
os.environ["NEMO_EVALUATOR_LOG_LEVEL"] = "DEBUG"

print("\n=== Environment Variables ===")
print(f"ADAPTER_PORT: {os.environ.get('ADAPTER_PORT')}")
print(f"NEMO_EVALUATOR_LOG_LEVEL: {os.environ.get('NEMO_EVALUATOR_LOG_LEVEL')}")

# Run the evaluation with the direct configuration
print("\n=== Running Evaluation ===")
try:
    result = evaluate(eval_cfg=eval_config, target_cfg=target_config)
    print(f"Evaluation completed successfully: {result}")
except Exception as e:
    print(f"Evaluation failed: {e}")
    print("Note: This is expected if the model endpoint is not accessible")
```

For full API reference, see [API](../api.md) page.
