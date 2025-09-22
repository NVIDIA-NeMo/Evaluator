# Python API

The NeMo Evaluator Python API provides programmatic access to evaluation capabilities through the `nemo-evaluator` package, allowing you to integrate evaluations into existing ML pipelines, automate workflows, and build custom evaluation applications.

## Overview

The Python API is built on top of NeMo Evaluator and provides:

- **Programmatic Evaluation**: Run evaluations from Python code using `evaluate`
- **Configuration Management**: Dynamic configuration and parameter management
- **Adapter Integration**: Access to the full adapter system capabilities
- **Result Processing**: Programmatic access to evaluation results
- **Pipeline Integration**: Seamless integration with existing ML workflows

## Supported PyPi Wheels

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

### Running Evaluations

Below is an example script to run evaluations through a Python script. Please ensure that the `nvidia-simple-evals` package is installed. If you have not installed it yet, you can find the installation instructions [here](https://pypi.org/project/nvidia-simple-evals/).


Firstly, import required packages:
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
```
### Minimal Example
You can use `EvaluationConfig` dataclass to provide your evaluation setup. 
- `type` specifies the type of evaluation to be used (for example, `mmlu_pro`, `ifeval`, and so on)
- `output_dir` indicates where the results, cache and other files should be stored

```python
eval_config = EvaluationConfig(
    type="mmlu_pro",
    output_dir="./results",
    params=ConfigParams(
        limit_samples=3,  # Limit to only 3 samples for testing
    )
)
```

Next, you need to define the EvaluationTarget:

```python
target_config = EvaluationTarget(
    api_endpoint=ApiEndpoint(
        model_id="meta/llama-3.1-8b-instruct",
        url="https://integrate.api.nvidia.com/v1/chat/completions",
        type=EndpointType.CHAT,
        api_key="MY_API_KEY",  # name of the env variable
    )
)
```

In the ApiEndpoint dataclass:
- `model_id` represents the name or identifier of the model
- `url` points to the endpoint URL where the model is hosted
- `type` refers to the type of the endpoint. It should be one of: `EndpointType.CHAT`, `EndpointType.COMPLETIONS`, `EndpointType.VLM`, `EndpointType.EMBEDDING`:
    - `CHAT` endpoint accepts structured input as a sequence of messages (for example, system, user, assistant roles) and returns a model-generated message, enabling controlled multi-turn interactions
    - `COMPLETIONS` endpoint takes a single prompt string and returns a text continuation, typically used for one-shot or single-turn tasks without conversational structure
    - `VLM` endpoint hosts a model that has vision capabilities
    - `EMBEDDING` endpoint hosts an embedding model
- `api_key` is the name of the environment variable that stores the API key required to access the gated endpoint

Having prepared `eval_config` and `target_config`, you can finally run the evaluation: 

```python
print("\n=== Running Evaluation ===")
try:
    result = evaluate(eval_cfg=eval_config, target_cfg=target_config)
    print(f"Evaluation completed successfully: {result}")
except Exception as e:
    print(f"Evaluation failed: {e}")
    print("Note: This is expected if the model endpoint is not accessible")
```

## Advanced Usage

### Direct Configuration in Dataclasses

You can use the `params` field in `EvaluationConfig` dataclass to override the default parameters, such as `temperature` or `max_new_tokens`: 

```python
eval_config = EvaluationConfig(
    type="mmlu_pro",
    output_dir="./results",
    params=ConfigParams(
        limit_samples=3,  # Limit to only 3 samples for testing
        temperature=0.0,
        max_new_tokens=1024,
        parallelism=1  # number of asynchronous requests sent to the model
    )
)
```

You can define the Adapters to be used: 

```python
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
```

To use the adapter configuration, include it in the ApiEndpoint configuration when creating your EvaluationTarget:

```python
target_config = EvaluationTarget(
    api_endpoint=ApiEndpoint(
        model_id="meta/llama-3.1-8b-instruct",
        url="https://integrate.api.nvidia.com/v1/chat/completions",
        type=EndpointType.CHAT,
        api_key="MY_API_KEY",
        adapter_config=adapter_config
    )
)
```

You can run the evaluation, just like before:

```python
print("\n=== Running Evaluation ===")
try:
    result = evaluate(eval_cfg=eval_config, target_cfg=target_config)
    print(f"Evaluation completed successfully: {result}")
except Exception as e:
    print(f"Evaluation failed: {e}")
    print("Note: This is expected if the model endpoint is not accessible")
```

### Using Overrides

Another way to customize your evaluation setup is through the `overrides` field. First, create a `base_config` by converting your existing EvaluationConfig and EvaluationTarget configurations into a dictionary format:

```python
base_config = {
    "config": eval_config.model_dump(),
    "target": target_config.model_dump()
}
```

Prepare your dictionary of overrides:

```python
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
```

And update your base_dict with new values as follows:

```python
from nemo_evaluator.core.utils import deep_update
final_config = deep_update(base_config, overrides, skip_nones=True)

print("\n=== Method 2: Using Overrides ===")
print(f"Base config limit_samples: {eval_config.params.limit_samples}")
print(f"After override limit_samples: {final_config['config']['params']['limit_samples']}")
```

Finally, run the evaluation:

```python
new_config = EvaluationConfig(**final_config['config'])
new_target = EvaluationTarget(**final_config['target'])

print("\n=== Running Evaluation ===")
try:
    result = evaluate(eval_cfg=new_config, target_cfg=new_target)
    print(f"Evaluation completed successfully: {result}")
except Exception as e:
    print(f"Evaluation failed: {e}")
    print("Note: This is expected if the model endpoint is not accessible")
```

### Environment Variable Overrides

You can also set environment variables for dynamic configuration:

```python
import os
os.environ["ADAPTER_PORT"] = "3828" 
os.environ["NEMO_EVALUATOR_LOG_LEVEL"] = "DEBUG"

print("\n=== Environment Variables ===")
print(f"ADAPTER_PORT: {os.environ.get('ADAPTER_PORT')}")
print(f"NEMO_EVALUATOR_LOG_LEVEL: {os.environ.get('NEMO_EVALUATOR_LOG_LEVEL')}")
```

For full API reference, refer to [API](../reference/api.md) page.
