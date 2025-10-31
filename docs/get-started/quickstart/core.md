(gs-quickstart-core)=
# NeMo Evaluator Core

**Best for**: Developers who need programmatic control

The NeMo Evaluator Core provides direct Python API access for custom configurations and integration into existing Python workflows.

## Prerequisites

- Python environment
- OpenAI-compatible endpoint (hosted or self-deployed) and an API key (if the endpoint is gated)
- Verify endpoint compatibility using our {ref}`deployment-testing-compatibility` guide

## Quick Start

```bash
# 1. Install the nemo-evaluator and nvidia-simple-evals
pip install nemo-evaluator nvidia-simple-evals

# 2. List available benchmarks and tasks
nemo-evaluator ls

# 3. Run evaluation
# Prerequisites: Set your API key
export NGC_API_KEY="nvapi-..."

# Launch using python:
```

```{literalinclude} ../_snippets/core_basic.py
:language: python
:start-after: "# [snippet-start]"
:end-before: "# [snippet-end]"
```

## Complete Working Example

### Using Python API

```{literalinclude} ../_snippets/core_full_example.py
:language: python
:start-after: "# [snippet-start]"
:end-before: "# [snippet-end]"
```

### Using CLI

```{literalinclude} ../_snippets/core_full_cli.sh
:language: bash
:start-after: "# [snippet-start]"
:end-before: "# [snippet-end]"
```

## Key Features

### Programmatic Integration

- Direct Python API access
- Pydantic-based configuration with type hints
- Integration with existing Python workflows

### Evaluation Configuration

- Fine-grained parameter control via `ConfigParams`
- Multiple evaluation types: `mmlu_pro`, `gsm8k`, `hellaswag`, and more
- Configurable sampling, temperature, and token limits

### Endpoint Support

- Chat endpoints (`EndpointType.CHAT`)
- Completion endpoints (`EndpointType.COMPLETIONS`)
- VLM endpoints (`EndpointType.VLM`)
- Embedding endpoints (`EndpointType.EMBEDDING`)

## Advanced Usage Patterns

### Multi-Benchmark Evaluation

```{literalinclude} ../_snippets/core_multi_benchmark.py
:language: python
:start-after: "# [snippet-start]"
:end-before: "# [snippet-end]"
```

### Discovering Installed Benchmarks

```python
from nemo_evaluator import show_available_tasks

# List all installed evaluation tasks
show_available_tasks()
```

:::{tip}
To extend the list of benchmarks install additional harnesses. See the list of evaluation harnesses available as PyPI wheels: {ref}`core-wheels`.
:::

### Using Adapters and Interceptors

For advanced evaluation scenarios, configure the adapter system with interceptors for request/response processing, caching, logging, and more:

```python
from nemo_evaluator.core.evaluate import evaluate
from nemo_evaluator.api.api_dataclasses import (
    ApiEndpoint, EvaluationConfig, EvaluationTarget, ConfigParams, EndpointType
)
from nemo_evaluator.adapters.adapter_config import AdapterConfig, InterceptorConfig

# Configure evaluation target with adapter
api_endpoint = ApiEndpoint(
    url="http://0.0.0.0:8080/v1/completions/",
    type=EndpointType.COMPLETIONS,
    model_id="my_model"
)

# Create adapter configuration with interceptors
api_endpoint.adapter_config = AdapterConfig(
    interceptors=[
        InterceptorConfig(
            name="system_message",
            config={"system_message": "You are a helpful AI assistant. Think step by step."}
        ),
        InterceptorConfig(
            name="request_logging",
            config={"max_requests": 50}
        ),
        InterceptorConfig(
            name="caching",
            config={
                "cache_dir": "./evaluation_cache",
                "reuse_cached_responses": True
            }
        ),
        InterceptorConfig(
            name="endpoint",
        ),
        InterceptorConfig(
            name="response_logging",
            config={"max_responses": 50}
        ),
        InterceptorConfig(
            name="reasoning",
            config={
                "start_reasoning_token": "<think>",
                "end_reasoning_token": "</think>"
            }
        ),
        InterceptorConfig(
            name="progress_tracking",
            config={"progress_tracking_url": "http://localhost:3828/progress"}
        )
    ]
)

target = EvaluationTarget(api_endpoint=api_endpoint)

# Run evaluation with full adapter pipeline
config = EvaluationConfig(
    type="gsm8k",
    output_dir="./results/gsm8k",
    params=ConfigParams(
        limit_samples=10,
        temperature=0.0,
        max_new_tokens=512,
        parallelism=1
    )
)

result = evaluate(eval_cfg=config, target_cfg=target)
print(f"Evaluation completed: {result}")
```

**Available Interceptors:**

- `system_message`: Add custom system prompts to chat requests
- `request_logging`: Log incoming requests for debugging
- `response_logging`: Log outgoing responses for debugging
- `caching`: Cache responses to reduce API costs and speed up reruns
- `reasoning`: Extract chain-of-thought reasoning from model responses
- `progress_tracking`: Track evaluation progress and send updates

For complete adapter documentation, refer to {ref}`adapters-usage`.

## Next Steps

- Integrate into your existing Python workflows
- Run multiple benchmarks in sequence
- Explore available evaluation types with `show_available_tasks()`
- Configure adapters and interceptors for advanced evaluation scenarios
- Consider {ref}`gs-quickstart-launcher` for CLI workflows
- Try {ref}`gs-quickstart-container` for containerized environments
