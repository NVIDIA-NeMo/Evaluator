(gs-quickstart-core)=
# NeMo Evaluator Core

**Best for**: Developers who need programmatic control

The NeMo Evaluator Core provides direct Python API access with full adapter features, custom configurations, and integration capabilities for existing workflows.

## Prerequisites

- Python environment with nemo-evaluator installed
- OpenAI-compatible endpoint

## Quick Start

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

# Configure evaluation
eval_config = EvaluationConfig(
    type="mmlu_pro",
    output_dir="./results",
    params=ConfigParams(
        limit_samples=10,  # Small test run
        temperature=0.0,
        max_new_tokens=1024,
        parallelism=1
    )
)

# Configure target endpoint with adapter features
adapter_config = AdapterConfig(
    use_request_logging=True,      # Log all requests
    use_response_logging=True,     # Log all responses
    use_caching=True,              # Enable caching
    caching_dir="./cache"          # Cache directory
)

target_config = EvaluationTarget(
    api_endpoint=ApiEndpoint(
        url="https://integrate.api.nvidia.com/v1/chat/completions",
        model_id="meta/llama-3.1-8b-instruct",
        api_key="your_api_key_here",
        type=EndpointType.CHAT,
        adapter_config=adapter_config
    )
)

# Run evaluation
result = evaluate(eval_cfg=eval_config, target_cfg=target_config)
print(f"Evaluation completed: {result}")
```

## Complete Working Example

Here's a comprehensive example with error handling and environment setup:

```python
import os
from nemo_evaluator.core.evaluate import evaluate
from nemo_evaluator.api.api_dataclasses import (
    EvaluationConfig, EvaluationTarget, ApiEndpoint, 
    EndpointType, ConfigParams, AdapterConfig
)

# Set up environment
os.environ["NGC_API_KEY"] = "nvapi-your-key-here"

# Configure evaluation with comprehensive settings
eval_config = EvaluationConfig(
    type="mmlu_pro",
    output_dir="./results",
    params=ConfigParams(
        limit_samples=3,           # Small test for quick validation
        temperature=0.0,           # Deterministic results
        max_new_tokens=1024,
        parallelism=1,
        max_retries=5
    )
)

# Advanced adapter configuration
adapter_config = AdapterConfig(
    use_request_logging=True,
    use_response_logging=True,
    use_caching=True,
    caching_dir="./cache",
    save_responses=True,
    log_failed_requests=True
)

target_config = EvaluationTarget(
    api_endpoint=ApiEndpoint(
        model_id="meta/llama-3.1-8b-instruct",
        url="https://integrate.api.nvidia.com/v1/chat/completions",
        type=EndpointType.CHAT,
        api_key=os.environ["NGC_API_KEY"],
        adapter_config=adapter_config
    )
)

# Run evaluation with error handling
try:
    print("Starting evaluation...")
    result = evaluate(eval_cfg=eval_config, target_cfg=target_config)
    print(f"✅ Evaluation completed successfully!")
    print(f"Results saved to: {eval_config.output_dir}")
    print(f"Result summary: {result}")
except Exception as e:
    print(f"❌ Evaluation failed: {e}")
    print("Check your API key and endpoint configuration")
```

## Key Features

### Full Adapter Control

- Request and response logging
- Response caching for efficiency
- Custom system prompts
- Reasoning extraction capabilities
- Progress tracking integration

### Programmatic Integration

- Direct Python API access
- Custom configuration objects
- Error handling and retry logic
- Integration with existing Python workflows

### Advanced Configuration

- Fine-grained parameter control
- Various evaluation types in sequence
- Custom output formatting
- Environment variable integration

### Development Benefits

- Type hints for better IDE support
- Comprehensive error messages
- Debugging capabilities
- Unit test integration

## Advanced Usage Patterns

### Multi-Benchmark Evaluation

```python
benchmarks = ["gsm8k", "hellaswag", "arc_easy"]
results = {}

for benchmark in benchmarks:
    config = EvaluationConfig(
        type=benchmark,
        output_dir=f"./results/{benchmark}",
        params=ConfigParams(limit_samples=10)
    )
    
    result = evaluate(eval_cfg=config, target_cfg=target_config)
    results[benchmark] = result
```

### Custom Adapter Configuration

```python
# Advanced adapter with reasoning extraction
adapter_config = AdapterConfig(
    use_reasoning=True,
    start_reasoning_token="<think>",
    end_reasoning_token="</think>",
    custom_system_prompt="You are a helpful AI assistant. Think step by step.",
    use_progress_tracking=True,
    progress_tracking_url="http://localhost:3828/progress"
)
```

## Next Steps

- Integrate into your existing Python workflows
- Explore advanced adapter features
- Build custom evaluation pipelines
- Consider {ref}`gs-quickstart-launcher` for simpler CLI workflows
- Try {ref}`gs-quickstart-container` for containerized environments
