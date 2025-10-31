(configuration-issues)=

# Configuration Issues

Solutions for configuration parameters, tokenizer setup, and endpoint configuration problems.

## Log-Probability Evaluation Issues

###  Problem: Log-probability evaluation fails

**Required Configuration**:

```python
from nemo_evaluator import EvaluationConfig, ConfigParams

config = EvaluationConfig(
    type="arc_challenge",
    params=ConfigParams(
        extra={
            "tokenizer": "/path/to/checkpoint/context/nemo_tokenizer",
            "tokenizer_backend": "huggingface"
        }
    )
)
```

**Common Issues**:

- Missing tokenizer path
- Incorrect tokenizer backend
- Tokenizer version mismatch

### Tokenizer Configuration

**Verify Tokenizer Path**:

```python
import os
tokenizer_path = "/path/to/checkpoint/context/nemo_tokenizer"
if os.path.exists(tokenizer_path):
    print(" Tokenizer path exists")
else:
    print(" Tokenizer path not found")
    # Check alternative locations
```

## Chat vs. Completions Configuration

Before troubleshooting endpoint issues, verify your endpoint supports the required OpenAI API format using our {ref}`deployment-testing-compatibility` guide.

###  Problem: Chat evaluation fails with base model

:::{admonition} Issue
:class: error
Base models don't have chat templates
:::

:::{admonition} Solution
:class: tip
Use completions endpoint instead:

```python
from nemo_evaluator import ApiEndpoint, EvaluationConfig, EndpointType

# Change from chat to completions
api_endpoint = ApiEndpoint(
    url="http://0.0.0.0:8080/v1/completions/",
    type=EndpointType.COMPLETIONS
)

# Use completion-based tasks
config = EvaluationConfig(type="mmlu")
```
:::

### Endpoint Configuration Examples

**For Completions (Base Models)**:

```python
from nemo_evaluator import EvaluationTarget, ApiEndpoint, EndpointType

target_cfg = EvaluationTarget(
    api_endpoint=ApiEndpoint(
        url="http://0.0.0.0:8080/v1/completions/",
        type=EndpointType.COMPLETIONS,
        model_id="megatron_model"
    )
)
```

**For Chat (Instruct Models)**:

```python
from nemo_evaluator import EvaluationTarget, ApiEndpoint, EndpointType

target_cfg = EvaluationTarget(
    api_endpoint=ApiEndpoint(
        url="http://0.0.0.0:8080/v1/chat/completions/",
        type=EndpointType.CHAT,
        model_id="megatron_model"
    )
)
```

## Timeout and Parallelism Issues

###  Problem: Evaluation hangs, times out or crashes with "Too many requests" error

**Diagnosis**:

- Check `parallelism` setting (start with 1)
- Monitor resource usage
- Verify network connectivity

**Solutions**:

```python
from nemo_evaluator import ConfigParams

# Reduce concurrency
params = ConfigParams(
    parallelism=1,  # Start with single-threaded
    limit_samples=10,  # Test with small sample
    request_timeout=600  # Increase timeout for large models (seconds)
)
```


## Configuration Validation

### Pre-Evaluation Checks

```python
from nemo_evaluator import show_available_tasks

# Verify task exists
print("Available tasks:")
show_available_tasks()

# Test endpoint connectivity with curl before running evaluation:
# curl -X POST http://0.0.0.0:8080/v1/completions/ \
#   -H "Content-Type: application/json" \
#   -d '{"prompt": "test", "model": "megatron_model", "max_tokens": 1}'
```

### Common Configuration Issues

- Wrong endpoint type (using `EndpointType.CHAT` for base models or `EndpointType.COMPLETIONS` for instruct models)
- Missing tokenizer (log-probability tasks require explicit tokenizer configuration in `params.extra`)
- High parallelism (starting with `parallelism > 1` can mask underlying issues; use `parallelism=1` for initial debugging)
- Incorrect model ID (model ID must match what the deployment expects)
- Missing output directory (ensure output path exists and is writable)

### Task-Specific Configuration

**MMLU (Choice-Based)**:

```python
from nemo_evaluator import EvaluationConfig, ConfigParams

config = EvaluationConfig(
    type="mmlu",
    params=ConfigParams(
        extra={
            "tokenizer": "/path/to/tokenizer",
            "tokenizer_backend": "huggingface"
        }
    )
)
```

**Generation Tasks**:

```python
from nemo_evaluator import EvaluationConfig, ConfigParams

config = EvaluationConfig(
    type="hellaswag",
    params=ConfigParams(
        max_new_tokens=100,
        limit_samples=50
    )
)
```
