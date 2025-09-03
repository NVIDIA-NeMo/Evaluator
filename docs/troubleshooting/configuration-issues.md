(configuration-issues)=

# Configuration Issues

Solutions for configuration parameters, tokenizer setup, and endpoint configuration problems.

## Log-Probability Evaluation Issues

### ❌ Problem: Log-probability evaluation fails

**Required Configuration**:

```python
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
    print("✅ Tokenizer path exists")
else:
    print("❌ Tokenizer path not found")
    # Check alternative locations
```

## Chat vs. Completions Configuration

### ❌ Problem: Chat evaluation fails with base model

**Issue**: Base models don't have chat templates

**Solution**: Use completions endpoint instead:

```python
# Change from chat to completions
api_endpoint = ApiEndpoint(
    url="http://0.0.0.0:8080/v1/completions/",  # Not chat/completions
    type="completions"  # Not "chat"
)

# Use completion-based tasks
config = EvaluationConfig(type="mmlu")  # Not "mmlu_instruct"
```

### Endpoint Configuration Examples

**For Completions (Base Models)**:

```python
from nemo_eval import EvaluationConfig, TargetConfig, ApiEndpoint

target_cfg = TargetConfig(
    api_endpoint=ApiEndpoint(
        url="http://0.0.0.0:8080/v1/completions/",
        type="completions",
        model_id="megatron_model"
    )
)
```

**For Chat (Instruct Models)**:

```python
target_cfg = TargetConfig(
    api_endpoint=ApiEndpoint(
        url="http://0.0.0.0:8080/v1/chat/completions/",
        type="chat",
        model_id="megatron_model"
    )
)
```

## Timeout and Parallelism Issues

### ❌ Problem: Evaluation hangs or times out

**Diagnosis**:

- Check `parallelism` setting (start with 1)
- Monitor resource usage
- Verify network connectivity

**Solutions**:

```python
# Reduce concurrency
params = ConfigParams(
    parallelism=1,  # Start with single-threaded
    limit_samples=10  # Test with small sample
)

# Increase timeout for large models
# (timeout settings depend on evaluation framework)
```

### Conservative Configuration Template

```python
def create_evaluation_config(task_name, output_dir, limit_samples=None):
    """Template for reliable configuration"""
    return EvaluationConfig(
        type=task_name,
        params=ConfigParams(
            limit_samples=limit_samples,
            parallelism=1,  # Conservative default
            temperature=0,   # Deterministic for testing
        ),
        output_dir=output_dir
    )
```

## Configuration Validation

### Pre-Evaluation Checks

```python
def validate_config(target_cfg, eval_cfg):
    """Test configuration before full evaluation"""
    
    # Test endpoint connectivity
    test_response = requests.post(
        target_cfg.api_endpoint.url,
        json={
            "prompt": "test", 
            "model": target_cfg.api_endpoint.model_id, 
            "max_tokens": 1
        }
    )
    assert test_response.status_code == 200, f"Endpoint failed: {test_response.status_code}"
    
    # Verify task exists
    from nemo_eval.utils.base import find_framework
    framework = find_framework(eval_cfg.type.split('.')[-1])
    print(f"Task {eval_cfg.type} found in {framework}")
    
    return True
```

### Common Configuration Mistakes

1. **Wrong endpoint type**: Using `chat` for base models or `completions` for instruct models
2. **Missing tokenizer**: Log-probability tasks require explicit tokenizer configuration  
3. **High parallelism**: Starting with parallelism > 1 can mask underlying issues
4. **Incorrect model ID**: Model ID must match what the deployment expects
5. **Missing output directory**: Ensure output path exists and is writable

### Task-Specific Configuration

**MMLU (Choice-Based)**:

```python
config = EvaluationConfig(
    type="lm-evaluation-harness.mmlu",
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
config = EvaluationConfig(
    type="simple-evals.hellaswag",
    params=ConfigParams(
        max_tokens=100,
        temperature=0,
        limit_samples=50
    )
)
```
