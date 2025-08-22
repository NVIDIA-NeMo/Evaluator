(troubleshooting)=

# Evaluation Troubleshooting

Common issues, solutions, and debugging techniques for NeMo Eval evaluations.

## Quick Diagnostics

### Check System Status

**Verify Model Deployment**:
```python
import requests

# Check health endpoint
health_response = requests.get("http://0.0.0.0:8080/v1/triton_health")
print(f"Health Status: {health_response.status_code}")

# Test completions endpoint
test_payload = {
    "prompt": "Hello",
    "model": "megatron_model", 
    "max_tokens": 5
}
response = requests.post("http://0.0.0.0:8080/v1/completions/", json=test_payload)
print(f"Completions Status: {response.status_code}")
```

**Check Available Tasks**:
```python
from nemo_eval.utils.base import list_available_evaluations

try:
    tasks = list_available_evaluations()
    print("Available frameworks:", list(tasks.keys()))
except ImportError as e:
    print(f"Missing dependency: {e}")
```

## Common Issues & Solutions

### 1. Import and Installation Issues

#### ❌ Problem: `ModuleNotFoundError: No module named 'core_evals'`

**Solution**:
```bash
# Install missing core evaluation framework
pip install nvidia-lm-eval

# For additional frameworks
pip install nvidia-simple-evals nvidia-bigcode nvidia-bfcl
```

#### ❌ Problem: `Framework for task X not found`

**Diagnosis**:
```python
from nemo_eval.utils.base import list_available_evaluations
tasks = list_available_evaluations()
print("Available tasks:", [t for task_list in tasks.values() for t in task_list])
```

**Solution**:
```bash
# Install the framework containing the missing task
pip install nvidia-<framework-name>

# Restart Python session to reload frameworks
```

#### ❌ Problem: `Multiple frameworks found for task X`

**Solution**:
```python
# Use explicit framework specification
config = EvaluationConfig(
    type="lm-evaluation-harness.mmlu",  # Instead of just "mmlu"
    # ... other config
)
```

### 2. Authentication and Access Issues

#### ❌ Problem: `401 Unauthorized` for gated datasets

**Solution**:
```bash
# Set HuggingFace token
export HF_TOKEN=your_huggingface_token

# Or authenticate via CLI
huggingface-cli login

# Verify authentication
huggingface-cli whoami
```

**In Python**:
```python
import os
os.environ["HF_TOKEN"] = "your_token_here"
```

#### ❌ Problem: `403 Forbidden` for specific datasets

**Solution**:
1. Request access to the gated dataset on HuggingFace
2. Wait for approval from dataset maintainers
3. Ensure your token has the required permissions

### 3. Model Deployment Issues

#### ❌ Problem: `Connection refused` or `Server not ready`

**Diagnosis**:
```python
from nemo_eval.utils.base import wait_for_fastapi_server

# Wait for server with detailed logging
success = wait_for_fastapi_server(
    base_url="http://0.0.0.0:8080",
    max_retries=60,
    retry_interval=5
)
print(f"Server ready: {success}")
```

**Solutions**:
1. **Check deployment process**: Ensure deploy command completed successfully
2. **Verify ports**: Confirm no port conflicts (default: 8080 for FastAPI, 8000 for Triton)
3. **Resource availability**: Check GPU memory and system resources
4. **Firewall settings**: Ensure ports are accessible

#### ❌ Problem: Model loads but inference fails

**Diagnosis**:
```python
# Test with minimal request
minimal_test = {
    "prompt": "Hi",
    "model": "megatron_model",
    "max_tokens": 1,
    "temperature": 0
}
```

**Solutions**:
1. **Reduce batch size**: Lower `max_batch_size` in deployment
2. **Check input length**: Verify `max_input_len` is appropriate
3. **Memory issues**: Monitor GPU memory during inference

### 4. Configuration Issues

#### ❌ Problem: Log-probability evaluation fails

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

#### ❌ Problem: Chat evaluation fails with base model

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

#### ❌ Problem: Evaluation hangs or times out

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

### 5. Performance Issues

#### ❌ Problem: Evaluation too slow

**Optimization Strategies**:

1. **Increase Parallelism**:
```python
params = ConfigParams(parallelism=4)  # Adjust based on resources
```

2. **Use Ray Multi-Instance**:
```python
# Deploy with multiple replicas
deploy(
    serving_backend="ray",
    num_replicas=2,
    num_gpus=4
)
```

3. **Limit Evaluation Scope**:
```python
params = ConfigParams(
    limit_samples=100,  # Or 0.1 for 10%
    max_tokens=256      # Reduce response length
)
```

#### ❌ Problem: GPU memory issues

**Solutions**:
1. **Reduce batch size**: Lower `max_batch_size` in deployment
2. **Use tensor parallelism**: Split model across GPUs
3. **Enable optimizations**: Use `enable_cuda_graphs=True`

```python
deploy(
    max_batch_size=4,  # Reduce from default 8
    tensor_parallelism_size=2,  # Use 2 GPUs
    enable_cuda_graphs=True,
    enable_flash_decode=True
)
```

### 6. Results and Output Issues

#### ❌ Problem: Unexpected evaluation results

**Debugging Steps**:

1. **Verify task configuration**:
```python
# Check task parameters
from nvidia_eval_commons.core.entrypoint import show_available_tasks
show_available_tasks()
```

2. **Test with known baseline**:
```python
# Run on small sample first
params = ConfigParams(limit_samples=5)
```

3. **Check model responses**:
```python
# Manual verification
test_response = requests.post(
    "http://0.0.0.0:8080/v1/completions/",
    json={"prompt": "What is 2+2?", "model": "megatron_model", "max_tokens": 10}
)
print(test_response.json())
```

#### ❌ Problem: Missing output files

**Common Causes**:
- Insufficient permissions in output directory
- Evaluation terminated early
- Incorrect output path specification

**Solution**:
```python
import os
output_dir = "./evaluation_results"
os.makedirs(output_dir, exist_ok=True)

config = EvaluationConfig(
    type="mmlu",
    output_dir=os.path.abspath(output_dir)  # Use absolute path
)
```

## Debugging Techniques

### Enable Detailed Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Framework-specific logging
logging.getLogger("nvidia_eval_commons").setLevel(logging.DEBUG)
logging.getLogger("nemo_eval").setLevel(logging.DEBUG)
```

### Validate Configuration

```python
# Test configuration before full evaluation
def validate_config(target_cfg, eval_cfg):
    # Test endpoint connectivity
    test_response = requests.post(
        target_cfg.api_endpoint.url,
        json={"prompt": "test", "model": target_cfg.api_endpoint.model_id, "max_tokens": 1}
    )
    assert test_response.status_code == 200, f"Endpoint failed: {test_response.status_code}"
    
    # Verify task exists
    from nemo_eval.utils.base import find_framework
    framework = find_framework(eval_cfg.type.split('.')[-1])
    print(f"Task {eval_cfg.type} found in {framework}")
    
    return True
```

### Monitor Resources

```bash
# Monitor GPU usage
nvidia-smi -l 1

# Monitor system resources
htop

# Check network connectivity
netstat -tlnp | grep :8080
```

## Prevention Best Practices

### Pre-Evaluation Checklist

- [ ] Model successfully deployed and responding
- [ ] Required evaluation frameworks installed
- [ ] Authentication tokens configured (if needed)
- [ ] Sufficient compute resources available
- [ ] Output directory exists and writable
- [ ] Network connectivity verified

### Testing Strategy

1. **Start Small**: Test with `limit_samples=1` first
2. **Single Threaded**: Begin with `parallelism=1`
3. **Known Tasks**: Use pre-defined configurations initially
4. **Incremental Scale**: Gradually increase scope and parallelism

### Configuration Management

```python
# Template for reliable configuration
def create_evaluation_config(task_name, output_dir, limit_samples=None):
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

## Getting Help

### Log Collection

When reporting issues, include:

1. **System Information**:
```bash
python --version
pip list | grep nvidia
nvidia-smi
```

2. **Configuration Details**:
```python
print(f"Task: {eval_cfg.type}")
print(f"Endpoint: {target_cfg.api_endpoint.url}")
print(f"Model: {target_cfg.api_endpoint.model_id}")
```

3. **Error Messages**: Full stack traces and error logs

### Community Resources

- **GitHub Issues**: [NeMo Eval Issues](https://github.com/NVIDIA-NeMo/Eval/issues)
- **Discussions**: [GitHub Discussions](https://github.com/NVIDIA-NeMo/Eval/discussions)
- **Documentation**: [Complete Documentation](../index.md)

### Professional Support

For enterprise support, contact: [nemo-toolkit@nvidia.com](mailto:nemo-toolkit@nvidia.com)
