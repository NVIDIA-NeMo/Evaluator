(deployment-issues)=

# Deployment Issues

Solutions for model deployment problems, server connectivity, and inference failures.

## Connection and Server Issues

### ❌ Problem: `Connection refused` or `Server not ready`

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

## Model Loading and Inference Issues

### ❌ Problem: Model loads but inference fails

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

## Deployment Configuration

### Recommended Settings for Troubleshooting

```python
deploy(
    max_batch_size=4,  # Reduce from default 8
    tensor_parallelism_size=2,  # Use 2 GPUs if available
    enable_cuda_graphs=True,
    enable_flash_decode=True
)
```

### Health Check Verification

```python
import requests

def verify_deployment(base_url="http://0.0.0.0:8080"):
    """Comprehensive deployment verification"""
    
    # Check health endpoint
    try:
        health_response = requests.get(f"{base_url}/v1/triton_health")
        print(f"Health endpoint: {health_response.status_code}")
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server")
        return False
    
    # Test completions endpoint
    test_payload = {
        "prompt": "Test",
        "model": "megatron_model",
        "max_tokens": 1
    }
    
    try:
        comp_response = requests.post(f"{base_url}/v1/completions/", json=test_payload)
        print(f"Completions endpoint: {comp_response.status_code}")
        if comp_response.status_code == 200:
            print("✅ Deployment successful")
            return True
    except Exception as e:
        print(f"❌ Completions test failed: {e}")
    
    return False
```

## Resource Monitoring

### GPU Memory Issues

```bash
# Monitor GPU usage during deployment
nvidia-smi -l 1

# Check for memory leaks
watch -n 1 'nvidia-smi --query-gpu=memory.used,memory.free --format=csv'
```

### System Resource Checks

```bash
# Monitor system resources
htop

# Check port availability
netstat -tlnp | grep :8080

# Verify process status
ps aux | grep python
```

## Common Deployment Patterns

### Single GPU Deployment
```python
deploy(
    serving_backend="fastapi",
    max_batch_size=2,
    max_input_len=2048,
    max_output_len=512
)
```

### Multi-GPU Deployment
```python
deploy(
    serving_backend="ray",
    num_replicas=2,
    num_gpus=4,
    tensor_parallelism_size=2
)
```

### Conservative Resource Usage
```python
deploy(
    max_batch_size=1,  # Single request at a time
    enable_cuda_graphs=False,  # Disable for debugging
    max_input_len=1024,  # Shorter inputs
    max_output_len=256   # Shorter outputs
)
```
