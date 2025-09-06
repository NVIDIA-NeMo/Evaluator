(debugging-guide)=

# Debugging Guide

Debugging techniques, monitoring strategies, and best practices for preventing evaluation issues.

## Debugging Techniques

### Enable Detailed Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Framework-specific logging
logging.getLogger("nvidia_eval_commons").setLevel(logging.DEBUG)
logging.getLogger("nemo_eval").setLevel(logging.DEBUG)
```

### Step-by-Step Debugging Process

1. **Start with Minimal Configuration**:

```python
# Test with smallest possible configuration
config = EvaluationConfig(
    type="mmlu",
    params=ConfigParams(
        limit_samples=1,
        parallelism=1,
        temperature=0
    )
)
```

2. **Verify Each Component Individually**:

```python
# Test model endpoint
response = requests.post(
    "http://0.0.0.0:8080/v1/completions/",
    json={"prompt": "test", "model": "megatron_model", "max_tokens": 1}
)
assert response.status_code == 200

# Test task availability
from nemo_eval.utils.base import find_framework
framework = find_framework("mmlu")
print(f"MMLU found in: {framework}")
```

3. **Gradually Increase Complexity**:

```python
# Progressive testing
test_configs = [
    {"limit_samples": 1, "parallelism": 1},
    {"limit_samples": 5, "parallelism": 1},
    {"limit_samples": 5, "parallelism": 2},
    {"limit_samples": 20, "parallelism": 4}
]
```

## Monitoring and Observability

### Resource Monitoring

```bash
# Monitor GPU usage
nvidia-smi -l 1

# Monitor system resources
htop

# Check network connectivity
netstat -tlnp | grep :8080

# Monitor disk I/O
iotop
```

### Application Monitoring

```python
def monitor_evaluation_progress(log_file="evaluation.log"):
    """Monitor evaluation progress and resource usage"""
    import psutil
    import time
    
    start_time = time.time()
    
    with open(log_file, "w") as f:
        while True:  # Until evaluation completes
            # System resources
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # GPU resources (if available)
            gpu_info = []
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                for gpu in gpus:
                    gpu_info.append(f"GPU{gpu.id}: {gpu.memoryUtil*100:.1f}%mem {gpu.load*100:.1f}%util")
            except:
                gpu_info = ["No GPU monitoring"]
            
            # Log status
            elapsed = time.time() - start_time
            status = f"T+{elapsed:.0f}s CPU:{cpu_percent:.1f}% MEM:{memory.percent:.1f}% {' '.join(gpu_info)}"
            
            print(status)
            f.write(f"{status}\n")
            f.flush()
            
            time.sleep(10)  # Log every 10 seconds
```

## Error Pattern Analysis

### Common Error Patterns and Solutions

**Connection Errors**:

```python
# Pattern: "Connection refused", "Server not ready"
# Solution: Check deployment status and wait for server
from nemo_eval.utils.base import wait_for_fastapi_server
wait_for_fastapi_server("http://0.0.0.0:8080", max_retries=60)
```

**Memory Errors**:

```python
# Pattern: "CUDA out of memory", "RuntimeError: out of memory"
# Solution: Reduce batch size and input length
deploy(max_batch_size=1, max_input_len=512)
```

**Import Errors**:

```python
# Pattern: "ModuleNotFoundError", "No module named"
# Solution: Install missing frameworks
# pip install nvidia-lm-eval nvidia-simple-evals
```

### Diagnostic Scripts

```python
def comprehensive_health_check():
    """Complete system health check before evaluation"""
    checks = {}
    
    # Check 1: Python environment
    try:
        import nemo_eval
        checks["nemo_eval"] = " Installed"
    except ImportError:
        checks["nemo_eval"] = " Not installed"
    
    # Check 2: Frameworks
    frameworks = ["nvidia_lm_eval", "nvidia_simple_evals", "nvidia_bigcode"]
    for framework in frameworks:
        try:
            __import__(framework)
            checks[framework] = " Available"
        except ImportError:
            checks[framework] = " Missing"
    
    # Check 3: Server connectivity
    try:
        response = requests.get("http://0.0.0.0:8080/v1/triton_health")
        checks["server"] = f" Responding ({response.status_code})"
    except:
        checks["server"] = " Not accessible"
    
    # Check 4: GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            checks["gpu"] = f" {gpu_count} GPU(s) available"
        else:
            checks["gpu"] = " No CUDA GPUs"
    except:
        checks["gpu"] = " PyTorch not available"
    
    # Print results
    print(" Health Check Results:")
    for component, status in checks.items():
        print(f"  {component:20s}: {status}")
    
    return checks
```

## Best Practices for Prevention

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

### Error Recovery Patterns

```python
def robust_evaluation_wrapper(target_cfg, eval_cfg, max_retries=3):
    """Wrapper with automatic retry and error recovery"""
    
    for attempt in range(max_retries):
        try:
            # Run evaluation
            result = evaluate(target_cfg, eval_cfg)
            return result
        
        except ConnectionError:
            print(f"Attempt {attempt+1}: Connection failed, retrying...")
            time.sleep(30)  # Wait for server recovery
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"Attempt {attempt+1}: OOM error, reducing batch size...")
                # Reduce resource usage
                if hasattr(eval_cfg.params, 'parallelism'):
                    eval_cfg.params.parallelism = max(1, eval_cfg.params.parallelism // 2)
            else:
                raise  # Re-raise if not memory related
                
        except Exception as e:
            print(f"Attempt {attempt+1}: Unexpected error: {e}")
            if attempt == max_retries - 1:
                raise  # Re-raise on final attempt
    
    raise RuntimeError(f"Evaluation failed after {max_retries} attempts")
```

### Configuration Validation

```python
def validate_before_evaluation(target_cfg, eval_cfg):
    """Comprehensive pre-evaluation validation"""
    
    validations = []
    
    # Validate endpoint
    try:
        response = requests.post(
            target_cfg.api_endpoint.url,
            json={"prompt": "test", "model": target_cfg.api_endpoint.model_id, "max_tokens": 1},
            timeout=30
        )
        assert response.status_code == 200
        validations.append(" Endpoint responding")
    except Exception as e:
        validations.append(f" Endpoint failed: {e}")
        return False, validations
    
    # Validate task
    try:
        from nemo_eval.utils.base import find_framework
        framework = find_framework(eval_cfg.type.split('.')[-1])
        validations.append(f" Task found in {framework}")
    except Exception as e:
        validations.append(f" Task validation failed: {e}")
        return False, validations
    
    # Validate resources
    try:
        import psutil
        memory = psutil.virtual_memory()
        if memory.percent > 90:
            validations.append("  High memory usage detected")
        else:
            validations.append(" Memory usage acceptable")
    except:
        validations.append("  Could not check memory usage")
    
    return True, validations
```
