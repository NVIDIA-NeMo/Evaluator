# Common Issues & Solutions

Practical troubleshooting guide based on frequent user issues and test patterns in NeMo Eval deployments and evaluations.

## Deployment Issues

### Server Not Ready

**Symptom**: `Server is not ready. Please look at the deploy process log for the error`

**Source**: Based on test pattern in `tests/functional_tests/test_evaluation.py:114`

**Diagnosis**:
```python
from nemo_eval.utils.base import wait_for_fastapi_server

# Test server readiness with detailed logging
server_ready = wait_for_fastapi_server(
    base_url="http://0.0.0.0:8080",
    max_retries=60,  # 10 minutes 
    retry_interval=10
)
if not server_ready:
    print(" Server failed to start within timeout")
```

**Solutions**:

1. **Check GPU Memory**:
```bash
# Monitor GPU usage
nvidia-smi -l 1

# If OOM, reduce batch size or sequence length
deploy(
    nemo_checkpoint=checkpoint_path,
    max_batch_size=2,        # Reduce from default 8
    max_input_len=2048       # Reduce from default 4096
)
```

2. **Verify Checkpoint Path**:
```python
import os
checkpoint_path = "/path/to/checkpoint"
if not os.path.exists(checkpoint_path):
    print(f" Checkpoint not found: {checkpoint_path}")
else:
    print(f" Checkpoint found: {checkpoint_path}")
```

3. **Check Port Availability**:
```bash
# Check if port is in use
netstat -tulpn | grep 8080

# Kill existing processes if needed  
sudo fuser -k 8080/tcp

# Or use different port
deploy(checkpoint_path, server_port=8081)
```

---

## Evaluation Configuration Issues

### Invalid Configuration Parameters

**Symptom**: Configuration validation errors or unexpected parameter behavior

**Source**: Based on test patterns in `tests/unit_tests/test_eval.py:26-46`

**Common Valid Configurations**:
```python
#  Correct parameter combinations
valid_configs = [
    {"top_p": 0.1, "temperature": 0.001},
    {"limit_samples": 10},
    {"limit_samples": 0.1},  # Fraction of dataset
    {"max_new_tokens": 64},
    {"max_retries": 10, "parallelism": 16, "request_timeout": 100},
    {"task": "my_task", "extra": {"num_fewshot": 5}}
]

for params in valid_configs:
    config = EvaluationConfig(type="mmlu", params=params)
    print(f" Valid config: {params}")
```

**Parameter Validation**:
```python
def validate_eval_config(params: dict) -> bool:
    """Validate evaluation configuration parameters."""
    try:
        config = EvaluationConfig(type="test", params=params)
        print(f" Configuration valid: {params}")
        return True
    except Exception as e:
        print(f" Configuration invalid: {e}")
        return False

# Test your configuration
my_params = {"limit_samples": 100, "temperature": 0.7}
validate_eval_config(my_params)
```

### Tokenizer Configuration Issues

**Symptom**: Tokenizer-related errors in log-probability evaluations

**Source**: Test pattern from `tests/functional_tests/test_evaluation.py:155-164`

**Solution**:
```python
# For tasks requiring specific tokenizers
eval_params = {
    "limit_samples": 10,
    "extra": {
        "tokenizer_backend": "huggingface",
        "tokenizer": "/path/to/tokenizer",  # Path to tokenizer directory
    },
}

eval_config = EvaluationConfig(
    type="arc_challenge", 
    params=ConfigParams(**eval_params)
)
```

---

## Environment Configuration

### Dataset Access Issues

**Symptom**: Unable to download or access evaluation datasets

**Source**: Environment setup from `tests/functional_tests/test_evaluation.py:107-109`

**Solution**:
```python
import os

# Configure Hugging Face environment
os.environ["HF_DATASETS_OFFLINE"] = "1"           # Use cached datasets only
os.environ["HF_HOME"] = "/path/to/hf_cache"       # Set cache location
os.environ["HF_DATASETS_CACHE"] = f"{os.environ['HF_HOME']}/datasets"

# For gated datasets, set token
os.environ["HF_TOKEN"] = "your_huggingface_token"
```

**Verify Dataset Access**:
```python
def check_dataset_access():
    """Check if required datasets are accessible."""
    try:
        from nemo_eval.utils.base import list_available_evaluations
        evals = list_available_evaluations()
        print(f" Available evaluations: {len(evals)} frameworks")
        return True
    except ImportError as e:
        print(f" Missing evaluation packages: {e}")
        return False
    except Exception as e:
        print(f" Dataset access error: {e}")
        return False

check_dataset_access()
```

### CUDA Visibility Issues

**Source**: GPU configuration from `tests/functional_tests/test_evaluation.py:106`

**Solution**:
```bash
# Set specific GPU(s)
export CUDA_VISIBLE_DEVICES=0

# For multi-GPU setup
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Verify in Python
import torch
print(f" CUDA available: {torch.cuda.is_available()}")
print(f" GPU count: {torch.cuda.device_count()}")
```

---

## Request Timeout Issues

### Long-Running Evaluations

**Symptom**: Request timeouts during evaluation

**Source**: Timeout configuration from `tests/functional_tests/test_evaluation.py:122`

**Solution**:
```python
# Increase timeout for complex evaluations
eval_params = {
    "limit_samples": 100,
    "request_timeout": 360,    # 6 minutes per request
    "parallelism": 1,          # Reduce concurrency to avoid overload
    "max_retries": 5           # Retry failed requests
}

eval_config = EvaluationConfig(
    type="complex_reasoning_task",
    params=ConfigParams(**eval_params)
)
```

**Dynamic Timeout Configuration**:
```python
def get_timeout_for_task(task_type: str, model_size: str) -> int:
    """Get appropriate timeout based on task and model complexity."""
    base_timeouts = {
        "mmlu": 60,
        "gsm8k": 120, 
        "humaneval": 180,
        "safety": 240
    }
    
    multipliers = {
        "7b": 1.0,
        "13b": 1.5,
        "70b": 3.0,
        "405b": 5.0
    }
    
    base = base_timeouts.get(task_type, 120)
    multiplier = multipliers.get(model_size, 1.0)
    return int(base * multiplier)

# Usage
timeout = get_timeout_for_task("gsm8k", "70b")  # 180 seconds
eval_params = {"request_timeout": timeout}
```

---

## Import and Dependency Issues

### Package Import Errors

**Symptom**: `ImportError: Please ensure that core_evals is installed`

**Source**: Error handling from `src/nemo_eval/utils/base.py:122-123`

**Diagnosis**:
```python
def diagnose_imports():
    """Diagnose common import issues."""
    packages_to_check = [
        "core_evals",
        "nvidia_eval_commons", 
        "nemo_eval",
        "torch",
        "transformers"
    ]
    
    for package in packages_to_check:
        try:
            __import__(package)
            print(f" {package}")
        except ImportError as e:
            print(f" {package}: {e}")

diagnose_imports()
```

**Solutions**:
```bash
# Install missing core evaluation package
pip install nvidia-lm-eval

# Verify installation
python -c "import core_evals; print(' core_evals installed')"

# For additional harnesses
pip install nvidia-simple-evals==25.7.1
pip install nvidia-bigcode-eval==25.7.1
pip install nvidia-bfcl==25.7.1
```

### Version Compatibility Issues

**Solution**:
```bash
# Check package versions
pip list | grep nvidia

# Install compatible versions (from pyproject.toml)
pip install nvidia-lm-eval==25.7.1
pip install nvidia-eval-commons~=1.0.0
```

---

## Performance Issues

### Low Evaluation Throughput

**Symptom**: Evaluations taking unexpectedly long

**Diagnosis**:
```python
import time
from nvidia_eval_commons.core.evaluate import evaluate

# Time your evaluation
start_time = time.time()
results = evaluate(target_cfg=target, eval_cfg=config)
duration = time.time() - start_time

samples = config.params.limit_samples or 1000
throughput = samples / duration
print(f" Throughput: {throughput:.2f} samples/second")
```

**Optimization**:
```python
# High-throughput configuration
optimized_params = {
    "parallelism": 8,              # Increase concurrent requests
    "max_new_tokens": 128,         # Limit response length
    "temperature": 0.0,            # Deterministic (faster)
    "request_timeout": 60,         # Reasonable timeout
    "limit_samples": 100           # Start with subset for testing
}

# For development/testing
dev_params = {
    "limit_samples": 10,           # Quick validation
    "parallelism": 1,              # Sequential for debugging
    "temperature": 0.0             # Consistent results
}
```

### Memory Issues During Evaluation

**Symptom**: Out of memory errors during long evaluations

**Solution**:
```python
# Memory-conscious configuration
memory_efficient_params = {
    "parallelism": 2,              # Reduce concurrent requests
    "max_new_tokens": 256,         # Limit generation length
    "limit_samples": 50,           # Process in smaller batches
    "request_timeout": 120         # Allow more time per request
}

# Clear cache between evaluations
import torch
torch.cuda.empty_cache()
```

---

## Debugging Techniques

### Comprehensive Health Check

```python
def comprehensive_health_check():
    """Perform comprehensive system health check."""
    print(" NeMo Eval Health Check")
    print("=" * 50)
    
    # 1. Import check
    try:
        from nemo_eval.utils.base import list_available_evaluations
        evals = list_available_evaluations()
        print(f" Available frameworks: {len(evals)}")
    except Exception as e:
        print(f" Import error: {e}")
        return False
    
    # 2. Server check
    try:
        from nemo_eval.utils.base import wait_for_fastapi_server
        ready = wait_for_fastapi_server(
            base_url="http://0.0.0.0:8080",
            max_retries=5,
            retry_interval=2
        )
        print(f" Server ready: {ready}")
    except Exception as e:
        print(f" Server check failed: {e}")
    
    # 3. CUDA check
    try:
        import torch
        print(f" CUDA available: {torch.cuda.is_available()}")
        print(f" GPU count: {torch.cuda.device_count()}")
    except Exception as e:
        print(f" CUDA error: {e}")
    
    # 4. Environment check
    import os
    env_vars = ["HF_TOKEN", "CUDA_VISIBLE_DEVICES", "HF_HOME"]
    for var in env_vars:
        value = os.environ.get(var, "Not set")
        print(f" {var}: {value}")
    
    return True

# Run health check
comprehensive_health_check()
```

### Minimal Working Example

```python
def minimal_evaluation_test():
    """Minimal test to verify evaluation pipeline."""
    try:
        from nvidia_eval_commons.core.evaluate import evaluate
        from nvidia_eval_commons.api.api_dataclasses import (
            ApiEndpoint, EvaluationConfig, EvaluationTarget, ConfigParams
        )
        
        # Minimal configuration
        api_endpoint = ApiEndpoint(
            url="http://0.0.0.0:8080/v1/completions/",
            type="completions",
            model_id="megatron_model"
        )
        
        target = EvaluationTarget(api_endpoint=api_endpoint)
        config = EvaluationConfig(
            type="gsm8k",
            params=ConfigParams(limit_samples=1)  # Single sample test
        )
        
        print(" Running minimal evaluation test...")
        results = evaluate(target_cfg=target, eval_cfg=config)
        print(" Minimal evaluation successful")
        return True
        
    except Exception as e:
        print(f" Minimal evaluation failed: {e}")
        return False

# Test minimal setup
minimal_evaluation_test()
```

---

**Source**: `tests/functional_tests/test_evaluation.py`, `tests/unit_tests/test_eval.py`  
**Evidence**: Troubleshooting patterns derived from actual test implementations and common failure points
