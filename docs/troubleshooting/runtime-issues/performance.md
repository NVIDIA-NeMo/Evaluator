(performance-issues)=

# Performance Issues

Solutions for memory optimization, scaling issues, and resource management problems.

## Speed Optimization

###  Problem: Evaluation too slow

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

### Performance Monitoring

```python
import time
import psutil
import GPUtil

def monitor_evaluation_performance():
    """Monitor system resources during evaluation"""
    start_time = time.time()
    
    # Monitor CPU and memory
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    
    # Monitor GPU if available
    try:
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            print(f"GPU {gpu.id}: {gpu.memoryUtil*100:.1f}% memory, {gpu.load*100:.1f}% utilization")
    except:
        print("No GPU monitoring available")
    
    print(f"CPU: {cpu_percent}%, Memory: {memory.percent}%")
    return time.time() - start_time
```

## Memory Management

###  Problem: GPU memory issues

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

### Memory Optimization Strategies

**Conservative Memory Usage**:

```python
deploy(
    max_batch_size=1,    # Single request processing
    max_input_len=1024,  # Limit input length
    max_output_len=256,  # Limit output length
    enable_cuda_graphs=False  # Disable for lower memory usage
)
```

**Progressive Memory Scaling**:

```python
# Start conservative, then scale up
memory_configs = [
    {"max_batch_size": 1, "max_input_len": 512},
    {"max_batch_size": 2, "max_input_len": 1024},
    {"max_batch_size": 4, "max_input_len": 2048},
    {"max_batch_size": 8, "max_input_len": 4096}
]

for config in memory_configs:
    try:
        deploy(**config)
        print(f" Successfully deployed with: {config}")
        break
    except Exception as e:
        print(f" Failed with {config}: {e}")
        continue
```

## Resource Scaling

### Multi-GPU Deployment

**Tensor Parallelism**:

```python
# Split model across multiple GPUs
deploy(
    tensor_parallelism_size=4,  # Use 4 GPUs for one model instance
    max_batch_size=8
)
```

**Pipeline Parallelism with Ray**:

```python
# Multiple model instances across GPUs  
deploy(
    serving_backend="ray",
    num_replicas=4,      # 4 model instances
    num_gpus=8,         # 8 total GPUs (2 per instance)
    tensor_parallelism_size=2
)
```

### Evaluation Parallelism

**Task-Level Parallelism**:

```python
# Run multiple evaluations in parallel
configs = [
    EvaluationConfig(type="mmlu", params=ConfigParams(parallelism=2)),
    EvaluationConfig(type="hellaswag", params=ConfigParams(parallelism=2)),
    EvaluationConfig(type="arc_challenge", params=ConfigParams(parallelism=2))
]

# Process in parallel (implementation-dependent)
```

**Sample-Level Parallelism**:

```python
# Increase parallelism for individual evaluations
params = ConfigParams(
    parallelism=8,  # Process 8 samples concurrently
    limit_samples=100  # Total samples to process
)
```

## Performance Tuning Guidelines

### Resource Planning

1. **Start Conservative**: Begin with minimal resource allocation
2. **Monitor Utilization**: Track GPU memory, CPU usage, and network I/O
3. **Scale Incrementally**: Increase resources gradually based on bottlenecks
4. **Test with Subsets**: Use `limit_samples` for performance testing

### Bottleneck Identification

**Memory Bottleneck Signs**:

- OOM (Out of Memory) errors
- High GPU memory utilization (>90%)
- Slow garbage collection

**Compute Bottleneck Signs**:

- Low GPU utilization (<50%)
- High CPU usage
- Long inference times

**I/O Bottleneck Signs**:

- Network timeouts
- Slow data loading
- High disk usage

### Performance Testing Framework

```python
def benchmark_configuration(config_params, sample_sizes=[10, 50, 100]):
    """Benchmark different configurations"""
    results = {}
    
    for sample_size in sample_sizes:
        start_time = time.time()
        
        # Run evaluation with limited samples
        config = EvaluationConfig(
            type="mmlu",
            params=ConfigParams(
                limit_samples=sample_size,
                **config_params
            )
        )
        
        # Time the evaluation
        # evaluate(target_cfg, config)  # Implementation-dependent
        
        elapsed = time.time() - start_time
        samples_per_second = sample_size / elapsed
        
        results[sample_size] = {
            "elapsed": elapsed,
            "samples_per_second": samples_per_second
        }
        
        print(f"Samples: {sample_size}, Time: {elapsed:.1f}s, Rate: {samples_per_second:.2f} samples/s")
    
    return results
```
