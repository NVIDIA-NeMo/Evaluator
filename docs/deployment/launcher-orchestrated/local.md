(launcher-orchestrated-local)=

# Local Deployment via Launcher

Deploy and evaluate models on your local machine using NeMo Evaluator Launcher's orchestration. This approach is ideal for development, testing, and small-scale evaluations.

## Overview

Local launcher-orchestrated deployment:
- Deploys models on your local machine
- Manages the complete lifecycle automatically
- Supports all deployment types (vLLM, NIM, SGLang)
- Handles resource allocation and cleanup

Based on PR #108's local execution backend implementation.

## Quick Start

```bash
# Deploy vLLM model and run evaluation locally
nemo-evaluator-launcher run \
    --config-dir examples \
    --config-name local_llama_3_1_8b_instruct \
    -o deployment.model_path=/path/to/your/model \
    -o deployment.type=vllm
```

## Deployment Types

### vLLM Local Deployment

Fast inference with optimized attention mechanisms:

```yaml
# config/local_vllm.yaml
deployment:
  type: vllm
  model_path: /models/llama-3.1-8b-instruct
  port: 8080
  gpu_memory_utilization: 0.9
  max_model_len: 4096
  tensor_parallel_size: 1
  served_model_name: llama-3.1-8b

execution:
  backend: local
  timeout: 3600

evaluation:
  tasks:
    - name: mmlu_pro
      params:
        limit_samples: 100
    - name: gsm8k
      params:
        limit_samples: 50

target:
  api_endpoint:
    url: http://localhost:8080/v1/chat/completions
    model_id: llama-3.1-8b
```

**Key parameters:**
- `gpu_memory_utilization`: Fraction of GPU memory to use (0.8-0.95 recommended)
- `tensor_parallel_size`: Number of GPUs for tensor parallelism
- `max_model_len`: Maximum sequence length

### NIM Local Deployment

NVIDIA Inference Microservices for production-ready serving:

```yaml
# config/local_nim.yaml
deployment:
  type: nim
  model_path: /models/llama-3.1-8b.nemo
  container_image: nvcr.io/nim/llama-3.1-8b-instruct
  port: 8000
  max_batch_size: 8

execution:
  backend: local

evaluation:
  tasks:
    - name: mmlu_pro
    - name: gsm8k

target:
  api_endpoint:
    url: http://localhost:8000/v1/chat/completions
    model_id: llama-3.1-8b-instruct
```

**Key features:**
- Enterprise-grade reliability
- Built-in monitoring and logging
- Optimized NVIDIA container runtime

### SGLang Local Deployment

Structured generation language serving:

```yaml
# config/local_sglang.yaml
deployment:
  type: sglang
  model_path: /models/llama-3.1-8b-instruct
  port: 30000
  tensor_parallel_size: 1
  max_batch_size: 16

execution:
  backend: local

evaluation:
  tasks:
    - name: ifeval  # Structured generation tasks
    - name: gsm8k
```

**Best for:**
- Function calling evaluations
- Structured output tasks
- JSON mode requirements

## Configuration Examples

### Multi-Task Evaluation

```yaml
# config/local_multi_task.yaml
deployment:
  type: vllm
  model_path: /models/llama-3.1-8b-instruct
  gpu_memory_utilization: 0.9

execution:
  backend: local

evaluation:
  tasks:
    - name: mmlu_pro
      params:
        limit_samples: 200
    - name: gsm8k
      params:
        limit_samples: 100
    - name: humaneval
      params:
        limit_samples: 50
    - name: hellaswag
      params:
        limit_samples: 1000
  
  # Run tasks sequentially or in parallel
  parallelism: 1  # Sequential for memory constraints
```

### With Adapters

```yaml
# config/local_with_adapters.yaml
deployment:
  type: vllm
  model_path: /models/reasoning-model
  
execution:
  backend: local

evaluation:
  tasks:
    - name: gsm8k

target:
  api_endpoint:
    url: http://localhost:8080/v1/chat/completions
    model_id: reasoning-model
    adapter_config:
      # Reasoning cleanup
      use_reasoning: true
      start_reasoning_token: "<think>"
      end_reasoning_token: "</think>"
      
      # Caching for repeated evaluations
      use_caching: true
      caching_dir: ./local_cache
      
      # Request logging
      use_request_logging: true
      max_logged_requests: 10
```

### Resource-Constrained Setup

```yaml
# config/local_low_memory.yaml
deployment:
  type: vllm
  model_path: /models/small-model
  gpu_memory_utilization: 0.7  # Conservative memory usage
  max_model_len: 2048          # Shorter sequences
  tensor_parallel_size: 1

execution:
  backend: local
  timeout: 7200  # Longer timeout for slower inference

evaluation:
  tasks:
    - name: mmlu_pro
      params:
        limit_samples: 50  # Smaller sample size
        
  parallelism: 1  # Sequential processing
```

## Command-Line Usage

### Basic Commands

```bash
# Run with default configuration
nemo-evaluator-launcher run --config-dir examples --config-name local_llama_3_1_8b_instruct

# Override deployment type
nemo-evaluator-launcher run \
    --config-dir examples \
    --config-name local_llama_3_1_8b_instruct \
    -o deployment.type=nim

# Override model path and tasks
nemo-evaluator-launcher run \
    --config-dir examples \
    --config-name local_llama_3_1_8b_instruct \
    -o deployment.model_path=/path/to/my/model \
    -o 'evaluation.tasks=["mmlu_pro", "gsm8k"]'
```

### Monitoring and Management

```bash
# Check job status
nemo-evaluator-launcher status <job_id>

# View logs
nemo-evaluator-launcher logs <job_id>

# Kill running job
nemo-evaluator-launcher kill <job_id>

# List all local runs
nemo-evaluator-launcher ls-runs --backend local
```

## Hardware Requirements

### Minimum Requirements
- **GPU**: 1x NVIDIA GPU with 8GB+ VRAM
- **RAM**: 16GB system memory
- **Storage**: 50GB+ for model and results
- **CUDA**: Compatible CUDA version

### Recommended Setup
- **GPU**: 1-2x NVIDIA A100/H100 GPUs
- **RAM**: 32GB+ system memory  
- **Storage**: NVMe SSD with 100GB+ free space
- **Network**: Fast internet for model downloads

### Multi-GPU Configuration

```yaml
# config/local_multi_gpu.yaml
deployment:
  type: vllm
  model_path: /models/large-model
  tensor_parallel_size: 2  # Use 2 GPUs
  gpu_memory_utilization: 0.9

execution:
  backend: local
  
# Launcher will automatically detect and use available GPUs
```

## Performance Optimization

### Memory Management
```yaml
deployment:
  type: vllm
  gpu_memory_utilization: 0.85  # Leave memory for system
  swap_space: 4  # GB of CPU memory for overflow
  cpu_offload: true  # Offload to CPU when needed
```

### Batch Processing
```yaml
deployment:
  type: vllm
  max_batch_size: 32  # Larger batches for throughput
  max_waiting_time: 0.1  # Reduce latency for small batches
```

### Caching Strategy
```yaml
target:
  api_endpoint:
    adapter_config:
      use_caching: true
      caching_dir: /fast/ssd/cache  # Use fast storage
      cache_key_fields: ["messages", "temperature"]
```

## Troubleshooting

### Common Issues

**Out of Memory Errors:**
```bash
# Reduce GPU memory utilization
-o deployment.gpu_memory_utilization=0.7

# Use smaller model or reduce sequence length
-o deployment.max_model_len=2048

# Enable CPU offloading
-o deployment.cpu_offload=true
```

**Port Already in Use:**
```bash
# Change deployment port
-o deployment.port=8081

# Check what's using the port
netstat -tulpn | grep 8080
```

**Model Loading Issues:**
```bash
# Verify model path
ls -la /path/to/model

# Check model format compatibility
file /path/to/model/*

# Use absolute paths
-o deployment.model_path=/absolute/path/to/model
```

**Evaluation Hangs:**
```bash
# Increase timeout
-o execution.timeout=7200

# Check endpoint connectivity
curl http://localhost:8080/health

# Reduce parallelism
-o evaluation.parallelism=1
```

### Debug Mode

```bash
# Run with verbose logging
nemo-evaluator-launcher run \
    --config-dir examples \
    --config-name local_llama_3_1_8b_instruct \
    --log-level DEBUG

# Dry run to check configuration
nemo-evaluator-launcher run \
    --config-dir examples \
    --config-name local_llama_3_1_8b_instruct \
    --dry-run
```

## Best Practices

### Development Workflow
1. **Start small**: Use `limit_samples` for quick iterations
2. **Test deployment**: Verify endpoint health before full evaluation
3. **Monitor resources**: Watch GPU memory and CPU usage
4. **Use caching**: Enable caching for repeated evaluations

### Resource Management
- **GPU memory**: Leave 10-15% headroom for system processes
- **Storage**: Use fast SSD for model loading and caching
- **Cooling**: Ensure adequate cooling for sustained workloads
- **Power**: Consider power consumption for long evaluations

### Configuration Management
- **Version control**: Store configurations in git
- **Environment variables**: Use env vars for paths and secrets
- **Validation**: Test configurations with dry runs
- **Documentation**: Comment complex configurations

## Next Steps

- **Scale up**: Move to {ref}`launcher-orchestrated-slurm` for larger evaluations
- **Cloud deployment**: Try {ref}`launcher-orchestrated-lepton` for cloud resources
- **Custom adapters**: Configure {ref}`adapters`
- **Result analysis**: Set up {ref}`exporters-overview`
