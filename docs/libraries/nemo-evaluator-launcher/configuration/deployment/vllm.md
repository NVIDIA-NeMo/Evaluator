(deployment-vllm)=

# vLLM Deployment

vLLM is a high-performance library for LLM inference and serving, optimized for maximum throughput and efficient GPU utilization with advanced parallelism strategies.

## When to Use vLLM

- **High-Performance Serving**: Need maximum throughput and low latency for LLM inference
- **Large Model Deployment**: Serving models that require multiple GPUs with advanced parallelism
- **Production Workloads**: Robust, battle-tested serving for production environments
- **Memory Optimization**: Efficient GPU memory utilization with PagedAttention
- **Broad Model Support**: Wide compatibility with popular LLM architectures

## Key Benefits

- **Performance**: Industry-leading throughput with optimized attention mechanisms
- **Scalability**: Advanced parallelism options (tensor, pipeline, data) for any scale
- **Memory Efficiency**: PagedAttention for optimal GPU memory utilization
- **Reliability**: Production-ready with extensive testing and optimization
- **Compatibility**: Supports most popular LLM architectures and formats

## Configuration

### Basic Settings

```yaml
deployment:
  type: vllm
  image: vllm/vllm-openai:latest
  served_model_name: your-model-name
  port: 8000
  
  # Model source (choose one)
  hf_model_handle: meta-llama/Llama-3.1-8B-Instruct  # HuggingFace model
  # OR
  checkpoint_path: /path/to/local/checkpoint          # Local checkpoint
```

### Parallelism Configuration

vLLM supports three types of parallelism for maximum scaling flexibility:

```yaml
deployment:
  tensor_parallel_size: 8      # Split model across GPUs (for large models)
  pipeline_parallel_size: 1    # Pipeline stages (for very large models)
  data_parallel_size: 1        # Multiple replicas (for high throughput)
```

**Parallelism Guidelines:**
- **Tensor Parallel**: Distribute model layers across multiple GPUs
- **Pipeline Parallel**: Split model into sequential stages across GPUs
- **Data Parallel**: Run multiple model instances for higher throughput
- **Total GPUs**: `tensor_parallel_size × pipeline_parallel_size × data_parallel_size`

### Advanced Configuration

```yaml
deployment:
  extra_args: "--max-model-len 4096 --block-size 16 --swap-space 4"
  env_vars:
    HF_HOME: "/cache/huggingface"
    CUDA_VISIBLE_DEVICES: "0,1,2,3,4,5,6,7"
    NCCL_DEBUG: "INFO"
  
  endpoints:
    chat: /v1/chat/completions
    completions: /v1/completions
    health: /health
```

## Model Loading Options

vLLM supports flexible model loading with automatic selection:

```yaml
# Option 1: HuggingFace Model (recommended)
deployment:
  hf_model_handle: meta-llama/Llama-3.1-8B-Instruct
  # Automatically downloads and caches model

# Option 2: Local Checkpoint
deployment:
  checkpoint_path: /models/llama-3.1-8b-instruct
  # Uses local model files

# Option 3: Automatic Selection
deployment:
  hf_model_handle: meta-llama/Llama-3.1-8B-Instruct
  checkpoint_path: /models/llama-3.1-8b-instruct
  # Uses HuggingFace if available, falls back to local path
```

The launcher uses `${oc.select:deployment.hf_model_handle,/checkpoint}` syntax to automatically choose the appropriate model source.

## Complete Example

```yaml
defaults:
  - execution: local
  - deployment: vllm
  - _self_

deployment:
  type: vllm
  image: vllm/vllm-openai:latest
  hf_model_handle: meta-llama/Llama-3.1-8B-Instruct
  served_model_name: llama-3.1-8b-instruct
  port: 8000
  
  # Parallelism for 8 GPUs
  tensor_parallel_size: 8
  pipeline_parallel_size: 1
  data_parallel_size: 1
  
  # Performance optimizations
  extra_args: "--max-model-len 4096 --gpu-memory-utilization 0.95 --enforce-eager"
  env_vars:
    HF_HOME: "/cache/huggingface"
    CUDA_VISIBLE_DEVICES: "0,1,2,3,4,5,6,7"

execution:
  output_dir: vllm_evaluation_results

evaluation:
  tasks:
    - name: gpqa_diamond
    - name: ifeval
```

## Performance Optimization

### GPU Configuration Strategies

**Single GPU:**
```yaml
tensor_parallel_size: 1
pipeline_parallel_size: 1
data_parallel_size: 1
```

**Multi-GPU (Large Model):**
```yaml
tensor_parallel_size: 8    # Split model across 8 GPUs
pipeline_parallel_size: 1
data_parallel_size: 1
```

**Multi-GPU (High Throughput):**
```yaml
tensor_parallel_size: 4    # Split model across 4 GPUs
pipeline_parallel_size: 1
data_parallel_size: 2      # Run 2 instances (total: 8 GPUs)
```

**Very Large Models (Pipeline Parallel):**
```yaml
tensor_parallel_size: 4    # Split layers across 4 GPUs
pipeline_parallel_size: 2  # 2 pipeline stages (total: 8 GPUs)
data_parallel_size: 1
```

### Memory and Performance Tuning

```yaml
deployment:
  extra_args: >-
    --gpu-memory-utilization 0.95
    --max-model-len 4096
    --block-size 16
    --swap-space 4
    --enforce-eager
    --trust-remote-code
```

**Key Parameters:**
- `--gpu-memory-utilization`: GPU memory fraction (0.8-0.95)
- `--max-model-len`: Maximum sequence length
- `--block-size`: Memory block size for PagedAttention
- `--enforce-eager`: Disable CUDA graphs for compatibility
- `--trust-remote-code`: Enable custom model code from HuggingFace

### Caching and Storage

```yaml
deployment:
  env_vars:
    HF_HOME: "/fast/storage/huggingface"
    TRANSFORMERS_CACHE: "/fast/storage/transformers"
    CUDA_VISIBLE_DEVICES: "0,1,2,3,4,5,6,7"
```

## Tips and Best Practices

- **Model Selection**: Use `hf_model_handle` for HuggingFace models, `checkpoint_path` for local models
- **GPU Allocation**: Start with tensor parallelism, add pipeline parallelism for very large models
- **Memory Management**: Adjust `--gpu-memory-utilization` based on available GPU memory
- **Performance**: Use `--enforce-eager` if encountering CUDA graph issues
- **Caching**: Set `HF_HOME` to fast storage to speed up model loading
- **Monitoring**: Check `/health` endpoint and GPU utilization during deployment
- **Scaling**: Use data parallelism to increase throughput with multiple model instances

## Reference

**Example Configurations:**
- `lepton_vllm_llama_3_1_8b_instruct.yaml` - vLLM deployment on Lepton platform
- `slurm_llama_3_1_8b_instruct.yaml` - vLLM deployment on SLURM cluster
- `slurm_llama_3_1_8b_instruct_hf.yaml` - vLLM with HuggingFace model
- `nv-eval-api.ipynb` - Python API usage with vLLM

**Related Documentation:**
- [vLLM Documentation](https://docs.vllm.ai/en/latest/) - Official vLLM documentation
- [Execution Platforms](../execution/index.md) - Execution platform configuration
- [vLLM Performance Guide](https://docs.vllm.ai/en/latest/performance_benchmark/benchmarks.html) - Performance optimization

**Validation:**
- Use `nemo-evaluator-launcher run --dry-run` to validate configuration
- Monitor GPU memory usage and utilization during deployment
- Check vLLM server logs for performance metrics and warnings
