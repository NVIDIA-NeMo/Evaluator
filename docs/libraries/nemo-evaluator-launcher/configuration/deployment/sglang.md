(deployment-sglang)=

# SGLang Deployment

SGLang is a fast serving framework for large language models and vision language models, optimized for high-throughput inference with efficient memory usage and structured generation capabilities.

## When to Use SGLang

- **High-Throughput Serving**: Need fast inference with optimized attention mechanisms
- **Structured Generation**: Require constrained or guided text generation
- **Memory Efficiency**: Want to maximize GPU memory utilization
- **Vision-Language Models**: Working with multimodal models
- **Custom Serving**: Need flexible serving configuration and optimization

## Key Benefits

- **Performance**: Optimized attention and memory management for faster inference
- **Flexibility**: Supports both HuggingFace models and local checkpoints
- **Scalability**: Configurable tensor and data parallelism
- **Compatibility**: OpenAI-compatible API endpoints
- **Efficiency**: Advanced memory optimization and batching strategies

## Configuration

### Basic Settings

```yaml
deployment:
  type: sglang
  image: lmsysorg/sglang:latest
  served_model_name: your-model-name
  port: 8000
  
  # Model source (choose one)
  hf_model_handle: meta-llama/Llama-3.1-8B-Instruct  # HuggingFace model
  # OR
  checkpoint_path: /path/to/local/checkpoint          # Local checkpoint
```

### Parallelism Configuration

```yaml
deployment:
  tensor_parallel_size: 8    # Number of GPUs for tensor parallelism
  data_parallel_size: 1      # Number of replicas for data parallelism
```

**Parallelism Guidelines:**
- **Tensor Parallel**: Split model across multiple GPUs (for large models)
- **Data Parallel**: Multiple model replicas (for high throughput)
- **Total GPUs**: `tensor_parallel_size × data_parallel_size`

### Advanced Configuration

```yaml
deployment:
  extra_args: "--disable-flashinfer --enable-torch-compile"
  env_vars:
    HF_HOME: "/cache/huggingface"
    CUDA_VISIBLE_DEVICES: "0,1,2,3"
  
  endpoints:
    chat: /v1/chat/completions
    completions: /v1/completions
    health: /health
```

## Model Loading Options

SGLang supports flexible model loading with automatic fallback:

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
  - deployment: sglang
  - _self_

deployment:
  type: sglang
  image: lmsysorg/sglang:latest
  hf_model_handle: meta-llama/Llama-3.1-8B-Instruct
  served_model_name: llama-3.1-8b-instruct
  port: 8000
  
  # Parallelism for 4 GPUs
  tensor_parallel_size: 4
  data_parallel_size: 1
  
  # Performance optimizations
  extra_args: "--disable-flashinfer --mem-fraction-static 0.8"
  env_vars:
    HF_HOME: "/cache/huggingface"
    CUDA_VISIBLE_DEVICES: "0,1,2,3"

execution:
  output_dir: sglang_evaluation_results

evaluation:
  tasks:
    - name: gpqa_diamond
    - name: ifeval
```

## Performance Optimization

### GPU Configuration
- **Single GPU**: `tensor_parallel_size: 1, data_parallel_size: 1`
- **Multi-GPU (Large Model)**: `tensor_parallel_size: 4, data_parallel_size: 1`
- **Multi-GPU (High Throughput)**: `tensor_parallel_size: 2, data_parallel_size: 2`

### Memory Optimization
```yaml
deployment:
  extra_args: "--mem-fraction-static 0.8 --max-running-requests 256"
  env_vars:
    PYTORCH_CUDA_ALLOC_CONF: "max_split_size_mb:128"
```

### Caching and Storage
```yaml
deployment:
  env_vars:
    HF_HOME: "/fast/storage/huggingface"  # Use fast storage for model cache
    TRANSFORMERS_CACHE: "/fast/storage/transformers"
```

## Tips and Best Practices

- **Model Selection**: Use `hf_model_handle` for HuggingFace models, `checkpoint_path` for local models
- **GPU Allocation**: Match `tensor_parallel_size` to your available GPUs for large models
- **Memory Management**: Adjust `--mem-fraction-static` based on your GPU memory
- **Performance**: Use `--disable-flashinfer` if encountering compatibility issues
- **Caching**: Set `HF_HOME` to fast storage to speed up model loading
- **Monitoring**: Check `/health` endpoint to verify deployment status

## Reference

**Related Documentation:**
- [SGLang Documentation](https://docs.sglang.ai/) - Official SGLang documentation
- [Execution Platforms](../execution/index.md) - Execution platform configuration
- [Model Optimization Guide](https://docs.sglang.ai/optimization) - Performance tuning

**Validation:**
- Use `nemo-evaluator-launcher run --dry-run` to validate configuration
- Check SGLang server logs for deployment issues
- Monitor GPU memory usage during deployment
