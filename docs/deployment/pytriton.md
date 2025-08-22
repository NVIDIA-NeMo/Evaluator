(pytriton-deployment)=

# PyTriton Deployment

Deploy NeMo Framework models using PyTriton backend for high-performance inference with multi-node model parallelism support.

## Overview

PyTriton deployment provides the highest performance option for serving NeMo Framework models in production environments. It leverages NVIDIA Triton Inference Server for optimized inference and supports advanced features like multi-node model parallelism.

## Key Features

- **High Performance**: Optimized inference with CUDA graphs and flash decoding
- **Multi-Node Support**: Distribute large models across multiple nodes
- **Production Ready**: Enterprise-grade reliability and monitoring
- **OpenAI Compatibility**: Standard `/v1/completions` and `/v1/chat/completions` endpoints

## Basic Deployment

Deploy a NeMo Framework checkpoint using PyTriton with the following Python command:

```python
from nemo_eval.api import deploy

if __name__ == "__main__":
    deploy(
        nemo_checkpoint='/workspace/llama3_8b_nemo2',
        max_input_len=4096,
        max_batch_size=4,
        server_port=8080,
        num_gpus=1,
    )
```

## Configuration Options

### Core Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `nemo_checkpoint` | `str` | Path to NeMo Framework checkpoint |
| `max_input_len` | `int` | Maximum input sequence length |
| `max_batch_size` | `int` | Maximum batch size for inference |
| `server_port` | `int` | Port for the inference server |
| `num_gpus` | `int` | Number of GPUs to use |

### Advanced Configuration

For larger models requiring model parallelism:

```python
deploy(
    nemo_checkpoint='/workspace/large_model',
    max_input_len=8192,
    max_batch_size=8,
    server_port=8080,
    num_gpus=8,
    tensor_parallelism_size=4,
    pipeline_parallelism_size=2,
)
```

### Multi-Node Deployment

For extremely large models that require multi-node deployment:

```python
deploy(
    nemo_checkpoint='/workspace/massive_model',
    max_input_len=4096,
    max_batch_size=4,
    server_port=8080,
    num_gpus=16,
    tensor_parallelism_size=8,
    pipeline_parallelism_size=2,
    num_nodes=2,
)
```

## API Endpoints

Once deployed, the server exposes OpenAI-compatible endpoints:

### Completions Endpoint
- **URL**: `http://localhost:8080/v1/completions`
- **Use Case**: Text completion tasks, few-shot prompting
- **Compatible Tasks**: MMLU, GSM8K, HellaSwag, etc.

### Chat Completions Endpoint  
- **URL**: `http://localhost:8080/v1/chat/completions`
- **Use Case**: Conversational tasks, instruction following
- **Compatible Tasks**: IFEval, GPQA, chat-based benchmarks

## Health Monitoring

The deployment includes built-in health checks:

```python
import requests

# Check server health
response = requests.get("http://localhost:8080/health")
print(response.status_code)  # Should return 200 when ready
```

## Performance Optimization

### Memory Management
- Use appropriate `max_batch_size` based on available GPU memory
- Set `max_input_len` to match your evaluation requirements
- Consider gradient checkpointing for memory-constrained scenarios

### Inference Acceleration
- PyTriton automatically enables CUDA graphs when possible
- Flash attention is used for supported model architectures
- KV-cache optimization reduces memory overhead

## Troubleshooting

### Common Issues

**Server fails to start:**
- Check GPU memory availability
- Verify checkpoint path is accessible
- Ensure port is not already in use

**High memory usage:**
- Reduce `max_batch_size`
- Lower `max_input_len`
- Use model parallelism for large models

**Slow inference:**
- Increase `max_batch_size` if memory allows
- Verify CUDA graphs are enabled
- Check for mixed precision usage

### Logging and Debugging

Enable verbose logging for troubleshooting:

```python
deploy(
    nemo_checkpoint='/workspace/model',
    server_port=8080,
    num_gpus=1,
    log_level="DEBUG"
)
```

## Next Steps

After successful deployment:

1. **Validate Endpoints**: Test both completions and chat endpoints
2. **Run Evaluations**: Use the deployed model for benchmark evaluation
3. **Monitor Performance**: Track inference latency and throughput
4. **Scale as Needed**: Adjust parallelism based on workload requirements

For evaluation instructions, see the [Model Evaluation](../evaluation/index.md) section.
