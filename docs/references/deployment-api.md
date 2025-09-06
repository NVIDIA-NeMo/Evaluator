# Deployment API Reference

Complete reference for the `deploy()` function and related deployment utilities in NeMo Eval.

## nemo_eval.api.deploy()

Deploy NeMo Framework models for evaluation using PyTriton or Ray Serve backends.

### Function Signature

```python
def deploy(
    nemo_checkpoint: Optional[AnyPath] = None,
    hf_model_id_path: Optional[AnyPath] = None,
    serving_backend: str = "pytriton",
    model_name: str = "megatron_model",
    server_port: int = 8080,
    server_address: str = "0.0.0.0",
    triton_address: str = "0.0.0.0",
    triton_port: int = 8000,
    num_gpus: int = 1,
    num_nodes: int = 1,
    tensor_parallelism_size: int = 1,
    pipeline_parallelism_size: int = 1,
    context_parallel_size: int = 1,
    expert_model_parallel_size: int = 1,
    max_input_len: int = 4096,
    max_batch_size: int = 8,
    # NeMo checkpoint specific
    enable_flash_decode: bool = True,
    enable_cuda_graphs: bool = True,
    legacy_ckpt: bool = False,
    # Hugging Face checkpoint specific  
    use_vllm_backend: bool = True,
    # Ray deployment specific
    num_replicas: int = 1,
    num_cpus: Optional[int] = None,
    include_dashboard: bool = True,
) -> None
```

### Parameters

#### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `nemo_checkpoint` | `Optional[AnyPath]` | `None` | Path to NeMo Framework checkpoint |
| `hf_model_id_path` | `Optional[AnyPath]` | `None` | Hugging Face model ID or local path (Ray backend only) |
| `serving_backend` | `str` | `"pytriton"` | Backend for serving: `"pytriton"` or `"ray"` |
| `model_name` | `str` | `"megatron_model"` | Name for the deployed model |

#### Server Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `server_port` | `int` | `8080` | HTTP port for FastAPI/Ray server |
| `server_address` | `str` | `"0.0.0.0"` | HTTP address for FastAPI/Ray server |
| `triton_address` | `str` | `"0.0.0.0"` | HTTP address for Triton server |
| `triton_port` | `int` | `8000` | Port for Triton server |

#### Hardware Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_gpus` | `int` | `1` | Number of GPUs per node |
| `num_nodes` | `int` | `1` | Number of nodes for deployment |
| `tensor_parallelism_size` | `int` | `1` | Tensor parallelism degree |
| `pipeline_parallelism_size` | `int` | `1` | Pipeline parallelism degree |
| `context_parallel_size` | `int` | `1` | Context parallelism degree |
| `expert_model_parallel_size` | `int` | `1` | Expert parallelism degree for MoE models |

#### Model Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_input_len` | `int` | `4096` | Maximum input sequence length |
| `max_batch_size` | `int` | `8` | Maximum batch size for inference |

#### NeMo Checkpoint Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_flash_decode` | `bool` | `True` | Enable flash decoding for faster inference |
| `enable_cuda_graphs` | `bool` | `True` | Enable CUDA graphs optimization |
| `legacy_ckpt` | `bool` | `False` | Use legacy checkpoint format |

#### Hugging Face Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_vllm_backend` | `bool` | `True` | Use VLLM backend for HF models |

#### Ray Deployment Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_replicas` | `int` | `1` | Number of model replicas |
| `num_cpus` | `Optional[int]` | `None` | CPUs to allocate (None = all available) |
| `include_dashboard` | `bool` | `True` | Include Ray dashboard |

### Usage Examples

#### Basic PyTriton Deployment

```python
from nemo_eval.api import deploy

deploy(
    nemo_checkpoint="/path/to/checkpoint",
    serving_backend="pytriton",
    num_gpus=1,
    max_input_len=8192
)
```

#### Multi-GPU Tensor Parallel Deployment

```python
deploy(
    nemo_checkpoint="/path/to/large_model",
    serving_backend="pytriton", 
    num_gpus=8,
    tensor_parallelism_size=8,
    max_input_len=32768,
    max_batch_size=16
)
```

#### Multi-Node Pipeline Parallel Deployment

```python
deploy(
    nemo_checkpoint="/path/to/very_large_model",
    serving_backend="pytriton",
    num_gpus=8,
    num_nodes=4,
    tensor_parallelism_size=2,
    pipeline_parallelism_size=4,
    max_input_len=65536
)
```

#### Ray Serve Multi-Instance Deployment

```python
deploy(
    nemo_checkpoint="/path/to/checkpoint",
    serving_backend="ray",
    num_gpus=4,
    num_replicas=2,
    tensor_parallelism_size=2,
    include_dashboard=True,
    num_cpus=16
)
```

#### Hugging Face Model with Ray

```python
deploy(
    hf_model_id_path="meta-llama/Llama-2-7b-chat-hf",
    serving_backend="ray",
    use_vllm_backend=True,
    num_gpus=2,
    max_input_len=4096
)
```

### Performance Optimization

#### Hardware-Specific Configurations

**Single GPU (RTX 4090, A100 80GB)**:
```python
deploy(
    nemo_checkpoint=checkpoint_path,
    num_gpus=1,
    max_batch_size=16,
    max_input_len=8192,
    enable_flash_decode=True,
    enable_cuda_graphs=True
)
```

**Multi-GPU A100/H100 Cluster**:
```python
deploy(
    nemo_checkpoint=checkpoint_path,
    num_gpus=8,
    tensor_parallelism_size=8,
    max_batch_size=32,
    max_input_len=32768,
    enable_flash_decode=True,
    enable_cuda_graphs=True
)
```

**High-Throughput Multi-Instance**:
```python
deploy(
    nemo_checkpoint=checkpoint_path,
    serving_backend="ray",
    num_gpus=16,
    num_replicas=4,
    tensor_parallelism_size=4,
    max_batch_size=8,
    num_cpus=64
)
```

### Backend Comparison

| Feature | PyTriton | Ray Serve |
|---------|----------|-----------|
| **Multi-Node Model Parallelism** |  Yes |  Coming soon |
| **Multi-Instance Evaluation** |  No |  Yes |
| **Hugging Face Models** |  NeMo only |  Yes |
| **Production Deployment** |  Optimized |  Good |
| **Development/Testing** |  Good |  Excellent |

### Error Handling

Common deployment errors and solutions:

**CUDA Out of Memory**:
```python
# Reduce batch size or sequence length
deploy(
    nemo_checkpoint=checkpoint_path,
    max_batch_size=4,      # Reduce from default 8
    max_input_len=2048     # Reduce from default 4096
)
```

**Port Already in Use**:
```python
# Use different ports
deploy(
    nemo_checkpoint=checkpoint_path,
    server_port=8081,      # Change from default 8080
    triton_port=8001       # Change from default 8000
)
```

**Insufficient GPU Memory for Parallelism**:
```python
# Reduce parallelism degree
deploy(
    nemo_checkpoint=checkpoint_path,
    num_gpus=4,
    tensor_parallelism_size=2,     # Reduce from 4
    pipeline_parallelism_size=2    # Use pipeline instead
)
```

### Health Checking

After deployment, verify model health:

```python
from nemo_eval.utils.base import wait_for_fastapi_server

# Wait for server to be ready
success = wait_for_fastapi_server(
    base_url="http://0.0.0.0:8080",
    model_name="megatron_model",
    max_retries=300,
    retry_interval=10
)

if success:
    print(" Model deployment successful")
else:
    print(" Model deployment failed")
```

## Related Functions

### nemo_eval.utils.base.wait_for_fastapi_server()

Wait for deployed model to be ready for evaluation.

```python
def wait_for_fastapi_server(
    base_url: str = "http://0.0.0.0:8080",
    model_name: str = "megatron_model", 
    max_retries: int = 600,
    retry_interval: int = 10,
) -> bool
```

### nemo_eval.utils.base.check_health()

Check server health status.

```python
def check_health(
    health_url: str, 
    max_retries: int = 600, 
    retry_interval: int = 2
) -> bool
```

### nemo_eval.utils.base.check_endpoint()

Test endpoint responsiveness.

```python
def check_endpoint(
    endpoint_url: str,
    endpoint_type: str,
    model_name: str,
    max_retries: int = 600,
    retry_interval: int = 2
) -> bool
```

---

**Source**: `src/nemo_eval/api.py:24-90`  
**Evidence**: Complete function signature with all 27 parameters and their default values
