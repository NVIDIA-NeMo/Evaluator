(deployment-vllm)=

# vLLM Deployment

Configure vLLM as the deployment backend for serving models during evaluation.

## Configuration Parameters

### Basic Settings

```yaml
deployment:
  type: vllm
  image: vllm/vllm-openai:latest
  hf_model_handle: hf-model/handle  # HuggingFace ID
  checkpoint_path: null             # or provide a path to the stored checkpoint
  served_model_name: your-model-name
  port: 8000
```

**Required Fields:**

- `checkpoint_path` or `hf_model_handle`: Model path or HuggingFace model ID (e.g., `meta-llama/Llama-3.1-8B-Instruct`)
- `served_model_name`: Name for the served model

### Performance Settings

```yaml
deployment:
  tensor_parallel_size: 8
  pipeline_parallel_size: 1
  data_parallel_size: 1
  gpu_memory_utilization: 0.95
```

- **tensor_parallel_size**: Number of GPUs to split the model across (default: 8)
- **pipeline_parallel_size**: Number of pipeline stages (default: 1)
- **data_parallel_size**: Number of model replicas (default: 1)
- **gpu_memory_utilization**: Fraction of GPU memory to use for the model (default: 0.95)

### Extra Arguments and Endpoints

```yaml
deployment:
  extra_args: "--max-model-len 4096"
  
  endpoints:
    chat: /v1/chat/completions
    completions: /v1/completions
    health: /health
```

The `extra_args` field passes extra arguments to the `vllm serve` command.

## Complete Example

```yaml
defaults:
  - execution: slurm/default
  - deployment: vllm
  - _self_

deployment:
  checkpoint_path: Qwen/Qwen3-4B-Instruct-2507
  served_model_name: qwen3-4b-instruct-2507
  tensor_parallel_size: 1
  data_parallel_size: 8
  extra_args: "--max-model-len 4096"

execution:
  hostname: your-cluster-headnode
  account: your-account
  output_dir: /path/to/output
  walltime: 02:00:00

evaluation:
  tasks:
    - name: ifeval
    - name: gpqa_diamond
      env_vars:
        HF_TOKEN: HF_TOKEN_FOR_GPQA_DIAMOND # Click request access for GPQA-Diamond: https://huggingface.co/datasets/Idavidrein/gpqa
```

## Multi-Node Deployment with Ray (`vllm_ray`)

For models requiring multiple nodes (e.g., pipeline parallelism across nodes), use the `vllm_ray` deployment config:

```yaml
defaults:
  - execution: slurm/default
  - deployment: vllm_ray
  - _self_

execution:
  num_nodes: 2              # Single instance spanning 2 nodes

deployment:
  tensor_parallel_size: 8
  pipeline_parallel_size: 2
```

The `vllm_ray` config inherits all fields from `vllm` and adds:

- **`distributed_executor_backend`**: Ray backend type (default: `ray`)
- **`ray_compiled_dag_channel_type`**: Ray channel type — `auto`, `shm`, or `nccl` (default: `shm`)
- **`command`**: Built-in Ray cluster setup script that starts a Ray head on rank 0, waits for workers, then launches vLLM with `--distributed-executor-backend`

The `base_command` field in the base `vllm` config contains the `vllm serve ...` invocation. The `vllm_ray` config references it via `${deployment.base_command}` to append Ray-specific flags.

## Reference

The following example configuration files are available in the `examples/` directory:

- `slurm_vllm_basic.yaml` - Basic single-node vLLM deployment
- `slurm_vllm_multinode_ray_tp_pp.yaml` - Multi-node Ray deployment with TP+PP
- `slurm_vllm_multinode_multiinstance_ray_tp_pp.yaml` - Multi-node multi-instance Ray with HAProxy
- `slurm_vllm_multinode_dp_haproxy.yaml` - Multi-node independent instances with HAProxy

Use `nemo-evaluator-launcher run --dry-run` to check your configuration before running.
