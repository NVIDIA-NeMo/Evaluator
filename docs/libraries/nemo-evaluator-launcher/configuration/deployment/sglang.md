(deployment-sglang)=

# SGLang Deployment

SGLang is a serving framework for large language models. This deployment type launches SGLang servers using the `lmsysorg/sglang` Docker image.

## Configuration

### Required Settings

See the complete configuration structure in `configs/deployment/sglang.yaml`.

```yaml
deployment:
  type: sglang
  image: lmsysorg/sglang:latest
  hf_model_handle: hf-model/handle  # HuggingFace ID
  checkpoint_path: null             # or provide a path to the stored checkpoint
  served_model_name: your-model-name
  port: 8000
```

**Required Fields:**

- `checkpoint_path` or `hf_model_handle`: Model path or HuggingFace model ID (e.g., `meta-llama/Llama-3.1-8B-Instruct`)
- `served_model_name`: Name for the served model

### Optional Settings

```yaml
deployment:
  tensor_parallel_size: 8    # Default: 8
  data_parallel_size: 1      # Default: 1
  extra_args: ""             # Extra SGLang server arguments
  env_vars: {}               # Environment variables (key: value dict)
```

**Configuration Fields:**

- `tensor_parallel_size`: Number of GPUs for tensor parallelism (default: 8)
- `data_parallel_size`: Number of data parallel replicas (default: 1)
- `extra_args`: Extra command-line arguments to pass to SGLang server
- `env_vars`: Environment variables for the container

### API Endpoints

The SGLang deployment exposes OpenAI-compatible endpoints:

```yaml
endpoints:
  chat: /v1/chat/completions
  completions: /v1/completions
  health: /health
```

## Example Configuration

```yaml
defaults:
  - execution: slurm/default
  - deployment: sglang
  - _self_

deployment:
  checkpoint_path: Qwen/Qwen3-4B-Instruct-2507
  served_model_name: qwen3-4b-instruct-2507
  tensor_parallel_size: 8
  data_parallel_size: 8
  extra_args: ""

execution:
  hostname: your-cluster-headnode
  account: your-account
  output_dir: /path/to/output
  walltime: 02:00:00

evaluation:
  tasks:
    - name: gpqa_diamond
    - name: ifeval
      env_vars:
        HF_TOKEN: HF_TOKEN_FOR_GPQA_DIAMOND  # or use HF_HOME
```

## Command Template

The launcher uses the following command template to start the SGLang server (from `configs/deployment/sglang.yaml`):

```bash
python3 -m sglang.launch_server \
  --model-path ${oc.select:deployment.hf_model_handle,/checkpoint} \
  --host 0.0.0.0 \
  --port ${deployment.port} \
  --served-model-name ${deployment.served_model_name} \
  --tp ${deployment.tensor_parallel_size} \
  --dp ${deployment.data_parallel_size} \
  ${deployment.extra_args}
```

:::{note}
The `${oc.select:deployment.hf_model_handle,/checkpoint}` syntax uses OmegaConf's select resolver. In practice, set `checkpoint_path` with your model path or HuggingFace model ID.
:::

## Reference

**Configuration File:**

- Source: `packages/nemo-evaluator-launcher/src/nemo_evaluator_launcher/configs/deployment/sglang.yaml`

**Related Documentation:**

- [Deployment Configuration Overview](index.md)
- [Execution Platform Configuration](../executors/index.md)
- [SGLang Documentation](https://docs.sglang.ai/)
