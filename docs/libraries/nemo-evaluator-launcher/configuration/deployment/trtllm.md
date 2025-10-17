(deployment-trtllm)=

# TensorRT LLM (TRT-LLM) Deployment

Configure TRT-LLM as the deployment backend for serving models during evaluation.

## Configuration Parameters

### Basic Settings

```yaml
deployment:
  type: trtllm
  image: nvcr.io/nvidia/tensorrt-llm/release:1.0.0
  checkpoint_path: /path/to/model
  served_model_name: your-model-name
  port: 8000
```

### Parallelism Configuration

```yaml
deployment:
  tensor_parallel_size: 4
  pipeline_parallel_size: 1
```

- **tensor_parallel_size**: Number of GPUs to split the model across (default: 4)
- **pipeline_parallel_size**: Number of pipeline stages (default: 1)

### Extra Arguments and Endpoints

```yaml
deployment:
  extra_args: "--ep_size 2"
  
  endpoints:
    chat: /v1/chat/completions
    completions: /v1/completions
    health: /health
```

The `extra_args` field passes extra arguments to the `trtllm-serve serve ` command.

## Complete Example

```yaml
defaults:
  - execution: slurm/default
  - deployment: trtllm
  - _self_

deployment:
  checkpoint_path: /path/to/checkpoint
  served_model_name: llama-3.1-8b-instruct
  tensor_parallel_size: 1
  extra_args: ""

execution:
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

<!-- ## Reference

The following example configuration files are available in the `examples/` directory:

- `config_name.yaml` - TRT-LLM deployment on XXX -->

Use `nemo-evaluator-launcher run --dry-run` to check your configuration before running.
