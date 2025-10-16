(deployment-vllm)=

# vLLM Deployment

Configure vLLM as the deployment backend for serving models during evaluation.

## Configuration Parameters

### Basic Settings

```yaml
deployment:
  type: vllm
  image: vllm/vllm-openai:latest
  checkpoint_path: /path/to/model  # Model path (local or HuggingFace ID)
  served_model_name: your-model-name
  port: 8000
```

### Parallelism Configuration

```yaml
deployment:
  tensor_parallel_size: 8
  pipeline_parallel_size: 1
  data_parallel_size: 1
```

- **tensor_parallel_size**: Number of GPUs to split the model across (default: 8)
- **pipeline_parallel_size**: Number of pipeline stages (default: 1)
- **data_parallel_size**: Number of model replicas (default: 1)

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
  checkpoint_path: /path/to/checkpoint
  served_model_name: llama-3.1-8b-instruct
  tensor_parallel_size: 1
  data_parallel_size: 8
  extra_args: "--max-model-len 4096"

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

## Reference

The following example configuration files are available in the `examples/` directory:

- `lepton_vllm_llama_3_1_8b_instruct.yaml` - vLLM deployment on Lepton platform
- `slurm_llama_3_1_8b_instruct.yaml` - vLLM deployment on SLURM cluster
- `slurm_llama_3_1_8b_instruct_hf.yaml` - vLLM deployment using HuggingFace model ID

Use `nemo-evaluator-launcher run --dry-run` to check your configuration before running.
