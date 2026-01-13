(deployment-none)=

# None Deployment

The "none" deployment option means **no model deployment is performed**. Instead, you provide an existing OpenAI-compatible endpoint. The launcher handles running evaluation tasks while connecting to your existing endpoint.

## When to Use None Deployment

- **Existing Endpoints**: You have a running model endpoint to evaluate
- **Third-Party Services**: Testing models from NVIDIA API Catalog, OpenAI, or other providers  
- **Custom Infrastructure**: Using your own deployment solution outside the launcher
- **Cost Optimization**: Reusing existing deployments across multiple evaluation runs
- **Separation of Concerns**: Keeping model deployment and evaluation as separate processes

## Key Benefits

- **No Resource Management**: No need to provision or manage model deployment resources
- **Platform Flexibility**: Works with Local, Lepton, and SLURM execution platforms
- **Quick Setup**: Minimal configuration required - just point to your endpoint
- **Cost Effective**: Leverage existing deployments without additional infrastructure

## Universal Configuration

These configuration patterns apply to all execution platforms when using "none" deployment.

### Target Endpoint Setup

```yaml
target:
  api_endpoint:
    model_id: meta/llama-3.1-8b-instruct    # Model identifier (required)
    url: https://your-endpoint.com/v1/chat/completions  # Endpoint URL (required)
    api_key_name: API_KEY                    # Environment variable name (recommended)
```

## Platform Examples

Choose your execution platform and see the specific configuration needed:

::::{tab-set}

:::{tab-item} Local
**Best for**: Development, testing, small-scale evaluations

```yaml
defaults:
  - execution: local
  - deployment: none
  - _self_

execution:
  output_dir: results

target:
  api_endpoint:
    model_id: meta/llama-3.2-3b-instruct
    url: https://integrate.api.nvidia.com/v1/chat/completions
    api_key_name: NGC_API_KEY

evaluation:
  tasks:
    - name: gpqa_diamond
      env_vars:
        HF_TOKEN: HF_TOKEN_FOR_GPQA_DIAMOND  # Click request access for GPQA-Diamond: https://huggingface.co/datasets/Idavidrein/gpqa
```

**Key Points:**
- Minimal configuration required
- Set environment variables in your shell
- Limited by local machine resources
:::

:::{tab-item} Lepton
**Best for**: Production evaluations, team environments, scalable workloads

```yaml
defaults:
  - execution: lepton/default
  - deployment: none
  - _self_

execution:
  output_dir: results

  lepton_platform:
    tasks:
      api_tokens:
        - value_from:
            token_name_ref: "ENDPOINT_API_KEY"

      env_vars:
        HF_TOKEN:
          value_from:
            secret_name_ref: "HUGGING_FACE_HUB_TOKEN"
        API_KEY:
          value_from:
            secret_name_ref: "ENDPOINT_API_KEY"

      node_group: "your-node-group"
      mounts:
        - from: "node-nfs:shared-fs"
          path: "/workspace/path"
          mount_path: "/workspace"

target:
  api_endpoint:
    model_id: meta/llama-3.1-8b-instruct
    url: https://your-endpoint.lepton.run/v1/chat/completions
    api_key_name: API_KEY

evaluation:
  tasks:
    - name: gpqa_diamond
```

**Key Points:**
- Requires Lepton credentials (`lep login`)
- Use `secret_name_ref` for secure credential storage
- Configure node groups and storage mounts
- Handles larger evaluation workloads
:::

:::{tab-item} SLURM
**Best for**: HPC environments, large-scale evaluations, batch processing

```yaml
defaults:
  - execution: slurm/default
  - deployment: none
  - _self_

execution:
  account: your-slurm-account
  output_dir: /shared/filesystem/results
  walltime: "02:00:00"
  partition: cpu_short
  gpus_per_node: null  # No GPUs needed

target:
  api_endpoint:
    model_id: meta/llama-3.2-3b-instruct
    url: https://integrate.api.nvidia.com/v1/chat/completions
    api_key_name: NGC_API_KEY # API Key with access to build.nvidia.com

evaluation:
  tasks:
    - name: gpqa_diamond
      env_vars:
        HF_TOKEN: HF_TOKEN_FOR_GPQA_DIAMOND # Click request access for GPQA-Diamond: https://huggingface.co/datasets/Idavidrein/gpqa
```

**Key Points:**
- Requires SLURM account and accessible output directory
- Creates one job per benchmark evaluation
- Uses CPU partitions (no GPUs needed for none deployment)
:::

::::
