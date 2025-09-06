(config-structure)=

# Structure & Sections

This page covers the core configuration sections that make up every NeMo Evaluator Launcher configuration file.

## defaults

Specifies which sub-configurations to include:

```yaml
defaults:
  - execution: local          # From configs/execution/local.yaml
  - deployment: none          # From configs/deployment/none.yaml
  - _self_                    # Include values from this file
```

**Available Options:**
- **execution**: `local`, `slurm`, `lepton`
- **deployment**: `none`, `vllm`, `nim`, `sglang`

## execution

Controls where and how evaluations run:

```yaml
execution:
  output_dir: results         # Output directory (required)
  # Executor-specific settings vary by type
```

**Local Executor:**
```yaml
execution:
  output_dir: ./results       # Local output directory
  docker_args: []             # Additional Docker arguments
```

**Slurm Executor:**
```yaml
execution:
  output_dir: /shared/results # Shared filesystem path
  partition: gpu              # Slurm partition
  nodes: 1                    # Number of nodes
  gpus_per_node: 8           # GPUs per node
  time_limit: "24:00:00"     # Time limit (HH:MM:SS)
  account: my_account        # Slurm account (optional)
```

**Lepton Executor:**
```yaml
execution:
  output_dir: results        # Results will be downloaded locally
  workspace: my_workspace    # Lepton workspace name
  region: us-west-1         # Lepton region (optional)
```

## target

Defines the model endpoint to evaluate:

```yaml
target:
  api_endpoint:
    url: https://integrate.api.nvidia.com/v1/chat/completions
    model_id: meta/llama-3.1-8b-instruct
    api_key_name: NGC_API_KEY    # Environment variable name
    api_key: ${oc.env:NGC_API_KEY}  # Direct environment reference
    
    # Adapter configuration
    adapter_config:
      use_reasoning: false        # Strip reasoning tokens
      use_system_prompt: true     # Include system prompts
      custom_system_prompt: "Think step by step."
      
    # Request parameters
    timeout: 60                   # Request timeout in seconds
    max_retries: 3               # Maximum retry attempts
```

## evaluation

Specifies which benchmarks to run and their configuration:

```yaml
evaluation:
  # Global overrides applied to all tasks
  overrides:
    config.params.temperature: 0.7
    config.params.max_new_tokens: 2048
    config.params.parallelism: 16
    target.api_endpoint.adapter_config.use_reasoning: false
    
  # List of evaluation tasks
  tasks:
    - name: hellaswag                    # Simple task
    
    - name: arc_challenge                # Task with overrides
      overrides:
        config.params.temperature: 0.0   # Task-specific overrides
        config.params.top_p: 1.0
      env_vars:
        HF_TOKEN: HF_TOKEN              # Task-specific env vars
        
    - name: mbpp                        # Code generation task
      overrides:
        config.params.temperature: 0.2
        config.params.max_new_tokens: 2048
        config.params.extra.n_samples: 5  # Multiple samples per prompt
        target.api_endpoint.adapter_config.custom_system_prompt: >-
          "You must only provide the code implementation"
```

## deployment

Controls model deployment (when using executors that support it):

```yaml
deployment:
  type: vllm                    # Deployment type
  image: nvcr.io/nvidia/vllm:latest
  model_path: /models/llama-3.1-8b
  command: vllm serve /models/llama-3.1-8b --port 8000
  
  # Environment variables for deployment
  envs:
    CUDA_VISIBLE_DEVICES: "0,1,2,3,4,5,6,7"
    VLLM_USE_FLASH_ATTN: "1"
    HF_TOKEN: ${oc.env:HF_TOKEN}        # Reference host environment
    
  # Resource requirements
  resources:
    gpus: 8
    memory: "64Gi"
    cpu: "16"
```

## See Also

- [Examples](examples.md) - Complete configuration examples
- [Parameter Reference](parameters.md) - Detailed parameter specifications
- [Overrides & Composition](overrides.md) - Configuration customization patterns
