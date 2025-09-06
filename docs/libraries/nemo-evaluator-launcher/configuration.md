# Configuration Reference

NeMo Evaluator Launcher uses [Hydra](https://hydra.cc/) for configuration management, allowing for flexible, composable, and overridable configurations.

## Configuration Structure

All configurations follow this general structure:

```yaml
# Specify default sub-configurations
defaults:
  - execution: local          # Execution backend (local, slurm, lepton)
  - deployment: none          # Model deployment (none, vllm, nim, sglang)
  - _self_                    # Include this config's values

# Execution settings
execution:
  output_dir: results         # Where to store results

# Model endpoint configuration  
target:
  api_endpoint:
    url: http://localhost:8000/v1/chat/completions
    model_id: my-model
    api_key_name: API_KEY     # Environment variable name

# Evaluation configuration
evaluation:
  overrides:                  # Global overrides for all tasks
    config.params.temperature: 0.7
  tasks:                      # List of evaluation tasks
    - name: hellaswag
    - name: arc_challenge
```

## Core Configuration Sections

### defaults

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

### execution

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

### target

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

### evaluation

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

### deployment

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

## Configuration Examples

### Local Development

```yaml
# configs/local_dev.yaml
defaults:
  - execution: local
  - deployment: none
  - _self_

execution:
  output_dir: ./dev_results

target:
  api_endpoint:
    url: http://localhost:8000/v1/chat/completions
    model_id: my-local-model

evaluation:
  overrides:
    config.params.limit_samples: 10    # Small test run
  tasks:
    - name: hellaswag
    - name: arc_challenge
```

### Production Slurm Cluster

```yaml
# configs/production_slurm.yaml
defaults:
  - execution: slurm
  - deployment: vllm
  - _self_

execution:
  output_dir: /shared/production_results
  partition: gpu
  nodes: 1
  gpus_per_node: 8
  time_limit: "12:00:00"
  account: research_project

deployment:
  type: vllm
  image: nvcr.io/nvidia/vllm:24.01
  model_path: /shared/models/llama-3.1-70b
  envs:
    CUDA_VISIBLE_DEVICES: "0,1,2,3,4,5,6,7"
    VLLM_USE_V2_BLOCK_MANAGER: "1"
    HF_TOKEN: ${oc.env:HF_TOKEN}

target:
  api_endpoint:
    url: http://localhost:8000/v1/chat/completions
    model_id: meta-llama/Llama-3.1-70B-Instruct

evaluation:
  overrides:
    config.params.temperature: 0.0
    config.params.max_new_tokens: 4096
    config.params.parallelism: 64
  tasks:
    - name: hellaswag
    - name: arc_challenge
    - name: winogrande
    - name: gsm8k
```

### Lepton AI Cloud

```yaml
# configs/lepton_cloud.yaml
defaults:
  - execution: lepton
  - deployment: nim
  - _self_

execution:
  output_dir: cloud_results

deployment:
  type: nim
  image: nvcr.io/nim/meta/llama-3.1-8b-instruct:latest
  envs:
    NGC_API_KEY: ${oc.env:NGC_API_KEY}

target:
  api_endpoint:
    model_id: meta/llama-3.1-8b-instruct

evaluation:
  tasks:
    - name: hellaswag
    - name: arc_challenge
    - name: winogrande
    - name: truthfulqa_mc2
```

## Configuration Overrides

### Command Line Overrides

Use Hydra's override syntax to modify configurations:

```bash
# Single override
nv-eval run --config-dir examples --config-name local_llama_3_1_8b_instruct \
  -o execution.output_dir=my_results

# Multiple overrides
nv-eval run --config-dir examples --config-name local_llama_3_1_8b_instruct \
  -o execution.output_dir=my_results \
  -o target.api_endpoint.model_id=my-model \
  -o evaluation.tasks[0].name=hellaswag

# Add new configuration (+ prefix)
nv-eval run --config-dir examples --config-name local_llama_3_1_8b_instruct \
  -o +config.params.limit_samples=10
```

### Environment Variable References

Reference environment variables in configurations:

```yaml
# Direct reference
target:
  api_endpoint:
    api_key: ${oc.env:NGC_API_KEY}
    
# With default value
target:
  api_endpoint:
    api_key: ${oc.env:NGC_API_KEY,nvapi-default}
    
# In deployment environment
deployment:
  envs:
    HF_TOKEN: ${oc.env:HF_TOKEN}
    CUSTOM_VAR: ${oc.env:CUSTOM_VAR,default_value}
```

### Configuration Composition

Create modular configurations using Hydra's composition:

```yaml
# configs/base.yaml
target:
  api_endpoint:
    timeout: 60
    max_retries: 3
    
evaluation:
  overrides:
    config.params.temperature: 0.7
```

```yaml
# configs/my_evaluation.yaml
defaults:
  - base                      # Include base configuration
  - execution: local
  - deployment: none
  - _self_

execution:
  output_dir: my_results

evaluation:
  tasks:
    - name: hellaswag
```

## Parameter Reference

### Common Task Parameters

These parameters can be overridden per task or globally:

```yaml
config.params:
  temperature: 0.7              # Sampling temperature (0.0-1.0)
  top_p: 0.95                   # Nucleus sampling parameter
  top_k: 50                     # Top-k sampling parameter
  max_new_tokens: 2048          # Maximum tokens to generate
  parallelism: 16               # Number of parallel requests
  request_timeout: 3600         # Request timeout in seconds
  limit_samples: null           # Limit number of samples (for testing)
  
  # Task-specific parameters
  extra:
    n_samples: 1                # Number of samples per prompt (for code gen)
    pass_at_k: [1, 5, 10]      # Pass@k metrics (for code gen)
```

### Adapter Configuration

Control how requests are formatted and processed:

```yaml
target.api_endpoint.adapter_config:
  use_reasoning: false          # Strip reasoning tokens from responses
  use_system_prompt: true       # Include system prompts
  custom_system_prompt: "Think step by step."
  
  # Request formatting
  add_bos_token: false         # Add beginning-of-sequence token
  add_eos_token: false         # Add end-of-sequence token
  
  # Response processing
  strip_whitespace: true        # Strip leading/trailing whitespace
  normalize_newlines: true      # Normalize newline characters
```

## Validation and Debugging

### Configuration Validation

```bash
# Dry run to see resolved configuration
nv-eval run --config-dir examples --config-name my_config --dry-run

# Print configuration tree
nv-eval run --config-dir examples --config-name my_config --cfg job
```

### Common Validation Errors

**Missing Required Fields:**
```
Error: Missing required field 'execution.output_dir'
```

**Invalid Task Names:**
```
Error: Unknown task 'invalid_task'. Available tasks: hellaswag, arc_challenge, ...
```

**Configuration Conflicts:**
```
Error: Cannot specify both 'api_key' and 'api_key_name' in target.api_endpoint
```

## Advanced Configuration Patterns

These real-world examples from the launcher package show advanced configuration techniques:

### Automatic Result Export
```yaml
# examples/local_auto_export_llama_3_1_8b_instruct.yaml
defaults:
  - execution: local
  - deployment: none
  - _self_

execution:
  output_dir: ./results
  auto_export: true
  exporters:
    - type: mlflow
      tracking_uri: ${oc.env:MLFLOW_TRACKING_URI}
    - type: wandb
      project: llm-evaluation
      entity: ${oc.env:WANDB_ENTITY}

target:
  api_endpoint:
    url: http://localhost:8080/v1/chat/completions
    model_id: meta/llama-3.1-8b-instruct
    
evaluation:
  tasks:
    - name: hellaswag
    - name: arc_challenge
    - name: winogrande
```

### Custom Metadata Injection
```yaml  
# examples/local_with_user_provided_metadata.yaml
defaults:
  - execution: local
  - deployment: none
  - _self_

execution:
  output_dir: ./results
  metadata:
    experiment_name: "baseline_evaluation"
    model_version: "v1.0"
    dataset_version: "2024-01"
    researcher: ${oc.env:USER}
    notes: "Initial baseline run for comparison"

target:
  api_endpoint:
    url: http://localhost:8080/v1/chat/completions
    model_id: meta/llama-3.1-8b-instruct
    
evaluation:
  tasks:
    - name: hellaswag
      params:
        temperature: 0.0
        max_tokens: 512
    - name: arc_challenge
      params:
        temperature: 0.0
        max_tokens: 512
```

### Development Testing with Sample Limits
```yaml
# examples/local_limit_samples.yaml  
defaults:
  - execution: local
  - deployment: none
  - _self_

execution:
  output_dir: ./dev_results

target:
  api_endpoint:
    url: http://localhost:8080/v1/chat/completions
    model_id: meta/llama-3.1-8b-instruct
    
evaluation:
  # Use small sample sizes for quick development testing
  tasks:
    - name: hellaswag
      params:
        limit_samples: 10
    - name: arc_challenge  
      params:
        limit_samples: 10
    - name: winogrande
      params:
        limit_samples: 10
    - name: gsm8k
      params:
        limit_samples: 5  # Math problems take longer
```

### Reasoning vs Non-Reasoning Task Separation
```yaml
# examples/local_aa_reasoning.yaml - For reasoning-heavy tasks
defaults:
  - execution: local
  - deployment: none
  - _self_

execution:
  output_dir: ./reasoning_results

target:
  api_endpoint:
    url: http://localhost:8080/v1/chat/completions
    model_id: meta/llama-3.1-8b-instruct
    
evaluation:
  # Tasks that benefit from chain-of-thought reasoning
  tasks:
    - name: gsm8k
      params:
        temperature: 0.1
        max_tokens: 1024
        enable_cot: true
    - name: math
      params:
        temperature: 0.1  
        max_tokens: 2048
        enable_cot: true
    - name: arc_challenge
      params:
        temperature: 0.1
        max_tokens: 512
        enable_cot: true
```

### Best Practices

1. **Use Environment Variables**: Store sensitive information in environment variables
2. **Modular Configs**: Create reusable base configurations  
3. **Descriptive Names**: Use clear, descriptive configuration file names
4. **Test with Dry Run**: Always test configurations with `--dry-run` first
5. **Version Control**: Store configurations in version control
6. **Documentation**: Document custom configurations and their purpose
7. **Sample Limits for Testing**: Use `limit_samples` for development and validation
8. **Metadata for Tracking**: Include experiment metadata for better result organization

## See Also

- [Quickstart](quickstart.md) - Getting started guide
- [CLI Reference](cli.md) - Command-line interface
- [Executors](executors/index.md) - Execution backend options
- [Python API](api.md) - Programmatic configuration management
