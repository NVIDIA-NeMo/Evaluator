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

/// note | Legacy Adapter Configuration
The following adapter configuration parameters use the legacy format and are maintained for backward compatibility. For new configurations, use the modern interceptor-based system documented in {ref}`interceptor-system-messages` and {ref}`interceptor-reasoning`.

```yaml
target:
  api_endpoint:
    # Legacy adapter configuration (supported but not recommended for new configs)
    adapter_config:
      use_reasoning: false                   # Strip reasoning tokens if true
      use_system_prompt: true                # Enable system prompt support
      custom_system_prompt: "Think step by step."  # Custom system prompt
```
///

### Evaluation Configuration

```yaml
evaluation:
  # Global overrides (apply to all tasks)
  overrides:
    config.params.request_timeout: 3600
    config.params.temperature: 0.7
  
  # Task-specific configuration
  tasks:
    - name: gpqa_diamond
      overrides:
        config.params.temperature: 0.6
        config.params.max_new_tokens: 8192
        config.params.parallelism: 32
      env_vars:
        HF_TOKEN: HF_TOKEN_FOR_GPQA_DIAMOND
    
    - name: mbpp
      overrides:
        config.params.extra.n_samples: 5
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
    model_id: meta/llama-3.1-8b-instruct
    url: https://integrate.api.nvidia.com/v1/chat/completions
    api_key_name: API_KEY

evaluation:
  tasks:
    - name: gpqa_diamond
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
  lepton_platform:
    tasks:
      env_vars:
        HF_TOKEN:
          value_from:
            secret_name_ref: "HUGGING_FACE_HUB_TOKEN_read"
        API_KEY: "UNIQUE_ENDPOINT_TOKEN"
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
    model_id: meta/llama-3.1-8b-instruct
    url: https://integrate.api.nvidia.com/v1/chat/completions
    api_key_name: API_KEY

evaluation:
  tasks:
    - name: gpqa_diamond
```

**Key Points:**
- Requires SLURM account and accessible output directory
- Creates one job per benchmark evaluation
- Uses CPU partitions (no GPUs needed for none deployment)
- Supports CLI overrides for flexible job submission
:::

::::

## Advanced Features

### CLI Overrides

Override any configuration value from the command line using dot notation:

```bash
# Override execution settings
nemo-evaluator-launcher run --config-name your_config execution.walltime="1:00:00"

# Override endpoint URL
nemo-evaluator-launcher run --config-name your_config target.api_endpoint.url="https://new-endpoint.com/v1/chat/completions"

# Override evaluation parameters  
nemo-evaluator-launcher run --config-name your_config evaluation.overrides.config.params.temperature=0.8
```

### Common Configuration Overrides

**Request Parameters:**
- `config.params.temperature`: Control randomness (0.0-1.0)
- `config.params.max_new_tokens`: Maximum response length
- `config.params.parallelism`: Concurrent request limit
- `config.params.request_timeout`: Request timeout in seconds

**Task-Specific:**
- `config.params.extra.n_samples`: Number of samples per prompt (for code tasks)
- Environment variables for dataset access (like `HF_TOKEN`)

## Automatic Result Export

Automatically export evaluation results to multiple destinations for experiment tracking and collaboration.

**Supported Destinations**: W&B, MLflow, Google Sheets

### Basic Configuration

```yaml
execution:
  auto_export:
    destinations: ["wandb", "mlflow", "gsheets"]
    configs:
      wandb:
        entity: "your-team"
        project: "llm-evaluation"
        name: "experiment-name"
        tags: ["llama-3.1", "baseline"]
        log_metrics: ["accuracy", "pass@1"]
        
      mlflow:
        tracking_uri: "http://mlflow.company.com:5000"
        experiment_name: "LLM-Baselines-2024"
        log_metrics: ["accuracy", "pass@1"]
        
      gsheets:
        spreadsheet_name: "LLM Evaluation Results"
        log_mode: "multi_task"
```

/// note
For detailed exporter configuration, see {ref}`exporters-overview`.
///

### Key Configuration Options

- **`log_metrics`**: Filter which metrics to export (e.g., `["accuracy", "pass@1"]`)
- **`log_mode`**: "multi_task" (all tasks together) or "per_task" (separate entries)
- **`extra_metadata`**: Additional experiment metadata and tags
- **Environment variables**: Use `${oc.env:VAR_NAME}` for secure credential handling
