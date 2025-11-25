(launcher-orchestrated-lepton)=

# Lepton AI Deployment via Launcher

Deploy and evaluate models on Lepton AI cloud platform using NeMo Evaluator Launcher orchestration. This approach provides scalable cloud inference with managed infrastructure.

## Overview

Lepton launcher-orchestrated deployment:

- Deploys models on Lepton AI cloud platform
- Provides managed infrastructure and scaling
- Supports various resource shapes and configurations
- Handles deployment lifecycle in the cloud

## Quick Start

```bash
# Deploy and evaluate on Lepton AI
nemo-evaluator-launcher run \
    --config packages/nemo-evaluator-launcher/examples/lepton_vllm_llama_3_1_8b_instruct.yaml \
    -o deployment.checkpoint_path=meta-llama/Llama-3.1-8B-Instruct \
    -o deployment.lepton_config.resource_shape=gpu.1xh200
```

This command:

1. Deploys a vLLM endpoint on Lepton AI
2. Runs the configured evaluation tasks
3. Returns an invocation ID for monitoring

The launcher handles endpoint creation, evaluation execution, and provides cleanup commands.

## Prerequisites

### Lepton AI Setup

```bash
# Install Lepton AI CLI
pip install leptonai

# Authenticate with Lepton AI
lep login
```

Refer to the [Lepton AI documentation](https://docs.nvidia.com/dgx-cloud/lepton/get-started/) for authentication and workspace configuration.

## Deployment Types

### vLLM Lepton Deployment

High-performance inference with cloud scaling:

Refer to the complete working configuration in `examples/lepton_vllm_llama_3_1_8b_instruct.yaml`. Key configuration sections:

```yaml
deployment:
  type: vllm
  checkpoint_path: meta-llama/Llama-3.1-8B-Instruct
  served_model_name: llama-3.1-8b-instruct
  tensor_parallel_size: 1
  
  lepton_config:
    resource_shape: gpu.1xh200
    min_replicas: 1
    max_replicas: 3
    auto_scaler:
      scale_down:
        no_traffic_timeout: 3600

execution:
  type: lepton
  evaluation_tasks:
    timeout: 3600

evaluation:
  tasks:
    - name: ifeval
```

The launcher automatically retrieves the endpoint URL after deployment, eliminating the need for manual URL configuration.

### NIM Lepton Deployment

Enterprise-grade serving in the cloud. Refer to the complete working configuration in `examples/lepton_nim_llama_3_1_8b_instruct.yaml`:

```yaml
deployment:
  type: nim
  image: nvcr.io/nim/meta/llama-3.1-8b-instruct:1.8.6
  served_model_name: meta/llama-3.1-8b-instruct
  
  lepton_config:
    resource_shape: gpu.1xh200
    min_replicas: 1
    max_replicas: 3
    auto_scaler:
      scale_down:
        no_traffic_timeout: 3600

execution:
  type: lepton

evaluation:
  tasks:
    - name: ifeval
```

### SGLang Deployment

SGLang is also supported as a deployment type. Use `deployment.type: sglang` with similar configuration to vLLM.

## Resource Shapes

Resource shapes are Lepton platform-specific identifiers that determine the compute resources allocated to your deployment. Available shapes depend on your Lepton workspace configuration and quota.

Configure in your deployment:

```yaml
deployment:
  lepton_config:
    resource_shape: gpu.1xh200  # Example: Check your Lepton workspace for available shapes
```

Refer to the [Lepton AI documentation](https://docs.nvidia.com/dgx-cloud/lepton/) or check your workspace settings for available resource shapes in your environment.

## Configuration Examples

### Auto-Scaling Configuration

Configure auto-scaling behavior through the `lepton_config.auto_scaler` section:

```yaml
deployment:
  lepton_config:
    min_replicas: 1
    max_replicas: 3
    auto_scaler:
      scale_down:
        no_traffic_timeout: 3600  # Seconds before scaling down
        scale_from_zero: false
```

### Using Existing Endpoints

To evaluate against an already-deployed Lepton endpoint without creating a new deployment, use `deployment.type: none` and provide the endpoint URL in the `target.api_endpoint` section.

Refer to `examples/lepton_none_llama_3_1_8b_instruct.yaml` for a complete example.

### Tasks Requiring Dataset Mounting

Some tasks require access to local datasets that must be mounted into the evaluation container:

```yaml
evaluation:
  tasks:
    - name: mteb.techqa
      dataset_dir: /path/to/shared/storage/techqa
```

The system will automatically:
- Mount the dataset directory into the evaluation container
- Set the `NEMO_EVALUATOR_DATASET_DIR` environment variable
- Validate that all required environment variables are configured

**Custom mount path example:**

```yaml
evaluation:
  tasks:
    - name: mteb.techqa
      dataset_dir: /lepton/shared/datasets/techqa
      dataset_mount_path: /data/techqa  # Optional: customize container mount point
```

:::{note}
Ensure the dataset directory is accessible from the Lepton platform's shared storage configured in your workspace.
:::

## Advanced Configuration

### Environment Variables

Pass environment variables to deployment containers through `lepton_config.envs`:

```yaml
deployment:
  lepton_config:
    envs:
      HF_TOKEN:
        value_from:
          secret_name_ref: "HUGGING_FACE_HUB_TOKEN"
      CUSTOM_VAR: "direct_value"
```

### Storage Mounts

Configure persistent storage for model caching:

```yaml
deployment:
  lepton_config:
    mounts:
      enabled: true
      cache_path: "/path/to/storage"
      mount_path: "/opt/nim/.cache"
```

## Monitoring and Management

### Check Evaluation Status

Use NeMo Evaluator Launcher commands to monitor your evaluations:

```bash
# Check status using invocation ID
nemo-evaluator-launcher status <invocation_id>

# Kill running evaluations and cleanup endpoints
nemo-evaluator-launcher kill <invocation_id>
```

### Monitor Lepton Resources

Use Lepton AI CLI commands to monitor platform resources:

```bash
# List all deployments in your workspace
lepton deployment list

# Get details about a specific deployment
lepton deployment get <deployment-name>

# View deployment logs
lepton deployment logs <deployment-name>

# Check resource availability
lepton resource list --available
```

Refer to the [Lepton AI CLI documentation](https://docs.nvidia.com/dgx-cloud/lepton/reference/cli/get-started/) for the complete command reference.

## Exporting Results

After evaluation completes, export results using the export command:

```bash
# Export results to MLflow
nemo-evaluator-launcher export <invocation_id> --dest mlflow
```

Refer to the {ref}`exporters-overview` for additional export options and configurations.

## Troubleshooting

### Common Issues

**Deployment Timeout:**

If endpoints take too long to become ready, check deployment logs:

```bash
# Check deployment logs via Lepton CLI
lepton deployment logs <deployment-name>

# Increase readiness timeout in configuration
# (in execution.lepton_platform.deployment.endpoint_readiness_timeout)
```

**Resource Unavailable:**

If your requested resource shape is unavailable:

```bash
# Check available resources in your workspace
lepton resource list --available

# Try a different resource shape in your config
```

**Authentication Issues:**

```bash
# Re-authenticate with Lepton
lep login
```

**Endpoint Not Found:**

If evaluation jobs cannot connect to the endpoint:

1. Verify endpoint is in "Ready" state using `lepton deployment get <deployment-name>`
2. Confirm the endpoint URL is accessible
3. Verify API tokens are properly set in Lepton secrets

## Next Steps

- Compare with {ref}`launcher-orchestrated-slurm` for HPC cluster deployments
- Explore {ref}`launcher-orchestrated-local` for local development and testing
- Review complete configuration examples in the `examples/` directory
