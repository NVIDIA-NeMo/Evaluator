(executor-slurm)=

# Slurm Executor

The Slurm executor runs evaluations on high‑performance computing (HPC) clusters managed by Slurm, an open‑source workload manager widely used in research and enterprise environments. It schedules and executes jobs across cluster nodes, enabling parallel, large‑scale evaluation runs while preserving reproducibility via containerized benchmarks.

See common concepts and commands in {ref}`executors-overview`.

Slurm can optionally host your model for the scope of an evaluation by deploying a serving container on the cluster and pointing the benchmark to that temporary endpoint. In this mode, two containers are used: one for the evaluation harness and one for the model server. The evaluation configuration includes a deployment section when this is enabled. See the examples in the examples/ directory for ready‑to‑use configurations.

If you do not require deployment on Slurm, simply omit the deployment section from your configuration and set the model's endpoint URL directly (any OpenAI‑compatible endpoint that you host elsewhere).

## Prerequisites
- Access to a Slurm cluster (with appropriate partitions/queues)
- [Pyxis SPANK plugin](https://github.com/NVIDIA/pyxis) installed on the cluster 

## Configuration Overview

### Connecting to Your Slurm Cluster

To run evaluations on Slurm, specify how to connect to your cluster

```yaml
execution:
  hostname: your-cluster-headnode      # Slurm headnode (login node)
  username: your_username            # Cluster username (defaults to $USER env var)
  account: your_allocation           # Slurm account or project name
  output_dir: /shared/scratch/your_username/eval_results  # Absolute, shared path
```

:::{note}
When specifying the parameters make sure to provide:
- `hostname`: Slurm headnode (login node) where you normally SSH to submit jobs.
- `output_dir`: must be an **absolute path** on a shared filesystem (e.g., /shared/scratch/ or /home/) accessible to both the headnode and compute nodes.
:::

### Model Deployment Options

When deploying models on Slurm, you have two options for specifying your model source:

#### Option 1: HuggingFace Models (Recommended - Automatic Download)

- Use valid Hugging Face model IDs for `hf_model_handle` (for example, `meta-llama/Llama-3.1-8B-Instruct`).  
- Browse model IDs: [Hugging Face Models](https://huggingface.co/models).

```yaml
deployment:
  checkpoint_path: null  # Set to null when using hf_model_handle
  hf_model_handle: meta-llama/Llama-3.1-8B-Instruct  # HuggingFace model ID
```

**Benefits:**
- Model is automatically downloaded during deployment
- No need to pre-download or manage model files
- Works with any HuggingFace model (public or private with valid access tokens)

**Requirements:**
- Set `HF_TOKEN` environment variable if accessing gated models
- Internet access from compute nodes (or model cached locally)

#### Option 2: Local Model Files (Manual Setup Required)

If you work with a checkpoint stored on locally on the cluster, use `checkpoint_path`:

```yaml
deployment:
  checkpoint_path: /shared/models/llama-3.1-8b-instruct  # model directory accessible to compute nodes
  # Do NOT set hf_model_handle when using checkpoint_path
```

**Note:**
- The directory must exist, be accessible from compute nodes, and contain model files
- Slurm does not automatically download models when using `checkpoint_path`

### Environment Variables

The Slurm executor supports environment variables through `execution.env_vars`:

```yaml
execution:
  env_vars:
    deployment:
      CUDA_VISIBLE_DEVICES: "0,1,2,3,4,5,6,7"
      USER: ${oc.env:USER}  # References host environment variable
    evaluation:
      CUSTOM_VAR: YOUR_CUSTOM_ENV_VAR_VALUE  # set the value directly
evaluation:
  env_vars:
    CUSTOM_VAR: ${oc.env:YOUR_CUSTOM_ENV_VAR_NAME}  # use YOUR_CUSTOM_ENV_VAR_NAME to populate CUSTOM_VAR
  tasks:
    - name: my_task
      env_vars:
        TASK_SPECIFIC_VAR: TASK_ENV_VAR_VALUE
```

**How to use environment variables:**

- **Deployment Variables**: Use `execution.env_vars.deployment` for model serving containers
- **Evaluation Variables**: Use `execution.env_vars.evaluation` for evaluation containers
- **Direct Values**: Use quoted strings for direct values
- **Hydra Environment Variables**: Use `${oc.env:VARIABLE_NAME}` to reference host environment variables

### Secrets and API Keys

API keys are handled the same way as environment variables - store them as environment variables on your machine and reference them in the `execution.env_vars` configuration.

**Security Considerations:**

- **No Hardcoding**: Never put API keys directly in configuration files, use `${oc.env:ENV_VAR_NAME}` instead.
- **SSH Security**: Ensure secure SSH configuration for key transmission to the cluster.
- **File Permissions**: Ensure configuration files have appropriate permissions (not world-readable).
- **Public Clusters**: Secrets in `execution.env_vars` are stored in plain text in the batch script and saved under `output_dir` on the login node. Use caution when handling sensitive data on public clusters.

### Mounting and Storage

The Slurm executor provides sophisticated mounting capabilities:

```yaml
execution:
  mounts:
    deployment:
      /path/to/checkpoints: /checkpoint
      /path/to/cache: /cache
    evaluation:
      /path/to/data: /data
      /path/to/results: /results
    mount_home: true  # Mount user home directory
```

**Mount Types:**:

- **Deployment Mounts**: For model checkpoints, cache directories, and model data.
- **Evaluation Mounts**: For input data, additional artifacts, and evaluation-specific files
- **Home Mount**: Optional mounting of user home directory (enabled by default)


## Complete Configuration Example

Here's a complete Slurm executor configuration using HuggingFace models:

```yaml
# examples/slurm_llama_3_1_8b_instruct.yaml
defaults:
  - execution: slurm/default
  - deployment: vllm
  - _self_

execution:
  hostname: your-cluster-headnode
  account: your_account
  output_dir: /shared/scratch/your_username/eval_results
  partition: gpu
  walltime: "04:00:00"
  gpus_per_node: 8
  env_vars:
    deployment:
      HF_TOKEN: ${oc.env:HF_TOKEN}   # Needed to access meta-llama/Llama-3.1-8B-Instruct gated model

deployment:
  hf_model_handle: meta-llama/Llama-3.1-8B-Instruct
  checkpoint_path: null
  served_model_name: meta-llama/Llama-3.1-8B-Instruct
  tensor_parallel_size: 1
  data_parallel_size: 8
    
evaluation:
  tasks:
    - name: hellaswag
    - name: arc_challenge  
    - name: winogrande
```

This configuration:
- Uses the Slurm execution backend
- Deploys a vLLM model server on the cluster
- Requests GPU resources (8 GPUs per node, 4-hour time limit)
- Runs three benchmark tasks in parallel
- Saves benchmark artifacts to `output_dir`


## Resuming

The Slurm executor includes advanced auto-resume capabilities:

### Automatic Resumption
- **Timeout Handling**: Jobs automatically resume after timeout
- **Preemption Recovery**: Automatic resumption after job preemption
- **Node Failure Recovery**: Jobs resume after node failures
- **Dependency Management**: Uses Slurm job dependencies for resumption

### How It Works
1. **Initial Submission**: Job is submitted with auto-resume handler
2. **Failure Detection**: Script detects timeout/preemption/failure
3. **Automatic Resubmission**: New job is submitted with dependency on previous job
4. **Progress Preservation**: Evaluation continues from where it left off

## Monitoring and Job Management

For monitoring jobs, checking status, and managing evaluations, see the [Executors Overview](overview.md#job-management) section.


