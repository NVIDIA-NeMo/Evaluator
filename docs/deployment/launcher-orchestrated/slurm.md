(launcher-orchestrated-slurm)=

# Slurm Deployment via Launcher

Deploy and evaluate models on HPC clusters using Slurm workload manager through NeMo Evaluator Launcher orchestration.

## Overview

Slurm launcher-orchestrated deployment:

- Submits jobs to Slurm-managed HPC clusters
- Supports multi-node evaluation runs
- Handles resource allocation and job scheduling
- Manages model deployment lifecycle within Slurm jobs

## Quick Start

```bash
# Deploy and evaluate on Slurm cluster
nemo-evaluator-launcher run \
    --config packages/nemo-evaluator-launcher/examples/slurm_vllm_checkpoint_path.yaml \
    -o deployment.checkpoint_path=/shared/models/llama-3.1-8b-instruct \
    -o execution.partition=gpu
```

## vLLM Deployment

```yaml
# Slurm with vLLM deployment
defaults:
  - execution: slurm/default
  - deployment: vllm
  - _self_

deployment:
  type: vllm
  checkpoint_path: /shared/models/llama-3.1-8b-instruct
  served_model_name: meta-llama/Llama-3.1-8B-Instruct
  tensor_parallel_size: 1
  data_parallel_size: 8
  port: 8000

execution:
  account: my-account
  output_dir: /shared/results
  partition: gpu
  num_nodes: 1
  ntasks_per_node: 1
  gres: gpu:8
  walltime: "02:00:00"

target:
  api_endpoint:
    url: http://localhost:8000/v1/chat/completions
    model_id: meta-llama/Llama-3.1-8B-Instruct

evaluation:
  tasks:
    - name: ifeval
    - name: gpqa_diamond
    - name: mbpp
```

## Slurm Configuration

### Supported Parameters

The following execution parameters are supported for Slurm deployments. See `configs/execution/slurm/default.yaml` in the launcher package for the base configuration:

```yaml
execution:
  # Required parameters
  hostname: ???                         # Slurm cluster hostname
  username: ${oc.env:USER}             # SSH username (defaults to USER environment variable)
  account: ???                          # Slurm account for billing
  output_dir: ???                       # Results directory
  
  # Resource allocation
  partition: batch                      # Slurm partition/queue
  num_nodes: 1                         # Number of nodes
  ntasks_per_node: 1                   # Tasks per node
  gres: gpu:8                          # GPU resources
  walltime: "01:00:00"                 # Wall time limit (HH:MM:SS)
  
  # Environment variables and mounts
  env_vars:
    deployment: {}                     # Environment variables for deployment container
    evaluation: {}                     # Environment variables for evaluation container
  mounts:
    deployment: {}                     # Mount points for deployment container (source:target format)
    evaluation: {}                     # Mount points for evaluation container (source:target format)
    mount_home: true                   # Whether to mount home directory
```

:::{note}
The `gpus_per_node` parameter can be used as an alternative to `gres` for specifying GPU resources. However, `gres` is the default in the base configuration.
:::

## Configuration Examples

### Benchmark Suite Evaluation

```yaml
# Run multiple benchmarks on a single model
defaults:
  - execution: slurm/default
  - deployment: vllm
  - _self_

deployment:
  type: vllm
  checkpoint_path: /shared/models/llama-3.1-8b-instruct
  served_model_name: meta-llama/Llama-3.1-8B-Instruct
  tensor_parallel_size: 1
  data_parallel_size: 8
  port: 8000

execution:
  account: my-account
  output_dir: /shared/results
  hostname: slurm.example.com
  partition: gpu
  num_nodes: 1
  ntasks_per_node: 1
  gres: gpu:8
  walltime: "06:00:00"

target:
  api_endpoint:
    url: http://localhost:8000/v1/chat/completions
    model_id: meta-llama/Llama-3.1-8B-Instruct

evaluation:
  tasks:
    - name: ifeval
    - name: gpqa_diamond
    - name: mbpp
    - name: hellaswag
```

### Tasks Requiring Dataset Mounting

Some tasks require access to local datasets stored on the cluster's shared filesystem:

```yaml
evaluation:
  tasks:
    - name: mteb.techqa
      dataset_dir: /shared/datasets/techqa  # Path on shared filesystem
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
      dataset_dir: /shared/datasets/techqa
      dataset_mount_path: /data/techqa  # Optional: customize container mount point
```

:::{note}
Ensure the dataset directory is accessible from all cluster nodes via shared storage (e.g., NFS, Lustre).
:::

## Job Management

### Submitting Jobs

```bash
# Submit job with configuration
nemo-evaluator-launcher run \
    --config packages/nemo-evaluator-launcher/examples/slurm_vllm_basic.yaml

# Submit with configuration overrides
nemo-evaluator-launcher run \
    --config packages/nemo-evaluator-launcher/examples/slurm_vllm_basic.yaml \
    -o execution.walltime="04:00:00" \
    -o execution.partition=gpu-long
```

### Monitoring Jobs

```bash
# Check job status
nemo-evaluator-launcher status <job_id>

# List all runs (optionally filter by executor)
nemo-evaluator-launcher ls runs --executor slurm
```

### Managing Jobs

```bash
# Cancel job
nemo-evaluator-launcher kill <job_id>
```

### Native Slurm Commands

You can also use native Slurm commands to manage jobs directly:

```bash
# View job details
squeue -j <slurm_job_id> -o "%.18i %.9P %.50j %.8u %.2t %.10M %.6D %R"

# Check job efficiency
seff <slurm_job_id>

# Cancel Slurm job directly
scancel <slurm_job_id>

# Hold/release job
scontrol hold <slurm_job_id>
scontrol release <slurm_job_id>

# View detailed job information
scontrol show job <slurm_job_id>
```

## Shared Storage

Slurm evaluations require shared storage accessible from all cluster nodes:

### Model Storage

Store models in a shared filesystem accessible to all compute nodes:

```bash
# Example shared model directory
/shared/models/
├── llama-3.1-8b-instruct/
├── llama-3.1-70b-instruct/
└── custom-model.nemo
```

Specify the model path in your configuration:

```yaml
deployment:
  checkpoint_path: /shared/models/llama-3.1-8b-instruct
```

### Results Storage

Evaluation results are written to the configured output directory:

```yaml
execution:
  output_dir: /shared/results
```

Results are organized by timestamp and invocation ID in subdirectories.

## Troubleshooting

### Common Issues

**Job Pending:**

```bash
# Check node availability
sinfo -p gpu

# Try different partition
-o execution.partition="gpu-shared"
```

**Job Failed:**

```bash
# Check job status
nemo-evaluator-launcher status <job_id>

# View Slurm job details
scontrol show job <slurm_job_id>

# Check job output logs (location shown in status output)
```

**Job Timeout:**

```bash
# Increase walltime
-o execution.walltime="08:00:00"

# Check current walltime limit for partition
sinfo -p <partition_name> -o "%P %l"
```

**Resource Allocation:**

```bash
# Adjust GPU allocation via gres
-o execution.gres=gpu:4
-o deployment.tensor_parallel_size=4
```

### Debugging with Slurm Commands

```bash
# View job details
scontrol show job <slurm_job_id>

# Monitor resource usage
sstat -j <slurm_job_id> --format=AveCPU,AveRSS,MaxRSS,AveVMSize

# Job accounting information
sacct -j <slurm_job_id> --format=JobID,JobName,State,ExitCode,DerivedExitCode

# Check job efficiency after completion
seff <slurm_job_id>
```
