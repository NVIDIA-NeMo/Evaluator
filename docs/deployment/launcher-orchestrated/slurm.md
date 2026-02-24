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
  
  # Multi-node topology (see Multi-Node Deployment section below)
  num_nodes_per_instance: 1            # Nodes per deployment instance (default: 1)
  num_instances: 1                     # Number of independent deployment instances (default: 1)

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

## Multi-Node Deployment

The launcher supports deploying models across multiple SLURM nodes. Two execution parameters control the multi-node topology:

| Parameter | Default | Description |
|---|---|---|
| `num_nodes_per_instance` | 1 | Nodes per deployment instance. When > 1, Ray is auto-injected for cross-node coordination (vLLM only). |
| `num_instances` | 1 | Number of independent deployment instances. When > 1, HAProxy is auto-started to load-balance requests across instances. |

Total SLURM nodes allocated = `num_nodes_per_instance × num_instances`.

### Single Instance, Multiple Nodes (Ray)

Use when a model is too large for a single node and requires tensor/pipeline parallelism across nodes.

```yaml
execution:
  num_nodes_per_instance: 2  # Instance spans 2 nodes → Ray auto-injected

deployment:
  tensor_parallel_size: 8    # Within-node GPU parallelism
  pipeline_parallel_size: 2  # Cross-node model parallelism
```

The launcher automatically:
- Injects a Ray cluster setup script (head on node 0, workers on remaining nodes)
- Adds `--distributed-executor-backend ray` to vLLM args
- Sets SLURM `--ntasks` to match total nodes

See: `examples/slurm_vllm_multinode_ray_tp_pp.yaml`

### Multiple Single-Node Instances (HAProxy)

Use when a model fits on a single node but you want to scale throughput with independent replicas behind a load balancer.

```yaml
execution:
  num_instances: 2  # 2 independent instances → HAProxy auto-enabled

deployment:
  data_parallel_size: 8  # Each node uses all 8 GPUs for data parallelism
```

The launcher automatically:
- Starts HAProxy to distribute requests across all instances
- Sets SLURM `--ntasks` to match total nodes

See: `examples/slurm_vllm_multinode_dp_haproxy.yaml`

### Multiple Multi-Node Instances (Ray + HAProxy)

Use for very large models that need cross-node parallelism **and** multiple replicas for throughput.

```yaml
execution:
  num_nodes_per_instance: 2  # Each instance spans 2 nodes → Ray auto-injected
  num_instances: 2           # 2 instances → HAProxy auto-enabled
  # Total: 4 SLURM nodes

deployment:
  tensor_parallel_size: 8
  pipeline_parallel_size: 2
```

The launcher automatically:
- Sets up a Ray cluster within each instance
- Starts HAProxy to load-balance across instances
- Sets SLURM `--ntasks` to match total nodes (4)

See: `examples/slurm_vllm_multinode_multiinstance_ray_tp_pp.yaml`

:::{note}
Multi-node deployment (num_nodes_per_instance > 1) is only supported for vLLM deployments. The launcher auto-injects Ray and the `--distributed-executor-backend ray` flag — you do not need to configure these manually.
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
