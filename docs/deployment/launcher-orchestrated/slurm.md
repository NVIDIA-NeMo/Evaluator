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

### Multi-Node Multi-Instance Deployment

For models too large to fit on a single node (e.g., DeepSeek-R1 requiring 2 nodes per instance), you can run multiple instances across nodes with HAProxy load balancing. Each instance spans multiple nodes using Ray tensor/pipeline parallelism.

**Architecture** (4 nodes total, 2 instances of 2 nodes each):
- Instance 0 (nodes 0,1): Ray head + worker, vLLM on :8000
- Instance 1 (nodes 2,3): Ray head + worker, vLLM on :8000
- HAProxy: distributes requests across both instances

```yaml
defaults:
  - execution: slurm/default
  - deployment: vllm
  - _self_

execution:
  hostname: slurm.example.com
  username: ${oc.env:USER}
  account: my-account
  output_dir: /shared/results
  num_nodes: 4  # Total SLURM nodes: 2 instances × 2 nodes each
  deployment:
    n_tasks: ${execution.num_nodes}  # Must match num_nodes
  mounts:
    deployment:
      /path/to/hf_home: /root/.cache/huggingface
    mount_home: false
  env_vars:
    deployment:
      VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE: 'shm'
      HF_TOKEN: ${oc.env:HF_TOKEN}  # Required if downloading model from HuggingFace

deployment:
  image: vllm/vllm-openai:v0.15.1
  multiple_instances: true   # Enable HAProxy load balancing across instances
  nodes_per_instance: 2      # Each instance spans 2 nodes
  checkpoint_path: null
  hf_model_handle: deepseek-ai/DeepSeek-R1
  served_model_name: deepseek-ai/DeepSeek-R1
  tensor_parallel_size: 8
  pipeline_parallel_size: 2
  data_parallel_size: 1
  port: 8000
  extra_args: "--disable-custom-all-reduce --distributed-executor-backend ray --enforce-eager"
  pre_cmd: |
      # Fixed ports for Ray services to avoid conflicts between instances
      RAY_PORT=6379
      NODE_PORT=8266
      OBJ_PORT=8267
      RAY_FIXED_PORTS="--node-manager-port=$NODE_PORT --object-manager-port=$OBJ_PORT --metrics-export-port=8269 --dashboard-agent-grpc-port=8270 --dashboard-agent-listen-port=8271 --runtime-env-agent-port=8272"

      # INSTANCE_RANK is injected by the launcher: 0 = head node, 1+ = worker nodes
      if [ "$INSTANCE_RANK" -eq 0 ]; then
          # Head node: set VLLM_HOST_IP so vLLM advertises the correct routable IP to Ray
          export VLLM_HOST_IP=$MASTER_IP
          # Start Ray head and wait for all worker nodes in this instance to join
          ray start --head --port=$RAY_PORT $RAY_FIXED_PORTS
          export RAY_ADDRESS=$MASTER_IP:$RAY_PORT
          until [ "$(ray status 2>/dev/null | grep -c 'node_')" -ge "$NODES_PER_INSTANCE" ]; do
              sleep 10
          done
          ray status
      else
          # Worker node: connect to the head node's Ray cluster and block until terminated
          until ray start --address=$MASTER_IP:$RAY_PORT $RAY_FIXED_PORTS --block 2>/dev/null; do
              sleep 5
          done
      fi

evaluation:
  nemo_evaluator_config:
    config:
      params:
        parallelism: 128
        request_timeout: 3600
        temperature: 0.6
        top_p: 0.95
        max_new_tokens: 32768
  tasks:
    - name: gsm8k_cot_instruct
```

Key parameters:
- **`deployment.nodes_per_instance`**: Number of nodes per vLLM instance (default: 1)
- **`deployment.multiple_instances: true`**: Required — enables HAProxy load balancing
- **`execution.deployment.n_tasks`**: Must equal `num_nodes` (one srun task per node)
- **`execution.env_vars.deployment`**: Environment variables passed into the deployment container

The launcher automatically injects per-task variables inside the container:
- **`INSTANCE_ID`**: Which instance this task belongs to (0, 1, ...)
- **`INSTANCE_RANK`**: Rank within the instance (0 = head, 1+ = worker)
- **`INSTANCE_MASTER_IP`**: IP of the head node for this instance
- **`MASTER_IP`**: Overridden to `INSTANCE_MASTER_IP` so Ray connects within the instance
- **`NODES_PER_INSTANCE`**: Number of nodes per instance
- **`NUM_INSTANCES`**: Total number of instances
- **`ALL_NODE_IPS`**: Comma-separated list of all node IPs

See the full example at `examples/slurm_vllm_multinode_multiinstance_ray_tp_pp.yaml`.

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
