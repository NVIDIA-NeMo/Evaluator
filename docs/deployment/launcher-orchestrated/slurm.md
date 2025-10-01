(launcher-orchestrated-slurm)=

# Slurm Deployment via Launcher

Deploy and evaluate models on HPC clusters using Slurm workload manager through NeMo Evaluator Launcher orchestration. This approach enables large-scale evaluations with multi-node model parallelism.

## Overview

Slurm launcher-orchestrated deployment:
- Submits jobs to Slurm-managed HPC clusters
- Supports multi-node model parallelism
- Handles resource allocation and job scheduling
- Manages deployment lifecycle within Slurm jobs

Based on PR #108's Slurm execution backend implementation.

## Quick Start

```bash
# Deploy and evaluate on Slurm cluster
nemo-evaluator-launcher run \
    --config-dir examples \
    --config-name slurm_llama_3_1_8b_instruct \
    -o deployment.model_path=/shared/models/llama-3.1-8b.nemo \
    -o execution.partition=gpu
```

## Deployment Types

### NIM Slurm Deployment

Production-grade serving with enterprise features:

```yaml
# config/slurm_nim.yaml
deployment:
  type: nim
  model_path: /shared/models/llama-3.1-8b.nemo
  container_image: nvcr.io/nim/llama-3.1-8b-instruct
  port: 8000
  max_batch_size: 16

execution:
  backend: slurm
  partition: gpu
  nodes: 1
  gpus_per_node: 4
  cpus_per_task: 8
  memory: 64G
  time_limit: "02:00:00"
  job_name: "nemo-eval-nim"

evaluation:
  tasks:
    - name: mmlu_pro
      params:
        limit_samples: 1000
    - name: gsm8k
      params:
        limit_samples: 500

target:
  api_endpoint:
    url: http://localhost:8000/v1/chat/completions
    model_id: llama-3.1-8b-instruct
```

### vLLM Slurm Deployment

High-performance inference with multi-GPU support:

```yaml
# config/slurm_vllm.yaml
deployment:
  type: vllm
  model_path: /shared/models/llama-3.1-8b-instruct
  port: 8080
  tensor_parallel_size: 4  # Use 4 GPUs
  gpu_memory_utilization: 0.9
  max_model_len: 4096

execution:
  backend: slurm
  partition: gpu
  nodes: 1
  gpus_per_node: 4
  cpus_per_task: 16
  memory: 128G
  time_limit: "04:00:00"
  exclusive: true  # Exclusive node access

evaluation:
  tasks:
    - name: mmlu_pro
    - name: gsm8k
    - name: humaneval
    - name: hellaswag
  
  parallelism: 8  # High parallelism for multi-GPU
```

### Multi-Node Deployment

Large model deployment across multiple nodes:

```yaml
# config/slurm_multi_node.yaml
deployment:
  type: vllm
  model_path: /shared/models/llama-3.1-70b-instruct
  tensor_parallel_size: 8  # 8 GPUs total
  pipeline_parallel_size: 2  # 2-way pipeline parallelism
  gpu_memory_utilization: 0.95

execution:
  backend: slurm
  partition: gpu
  nodes: 2  # 2 nodes
  gpus_per_node: 4  # 4 GPUs per node = 8 total
  cpus_per_task: 32
  memory: 256G
  time_limit: "08:00:00"
  constraint: "a100"  # Require specific GPU type

evaluation:
  tasks:
    - name: mmlu_pro
      params:
        limit_samples: 5000  # Large evaluation
```

## Slurm Configuration

### Resource Specifications

```yaml
execution:
  backend: slurm
  
  # Basic resource allocation
  partition: gpu              # Slurm partition
  nodes: 1                   # Number of nodes
  gpus_per_node: 4           # GPUs per node
  cpus_per_task: 16          # CPU cores
  memory: 64G                # Memory per node
  
  # Time and priority
  time_limit: "02:00:00"     # Wall time limit
  priority: high             # Job priority
  
  # Node selection
  constraint: "a100"         # Require specific hardware
  exclusive: true            # Exclusive node access
  
  # Job metadata
  job_name: "nemo-evaluation"
  account: "my-project"      # Billing account
  qos: "normal"              # Quality of service
```

### Advanced Slurm Options

```yaml
execution:
  backend: slurm
  
  # Array jobs for multiple evaluations
  array: "1-10"              # Job array indices
  
  # Dependencies
  dependency: "afterok:12345" # Wait for job 12345
  
  # Output and error files
  output: "/shared/logs/eval_%j.out"
  error: "/shared/logs/eval_%j.err"
  
  # Email notifications
  mail_type: "END,FAIL"
  mail_user: "user@domain.com"
  
  # Environment
  export: "ALL"              # Export all environment variables
  
  # Working directory
  chdir: "/shared/workspace"
```

## Configuration Examples

### Benchmark Suite Evaluation

```yaml
# config/slurm_benchmark_suite.yaml
deployment:
  type: vllm
  model_path: /shared/models/llama-3.1-8b-instruct
  tensor_parallel_size: 2

execution:
  backend: slurm
  partition: gpu
  nodes: 1
  gpus_per_node: 2
  time_limit: "06:00:00"
  job_name: "benchmark-suite"

evaluation:
  tasks:
    # Language understanding
    - name: mmlu_pro
    - name: arc_challenge
    - name: hellaswag
    
    # Reasoning
    - name: gsm8k
    - name: math
    
    # Code generation
    - name: humaneval
    - name: mbpp
    
    # Safety
    - name: toxigen
    - name: truthfulqa
  
  parallelism: 4
```

### Model Comparison Study

```yaml
# config/slurm_model_comparison.yaml
# Use Slurm job arrays to compare multiple models

deployment:
  type: vllm
  model_path: /shared/models/model_${SLURM_ARRAY_TASK_ID}
  tensor_parallel_size: 4

execution:
  backend: slurm
  partition: gpu
  nodes: 1
  gpus_per_node: 4
  time_limit: "03:00:00"
  array: "1-5"  # Compare 5 different models
  job_name: "model-comparison"

evaluation:
  tasks:
    - name: mmlu_pro
    - name: gsm8k
  
  # Add model identifier to results
  metadata:
    model_id: "${SLURM_ARRAY_TASK_ID}"
    experiment: "model-comparison-study"
```

### Long-Running Evaluation

```yaml
# config/slurm_long_eval.yaml
deployment:
  type: nim
  model_path: /shared/models/large-model.nemo

execution:
  backend: slurm
  partition: gpu
  nodes: 2
  gpus_per_node: 8
  time_limit: "24:00:00"  # 24 hours
  job_name: "long-evaluation"
  
  # Checkpointing for long jobs
  signal: "SIGUSR1@300"  # Signal 5 minutes before time limit
  requeue: true          # Allow job requeuing

evaluation:
  tasks:
    - name: mmlu_pro
      params:
        limit_samples: 10000  # Full dataset
    
  # Enable checkpointing
  checkpoint_interval: 1000  # Save every 1000 samples
  resume_from_checkpoint: true
```

## Job Management

### Submitting Jobs

```bash
# Submit job with configuration
nemo-evaluator-launcher run \
    --config-dir examples \
    --config-name slurm_llama_3_1_8b_instruct

# Submit with overrides
nemo-evaluator-launcher run \
    --config-dir examples \
    --config-name slurm_llama_3_1_8b_instruct \
    -o execution.nodes=2 \
    -o execution.time_limit="04:00:00"

# Submit job array
nemo-evaluator-launcher run \
    --config-dir examples \
    --config-name slurm_model_comparison \
    -o execution.array="1-10"
```

### Monitoring Jobs

```bash
# Check job status
nemo-evaluator-launcher status <job_id>

# List all Slurm runs
nemo-evaluator-launcher ls-runs --backend slurm

# View job details
squeue -j <slurm_job_id> -o "%.18i %.9P %.50j %.8u %.2t %.10M %.6D %R"

# Check job efficiency
seff <slurm_job_id>
```

### Managing Jobs

```bash
# Cancel job
nemo-evaluator-launcher kill <job_id>

# Cancel Slurm job directly
scancel <slurm_job_id>

# Hold/release job
scontrol hold <slurm_job_id>
scontrol release <slurm_job_id>

# Modify job parameters
scontrol update JobId=<slurm_job_id> TimeLimit=06:00:00
```

## Shared Storage Setup

### Model Storage

```bash
# Shared model directory structure
/shared/models/
├── llama-3.1-8b-instruct/
│   ├── config.json
│   ├── tokenizer.json
│   └── pytorch_model.bin
├── llama-3.1-70b-instruct/
└── custom-model.nemo

# Ensure proper permissions
chmod -R 755 /shared/models/
```

### Results Storage

```yaml
# config with shared results directory
evaluation:
  output_dir: "/shared/results/${SLURM_JOB_ID}"
  
# Automatic cleanup after export
cleanup:
  remove_temp_files: true
  keep_results_days: 30
```

### Cache Management

```yaml
target:
  api_endpoint:
    adapter_config:
      use_caching: true
      caching_dir: "/shared/cache/${USER}"
      
# Shared cache with proper permissions
cache:
  shared: true
  cleanup_policy: "lru"  # Least recently used
  max_size: "100GB"
```

## Performance Optimization

### Resource Allocation

```yaml
# Optimize for throughput
execution:
  backend: slurm
  nodes: 1
  gpus_per_node: 8
  cpus_per_task: 32  # High CPU:GPU ratio
  memory: 256G

deployment:
  type: vllm
  tensor_parallel_size: 8
  max_batch_size: 64  # Large batches
  gpu_memory_utilization: 0.95

evaluation:
  parallelism: 16  # High parallelism
```

### Network Optimization

```yaml
execution:
  # Use high-speed interconnect
  constraint: "infiniband"
  
  # Optimize for network-intensive workloads
  network: "ib"
  
deployment:
  # Multi-node communication settings
  distributed_backend: "nccl"
  nccl_socket_ifname: "ib0"
```

## Troubleshooting

### Common Slurm Issues

**Job Pending (QOSMaxJobsPerUserLimit):**
```bash
# Check limits
sacctmgr show qos format=Name,MaxJobsPerUser

# Use different QOS
-o execution.qos="normal"
```

**Job Failed (Out of Memory):**
```bash
# Increase memory allocation
-o execution.memory="128G"

# Use fewer GPUs per node
-o execution.gpus_per_node=2
-o deployment.tensor_parallel_size=2
```

**Job Timeout:**
```bash
# Increase time limit
-o execution.time_limit="08:00:00"

# Enable checkpointing
-o evaluation.checkpoint_interval=500
```

**Node Allocation Issues:**
```bash
# Check node availability
sinfo -p gpu

# Remove constraints
-o execution.constraint=""

# Use different partition
-o execution.partition="gpu-shared"
```

### Debugging Slurm Jobs

```bash
# View job script
scontrol show job <slurm_job_id>

# Check job output
cat /shared/logs/eval_<slurm_job_id>.out

# Monitor resource usage
sstat -j <slurm_job_id> --format=AveCPU,AveRSS,MaxRSS,AveVMSize

# Job accounting information
sacct -j <slurm_job_id> --format=JobID,JobName,State,ExitCode,DerivedExitCode
```

## Best Practices

### Resource Management
- **Request appropriate resources**: Don't over-allocate to avoid waste
- **Use exclusive access**: For consistent performance on shared clusters
- **Monitor efficiency**: Use `seff` to check resource utilization
- **Plan for queuing**: Account for queue wait times in planning

### Job Design
- **Use job arrays**: For parameter sweeps and model comparisons
- **Enable checkpointing**: For long-running evaluations
- **Set reasonable time limits**: Balance between completion and queue priority
- **Handle failures gracefully**: Use error handling and cleanup scripts

### Data Management
- **Use shared storage**: Avoid copying large models to local storage
- **Implement caching**: Share cache across jobs to reduce redundancy
- **Clean up temporary files**: Remove intermediate files after completion
- **Backup important results**: Copy results to permanent storage

### Cluster Etiquette
- **Follow cluster policies**: Respect resource limits and usage guidelines
- **Monitor your jobs**: Don't submit and forget
- **Use appropriate partitions**: Choose partitions that match your needs
- **Communicate with admins**: Report issues and ask for guidance

## Next Steps

- **Scale further**: Explore multi-cluster deployment strategies
- **Optimize costs**: Use [cloud deployment](lepton.md) for cost comparison
- **Automate workflows**: Set up CI/CD pipelines with Slurm integration
- **Advanced monitoring**: Implement comprehensive job monitoring and alerting
