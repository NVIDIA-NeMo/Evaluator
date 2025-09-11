(executor-slurm)=

# Slurm Executor

The Slurm executor runs evaluations on high‑performance computing (HPC) clusters managed by Slurm, an open‑source workload manager widely used in research and enterprise environments. It schedules and executes jobs across cluster nodes, enabling parallel, large‑scale evaluation runs while preserving reproducibility via containerized benchmarks.

See common concepts and commands in {ref}`executors-overview`.

Slurm can optionally host your model for the scope of an evaluation by deploying a serving container on the cluster and pointing the benchmark to that temporary endpoint. In this mode, two containers are used: one for the evaluation harness and one for the model server. The evaluation configuration includes a deployment section when this is enabled. See the examples in the examples/ directory for ready‑to‑use configurations.

If you do not require deployment on Slurm, simply omit the deployment section from your configuration and set the model's endpoint URL directly (any OpenAI‑compatible endpoint that you host elsewhere).

## Prerequisites
- Access to a Slurm cluster (with appropriate partitions/queues)
- Docker or container runtime available on worker nodes (per your environment)

## Configuration Example

Here's a complete Slurm executor configuration with model deployment:

```yaml
# examples/slurm_llama_3_1_8b_instruct.yaml
defaults:
  - execution: slurm
  - deployment: vllm
  - _self_

execution:
  output_dir: /shared/results
  partition: gpu
  nodes: 1
  gpus_per_node: 8
  time_limit: "04:00:00"

deployment:
  type: vllm
  model_path: /shared/models/llama-3.1-8b-instruct
  
target:
  api_endpoint:
    url: http://localhost:8000/v1/chat/completions
    model_id: meta/llama-3.1-8b-instruct
    
evaluation:
  tasks:
    - name: hellaswag
    - name: arc_challenge  
    - name: winogrande
```

This configuration:
- Uses the `slurm` execution backend
- Deploys a vLLM model server on the cluster
- Requests GPU resources (1 node, 8 GPUs, 4-hour time limit)
- Points the evaluation to the deployed model endpoint
- Runs three common benchmark tasks
