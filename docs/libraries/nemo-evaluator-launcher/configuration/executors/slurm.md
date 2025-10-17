(executor-slurm)=

# Slurm Executor

The Slurm executor runs evaluations on high‑performance computing (HPC) clusters managed by Slurm, an open‑source workload manager widely used in research and enterprise environments. It schedules and executes jobs across cluster nodes, enabling parallel, large‑scale evaluation runs while preserving reproducibility via containerized benchmarks.

See common concepts and commands in {ref}`executors-overview`.

Slurm can optionally host your model for the scope of an evaluation by deploying a serving container on the cluster and pointing the benchmark to that temporary endpoint. In this mode, two containers are used: one for the evaluation harness and one for the model server. The evaluation configuration includes a deployment section when this is enabled. See the examples in the examples/ directory for ready‑to‑use configurations.

If you do not require deployment on Slurm, simply omit the deployment section from your configuration and set the model's endpoint URL directly (any OpenAI‑compatible endpoint that you host elsewhere).

## Prerequisites
- Access to a Slurm cluster (with appropriate partitions/queues)
- [Pyxis SPANK plugin](https://github.com/NVIDIA/pyxis) installed on the cluster 

## Configuration Example

Here's a complete Slurm executor configuration with model deployment:

```yaml
# examples/slurm_llama_3_1_8b_instruct.yaml
defaults:
  - execution: slurm/default
  - deployment: vllm
  - _self_

execution:
  account: your_account
  output_dir: /shared/results
  partition: gpu
  walltime: "04:00:00"
  gpus_per_node: 8

deployment:
  checkpoint_path: /shared/models/llama-3.1-8b-instruct
  served_model_name: meta-llama/Llama-3.1-8B-Instruct
  tensor_parallel_size: 1
    
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
- Runs three benchmark tasks
