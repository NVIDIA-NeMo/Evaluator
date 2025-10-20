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

**Note:** 
- `hostname`: Slurm headnode (login node) where you normally SSH to submit jobs.
- `output_dir`: must be an absolute path on a shared filesystem (e.g., /shared/scratch/ or /home/) accessible to both the headnode and compute nodes.


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

For local model files, use `checkpoint_path`:

```yaml
deployment:
  checkpoint_path: /shared/models/llama-3.1-8b-instruct  # model directory accessible to compute nodes
  # Do NOT set hf_model_handle when using checkpoint_path
```

**Note:**
- The directory must exist, be accessible from compute nodes, and contain model files
- Slurm does not automatically download models when using `checkpoint_path`

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
- Runs three benchmark tasks in parallel
- Saves benchmark artifacts to `output_dir`