# SLURM Deployment

Distribute evaluations across an HPC cluster using SLURM.

## Quick start

```yaml
# slurm_eval.yaml
services:
  model:
    type: vllm
    model: nvidia/Llama-3.1-70B-Instruct
    protocol: chat_completions
    tensor_parallel_size: 4
    port: 8000
    node_pool: compute

benchmarks:
  - name: gsm8k
    repeats: 8
    solver:
      type: simple
      service: model

cluster:
  type: slurm
  walltime: "02:00:00"
  node_pools:
    compute:
      partition: batch
      nodes: 1
      ntasks_per_node: 1
      gres: "gpu:4"
```

```bash
nel eval run slurm_eval.yaml
```

This generates an sbatch script, deploys vLLM on the allocated GPU node, and runs the benchmark.

## Sharded evaluation

Split a large benchmark across multiple SLURM array tasks:

```yaml
# slurm_sharded.yaml
services:
  model:
    type: vllm
    model: nvidia/Llama-3.1-70B-Instruct
    protocol: chat_completions
    tensor_parallel_size: 4
    port: 8000
    node_pool: compute

benchmarks:
  - name: gsm8k
    repeats: 8
    solver:
      type: simple
      service: model

cluster:
  type: slurm
  walltime: "02:00:00"
  shards: 16
  node_pools:
    compute:
      partition: batch
      nodes: 1
      ntasks_per_node: 1
      gres: "gpu:4"
```

```bash
nel eval run slurm_sharded.yaml
```

This submits a 16-task job array. Each task evaluates a disjoint slice of the dataset. After all tasks complete, merge results:

```bash
nel eval merge ./eval_results
```

## Generated scripts

`nel eval run` with a SLURM cluster config generates an sbatch script in `--output-dir`:

### `eval.sbatch`

For sharded runs, the generated script includes:

```bash
#!/bin/bash
#SBATCH --job-name=nel-eval-gsm8k
#SBATCH --array=0-15
#SBATCH --partition=batch
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:4

export NEL_SHARD_IDX=$SLURM_ARRAY_TASK_ID
export NEL_TOTAL_SHARDS=$SLURM_ARRAY_TASK_COUNT
OUTPUT_DIR="$OUTPUT_DIR/shard_$SLURM_ARRAY_TASK_ID"
mkdir -p "$OUTPUT_DIR"

# ... vLLM startup, health check, eval run ...
```

## Resume after failure

If a benchmark fails within a multi-benchmark SLURM suite, re-submit with `--resume` to skip already-completed benchmarks:

```bash
nel eval run slurm_eval.yaml --resume
```

## ai-dynamo deployment

Deploy NVIDIA `ai-dynamo` (sglang engine) as a managed service. Dynamo is multi-process by design — NEL launches the NATS broker, etcd, the `dynamo.frontend` HTTP server, and the `dynamo.sglang` worker(s) in one sbatch job. The OpenAI-compatible endpoint is the frontend on `svc.port`.

**Aggregated** (one worker):

```yaml
services:
  glm:
    type: dynamo
    model: /lustre/path/to/GLM-5.1-FP8
    served_model_name: glm
    protocol: chat_completions
    port: 8000
    tensor_parallel_size: 8
```

For multi-node TP the worker rendezvouses via sglang's torch.distributed (`--nnodes/--node-rank/--dist-init-addr`); set `num_nodes: 2` and `tensor_parallel_size: 16` and NEL gates the NATS/etcd/frontend bootstrap on rank 0.

**Disaggregated** (separate prefill + decode workers, KV transfer over NIxl/UCX-CUDA):

```yaml
services:
  glm:
    type: dynamo
    model: /lustre/path/to/GLM-5.1-FP8
    served_model_name: glm
    protocol: chat_completions
    port: 8000
    prefill:
      tensor_parallel_size: 8
      num_nodes: 1
      node_pool: prefill_pool
      extra_env:
        UCX_TLS: "rc_x,rc,cuda_copy,cuda_ipc"
        UCX_NET_DEVICES: "mlx5_0:1"
    decode:
      tensor_parallel_size: 8
      num_nodes: 1
      node_pool: decode_pool
      extra_env:
        UCX_TLS: "rc_x,rc,cuda_copy,cuda_ipc"
        UCX_NET_DEVICES: "mlx5_0:1"

cluster:
  type: slurm
  node_pools:
    prefill_pool:
      partition: batch
      nodes: 1
      gres: "gpu:8"
    decode_pool:
      partition: batch
      nodes: 1
      gres: "gpu:8"
```

Mode is implicit: presence of `prefill` AND `decode` switches to disaggregated; their absence (default) is aggregated. Setting only one of the pair is a validation error.

Default image: `nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.1.1-cuda13` from NGC. Override per service with `image:` or globally via `containers.toml`.

## Heterogeneous jobs

For configurations that require separate GPU and CPU nodes (e.g., model on GPU + sandboxes on CPU), define multiple node pools:

```yaml
cluster:
  type: slurm
  walltime: "04:00:00"
  node_pools:
    gpu:
      partition: gpu
      nodes: 1
      gres: "gpu:4"
    sandbox:
      partition: cpu
      nodes: 2
      slots_per_node: 4
```

Services and sandboxes reference pools via `node_pool: gpu` or `node_pool: sandbox`.

Note: `shards` is incompatible with heterogeneous jobs (multiple node pools).

## Manual workflow

```bash
# 1. Generate scripts (set submit: false in config)
nel eval run slurm_eval.yaml

# 2. Review
cat ./eval_results/eval.sbatch

# 3. Submit eval
EVAL_JOB=$(sbatch ./eval_results/eval.sbatch | awk '{print $NF}')

# 4. For sharded runs, merge after all tasks complete
nel eval merge ./eval_results
```

## Serve on SLURM

Long-running environment server for Gym training:

```bash
nel serve -b gsm8k --gym-compat --port 9090
```

Wrap in an sbatch script for SLURM submission.

## Cluster config options

| Option | Default | Purpose |
|--------|---------|---------|
| `walltime` | `04:00:00` | Job wall time |
| `shards` | `null` | Number of SLURM array tasks (null = no sharding) |
| `account` | `null` | SLURM account for billing |
| `conda_env` | `null` | Conda environment to activate |
| `container_image` | `null` | Apptainer/Enroot image for containerized execution |
| `auto_resume` | `false` | Resubmit failed jobs via dependency chain |
| `hostname` | `null` | Remote SLURM head node (for SSH submission) |

## Node pool options

| Option | Default | Purpose |
|--------|---------|---------|
| `partition` | (required) | SLURM partition |
| `nodes` | `1` | Nodes to allocate |
| `ntasks_per_node` | `1` | Tasks per node |
| `gres` | `null` | Generic resources (e.g., `gpu:4`) |
| `slots_per_node` | `1` | Concurrent sandbox slots per node |
