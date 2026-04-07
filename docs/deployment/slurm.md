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
