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

## ai-dynamo deployment (srtctl passthrough)

`type: dynamo` delegates all Dynamo orchestration to `srtctl` (NVIDIA/srt-slurm). NEL is **agnostic to srtctl's recipe schema** — the user's `recipe:` block is passed through verbatim. NEL only:

- Injects the `benchmark` step (runs NEL eval inside the eval-runner container, pointed at `localhost:<port>`)
- Forces `dynamo.install: false` (the srtctl version is pinned via NEL's bootstrap)
- Sets a default `name` if the user didn't

This means **any srtctl feature works without NEL changes** — aggregated, disaggregated, KV-aware routing, vllm/sglang/trtllm, multi-frontend, EAGLE-3, wide-EP. Refer to the [srtctl recipe docs](https://github.com/NVIDIA/srt-slurm) for the schema.

Optional: set `cluster.srtctl_bin` + `cluster.srtctl_work_dir` to use a pre-installed srtctl, or omit both to auto-bootstrap from GitHub into `~/.nel/srtctl/`.

**Aggregated example:**

```yaml
services:
  model:
    type: dynamo
    # NEL-side: how to reach the deployed model
    served_model_name: my-model
    port: 8000
    protocol: chat_completions
    # srtctl recipe passthrough
    recipe:
      model:
        path: /lustre/path/to/checkpoint
        container: /lustre/containers/sglang-runtime.sqsh
        precision: bf16
      resources:
        gpu_type: gb200
        gpus_per_node: 4
        agg_nodes: 1
        agg_workers: 1
      backend:
        type: sglang
        aggregated_environment:
          SGLANG_ENABLE_JIT_DEEPGEMM: "false"
        sglang_config:
          aggregated:
            served-model-name: my-model
            tensor-parallel-size: 4
            disable-cuda-graph: true

cluster:
  type: slurm
  hostname: login-node.example.com
  eval_image: registry/nemo-evaluator-next:0.18.4
  srtctl_compute_arch: aarch64
  node_pools:
    gpu: {partition: batch, nodes: 1, gpus_per_node: 4}
```

**Disaggregated example** (separate prefill + decode, KV transfer via NIxl/UCX-CUDA):

```yaml
services:
  model:
    type: dynamo
    served_model_name: my-model
    port: 8000
    recipe:
      model: {path: ..., container: ..., precision: fp8}
      resources:
        gpu_type: gb200
        gpus_per_node: 4
        prefill_nodes: 4   # 2 workers × 2 nodes
        decode_nodes: 2    # 1 worker × 2 nodes
      backend:
        type: sglang
        prefill_environment: {UCX_TLS: "rc_x,rc,cuda_copy,cuda_ipc"}
        decode_environment: {UCX_TLS: "rc_x,rc,cuda_copy,cuda_ipc"}
        sglang_config:
          prefill:
            served-model-name: my-model
            disaggregation-mode: prefill
            tensor-parallel-size: 8
            disaggregation-transfer-backend: nixl
          decode:
            served-model-name: my-model
            disaggregation-mode: decode
            tensor-parallel-size: 8
            disaggregation-transfer-backend: nixl
```

**KV-aware routing example** — works without any NEL changes:

```yaml
recipe:
  frontend:
    type: dynamo
    enable_multiple_frontends: true
    num_additional_frontends: 1
    args:
      router-mode: kv
      kv-overlap-score-weight: 2.0
  backend:
    type: sglang
    dist_init_port: 49500
    # ...
```

Use `--dry-run` to inspect the generated recipe before submitting.

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
