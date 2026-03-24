# Distributed Evaluation

Scale evaluations across multiple nodes using SLURM, Kubernetes, Ray, or manual sharding.

## How Sharding Works

```{mermaid}
flowchart TB
    DS["Dataset<br/>14,000 problems"] --> SPLIT["Shard Splitter"]
    SPLIT --> S0["Shard 0<br/>[0, 1750)"]
    SPLIT --> S1["Shard 1<br/>[1750, 3500)"]
    SPLIT --> S2["..."]
    SPLIT --> S7["Shard 7<br/>[12250, 14000)"]

    S0 --> W0["Worker 0<br/>nel eval run"]
    S1 --> W1["Worker 1<br/>nel eval run"]
    S2 --> W2["..."]
    S7 --> W7["Worker 7<br/>nel eval run"]

    W0 --> M["nel eval merge"]
    W1 --> M
    W2 --> M
    W7 --> M

    M --> R["Merged Results<br/>pass@k, CI, trajectories"]
```

Each worker runs the same `nel eval run` command. Two environment variables control the split:

| Variable | Set by | Purpose |
|----------|--------|---------|
| `NEL_SHARD_IDX` | Orchestrator or `SLURM_ARRAY_TASK_ID` | This worker's shard (0-based) |
| `NEL_TOTAL_SHARDS` | Orchestrator or `SLURM_ARRAY_TASK_COUNT` | Total number of shards |

`nel eval run` auto-detects these and evaluates only its assigned problem range.

## SLURM

### SLURM config file

Use `cluster.type: slurm` with `shards:` to enable array job sharding:

```yaml
# slurm_gsm8k_sharded.yaml
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

### Run

```bash
nel eval run slurm_gsm8k_sharded.yaml
```

This:
1. Generates `eval.sbatch` with `#SBATCH --array=0-15`
2. Each array task exports `NEL_SHARD_IDX` and `NEL_TOTAL_SHARDS`
3. Each task writes results to `shard_N/` subdirectories

### Merge results

After all array tasks complete:

```bash
nel eval merge ./eval_results
```

This discovers all `shard_N/` directories, deduplicates any overlapping results, and produces merged bundles with combined scores.

## Kubernetes

Use a K8s Indexed Job for native parallelism:

```bash
kubectl apply -f deploy/k8s/eval-indexed-job.yaml
```

The manifest uses `completionMode: Indexed` with 8 completions. Each pod gets `JOB_COMPLETION_INDEX` mapped to `NEL_SHARD_IDX`:

```yaml
env:
  - name: NEL_SHARD_IDX
    valueFrom:
      fieldRef:
        fieldPath: metadata.annotations['batch.kubernetes.io/job-completion-index']
  - name: NEL_TOTAL_SHARDS
    value: "8"
```

After all pods complete, merge with `nel eval merge`.

## Ray

For Ray clusters (compatible with NeMo Gym's Ray infrastructure):

```bash
ray job submit --working-dir . -- python -m nemo_evaluator.engine.ray_launcher \
    --bench gsm8k --shards 8 --repeats 5 \
    --output-dir ./eval_results/ray
```

Each shard runs as a `@ray.remote` task. Results are merged in-process after all tasks complete.

### From Python

```python
import ray
from nemo_evaluator.engine.ray_launcher import run_shard

ray.init()
futures = [run_shard.remote("gsm8k", i, 8, ...) for i in range(8)]
results = ray.get(futures)
```

## Docker Compose

For local multi-container sharding:

```bash
# 4 shards
for i in 0 1 2 3; do
  NEL_SHARD_IDX=$i NEL_TOTAL_SHARDS=4 \
    docker compose -f deploy/docker-compose.yaml --profile sharded run -d eval-shard
done
```

## Manual Sharding (Any Orchestrator)

Works anywhere you can set environment variables:

```bash
# Terminal 1
NEL_SHARD_IDX=0 NEL_TOTAL_SHARDS=4 nel eval run config.yaml -o ./results/shard_0

# Terminal 2
NEL_SHARD_IDX=1 NEL_TOTAL_SHARDS=4 nel eval run config.yaml -o ./results/shard_1

# ... after all shards complete:
nel eval merge ./results
```

## Shard Size Distribution

Problems are distributed evenly. For 14,000 problems across 16 shards:

| Shards 0-7 | Shards 8-15 |
|------------|-------------|
| 875 problems each | 875 problems each |

When the total doesn't divide evenly, early shards get one extra problem (e.g., 14,001 problems / 16 shards = shard 0 gets 876, shards 1-15 get 875).

## Limitations

- `shards` is incompatible with heterogeneous SLURM jobs (multiple node pools). Use a single pool when sharding.
- `shards` and `auto_resume` cannot be used together. Use SLURM `--requeue` for per-task retries in array mode.
- `run_batch()` environments (e.g., legacy containers) are not shardable — a warning is emitted if shard env vars are detected.
