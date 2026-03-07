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

    W0 --> M["Auto-Merge"]
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

Create a YAML config with `cluster.type: slurm`:

```yaml
# slurm_gsm8k.yaml
benchmark: gsm8k
repeats: 8
output_dir: ./eval_results/gsm8k_distributed
cluster:
  type: slurm
  shards: 16
  partition: batch
  conda_env: gym
```

### Run

```bash
nel eval run slurm_gsm8k.yaml
```

This:
1. Generates `eval.sbatch` with `--array=0-15`
2. Submits the job array
3. Automatically merges results after all shards complete

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

Results are automatically merged after all pods complete.

## Ray

For Ray clusters (compatible with NeMo Gym's Ray infrastructure):

```bash
ray job submit --working-dir . -- python -m nemo_evaluator.runner.ray_launcher \
    --benchmark gsm8k --shards 8 --repeats 5 \
    --output-dir ./eval_results/ray
```

Each shard runs as a `@ray.remote` task. Results are merged in-process after all tasks complete.

### From Python

```python
import ray
from nemo_evaluator.runner.ray_launcher import run_shard

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
NEL_SHARD_IDX=0 NEL_TOTAL_SHARDS=4 nel eval run --bench gsm8k -o ./results/shard_0 --no-progress

# Terminal 2
NEL_SHARD_IDX=1 NEL_TOTAL_SHARDS=4 nel eval run --bench gsm8k -o ./results/shard_1 --no-progress
```

## Shard Size Distribution

Problems are distributed evenly. For 14,000 problems across 16 shards:

| Shards 0-7 | Shards 8-15 |
|------------|-------------|
| 875 problems each | 875 problems each |

When the total doesn't divide evenly, early shards get one extra problem (e.g., 14,001 problems / 16 shards = shards 0 get 876, shards 1-15 get 875).
