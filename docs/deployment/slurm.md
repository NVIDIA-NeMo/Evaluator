# SLURM Deployment

Distribute evaluations across an HPC cluster using SLURM job arrays.

## Quick start

```yaml
# slurm_eval.yaml
model:
  url: https://inference-api.nvidia.com/v1
  id: azure/openai/gpt-5.2

benchmarks:
  - name: gsm8k
    repeats: 8

executor:
  type: slurm
  shards: 16
  partition: batch
  time_limit: "2:00:00"
  conda_env: gym
  submit: true
```

```bash
nel eval run slurm_eval.yaml
```

This submits a 16-task job array. Shard results are merged automatically when all tasks complete.

## Generated scripts

`nel eval run` with a SLURM executor config generates an sbatch script in `--output-dir`:

### `eval.sbatch`

```bash
#!/bin/bash
#SBATCH --job-name=nel-eval-gsm8k
#SBATCH --array=0-15
#SBATCH --partition=batch
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

source activate gym
export NEL_SHARD_IDX=$SLURM_ARRAY_TASK_ID
export NEL_TOTAL_SHARDS=$SLURM_ARRAY_TASK_COUNT

nel eval run --bench gsm8k --repeats 8 \
    --output-dir $OUTPUT_DIR/shard_${SLURM_ARRAY_TASK_ID} \
    --no-progress
```

## Manual workflow

```bash
# 1. Generate scripts (set submit: false in config)
nel eval run slurm_eval.yaml

# 2. Review
cat ./eval_results/eval.sbatch

# 3. Submit eval
EVAL_JOB=$(sbatch ./eval_results/eval.sbatch | awk '{print $NF}')

# 4. Results are merged automatically after all shards complete
```

## Serve on SLURM

Long-running environment server for Gym training:

```bash
nel serve -b gsm8k --gym-compat --port 9090
```

Wrap in an sbatch script for SLURM submission. The hostname:port is written to `eval_results/endpoint.txt` once the job starts.

## Executor config options

| Option | Default | Purpose |
|--------|---------|---------|
| `shards` | 8 | Number of SLURM array tasks |
| `partition` | `batch` | SLURM partition |
| `cpus` | 4 | CPUs per task |
| `mem` | `16G` | Memory per task |
| `time_limit` | `2:00:00` | Wall time |
| `gpus` | 0 | GPUs per task |
| `conda_env` | `base` | Conda environment to activate |
| `submit` | false | Auto-submit via `sbatch` |
