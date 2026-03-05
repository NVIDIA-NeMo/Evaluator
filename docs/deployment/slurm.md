# SLURM Deployment

Distribute evaluations across an HPC cluster using SLURM job arrays.

## Quick start

```bash
nel slurm eval gsm8k \
    --shards 16 --repeats 8 \
    --partition batch --time-limit 2:00:00 \
    --conda-env gym --submit
```

This submits a 16-task job array plus a dependent merge job.

## Generated scripts

`nel slurm eval` generates two sbatch scripts in `--output-dir`:

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

nel run --benchmark gsm8k --repeats 8 \
    --output-dir $OUTPUT_DIR/shard_${SLURM_ARRAY_TASK_ID} \
    --no-progress
```

### `merge.sbatch`

```bash
#!/bin/bash
#SBATCH --job-name=nel-merge-gsm8k
#SBATCH --dependency=afterok:${EVAL_JOB_ID}

nel slurm merge -d $OUTPUT_DIR -o $OUTPUT_DIR/merged --repeats 8
```

## Manual workflow

```bash
# 1. Generate scripts
nel slurm eval gsm8k --shards 16 -o ./eval_results

# 2. Review
cat ./eval_results/eval.sbatch

# 3. Submit eval
EVAL_JOB=$(sbatch ./eval_results/eval.sbatch | awk '{print $NF}')

# 4. Submit merge with dependency
sbatch --dependency=afterok:$EVAL_JOB ./eval_results/merge.sbatch
```

## Serve on SLURM

Long-running environment server for Gym training:

```bash
nel slurm serve --benchmark gsm8k --gym-compat \
    --partition interactive --time-limit 24:00:00 --submit
```

The hostname:port is written to `eval_results/endpoint.txt` once the job starts.

## CLI options

| Option | Default | Purpose |
|--------|---------|---------|
| `--shards` | 8 | Number of SLURM array tasks |
| `--partition` | `batch` | SLURM partition |
| `--cpus` | 4 | CPUs per task |
| `--mem` | `16G` | Memory per task |
| `--time-limit` | `2:00:00` | Wall time |
| `--gpus` | 0 | GPUs per task |
| `--conda-env` | `base` | Conda environment to activate |
| `--submit` | off | Auto-submit via `sbatch` |
