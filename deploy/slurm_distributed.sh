#!/bin/bash
set -euo pipefail

# Distributed evaluation on SLURM: shard gsm8k across 8 nodes, then merge.
#
# Usage:
#   bash deploy/slurm_distributed.sh
#
# What happens:
#   1. nel slurm eval generates sbatch scripts + submits a job array (8 tasks)
#   2. Each task evaluates its shard of gsm8k (problems split evenly)
#   3. A merge job runs after all shards complete (SLURM dependency)
#   4. Final merged results appear in eval_results/gsm8k_distributed/merged/

OUTPUT_DIR=./eval_results/gsm8k_distributed

# --- Option A: generate + submit in one shot ---
nel slurm eval gsm8k \
    --shards 8 \
    --partition batch \
    --repeats 5 \
    --model-url https://inference-api.nvidia.com/v1 \
    --model-id azure/openai/gpt-5.2 \
    --conda-env gym \
    --output-dir "$OUTPUT_DIR" \
    --submit

# --- Option B: generate, inspect, then submit manually ---
# nel slurm eval gsm8k --shards 8 --output-dir "$OUTPUT_DIR"
# # inspect the generated scripts
# cat "$OUTPUT_DIR/eval.sbatch"
# cat "$OUTPUT_DIR/merge.sbatch"
# # submit
# EVAL_JOB=$(sbatch "$OUTPUT_DIR/eval.sbatch" | awk '{print $NF}')
# # update merge dependency and submit
# sed -i "s/\${EVAL_JOB_ID}/$EVAL_JOB/" "$OUTPUT_DIR/merge.sbatch"
# sbatch "$OUTPUT_DIR/merge.sbatch"

# --- Option C: serve an environment for Gym training ---
# nel slurm serve --benchmark gsm8k --gym-compat --output-dir "$OUTPUT_DIR" --submit
# # The allocated node+port will be in $OUTPUT_DIR/endpoint.txt
# # Gym training jobs can connect to that endpoint

echo "Done. Watch with: squeue -u \$USER"
