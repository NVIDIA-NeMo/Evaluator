#!/bin/bash
set -euo pipefail

# Distributed evaluation on SLURM via nel eval run.
#
# Usage:
#   bash deploy/slurm_distributed.sh
#
# What happens:
#   1. nel eval run generates a self-contained sbatch script from the config
#   2. The script starts model servers, runs benchmarks, generates reports
#   3. Results appear in eval_results/gsm8k_distributed/

OUTPUT_DIR=./eval_results/gsm8k_distributed

# --- Option A: generate + submit via SSH ---
nel eval run examples/configs/slurm.yaml --submit

# --- Option B: generate, inspect, then submit manually ---
# nel eval run examples/configs/slurm.yaml --dry-run
# cat "$OUTPUT_DIR/nel_eval.sbatch"
# sbatch "$OUTPUT_DIR/nel_eval.sbatch"

# --- Option C: serve a benchmark for Gym training ---
# nel serve -b gsm8k --gym-compat -p 9090

echo "Done. Watch with: squeue -u \$USER"
echo "Check status:     nel eval status -o $OUTPUT_DIR"
