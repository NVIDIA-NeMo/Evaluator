(continuous-checkpoint-evaluation)=
# Continuous Checkpoint Evaluation

During model training, you often want to evaluate checkpoints as they appear - without manually kicking off each run. The `nel-watch` command polls a directory for new checkpoint subdirectories and automatically submits evaluation jobs using your existing launcher config.

This is especially useful for:

- **Long training runs** where checkpoints land every few hours and you want evaluation results as early as possible.
- **Overnight / unattended campaigns** where you do not want to wait until morning to start evals.
- **Choosing the best checkpoint** by evaluating all of them and comparing scores afterward.

:::{note}
`nel-watch` currently requires **SLURM execution**. Evaluation configs must have an `execution` section with `type: slurm`.
:::

## Before You Start

Ensure you have:

- NeMo Evaluator Launcher installed (`pip install nemo-evaluator-launcher[all]`).
- A working eval config with SLURM execution (`nel run --config <eval-config.yaml> --dry-run` completes without errors).
- SSH access to the cluster login node from where you run `nel-watch`.
- Good understanding of your checkpoint naming pattern (e.g. `step_1000/`, `iter_500/`) and directory structure (to configure directory readiness markers).

## Quick Start

`nel-watch` is driven by a **watch config** YAML that combines cluster connection settings, monitoring parameters, an optional checkpoint conversion step, and one or more evaluation configs:

```bash
nel-watch --config my-watch-config.yaml
```

A minimal watch config looks like this:

```yaml
# my-watch-config.yaml
defaults:
  - cluster_config: default
  - monitoring_config: default
  - _self_

cluster_config:
  hostname: my-cluster-login.example.com
  account: my-slurm-account
  output_dir: /shared/results/watch-run

monitoring_config:
  directories:
    - /path/to/training/checkpoints

evaluation_configs:
  - /path/to/my-eval-config.yaml
```

This will:

1. Connect to the cluster via SSH.
2. Scan `/path/to/training/checkpoints` for subdirectories matching `iter_*` or `step_*`.
3. Check that each checkpoint is **ready** (contains `metadata.json` or `config.yaml`).
4. Submit a SLURM evaluation job for each new checkpoint, setting `deployment.checkpoint_path` automatically.
5. Sleep 5 minutes (default), then repeat.

Press `Ctrl+C` to stop gracefully — the current cycle finishes before exiting.

## Watch Config Reference

### `cluster_config`

Shared settings for all SLURM jobs submitted by nel-watch (monitoring, conversion, and evaluation).

```yaml
cluster_config:
  hostname: my-cluster-login.example.com  # Use 'localhost' when running on the login node directly
  username: ${oc.env:USER}                # Defaults to $USER
  account: my-slurm-account
  partition: batch                        # Default: batch
  output_dir: /shared/results/watch-run  # Per-checkpoint subdirectories are created here
  sbatch_extra_flags:                     # Additional #SBATCH directives for every job
    gres: "gpu:8"
    time: "04:00:00"
```

### `monitoring_config`

Controls how directories are scanned for new checkpoints.

```yaml
monitoring_config:
  directories:
    - /path/to/training/checkpoints       # Multiple directories are supported
  interval: 300                           # Seconds between scans; set to null to scan once and exit
  ready_markers:                          # Checkpoint is ready when ANY of these files exist inside it
    - metadata.json
    - config.yaml
    - config.json
  checkpoint_patterns:                    # Only subdirectories matching ANY pattern are considered
    - "iter_*"
    - "step_*"
  order: last                             # 'last' = highest name first (step_10000 before step_1000)
                                          # 'first' = lowest name first
```

### `conversion_config` (optional)

If you would like to evaluate your checkpoints in a different format than the one used for training, add a conversion step. A conversion SLURM job is submitted first; the evaluation job is held with `afterok` dependency until it completes.

```yaml
conversion_config:
  container: my-training-container:latest
  command_pattern: "convert -i {{ input_path }} -o {{ output_path }}"
```

:::{note}
`command_pattern` must contain {{ input_path }} and {{ output_path }} placeholders.
:::

Omit `conversion_config` (or set it to `null`) to use the raw checkpoint path directly in the evaluation.

See the Hydra config group templates in `configs/watcher/conversion_config/` for ready-to-use conversion presets (e.g. `mbridge`).

### `evaluation_configs`

A list of paths to standard NEL evaluation config files. Each config is run for every discovered checkpoint. `nel-watch` automatically sets `deployment.checkpoint_path` and `execution.output_dir` in each config before submission and there is no need to pre-define them.

```yaml
evaluation_configs:
  - /path/to/eval-config-A.yaml
  - /path/to/eval-config-B.yaml
```

## Using Hydra Config Groups

Watch config fields can be factored out using Hydra defaults lists. The package ships templates in `configs/watcher/`:

```yaml
# my-watch-config.yaml
defaults:
  - cluster_config: default
  - monitoring_config: default
  - conversion_config: mbridge   # conversion template for Megatron-Bridge -> Hugging Face
  - _self_

cluster_config:
  hostname: my-cluster.example.com
  account: my-account
  output_dir: /shared/results

monitoring_config:
  directories:
    - /path/to/checkpoints

conversion_config:
  container: /path/to/mbridge.sqsh
  command_params:
    hf_model: /path/to/reference-hf-model
    tp: 8

evaluation_configs:
  - /path/to/eval.yaml
```

## Troubleshooting

**No checkpoints found:** Verify that subdirectories match `checkpoint_patterns` and contain at least one `ready_markers` file. Run with `--dry-run` to see what the watcher detects.

**SSH errors:** Ensure your SSH key is loaded (`ssh-add`) and you can manually `ssh <username>@<hostname>` without a password prompt. When running directly on the login node, set `hostname: localhost` in `cluster_config`.

**Evaluation config validation error:** Each eval config must have both `deployment` and `execution` sections. `execution.type` must be `slurm`. Do not set `execution.output_dir` — `nel-watch` manages it per checkpoint.

**Re-evaluating a specific checkpoint:** Remove its entry from the state file, or use `--resubmit-previous-sessions` to resubmit all previously seen checkpoints.

## See Also

- {ref}`launcher-cli-dry-run` — Preview resolved configuration before running.
- [Python API](../references/api/nemo-evaluator-launcher/api.md) — Programmatic access to launcher functionality.
- [CLI](../references/api/nemo-evaluator-launcher/cli.md) — Commandline interface.
