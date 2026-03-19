(how-to-continuous-checkpoint-evaluation)=
# Continuous Checkpoint Evaluation

During model training, you often want to evaluate checkpoints as they appear — without manually kicking off each run. The `nel-watch` command polls a directory for new checkpoint subdirectories and automatically submits evaluation jobs using your existing launcher config.

This is especially useful for:

- **Long training runs** where checkpoints land every few hours and you want evaluation results as early as possible.
- **Overnight / unattended campaigns** where waiting until morning to start evals wastes GPU time.
- **Choosing the best checkpoint** by evaluating all of them and comparing scores afterward.

:::{note}
`nel-watch` currently requires **SLURM execution**. Evaluation configs must have an `execution` section with `type: slurm`.
:::

## Before You Start

Ensure you have:

- NeMo Evaluator Launcher installed (`pip install nemo-evaluator-launcher[all]`).
- A working eval config with SLURM execution (`nel run --config <eval-config.yaml> --dry-run` completes without errors).
- SSH access to the cluster login node from where you run `nel-watch`.
- Checkpoints saved with a recognizable naming pattern (e.g. `step_1000/`, `iter_500/`).

## Quick Start

`nel-watch` is driven by a **watch config** YAML that combines cluster connection settings, monitoring parameters, an optional checkpoint conversion step, and one or more evaluation configs:

```bash
nel-watch --config my-watch-config.yaml
```

A minimal watch config looks like this:

```yaml
# my-watch-config.yaml
cluster_config:
  hostname: my-cluster-login.example.com
  username: myuser
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

If your training checkpoints are in Megatron format and your evaluation requires HuggingFace format, add a conversion step. A conversion SLURM job is submitted first; the evaluation job is held with `afterok` dependency until it completes.

```yaml
conversion_config:
  container: /path/to/conversion.sqsh
  mounts:
    - source: /path/to/hf-reference-weights
      target: /path/to/hf-reference-weights
  command_params:
    hf_model: /path/to/hf-reference-weights
    tp: 8
    pp: 1
    ep: 1
    etp: 1
  command_pattern: >-
    bash -lc '
    python /opt/Megatron-Bridge/examples/conversion/convert_checkpoints.py export
    --megatron-path {{ input_path }}
    --hf-path {{ output_path }}
    --hf-model {{ hf_model }}
    --tp {{ tp }} --pp {{ pp }} --ep {{ ep }} --etp {{ etp }}'
```

Omit `conversion_config` (or set it to `null`) to use the raw checkpoint path directly in the evaluation.

See the Hydra config group templates in `configs/watcher/conversion_config/` for ready-to-use conversion presets (e.g. `mbridge`).

### `evaluation_configs`

A list of paths to standard NEL evaluation config files. Each config is run for every discovered checkpoint. `nel-watch` automatically sets `deployment.checkpoint_path` and `execution.output_dir` in each config before submission — do not pre-define `execution.output_dir` in your eval configs.

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
  - conversion_config: mbridge   # uses the Megatron-Bridge preset
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

## CLI Reference

```text
nel-watch --config WATCH_CONFIG [options]
```

```{list-table}
:header-rows: 1
:widths: 35 15 50

* - Option
  - Default
  - Description
* - `--config`
  - *(required)*
  - Path to the watch config YAML file.
* - `--interval`
  - *(from config)*
  - Override the polling interval (seconds). Pass `0` to scan once and exit.
* - `--order`
  - *(from config)*
  - Override processing order: `last` (highest step first) or `first` (lowest step first).
* - `--resubmit-previous-sessions`
  - `false`
  - Re-evaluate checkpoints that were already submitted in earlier watcher sessions.
* - `-n`, `--dry-run`
  - `false`
  - Show what would be submitted without actually submitting.
* - `-o`, `--override`
  - —
  - Hydra-style override for the watch config (e.g. `-o cluster_config.account=other`). Repeatable. Overrides to individual `evaluation_configs` entries are not supported.
```

## Examples

### Preview Before Running

```bash
nel-watch --config my-watch-config.yaml --dry-run
```

### Scan Once and Exit

Set `interval: null` in `monitoring_config`, or pass `--interval 0` on the command line:

```bash
nel-watch --config my-watch-config.yaml --interval 0
```

### Re-evaluate Previous Checkpoints

```bash
nel-watch --config my-watch-config.yaml --resubmit-previous-sessions
```

### Override Config Values at the Command Line

```bash
nel-watch --config my-watch-config.yaml \
  -o cluster_config.account=other-account \
  -o monitoring_config.interval=60
```

### Process Oldest Checkpoints First

```bash
nel-watch --config my-watch-config.yaml --order first
```

## State Persistence

`nel-watch` records every submission in an append-only JSONL log:

```
~/.nemo-evaluator/watch-state/watch-state.v1.jsonl
```

Each line is a JSON record containing the checkpoint path, invocation ID, session ID, and the resolved configs used at submission time. If you restart `nel-watch`, already-submitted checkpoints are automatically skipped. Use `--resubmit-previous-sessions` to reprocess them.

To use a custom location, set the `NEMO_EVALUATOR_WATCH_STATE_FILE` environment variable.

## Python API

```python
from pathlib import Path

from nemo_evaluator_launcher.watcher.configs import WatchConfig
from nemo_evaluator_launcher.watcher.run import watch_and_evaluate

watch_config = WatchConfig.from_hydra(path=Path("my-watch-config.yaml"))

submissions = watch_and_evaluate(
    watch_config=watch_config,
    resubmit_previous_sessions=False,
    dry_run=False,
)

for s in submissions:
    print(f"Checkpoint: {s.checkpoint} -> Invocation: {s.invocation_id}")
```

## Troubleshooting

**No checkpoints found:** Verify that subdirectories match `checkpoint_patterns` and contain at least one `ready_markers` file. Run with `--dry-run` to see what the watcher detects.

**SSH errors:** Ensure your SSH key is loaded (`ssh-add`) and you can manually `ssh <username>@<hostname>` without a password prompt. When running directly on the login node, set `hostname: localhost` in `cluster_config`.

**Evaluation config validation error:** Each eval config must have both `deployment` and `execution` sections. `execution.type` must be `slurm`. Do not set `execution.output_dir` — `nel-watch` manages it per checkpoint.

**Re-evaluating a specific checkpoint:** Remove its entry from the state file, or use `--resubmit-previous-sessions` to resubmit all previously seen checkpoints.

## See Also

- {ref}`launcher-cli-dry-run` — Preview resolved configuration before running.
- [Python API](../../references/api/nemo-evaluator-launcher/api.md) — Programmatic access to launcher functionality.
