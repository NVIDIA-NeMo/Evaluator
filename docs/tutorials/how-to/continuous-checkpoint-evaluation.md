(how-to-continuous-checkpoint-evaluation)=
# Continuous Checkpoint Evaluation

During model training, you often want to evaluate checkpoints as they appear — without manually kicking off each run. The `nel-watch` command polls a directory for new checkpoint subdirectories and automatically submits evaluation jobs using your existing launcher config.

This is especially useful for:

- **Long training runs** where checkpoints land every few hours and you want evaluation results as early as possible.
- **Overnight / unattended campaigns** where waiting until morning to start evals wastes GPU time.
- **Choosing the best checkpoint** by evaluating all of them and comparing scores afterward.

## Before You Start

Ensure you have:

- NeMo Evaluator Launcher installed (`pip install nemo-evaluator-launcher[all]`).
- A working eval config (`nel run --config <config.yaml> --dry-run` completes without errors).
- Access to the checkpoint directory (local or mounted filesystem).
- Checkpoints saved with a recognizable naming pattern (e.g. `step_1000/`, `iter_500/`).

## Quick Start

The simplest invocation watches a single directory and submits evals for every new checkpoint:

```bash
nel-watch \
  --config my-eval-config.yaml \
  --watch-dir /path/to/training/checkpoints
```

This will:

1. Scan `/path/to/training/checkpoints` for subdirectories matching `iter_*` or `step_*`.
2. Check that each checkpoint is **ready** (contains `metadata.json` or `config.yaml`).
3. Submit an evaluation for each new checkpoint by overriding the `deployment.hf_model_handle` config field with the checkpoint path.
4. Sleep for 5 minutes (default), then repeat.

Press `Ctrl+C` to stop gracefully — the current cycle finishes before exiting.

## How It Works

### Checkpoint Discovery

`nel-watch` scans the watch directory for subdirectories that match **checkpoint patterns** and contain **ready markers**:

- **Checkpoint patterns** (`--checkpoint-patterns`): Glob patterns for directory names. Default: `iter_*,step_*`. Only directories matching at least one pattern are considered.
- **Ready markers** (`--ready-markers`): Files whose presence signals a checkpoint is fully written. Default: `metadata.json,config.yaml`. A checkpoint is ready if **any** of the listed files exist.

:::{tip}
The ready marker check prevents evaluating half-written checkpoints. Training frameworks typically write the config/metadata file last, making it a reliable completion signal.
:::

### Processing Order

By default, checkpoints are processed **newest first** (`--order newest`), so the most recent checkpoint gets evaluated first. Use `--order oldest` to evaluate in chronological order.

### State Persistence

`nel-watch` tracks which checkpoints have been submitted in a `.watch_state.yaml` file. If you restart the watcher, it picks up where it left off — already-evaluated checkpoints are skipped. The state file location is auto-generated from the output directory, or you can set it explicitly with `--state-file`.

## CLI Reference

```text
nel-watch --config CONFIG [options]
```

```{list-table}
:header-rows: 1
:widths: 30 15 55

* - Option
  - Default
  - Description
* - `--config`
  - *(required)*
  - Path to eval config file (same as `nel run`).
* - `--watch-dir`
  - —
  - Directory to watch for checkpoint subdirectories.
* - `--watch-config`
  - —
  - YAML file defining multiple watch directories (mutually exclusive with `--watch-dir`).
* - `--interval`
  - `300`
  - Polling interval in seconds.
* - `--ready-markers`
  - `metadata.json,config.yaml`
  - Comma-separated list of marker files. Checkpoint is ready if ANY exist.
* - `--checkpoint-patterns`
  - `iter_*,step_*`
  - Comma-separated glob patterns for checkpoint directory names.
* - `--checkpoint-field`
  - `deployment.hf_model_handle`
  - Dot-separated config field to override with the checkpoint path.
* - `--order`
  - `newest`
  - Processing order: `newest` or `oldest`.
* - `--state-file`
  - *(auto)*
  - Path to state file for tracking submitted checkpoints.
* - `-n`, `--dry-run`
  - `false`
  - Show what would be submitted without actually submitting.
* - `--once`
  - `false`
  - Scan once and exit (no polling loop).
* - `-o`
  - —
  - Hydra config overrides (same as `nel run -o`). Repeatable.
```

## Examples

### Preview Before Running

Use `--dry-run --once` to see which checkpoints would be evaluated:

```bash
nel-watch \
  --config my-eval-config.yaml \
  --watch-dir /checkpoints/my-training-run \
  --dry-run --once
```

### Custom Checkpoint Patterns

If your checkpoints use a different naming scheme:

```bash
nel-watch \
  --config my-eval-config.yaml \
  --watch-dir /checkpoints/my-training-run \
  --checkpoint-patterns "checkpoint-*,ckpt_*"
```

### Shorter Polling Interval

For fast-paced training with frequent checkpoints:

```bash
nel-watch \
  --config my-eval-config.yaml \
  --watch-dir /checkpoints/my-training-run \
  --interval 60
```

### Evaluate Oldest First

Process checkpoints in chronological order:

```bash
nel-watch \
  --config my-eval-config.yaml \
  --watch-dir /checkpoints/my-training-run \
  --order oldest
```

### With Hydra Overrides

Pass additional config overrides, just like `nel run`:

```bash
nel-watch \
  --config my-eval-config.yaml \
  --watch-dir /checkpoints/my-training-run \
  -o execution.output_dir=/results/watch-run \
  -o +config.params.limit_samples=100
```

### Watching Multiple Directories

Create a watch config YAML to monitor several training runs simultaneously:

```yaml
# watch-config.yaml
checkpoint_field: deployment.hf_model_handle

watch_dirs:
  - checkpoint_dir: /training/run-A/checkpoints
    output_dir: /results/run-A

  - checkpoint_dir: /training/run-B/checkpoints
    output_dir: /results/run-B
    checkpoint_field: target.model_path  # per-dir override
```

Then run:

```bash
nel-watch \
  --config my-eval-config.yaml \
  --watch-config watch-config.yaml
```

## Python API

You can also use the watch functionality programmatically:

```python
from pathlib import Path

from nemo_evaluator_launcher.api.functional import RunConfig
from nemo_evaluator_launcher.api.watch import watch_checkpoints

config = RunConfig.from_hydra(config="my-eval-config.yaml")

submissions = watch_checkpoints(
    config=config,
    watch_dir=Path("/checkpoints/my-training-run"),
    interval=300,
    order="newest",
    dry_run=False,
)

for s in submissions:
    print(f"Checkpoint: {s.checkpoint} -> Invocation: {s.invocation_id}")
```

## Troubleshooting

**No checkpoints found:** Verify that subdirectories match the checkpoint patterns and contain at least one ready marker file:

```bash
# Check directory structure
ls /checkpoints/my-training-run/

# Check for marker files inside a checkpoint
ls /checkpoints/my-training-run/step_1000/
```

**Checkpoints already submitted:** Delete or move the `.watch_state.yaml` file to re-evaluate all checkpoints.

**Permission errors on Lustre/NFS:** Ensure the user running `nel-watch` has read access to the checkpoint directory and write access to the output/state file location.

## See Also

- {ref}`launcher-cli-dry-run` — Preview resolved configuration before running.
- [Python API](../../references/api/nemo-evaluator-launcher/api.md) — Programmatic access to launcher functionality.
