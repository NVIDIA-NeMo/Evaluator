# Weights & Biases Exporter (`wandb`)

## Overview

Export evaluation metrics and artifacts to Weights & Biases (W&B) for experiment tracking. The exporter inherits all core features from the [Local exporter](local.md)—artifact staging, multi-run support, and automatic export—and adds W&B run management.

## Key Features

- Log metrics and artifacts for each job to W&B.
- Choose per-task runs (default) or combine tasks into one run per invocation.
- Append to an existing run (multi-task resume) to keep all tasks in one place.
- Export automatically after evaluations finish (local or cluster).

## Configuration

- Required: `entity`, `project`.
- `log_mode`:
  - `per_task` (default): Creates one W&B run per task or benchmark (name: `eval-<invocation>-<benchmark>`).
  - `multi_task`: Creates one W&B run per invocation (name: `eval-<invocation>`) with resume capability.
- `name`: Custom run name (overrides defaults above).
- `group`: Run grouping (default: `invocation_id`).
- `tags`: List of tags for filtering.
- `description`: Run description.
- `job_type`: Run type classification (default: "evaluation").
- `log_metrics`: Filter specific metrics (default: all available metrics).
- `log_artifacts`: Include artifacts (default: true), including the run configuration file `config.yaml` for reproducibility.
- `extra_metadata`: User-defined fields merged into the W&B run configuration.
- Webhook fields: `triggered_by_webhook`, `webhook_source`, `source_artifact`, `config_source`.

## Credentials

- Run `wandb login` on the machine that performs the export.
- For non-interactive environments, set `WANDB_API_KEY` in the environment.
- Optional for on-prem or self-hosted W&B: set `WANDB_BASE_URL`.

## Examples

### YAML

```yaml
execution:
  auto_export:
    destinations: ["wandb"]
    configs:
      wandb:
        entity: "nvidia"
        project: "nemo-evaluator-launcher-test"
        group: "exporter-testing"
        job_type: "evaluation"
        tags: ["Nemotron-H", "multi-task", "exporter-test"]
        description: "Test Run"
        log_metrics: ["accuracy", "score", "pass@1"]
        log_mode: "multi_task"

        extra_metadata: 
          execution_type: "slurm"
          model_tag: "nemotrons"
          hardware: "super-gpu9000s"
```

### CLI

```bash
nemo-evaluator-launcher export 8abcd123 --dest wandb
```

### Python API

```python
from nemo_evaluator_launcher.api.functional import export_results

# Per-task
export_results("8abcd123", dest="wandb", config={"entity": "myorg", "project": "evals"})

# Multi-task
export_results("8abcd123", dest="wandb",
               config={"entity": "myorg", "project": "evals", "log_mode": "multi_task"})
```

## Tips

Use `auto_export` for the most flexible configuration and customization.
