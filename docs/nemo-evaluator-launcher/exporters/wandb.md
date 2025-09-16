# Weights & Biases Exporter (`wandb`)

Export metrics and artifacts to W&B for experiment tracking. Inherits all core features from the [Local](local.md) exporter (artifact staging, multi-run, auto-export), and adds W&B run management.

**What you can do:**
- Log metrics and artifacts for each job to W&B
- Choose per-task runs (default) or aggregate multiple tasks into one run per invocation
- Append to an existing run (multi-task resume) to keep all tasks in one place
- Auto-export after evaluations finish (local or cluster)

**Key configuration:**
- Required: `entity`, `project`
- `log_mode`: 
  - `per_task` (default): creates one W&B run per task/benchmark (name: `eval-<invocation>-<benchmark>`)
  - `multi_task`: creates one W&B run per invocation (name: `eval-<invocation>`) with resume capability
- `name`: custom run name (overrides defaults above)
- `group`: run grouping (default: invocation_id)
- `tags`: list of tags for filtering
- `description`: run description text
- `job_type`: run type classification (default: "evaluation")
- `log_metrics`: filter specific metrics (default: all available metrics)
- `log_artifacts`: include artifacts (default: true), including the run config `config.yaml` for reproducibility
- `extra_metadata`: user-defined custom fields merged into W&B run.config
- Webhook-related: `triggered_by_webhook`, `webhook_source`, `source_artifact`, `config_source`

**Tip:** We recommend `auto-export` for the extensive support of configuration and customization.

**Example (YAML excerpt):**
```yaml
execution:
  auto_export:
    destinations: ["wandb"]
    configs:
      wandb:
        entity: "nvidia"
        project: "nv-eval-test"
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

**Examples (CLI):**
```bash
nemo-evaluator-launcher export 8abcd123 --dest wandb
```

**Examples (API):**
```python
from nemo_evaluator_launcher.api.functional import export_results

# Per-task
export_results("8abcd123", dest="wandb", config={"entity": "myorg", "project": "evals"})

# Multi-task
export_results("8abcd123", dest="wandb",
               config={"entity": "myorg", "project": "evals", "log_mode": "multi_task"})
```