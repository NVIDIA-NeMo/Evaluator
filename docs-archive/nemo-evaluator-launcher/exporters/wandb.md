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
- `log_artifacts`: include artifacts and config.yaml (default: true)
- `log_logs`: include execution logs (default: false)
- `only_required`: copy only required artifacts (default: true)
- `extra_metadata`: user-defined custom fields merged into W&B run.config
- Webhook-related: `triggered_by_webhook`, `webhook_source`, `source_artifact`, `config_source`

**Tip:** We recommend `auto-export` for extensive configuration and customization support.

**Example (YAML):**
```yaml
execution:
  auto_export:
    destinations: ["wandb"]
  
  # Export-related env vars (placeholders expanded at runtime)
  env_vars:
    export:
      WANDB_API_KEY: WANDB_API_KEY
      PATH: "/path/to/conda/env/bin:$PATH"

export:
  wandb:
    entity: "nvidia"
    project: "nemo-evaluator-launcher-test"
    group: "exporter-testing"
    job_type: "evaluation"
    tags: ["Nemotron-H", "multi-task"]
    description: "Test Run"
    log_metrics: ["accuracy", "score", "pass@1"]
    log_mode: "multi_task"
    log_logs: true
    only_required: false
    
    extra_metadata: 
      execution_type: "slurm"
      model_tag: "nemotrons"
      hardware: "H100"
```

**Examples (CLI):**
```bash
# Basic export
nemo-evaluator-launcher export 8abcd123 --dest wandb

# With overrides
nemo-evaluator-launcher export 8abcd123 --dest wandb \
  -o export.wandb.entity=myorg \
  -o export.wandb.project=evals
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