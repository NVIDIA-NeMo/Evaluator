# Exporters Overview

Exporters move evaluation results and artifacts from completed runs to external destinations for tracking experiments, analysis, sharing and reporting.


## Core Features

- **Multi-Run Support**: Export multiple invocations and job IDs in one command
- **Cross-Executor Compatibility**: Works with local and Slurm execution backends.
- **Auto-Export**: Triggers exports automatically after successful evaluations
- **Multi-Destination Support**: Export to multiple destinations simultaneously via YAML config
- **Metric Filtering**: Include only specific metrics via `log_metrics` (default: export all available metrics)
- **Run Management**: New exports for same task/harness override existing runs; different tasks append to multi-task runs

## Supported Exporters

| Exporter | Best for | Key features |
|----------|----------|--------------|
| [`local`](local.md) | File-based workflows, debugging | Artifacts staging, single/multi-runs JSON/CSV summaries|
| [`wandb`](wandb.md) | Experiment tracking, collaboration | Run management, artifact logging, dashboards |
| [`mlflow`](mlflow.md) | Experiment tracking, collaboration | Run management, artifact logging, dashboards |
| [`gsheets`](gsheets.md) | Results sharing, quick analysis | Single/multi-runs spreadsheets summaries|


## Getting Started

1. **Install**: `pip install nemo-evaluator-launcher[exporters]`
2. **Credentials setup:**
   - **WandB**: run `wandb login` on the job submission machine; for non-interactive use set env var `WANDB_API_KEY`
   - **MLflow**: set `tracking_uri` in the exporter config (no env var required)
   - **GSheets**: set `service_account_file` in the exporter config (optional: use env var `GOOGLE_APPLICATION_CREDENTIALS`)
   - **Local**: none
3. **Test**: `nemo-evaluator-launcher export <invocation_or_job_id> --dest <exporter-name>`


## Usage Patterns

**Manual export (CLI):**
```bash
# Single invocation
nemo-evaluator-launcher export 8abcd123 --dest wandb

# Multiple invocations
nemo-evaluator-launcher export 8abcd123 9def4567 --dest local --format json

# Specific task only
nemo-evaluator-launcher export 8abcd123.0 --dest mlflow
```

**Manual export (API):**
```python
from nemo_evaluator_launcher.api.functional import export_results

# Single export
export_results("8abcd123", dest="wandb", config={"entity": "myorg", "project": "evals"})

# Multiple exports
export_results(["8abcd123", "9def4567"], dest="local", config={"format": "json"})
```

**Auto-export:** add to your YAML configs and customize
```yaml
# Auto-export config under execution
execution:
  auto_export:
    destinations: ["wandb", "local", "gsheets"]
    configs:
      wandb:
        entity: "myorg"
        project: "llm-benchmarks"
      local:
        format: "json"
        output_dir: "./results"
      gsheets:
        spreadsheet_name: "LLM Evaluation Results"

  # Ensure credentials available to job/auto-export
  env_vars:
    evaluation:
      WANDB_API_KEY: WANDB_API_KEY
      # Optional for on-prem W&B:
      # WANDB_BASE_URL: WANDB_BASE_URL
      # Optional GSheets env alternative:
      # GOOGLE_APPLICATION_CREDENTIALS: GOOGLE_APPLICATION_CREDENTIALS
```

## Tip: Remote Auto-Export (Slurm)

**Prerequisites on the cluster:**

- Install `nemo-evaluator-launcher[all]` in an env visible to compute nodes (on shared FS such as Lustre).
- Ensure binaries are resolvable at export time. Example YAML override:

  ```yaml
  execution:
    env_vars:
      evaluation:
        PATH: "/shared/envs/nemo/bin:$PATH"
  ```
- Credentials:
  - WandB: either `wandb login` on shared HOME or pass `WANDB_API_KEY` via `execution.env_vars.evaluation`
  - MLflow: set `tracking_uri` in exporter config and ensure it's reachable from compute nodes (open firewall/ACL)
  - GSheets: `service_account_file` must be a valid cluster path or pass `GOOGLE_APPLICATION_CREDENTIALS` via `execution.env_vars.evaluation`


**Notes:**
- Avoid “local” export paths that point to your laptop when running on Slurm; use cluster paths, or run a post‑hoc export from your laptop:
  - `nemo-evaluator-launcher export <invocation_id> --dest local --config '{"output_dir": "/your/local/path","format":"json"}'`

  ```bash
  nemo-evaluator-launcher export <invocation_id> --dest local --format json
  ```


## Add your own exporter
It’s straightforward to add a custom exporter to fit your tools:
- Inherit from `BaseExporter`
- Use `LocalExporter` to stage artifacts
- Add your custom upload/processing logic
- Register with `@register_exporter("name")` while `name` is your new custom exporter

See individual exporter pages for details and examples:
- [Local](local.md)
- [Weights & Biases](wandb.md)
- [MLflow](mlflow.md)
- [Google Sheets](gsheets.md)
