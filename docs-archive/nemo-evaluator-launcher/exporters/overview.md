# Exporters Overview

Exporters move evaluation results and artifacts from completed runs to external destinations for tracking experiments, analysis, sharing and reporting.


## Core Features

- **Multi-Run Support**: Export multiple invocations and job IDs in one command
- **Cross-Executor Compatibility**: Works with local and Slurm execution backends
- **Auto-Export**: Triggers exports automatically after successful evaluations
- **Multi-Destination Support**: Export to multiple destinations simultaneously via YAML config
- **Metric Filtering**: Include only specific metrics via `log_metrics` (default: export all available metrics)
- **CLI Overrides**: Use `-o export.<dest>.key=value` to pass any supported config parameter at export time or override values from YAML configs
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
   - **WandB**: run `wandb login` or set env var `WANDB_API_KEY`
   - **MLflow**: set `export.mlflow.tracking_uri` in config or env var `MLFLOW_TRACKING_URI`
   - **GSheets**: set `export.gsheets.service_account_file` or use `GOOGLE_APPLICATION_CREDENTIALS`
   - **Local**: none
3. **Test**: `nemo-evaluator-launcher export <invocation_or_job_id> --dest <exporter-name>`


## Usage Patterns

**Manual export (CLI):**
```bash
# Single invocation
nemo-evaluator-launcher export 8abcd123 --dest wandb

# With CLI overrides
nemo-evaluator-launcher export 8abcd123 --dest mlflow \
  -o export.mlflow.tracking_uri=http://mlflow:5000 \
  -o export.mlflow.experiment_name=my-exp

# Multiple invocations with output dir
nemo-evaluator-launcher export 8abcd123 9def4567 --dest local \
  --format json --output-dir ./results

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

**Auto-export:** add to your YAML configs
```yaml
execution:
  auto_export:
    destinations: ["wandb", "mlflow", "gsheets"]
  
  # Export-related env vars (for auto-export step)
  env_vars:
    export:
      WANDB_API_KEY: WANDB_API_KEY
      MLFLOW_TRACKING_URI: MLFLOW_TRACKING_URI
      PATH: "/path/to/your/conda/env/bin:$PATH"

# Exporter configurations
export:
  wandb:
    entity: "myorg"
    project: "llm-benchmarks"
    log_mode: "multi_task"
  mlflow:
    tracking_uri: "http://mlflow.company.com:5000"
    experiment_name: "llm-evals"
  gsheets:
    spreadsheet_name: "LLM Evaluation Results"
```

## Tip: Remote Auto-Export (Slurm)

**Prerequisites on the cluster:**

- Install `nemo-evaluator-launcher[all]` in an env on shared FS (e.g., Lustre)
- Make the launcher accessible during auto-export via `execution.env_vars.export.PATH`
- Set exporter credentials as placeholders (no secrets in configs):

  ```yaml
  execution:
    env_vars:
      export:
        WANDB_API_KEY: WANDB_API_KEY
        MLFLOW_TRACKING_URI: MLFLOW_TRACKING_URI
        PATH: "/lustre/envs/eval/bin:$PATH"
  ```

- Ensure those env vars are set in your remote shell (e.g., via ~/.bashrc or job profile)

**How placeholder expansion works:**
- Config values that look like env var names (e.g., `WANDB_API_KEY: WANDB_API_KEY`) are expanded at runtime from the job's environment
- This prevents secrets from being embedded in configs or logged to MLflow/W&B
- Literal values (e.g., URLs, paths) are used as-is

**Notes:**
- Avoid "local" export paths pointing to your laptop when running on Slurm; use cluster paths or run a post-hoc export from your machine:

  ```bash
  nemo-evaluator-launcher export <invocation_id> --dest local --format json --output-dir ./results
  ```


## Add your own exporter
It's straightforward to add a custom exporter to fit your tools:
- Inherit from `BaseExporter`
- Use `LocalExporter` to stage artifacts
- Add your custom upload/processing logic
- Register with `@register_exporter("name")`

See individual exporter pages for details and examples:
- [Local](local.md)
- [Weights & Biases](wandb.md)
- [MLflow](mlflow.md)
- [Google Sheets](gsheets.md)