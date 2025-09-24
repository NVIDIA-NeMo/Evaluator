# Exporters Overview

Exporters move evaluation results and artifacts from completed runs to external destinations for experiment tracking, analysis, sharing, and reporting.


## Core Features

- **Multi-Run Support**: Export multiple invocations and job IDs in one command
- **Cross-Executor Compatibility**: Works with local and Slurm execution backends
- **Auto-Export**: Triggers exports automatically after successful evaluations
- **Multi-Destination Support**: Export to multiple destinations simultaneously using YAML configuration
- **Metric Filtering**: Include only specific metrics using `log_metrics` (default: export all available metrics)
- **Run Management**: New exports for the same task or harness override existing runs, and exports for different tasks append to multi-task runs

## Supported Exporters

| Exporter | Best For | Key Features |
|----------|----------|--------------|
| [`local`](local.md) | File-based workflows, debugging | Artifact staging, JSON and CSV summaries for single or multiple runs |
| [`wandb`](wandb.md) | Experiment tracking, collaboration | Run management, artifact logging, dashboards |
| [`mlflow`](mlflow.md) | Experiment tracking, collaboration | Run management, artifact logging, dashboards |
| [`gsheets`](gsheets.md) | Results sharing, quick analysis | Spreadsheet summaries for single or multiple runs |


## Getting Started

1. **Install**: `pip install nemo-evaluator-launcher[exporters]`
2. **Credentials Setup:**
   - **Weights & Biases (W&B)**: Run `wandb login` on the job submission machine. For noninteractive use, set the `WANDB_API_KEY` environment variable.
   - **MLflow**: Set `tracking_uri` in the exporter configuration (no environment variable required).
   - **Google Sheets**: Set `service_account_file` in the exporter configuration (optionally set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable).
   - **Local**: None
3. **Test**: `nemo-evaluator-launcher export <invocation_or_job_id> --dest <exporter-name>`


## Usage Patterns

**Manual Export (CLI)**
```bash
# Single invocation
nemo-evaluator-launcher export 8abcd123 --dest wandb

# Multiple invocations
nemo-evaluator-launcher export 8abcd123 9def4567 --dest local --format json

# Specific task only
nemo-evaluator-launcher export 8abcd123.0 --dest mlflow
```

**Manual Export (API):**
```python
from nemo_evaluator_launcher.api.functional import export_results

# Single export
export_results("8abcd123", dest="wandb", config={"entity": "myorg", "project": "evals"})

# Multiple exports
export_results(["8abcd123", "9def4567"], dest="local", config={"format": "json"})
```

**Auto-Export:** Add the following to your YAML configuration and customize
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

  # Ensure credentials are available to the job and auto-export
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

- Install `nemo-evaluator-launcher[all]` in an environment that is visible to compute nodes (for example, on a shared file system such as Lustre).
- Ensure binaries are resolvable at export time. For example, add the following YAML override:

  ```yaml
  execution:
    env_vars:
      evaluation:
        PATH: "/shared/envs/nemo/bin:$PATH"
  ```
- Credentials:
  - Weights & Biases (W&B): Either run `wandb login` on a shared home directory or pass `WANDB_API_KEY` using `execution.env_vars.evaluation`.
  - MLflow: Set `tracking_uri` in the exporter configuration and ensure that it is reachable from compute nodes (open the firewall or access control list [ACL]).
  - Google Sheets: The `service_account_file` path must be valid on the cluster, or pass `GOOGLE_APPLICATION_CREDENTIALS` using `execution.env_vars.evaluation`.


**Notes:**
- Avoid "local" export paths that point to your local machine when running on Slurm. Use cluster paths, or run a post hoc export from your local machine:

  ```bash
  nemo-evaluator-launcher export <invocation_id> --dest local --format json
  ```


## Add Your Own Exporter
It is straightforward to add a custom exporter to fit your tools:
- Inherit from `BaseExporter`
- Use `LocalExporter` to stage artifacts
- Add your custom upload and processing logic
- Register with `@register_exporter("name")`, where `name` is your exporter name

Refer to the individual exporter pages for details and examples:
- [Local](local.md)
- [Weights & Biases](wandb.md)
- [MLflow](mlflow.md)
- [Google Sheets](gsheets.md)
