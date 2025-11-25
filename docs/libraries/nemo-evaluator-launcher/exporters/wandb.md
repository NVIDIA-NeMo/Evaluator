(exporter-wandb)=

# Weights & Biases Exporter (`wandb`)

Exports accuracy metrics and artifacts to W&B. Supports either per-task runs or a single multi-task run per invocation, with artifact logging and run metadata.

- **Purpose**: Track runs, metrics, and artifacts in W&B
- **Requirements**: `wandb` installed and credentials configured

## Usage

Export evaluation results to Weights & Biases for experiment tracking, visualization, and collaboration.

::::{tab-set}

:::{tab-item} CLI

Basic export to W&B using credentials and project settings from your evaluation configuration:

```bash
# Export to W&B (uses config from evaluation run)
nemo-evaluator-launcher export 8abcd123 --dest wandb

# Filter metrics to export specific measurements
nemo-evaluator-launcher export 8abcd123 --dest wandb --log-metrics accuracy f1_score
```

```{note}
Specify W&B configuration (entity, project, tags, etc.) in your evaluation YAML configuration file under `execution.auto_export.configs.wandb`. The CLI export command reads these settings from the stored job configuration.
```

:::

:::{tab-item} Python

Export results programmatically with W&B configuration:

```python
from nemo_evaluator_launcher.api.functional import export_results

# Basic W&B export
export_results(
    invocation_ids=["8abcd123"], 
    dest="wandb", 
    config={
        "entity": "myorg", 
        "project": "model-evaluations"
    }
)

# Export with metadata and organization
export_results(
    invocation_ids=["8abcd123"], 
    dest="wandb", 
    config={
        "entity": "myorg",
        "project": "llm-benchmarks",
        "name": "llama-3.1-8b-eval",
        "group": "llama-family-comparison",
        "description": "Evaluation of Llama 3.1 8B on benchmarks",
        "tags": ["llama-3.1", "8b"],
        "log_mode": "per_task",
        "log_metrics": ["accuracy"],
        "log_artifacts": True,
        "extra_metadata": {
            "hardware": "A100-80GB"
        }
    }
)

# Multi-task mode: single run for all tasks
export_results(
    invocation_ids=["8abcd123"], 
    dest="wandb", 
    config={
        "entity": "myorg",
        "project": "model-comparison",
        "log_mode": "multi_task",
        "log_artifacts": False
    }
)
```

:::

:::{tab-item} YAML Config

Configure W&B export in your evaluation YAML file for automatic export on completion:

```yaml
execution:
  auto_export:
    destinations: ["wandb"]
  
  # Export-related env vars (placeholders expanded at runtime)
  env_vars:
    export:
      WANDB_API_KEY: WANDB_API_KEY

export:
  wandb:
    entity: "myorg"
    project: "llm-benchmarks"
    name: "llama-3.1-8b-instruct-v1"
    group: "baseline-evals"
    tags: ["llama-3.1", "baseline"]
    description: "Baseline evaluation"
    log_mode: "multi_task"
    log_metrics: ["accuracy"]
    log_artifacts: true
    log_logs: true
    only_required: false
    
    extra_metadata:
      hardware: "H100"
      checkpoint: "path/to/checkpoint"

```

:::

::::

## Configuration Parameters

```{list-table}
:header-rows: 1
:widths: 20 15 50 15

* - Parameter
  - Type
  - Description
  - Default
* - `entity`
  - str
  - W&B entity (organization or username)
  - Required
* - `project`
  - str
  - W&B project name
  - Required
* - `log_mode`
  - str
  - Logging mode: `per_task` creates separate runs for each evaluation task; `multi_task` creates a single run for all tasks
  - `per_task`
* - `name`
  - str
  - Run display name. If not specified, auto-generated as `eval-{invocation_id}-{benchmark}` (per_task) or `eval-{invocation_id}` (multi_task)
  - Auto-generated
* - `group`
  - str
  - Run group for organizing related runs
  - Invocation ID
* - `tags`
  - list[str]
  - Tags for categorizing the run
  - None
* - `description`
  - str
  - Run description (stored as W&B notes)
  - None
* - `log_metrics`
  - list[str]
  - Metric name patterns to filter (e.g., `["accuracy", "f1"]`). Logs only metrics containing these substrings
  - All metrics
* - `log_artifacts`
  - bool
  - Whether to upload evaluation artifacts (results files, configs) to W&B
  - `true`
* - `log_logs`
  - bool
  - Upload execution logs
  - `false`
* - `only_required`
  - bool
  - Copy only required artifacts
  - `true`
* - `extra_metadata`
  - dict
  - Additional metadata stored in run config (e.g., hardware, hyperparameters)
  - `{}`
* - `job_type`
  - str
  - W&B job type classification
  - `evaluation`
```
