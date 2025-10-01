(exporter-wandb)=

# Weights & Biases Exporter (`wandb`)

Exports accuracy metrics and artifacts to W&B. Supports either per-task runs or a single multi-task run per invocation, with artifact logging and run metadata.

- **Purpose**: Track runs, metrics, and artifacts in W&B
- **Requirements**: `wandb` installed and credentials configured

## Usage

Export evaluation results to Weights & Biases for comprehensive experiment tracking, visualization, and collaboration.

::::{tab-set}

:::{tab-item} CLI

Export results to W&B with flexible logging options:

```bash
# Basic export to W&B project
nv-eval export 8abcd123 --dest wandb \
  --config '{"entity": "myorg", "project": "model-evaluations"}'

# Export with custom run name and grouping
nv-eval export 8abcd123 --dest wandb \
  --config '{
    "entity": "myorg",
    "project": "llm-benchmarks",
    "name": "gpt-4-mmlu-eval",
    "group": "mmlu-experiments",
    "tags": {"model": "gpt-4", "benchmark": "mmlu"}
  }'

# Export multiple runs in multi-task mode with filtered metrics
nv-eval export 8abcd123 9def4567 --dest wandb \
  --config '{
    "entity": "myorg",
    "project": "model-comparison",
    "log_mode": "multi_task",
    "log_metrics": ["accuracy", "f1_score"],
    "log_artifacts": false
  }'
```

:::

:::{tab-item} Python

Export results programmatically with comprehensive W&B integration:

```python
from nemo_evaluator_launcher.api.functional import export_results

# Basic W&B export
export_results(
    ["8abcd123"], 
    dest="wandb", 
    config={
        "entity": "myorg", 
        "project": "model-evaluations"
    }
)

# Comprehensive export with metadata and organization
export_results(
    ["8abcd123"], 
    dest="wandb", 
    config={
        "entity": "myorg",
        "project": "llm-benchmarks",
        "name": "gpt-4-comprehensive-eval",
        "group": "gpt-family-comparison",
        "description": "Comprehensive evaluation of GPT-4 across multiple benchmarks",
        "tags": {
            "model_family": "gpt",
            "model_version": "4",
            "evaluation_suite": "comprehensive"
        },
        "log_mode": "per_task",  # Create separate runs for each task
        "log_metrics": ["accuracy", "precision", "recall", "f1_score", "bleu"],
        "log_artifacts": True,
        "extra_metadata": {
            "hardware": "A100-80GB",
            "batch_size": 32,
            "temperature": 0.0,
            "max_tokens": 2048
        }
    }
)

# Multi-task export for comparative analysis
export_results(
    ["8abcd123", "9def4567", "abc12345"], 
    dest="wandb", 
    config={
        "entity": "myorg",
        "project": "model-comparison",
        "log_mode": "multi_task",  # Single run with all tasks
        "name": "three-model-comparison",
        "group": "comparative-analysis",
        "description": "Comparing three different models on the same evaluation tasks",
        "log_artifacts": False,  # Metrics only for comparison
        "triggered_by_webhook": True,
        "webhook_source": "automated-pipeline"
    }
)
```

:::

::::

## Key Configuration

```{list-table}
:header-rows: 1
:widths: 25 15 45 15

* - Parameter
  - Type
  - Description
  - Default/Notes
* - `entity`
  - str
  - W&B entity
  - Required
* - `project`
  - str
  - W&B project
  - Required
* - `log_mode`
  - str
  - Logging mode: `per_task` or `multi_task`
  - `per_task`
* - `name`
  - str, optional
  - Run display name
  - Auto-generated
* - `group`
  - str, optional
  - Run group
  - None
* - `tags`
  - dict[str,str], optional
  - Run tags
  - None
* - `description`
  - str, optional
  - Run description
  - None
* - `log_metrics`
  - list[str], optional
  - Filter metrics to log
  - All metrics
* - `log_artifacts`
  - bool
  - Log artifacts in addition to metrics
  - `true`
* - `extra_metadata`
  - dict, optional
  - Additional key/value metadata
  - None
* - `triggered_by_webhook`
  - bool, optional
  - Webhook trigger flag
  - `false`
* - `webhook_source`
  - str, optional
  - Webhook source identifier
  - None
* - `source_artifact`
  - str, optional
  - Source artifact path
  - None
* - `config_source`
  - str, optional
  - Configuration source
  - None
```