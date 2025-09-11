(exporter-mlflow)=

# MLflow Exporter (`mlflow`)

Exports accuracy metrics (and optionally artifacts) to an MLflow Tracking Server. Can create a run per job or a single run for an entire invocation.

- **Purpose**: Centralize metrics, parameters, and artifacts in MLflow
- **Requirements**: `mlflow` installed and a reachable tracking server

## Usage

Export evaluation results to MLflow Tracking Server for centralized experiment management and model lifecycle tracking.


::::{tab-set}

:::{tab-item} CLI

Export results to MLflow with various configuration options:

```bash
# Basic export to MLflow server
nv-eval export 8abcd123 --dest mlflow \
  --config '{"tracking_uri": "http://mlflow:5000", "experiment_name": "model-evaluation"}'

# Export with custom run name and tags
nv-eval export 8abcd123 --dest mlflow \
  --config '{
    "tracking_uri": "http://mlflow:5000",
    "experiment_name": "llm-benchmarks",
    "run_name": "gpt-4-evaluation",
    "tags": {"model": "gpt-4", "dataset": "mmlu"}
  }'

# Export multiple runs with artifact logging disabled
nv-eval export 8abcd123 9def4567 --dest mlflow \
  --config '{
    "tracking_uri": "http://mlflow:5000",
    "experiment_name": "model-comparison",
    "log_artifacts": false,
    "log_metrics": ["accuracy", "f1_score"]
  }'
```

:::

:::{tab-item} Python

Export results programmatically with comprehensive MLflow integration:

```python
from nemo_evaluator_launcher.api.functional import export_results

# Basic MLflow export
export_results(
    ["8abcd123"], 
    dest="mlflow", 
    config={
        "tracking_uri": "http://mlflow:5000",
        "experiment_name": "model-evaluation"
    }
)

# Comprehensive export with metadata and tags
export_results(
    ["8abcd123"], 
    dest="mlflow", 
    config={
        "tracking_uri": "http://mlflow:5000",
        "experiment_name": "llm-benchmarks",
        "run_name": "gpt-4-mmlu-evaluation",
        "description": "Evaluation of GPT-4 on MMLU benchmark dataset",
        "tags": {
            "model_family": "gpt",
            "model_version": "4",
            "benchmark": "mmlu",
            "evaluation_date": "2024-01-15"
        },
        "log_metrics": ["accuracy", "precision", "recall", "f1_score"],
        "log_artifacts": True,
        "extra_metadata": {
            "hardware": "A100-80GB",
            "batch_size": 32,
            "temperature": 0.0
        }
    }
)

# Export with webhook integration
export_results(
    ["8abcd123"], 
    dest="mlflow", 
    config={
        "tracking_uri": "http://mlflow:5000",
        "experiment_name": "automated-evaluations",
        "triggered_by_webhook": True,
        "webhook_source": "github-actions",
        "skip_existing": True  # Avoid duplicate runs
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
* - `tracking_uri`
  - str
  - MLflow tracking server URI
  - Required
* - `experiment_name`
  - str
  - Target experiment
  - Required
* - `run_name`
  - str, optional
  - Run display name
  - Auto-generated
* - `description`
  - str, optional
  - Run description
  - None
* - `tags`
  - dict[str,str], optional
  - Run tags
  - None
* - `extra_metadata`
  - dict, optional
  - Additional key/value metadata
  - None
* - `skip_existing`
  - bool
  - Skip if existing run for same invocation found
  - `false`
* - `log_metrics`
  - list[str], optional
  - Filter metrics to log
  - All metrics
* - `log_artifacts`
  - bool
  - Upload artifacts for each job
  - `true`
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

