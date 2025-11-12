(exporter-mlflow)=

# MLflow Exporter (`mlflow`)

Exports accuracy metrics and artifacts to an MLflow Tracking Server.

- **Purpose**: Centralize metrics, parameters, and artifacts in MLflow for experiment tracking
- **Requirements**: `mlflow` package installed and a reachable MLflow tracking server

:::{dropdown} **Prerequisites: MLflow Server Setup**
:open:

Before exporting results, ensure that an **MLflow Tracking Server** is running and reachable.  
If no server is active, export attempts will fail with connection errors.

### Quick Start: Local Tracking Server

For local development or testing:

```bash
# Install MLflow
pip install nemo-evaluator-launcher[mlflow]

# Start a local tracking server (runs on: http://127.0.0.1:5000)
mlflow server --host 127.0.0.1 --port 5000
```

This starts MLflow with a local SQLite backend and a file-based artifact store under current directory.

### Production Deployments

For production or multi-user setups:

* **Remote MLflow Server**: Deploy MLflow on a dedicated VM or container.
* **Docker**:

  ```bash
  docker run -p 5000:5000 ghcr.io/mlflow/mlflow:latest \
    mlflow server --host 0.0.0.0
  ```
* **Cloud-Managed Services**: Use hosted options such as **Databricks MLflow** or **AWS SageMaker MLflow**.

For detailed deployment and configuration options, see the
[official MLflow Tracking Server documentation](https://mlflow.org/docs/latest/tracking/server.html).

:::

## Usage

Export evaluation results to MLflow Tracking Server for centralized experiment management.

::::{tab-set}

:::{tab-item} Auto-Export (Recommended)

Configure MLflow export to run automatically after evaluation completes. Add MLflow configuration to your run config YAML file:

```yaml
execution:
  auto_export:
    destinations: ["mlflow"]
  
  # Export-related env vars (placeholders expanded at runtime)
  env_vars:
    export:
      MLFLOW_TRACKING_URI: MLFLOW_TRACKING_URI # or set tracking_uri under export.mflow
      PATH: "/path/to/conda/env/bin:$PATH" # set for slurm executor jobs

export:
  mlflow:
    tracking_uri: "http://mlflow.example.com:5000"
    experiment_name: "llm-evaluation"
    description: "Llama 3.1 8B evaluation"
    log_metrics: ["mmlu_score_macro", "mmlu_score_micro"]
    tags:
      model_family: "llama"
      version: "3.1"
    extra_metadata:
      hardware: "A100"
      batch_size: 32
    log_artifacts: true

target:
  api_endpoint:
    model_id: meta/llama-3.1-8b-instruct
    url: https://integrate.api.nvidia.com/v1/chat/completions

evaluation:
  tasks:
    - name: simple_evals.mmlu
```

Run the evaluation with auto-export enabled:

```bash
nemo-evaluator-launcher run --config-dir . --config-name my_config
```

:::

:::{tab-item} Manual Export (Python API)

Export results programmatically after evaluation completes:

```python
from nemo_evaluator_launcher.api.functional import export_results

# Basic MLflow export
export_results(
    invocation_ids=["8abcd123"], 
    dest="mlflow", 
    config={
        "tracking_uri": "http://mlflow:5000",
        "experiment_name": "model-evaluation"
    }
)

# Export with metadata and tags
export_results(
    invocation_ids=["8abcd123"], 
    dest="mlflow", 
    config={
        "tracking_uri": "http://mlflow:5000",
        "experiment_name": "llm-benchmarks",
        "run_name": "llama-3.1-8b-mmlu",
        "description": "Evaluation of Llama 3.1 8B on MMLU",
        "tags": {
            "model_family": "llama",
            "model_version": "3.1",
            "benchmark": "mmlu"
        },
        "log_metrics": ["accuracy"],
        "extra_metadata": {
            "hardware": "A100-80GB",
            "batch_size": 32
        }
    }
)

# Export with artifacts disabled
export_results(
    invocation_ids=["8abcd123"], 
    dest="mlflow", 
    config={
        "tracking_uri": "http://mlflow:5000",
        "experiment_name": "model-comparison",
        "log_artifacts": False
    }
)

# Skip if run already exists
export_results(
    invocation_ids=["8abcd123"], 
    dest="mlflow", 
    config={
        "tracking_uri": "http://mlflow:5000",
        "experiment_name": "nightly-evals",
        "skip_existing": True
    }
)
```

:::

:::{tab-item} Manual Export (CLI)

Export results after evaluation completes:

```shell
# Default export
nemo-evaluator-launcher export 8abcd123 --dest mlflow

# With overrides
nemo-evaluator-launcher export 8abcd123 --dest mlflow \
  -o export.mlflow.tracking_uri=http://mlflow:5000 \
  -o export.mlflow.experiment_name=my-exp

# With metric filtering
nemo-evaluator-launcher export 8abcd123 --dest mlflow --log-metrics accuracy pass@1
```

:::


::::

## Configuration Parameters

```{list-table}
:header-rows: 1
:widths: 25 15 45 15

* - Parameter
  - Type
  - Description
  - Default
* - `tracking_uri`
  - str
  - MLflow tracking server URI
  - Required if env var `MLFLOW_TRACKING_URI` is not set
* - `experiment_name`
  - str
  - MLflow experiment name
  - `"nemo-evaluator-launcher"`
* - `run_name`
  - str
  - Run display name
  - Auto-generated
* - `description`
  - str
  - Run description
  - None
* - `tags`
  - dict[str, str]
  - Custom tags for the run
  - None
* - `extra_metadata`
  - dict
  - Additional parameters logged to MLflow
  - None
* - `skip_existing`
  - bool
  - Skip export if run exists for invocation. Useful to avoid creating duplicate runs when re-exporting.
  - `false`
* - `log_metrics`
  - list[str]
  - Filter metrics by substring match
  - All metrics
* - `log_artifacts`
  - bool
  - Upload evaluation artifacts
  - `true`
* - `log_logs`
  - bool
  - Upload execution logs
  - `false`
* - `only_required`
  - bool
  - Copy only required artifacts
  - `true`
```
