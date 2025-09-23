<!-- vale off -->
<!-- cSpell:ignore MLflow Nemotron vLLM -->

# MLflow Exporter (`mlflow`)

## Overview

Exports metrics and artifacts to MLflow for experiment tracking and run management. Inherits core features from the [Local](local.md) exporter (artifact staging, multi-run, auto-export) and adds MLflow tracking.

## Key Features

- Log metrics and artifacts for each job as separate MLflow runs (per task)
- Organize runs under MLflow experiments
- Auto-export after evaluations finish (local or cluster)

## Configuration

- Required: `tracking_uri`
- `experiment_name`: MLflow experiment name (default: `nemo-evaluator-launcher`)
- `run_name`: Custom run name (default: `eval-<invocation>-<benchmark>`)
- `description`: Run description
- `skip_existing`: Safety check to prevent duplicate runs for the same invocation (default: false). MLflow runs are immutable; this setting prevents creating duplicates when re-exporting
- `log_metrics`: Filter specific metrics (default: all available metrics)
- `log_artifacts`: Include artifacts (default: true), including the run configuration file, `config.yaml`, for reproducibility
- `extra_metadata`: User-defined fields logged as MLflow parameters
- `tags`: MLflow tags for run organization
- Webhook settings: `triggered_by_webhook`, `webhook_source`, `source_artifact`, `config_source`

## Tips

Use `skip_existing: true` to avoid duplicate runs when re-exporting the same results. Use auto-export for robust configuration support.

## Examples

### YAML

```yaml
execution:
  auto_export:
    destinations: ["mlflow"]
    configs:
      mlflow:
        tracking_uri: "http://mlflow.company.com:5000"
        experiment_name: "llm-benchmarks"
        skip_existing: true
        tags:
          framework: "vLLM"
          precision: "bf16"
        log_metrics: ["accuracy", "pass@1"]
        extra_metadata:
          model_size: "8B"
          hardware: "H100"
```

### CLI

```bash
# Default export
nemo-evaluator-launcher export 8abcd123 --dest mlflow

# With metric filtering
nemo-evaluator-launcher export 8abcd123 --dest mlflow --log-metrics accuracy pass@1
```

### Python API

```python
from nemo_evaluator_launcher.api.functional import export_results

# Default export
export_results("8abcd123", dest="mlflow", config={"tracking_uri": "http://mlflow:5000"})

# Customized
export_results("8abcd123", dest="mlflow", config={
    "tracking_uri": "http://mlflow:5000",
    "experiment_name": "llm-evals",
    "skip_existing": True,
    "tags": {"model": "Nemotron", "env": "production"}
})
```
<!-- vale on -->