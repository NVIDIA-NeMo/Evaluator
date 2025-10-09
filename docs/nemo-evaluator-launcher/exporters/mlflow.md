# MLflow Exporter (`mlflow`)

Export metrics and artifacts to MLflow for experiment tracking and run management. Inherits all core features from the [Local](local.md) exporter (artifact staging, multi-run, auto-export), and adds MLflow tracking.

**What you can do:**
- Log metrics and artifacts for each job as separate MLflow runs (per-task only)
- Organize runs under MLflow experiments 
- Auto-export after evaluations finish (local or cluster)

**Key configuration:**
- Required: `tracking_uri` (or set env var `MLFLOW_TRACKING_URI`)
- `experiment_name`: MLflow experiment name (default: `nemo-evaluator-launcher`)
- `run_name`: custom run name (default: `eval-<invocation>-<benchmark>`)
- `description`: run description text
- `skip_existing`: skip logging if a run already exists for this invocation (default: false). Useful to avoid creating duplicate runs when re-exporting; existing runs can be updated by logging again without this flag
- `log_metrics`: filter specific metrics (default: all available metrics)
- `log_artifacts`: include artifacts and config.yaml (default: true)
- `log_logs`: include execution logs (default: false)
- `only_required`: copy only required artifacts (default: true)
- `extra_metadata`: user-defined fields logged as MLflow parameters
- `tags`: MLflow tags for run organization
- Webhook-related: `triggered_by_webhook`, `webhook_source`, `source_artifact`, `config_source`

**Tip:** Use `skip_existing: true` to avoid duplicate runs when re-exporting. We recommend `auto-export` for extensive configuration support.

**Example (YAML):**
```yaml
execution:
  auto_export:
    destinations: ["mlflow"]
  
  # Export-related env vars (placeholders expanded at runtime)
  env_vars:
    export:
      MLFLOW_TRACKING_URI: MLFLOW_TRACKING_URI # or set tracking_uri under export.mflow
      PATH: "/path/to/conda/env/bin:$PATH"

export:
  mlflow:
    tracking_uri: "http://mlflow.company.com:5000" # if this is missing fallback to MLFLOW_TRACKING_URI
    experiment_name: "llm-benchmarks"
    skip_existing: true
    log_logs: true
    only_required: false
    tags:
      framework: "vLLM"
      precision: "bf16"
    log_metrics: ["accuracy", "pass@1"]
    extra_metadata:
      model_size: "8B"
      hardware: "H100"
```

**Examples (CLI):**
```bash
# Default export
nemo-evaluator-launcher export 8abcd123 --dest mlflow

# With overrides
nemo-evaluator-launcher export 8abcd123 --dest mlflow \
  -o export.mlflow.tracking_uri=http://mlflow:5000 \
  -o export.mlflow.experiment_name=my-exp

# With metric filtering
nemo-evaluator-launcher export 8abcd123 --dest mlflow --log-metrics accuracy pass@1
```

**Examples (API):**
```python
from nemo_evaluator_launcher.api.functional import export_results

# Default export
export_results("8abcd123", dest="mlflow", config={"tracking_uri": "http://mlflow:5000"})

# Customized
export_results("8abcd123", dest="mlflow", config={
    "tracking_uri": "http://mlflow:5000",
    "experiment_name": "llm-evals",
    "skip_existing": True,
    "tags": {"model": "nemotrons", "env": "production"}
})
```