# MLflow Exporter (`mlflow`)

Exports accuracy metrics (and optionally artifacts) to an MLflow Tracking Server. Can create a run per job or a single run for an entire invocation.

- **Purpose**: Centralize metrics, parameters, and artifacts in MLflow
- **Requirements**: `mlflow` installed and a reachable tracking server

See common behavior and usage in [overview](overview.md).

## Key configuration
- `tracking_uri` (str): MLflow tracking server URI (required)
- `experiment_name` (str): Target experiment.
- `run_name` (str, optional), `description` (str, optional)
- `tags` (dict[str,str], optional), `extra_metadata` (dict, optional)
- `skip_existing` (bool): Skip if an existing run for the same invocation is found
- `log_metrics` (list[str], optional): Filter metrics
- `log_artifacts` (bool): Upload artifacts for each job. Default: `true`
- Webhook (optional): `triggered_by_webhook`, `webhook_source`, `source_artifact`, `config_source`

## Examples

CLI:
```bash
nv-eval export 8abcd123 --dest mlflow -o . --output-filename processed_results.json
```

Python:
```python
from nemo_evaluator_launcher.api.functional import export_results
export_results(["8abcd123"], dest="mlflow", config={"tracking_uri": "http://mlflow:5000"})
```
